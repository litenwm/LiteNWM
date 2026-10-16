# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse
import csv
import json
import math
import yaml
import os
import sys
import time
import random
import pickle
import numpy as np
import lpips
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from diffusers.models import AutoencoderKL

### evo evaluation library ###
from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync, metrics
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation

from diffusion import create_diffusion
from datasets import TrajectoryEvalDataset
from isolated_nwm_infer import encode_rgb_to_latent, model_forward_wrapper, model_forward_wrapper_from_latents
from misc import calculate_delta_yaw, get_action_torch, save_planning_pred, log_viz_single, transform, unnormalize_data
from isolated_nwm_eval import save_metric_to_disk
import distributed as dist
from models import CDiT_models
from scripts.teacher_baseline.teacher_collection_v2_utils import candidate_action_hashes, candidate_set_hash


with open("config/data_config.yaml", "r") as f:
    data_config = yaml.safe_load(f)

with open("config/data_hyperparams_plan.yaml", "r") as f:
    data_hyperparams = yaml.safe_load(f)

ACTION_STATS_TORCH = {}
for key in data_config['action_stats']:
    ACTION_STATS_TORCH[key] = torch.tensor(data_config['action_stats'][key])

def load_checkpoint_portable(checkpoint_path):
    import os
    import pathlib
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)

def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state

def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])

def wrap_to_pi(angle):
    return torch.remainder(angle + np.pi, 2 * np.pi) - np.pi

def compute_action_yaw_smoothness_penalty(deltas):
    actions = deltas[..., :2].float()
    if actions.shape[1] < 2:
        return torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)

    headings = torch.atan2(actions[..., 1], actions[..., 0])
    delta_heading = wrap_to_pi(headings[:, 1:] - headings[:, :-1])

    eps = torch.tensor(1e-6, device=actions.device, dtype=actions.dtype)
    norms = torch.linalg.norm(actions, dim=-1)
    valid = (norms[:, 1:] > eps) & (norms[:, :-1] > eps)
    delta_heading = torch.where(valid, delta_heading, torch.zeros_like(delta_heading))
    denom = valid.sum(dim=1).clamp_min(1).to(delta_heading.dtype)
    return (delta_heading ** 2).sum(dim=1) / denom


def tensor_stats_dict(tensor):
    tensor = tensor.detach().float().reshape(-1).cpu()
    if tensor.numel() == 0:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
    }


def add_action_feature_columns(row, candidate_action, baseline_action):
    cand = candidate_action.detach().float()
    base = baseline_action.detach().float()
    delta = cand - base
    cand_cpu = cand.detach().cpu()
    base_cpu = base.detach().cpu()
    delta_cpu = delta.detach().cpu()
    steps = int(cand_cpu.shape[0])
    dims = int(cand_cpu.shape[1])
    for t in range(steps):
        for d in range(dims):
            row[f"candidate_action_t{t}_d{d}"] = float(cand_cpu[t, d])
            row[f"fast_action_t{t}_d{d}"] = float(base_cpu[t, d])
            row[f"action_delta_t{t}_d{d}"] = float(delta_cpu[t, d])
    row["candidate_action_norm"] = float(torch.linalg.vector_norm(cand).detach().cpu())
    row["fast_action_norm"] = float(torch.linalg.vector_norm(base).detach().cpu())
    row["action_delta_norm"] = float(torch.linalg.vector_norm(delta).detach().cpu())
    row["action_delta_xy_norm"] = float(torch.linalg.vector_norm(delta[:, :2]).detach().cpu())
    row["action_delta_yaw_norm"] = float(torch.linalg.vector_norm(delta[:, 2]).detach().cpu())
    row["action_delta_abs_mean"] = float(delta.abs().mean().detach().cpu())
    row["action_delta_abs_max"] = float(delta.abs().max().detach().cpu())
    row["action_delta_xy_abs_mean"] = float(delta[:, :2].abs().mean().detach().cpu())
    row["action_delta_yaw_abs_mean"] = float(delta[:, 2].abs().mean().detach().cpu())
    row["action_delta_yaw_abs_max"] = float(delta[:, 2].abs().max().detach().cpu())
    row["final_action_delta_norm"] = float(torch.linalg.vector_norm(delta[-1]).detach().cpu())
    row["final_xy_delta_norm"] = float(torch.linalg.vector_norm(delta[-1, :2]).detach().cpu())
    row["final_yaw_delta_abs"] = float(delta[-1, 2].abs().detach().cpu())
    row["candidate_action_smoothness"] = float(torch.linalg.vector_norm(cand[1:] - cand[:-1]).detach().cpu())
    row["fast_action_smoothness"] = float(torch.linalg.vector_norm(base[1:] - base[:-1]).detach().cpu())
    row["action_delta_smoothness"] = float(torch.linalg.vector_norm(delta[1:] - delta[:-1]).detach().cpu())
    row["candidate_yaw_abs_mean"] = float(cand[:, 2].abs().mean().detach().cpu())
    row["fast_yaw_abs_mean"] = float(base[:, 2].abs().mean().detach().cpu())
    row["candidate_xy_abs_mean"] = float(cand[:, :2].abs().mean().detach().cpu())
    row["fast_xy_abs_mean"] = float(base[:, :2].abs().mean().detach().cpu())


def apply_planner_final_yaw_policy(final_yaw, policy):
    if policy == "legacy":
        return final_yaw
    if policy == "clip":
        return final_yaw.clamp(-np.pi, np.pi)
    if policy == "wrap":
        return torch.atan2(torch.sin(final_yaw), torch.cos(final_yaw))
    if policy == "zero":
        return torch.zeros_like(final_yaw)
    raise ValueError(f"Unsupported planner_final_yaw_policy={policy}")

class SpatialCNNScorer(nn.Module):
    def __init__(self, action_dim, cnn_channels=32, action_embed_dim=64, dropout=0.1):
        super().__init__()
        self.latent_encoder = nn.Sequential(
            nn.Conv2d(16, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        latent_embed_dim = cnn_channels * 2
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(action_embed_dim, action_embed_dim),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(latent_embed_dim + action_embed_dim, action_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(action_embed_dim, 1),
        )

    def forward(self, pred_latent_final, goal_latent, actions):
        latent_diff = pred_latent_final - goal_latent
        spatial = torch.cat([
            pred_latent_final,
            goal_latent,
            latent_diff,
            latent_diff.abs(),
        ], dim=1).float()
        latent_embedding = self.latent_encoder(spatial)
        action_embedding = self.action_encoder(actions.float().flatten(1))
        return self.scorer(torch.cat([latent_embedding, action_embedding], dim=1)).squeeze(-1)

def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

def load_latent_scorer_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)

def mean_or_none(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None

def plot_images_with_losses(preds, losses, save_path="predictions_with_losses.png"):
    # Denormalize images from [-1, 1] to [0, 1]
    preds = (preds + 1) / 2
    ncol = int(preds.size(0)**0.5)
    nrow = preds.size(0) // ncol
    if ncol * nrow < preds.size(0):
        nrow += 1
    grid_img = vutils.make_grid(preds, nrow=ncol, padding=2)
    np_grid = grid_img.to(torch.float32).permute(1, 2, 0).cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(50, 50))
    ax.imshow(np_grid)
    ax.axis("off")

    img_height, img_width = np_grid.shape[0] // nrow, np_grid.shape[1] // ncol

    # Overlay the losses on each image
    for idx, loss in enumerate(losses):
        row = idx // ncol
        col = idx % ncol
        x = col * img_width
        y = row * img_height
        if idx == 0:
            text = f"GT Goal"
        else:
            text = f"ID: {idx - 1}  Loss: {loss:.2f}"
        ax.text(x + img_width / 2, y + 15, text, color="white", 
                ha="center", va="top", fontsize=50, backgroundcolor="black")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_batch_final(init_imgs, pred_imgs, goal_imgs, idxs, losses, save_path="final_plan.png"):
    # images are (B, c, h, w)
    imgs_for_plotting = torch.cat([init_imgs, pred_imgs, goal_imgs])
    imgs_for_plotting = (imgs_for_plotting + 1) / 2
    ncol = init_imgs.shape[0]
    grid_img = vutils.make_grid(imgs_for_plotting, nrow=ncol, padding=2)
    np_grid = grid_img.to(torch.float32).permute(1, 2, 0).cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(ncol * 10, 30))  # Adjust size as needed
    ax.imshow(np_grid)
    ax.axis("off")

    img_height, img_width = np_grid.shape[0] // 3, np_grid.shape[1] // ncol

    # Overlay the IDs and losses on each image pair in the grid
    for i in range(ncol):
        x = i * img_width
        y_pred = img_height
        ax.text(x + img_width / 2, y_pred + 15, f"ID: {int(idxs[i].item())} Loss: {losses[i]:.2f}", 
                color="white", ha="center", va="top", fontsize=40, backgroundcolor="black")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def get_dataset_eval(config, dataset_name, predefined_index=True, navigation_eval_index=None):
    data_config = config["eval_datasets"][dataset_name]
    if navigation_eval_index is not None:
        predefined_index = navigation_eval_index
    elif predefined_index:
        predefined_index = f"data_splits/{dataset_name}/test/navigation_eval.pkl"
    else:
        predefined_index = None

    dataset = TrajectoryEvalDataset(
                data_folder=data_config["data_folder"],
                data_split_folder=data_config["test"],
                dataset_name=dataset_name,
                image_size=config["image_size"],
                min_dist_cat=config["trajectory_eval_distance"]["min_dist_cat"],
                max_dist_cat=config["trajectory_eval_distance"]["max_dist_cat"],
                len_traj_pred=config["trajectory_eval_len_traj_pred"],
                traj_stride=config["traj_stride"], 
                context_size=config["trajectory_eval_context_size"],
                normalize=config["normalize"],
                transform=transform,
                predefined_index=predefined_index,
                traj_names="rollout_traj_names.txt"
            )
    
    return dataset

class WM_Planning_Evaluator:
    def __init__(self, args):
        super().__init__()  
        self.args = args
        self.exp = args.exp
        _, _, device, distributed_enabled = dist.init_distributed()
        self.device = torch.device(device)
        
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        
        # Setting up Config
        # self.exp_eval = f'{self.exp}_nomad_eval' # local paths etc.
        self.exp_eval = self.exp
        self.get_eval_name()

        with open("config/eval_config.yaml", "r") as f:
            default_config = yaml.safe_load(f)
        self.config = default_config

        with open(self.exp_eval, "r") as f:
            user_config = yaml.safe_load(f)
        self.config.update(user_config)
        if self.args.trajectory_eval_horizon is not None:
            if self.args.trajectory_eval_horizon < 1:
                raise ValueError("--trajectory_eval_horizon must be >= 1")
            self.config["trajectory_eval_len_traj_pred"] = self.args.trajectory_eval_horizon
        if self.args.eval_data_folder is not None:
            for dataset_name in self.args.datasets.split(","):
                self.config["eval_datasets"][dataset_name]["data_folder"] = self.args.eval_data_folder


        latent_size = self.config['image_size'] // 8
        self.latent_size = self.config['image_size'] // 8
        self.num_cond = self.config['eval_context_size']
        
        # logging directory
        if self.args.save_preds:
            exp_name = os.path.basename(self.args.exp).split('.')[0]
            self.args.save_output_dir = os.path.join(args.output_dir, exp_name)
            os.makedirs(self.args.save_output_dir, exist_ok=True)
                
        # Loading Datasets
        self.dataset_names = self.args.datasets.split(',')
        self.datasets = {}
        for dataset_name in self.dataset_names:
            dataset_val = get_dataset_eval(
                self.config,
                dataset_name,
                predefined_index=True,
                navigation_eval_index=self.args.navigation_eval_index,
            )
            
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)

            curr_data_loader = torch.utils.data.DataLoader(
                                dataset_val, sampler=sampler_val,
                                batch_size=self.args.batch_size,
                                num_workers=self.args.num_workers,
                                pin_memory=True,
                                drop_last=False
                            )
            self.datasets[dataset_name] = curr_data_loader
        
        # Loading Model
        print("loading")
        model = CDiT_models[self.config['model']](
            context_size=self.num_cond,
            input_size=latent_size,
        )

        checkpoint_path = f'{self.config["results_dir"]}/{self.config["run_name"]}/checkpoints/{args.ckp}.pth.tar'
        ckp = load_checkpoint_portable(checkpoint_path)
        model.load_state_dict(ckp["ema"], strict=True)
        model.eval()
        model.to(self.device)
        self.model = model  # torch.compile disabled (no Triton on Windows)
        print(f"Using diffusion_steps={args.diffusion_steps}")
        self.diffusion = create_diffusion(str(args.diffusion_steps))
        self.diffusion.use_ddim = True  # use DDIM for faster inference
        self.fast_diffusion = None
        self.rerank_diffusion = None
        self._diagnostic_diffusions = {}
        if args.planner_variant in [
            "fast_latent",
            "hybrid_latent_rerank",
            "coarse_to_fine_latent",
            "adaptive_latent_rerank",
            "adaptive_two_gate_low_high",
        ]:
            self.fast_diffusion = create_diffusion(str(args.fast_diffusion_steps))
            self.fast_diffusion.use_ddim = True
        if args.planner_variant in ["hybrid_latent_rerank", "coarse_to_fine_latent", "adaptive_latent_rerank"]:
            self.rerank_diffusion = create_diffusion(str(args.rerank_diffusion_steps))
            self.rerank_diffusion.use_ddim = True
            self._diagnostic_diffusions[int(args.rerank_diffusion_steps)] = self.rerank_diffusion
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
        if distributed_enabled:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device],
                find_unused_parameters=False,
            )
            self.model_without_ddp = self.model.module
        else:
            self.model_without_ddp = self.model

        self.score_mode = self.args.score_mode
        self.loss_fn = None
        needs_lpips = (
            self.score_mode == "lpips"
            or self.args.rerank_score_mode == "lpips"
            or self.args.compare_score_modes
        )
        if needs_lpips:
            self.loss_fn = lpips.LPIPS(net='alex').to(self.device)
        self.latent_scorer = None
        self.latent_scorer_device = self.device
        if self.score_mode == "learned_spatial_scorer":
            self.load_latent_scorer()
        self.score_rows = []
        self.yaw_debug_rows = []
        self._last_yaw_debug = None
        self.summary_rows = []
        self.eval_timing = self.new_eval_timing()
        self.route2_timing = self.new_route2_timing()
        self.route2_cache_stats = self.new_route2_cache_stats()
        self.adaptive_stats = self.new_adaptive_stats()
        self.adaptive_two_gate_stats = self.new_adaptive_two_gate_stats()
        self.adaptive_policy = None
        if args.planner_variant == "adaptive_two_gate_low_high":
            self.load_adaptive_two_gate_policy()
        self.student_rerank_model = None
        self.student_rerank_feature_mean = None
        self.student_rerank_feature_std = None
        self.student_rerank_feature_indices = None
        self.student_rerank_device = self.device
        self.last_student_rerank_score_debug = {}
        self.top5_boundary_gate_model = None
        self.top5_boundary_gate_feature_mean = None
        self.top5_boundary_gate_feature_std = None
        self.top5_boundary_gate_args = {}
        self.b18_challenger_model = None
        self.b18_challenger_feature_mean = None
        self.b18_challenger_feature_std = None
        self.b18_challenger_args = {}
        self.b48_outcome_student = None
        self.b48_outcome_feature_cols = None
        self.b48_outcome_policy = {}
        self.b48_dynamic_policy = None
        self.s_combined_temporal_state = None
        self.s_groupfirst_ensemble_state = None
        self.s_hybrid_policy = {}
        self.learned_student_rerank_backends = {"student", "B3_student", "zero_residual_student"}
        self.b18_challenger_backends = {"b18_direct_challenger", "b51_safegate"}
        self.b48_outcome_backends = {"b48_outcome_student"}
        self.s_combined_temporal_backends = {"s_combined_temporal", "s_groupfirst_ensemble", "s_hybrid_groupfirst_combined"}
        self.top5_boundary_backends = {"top5_boundary_gate"}
        self.force_fast_rank_backends = {"force_fast_rank"}
        self.shortcut_rerank_backends = (
            self.learned_student_rerank_backends
            | self.b18_challenger_backends
            | self.b48_outcome_backends
            | self.s_combined_temporal_backends
            | self.top5_boundary_backends
            | self.force_fast_rank_backends
            | {"identity_fast_loss"}
        )
        self.diffusion_rerank_backends = {"diffusion", "teacher_diffusion_rerank"}
        if args.rerank_backend in self.learned_student_rerank_backends or args.student_rerank_debug_compare_teacher:
            rng_state = capture_rng_state()
            self.load_student_reranker()
            restore_rng_state(rng_state)
        if args.rerank_backend in self.b18_challenger_backends:
            rng_state = capture_rng_state()
            self.load_b18_direct_challenger()
            restore_rng_state(rng_state)
        if args.rerank_backend in self.b48_outcome_backends:
            rng_state = capture_rng_state()
            self.load_b18_direct_challenger()
            self.load_b48_outcome_student()
            restore_rng_state(rng_state)
        if args.rerank_backend in self.s_combined_temporal_backends:
            rng_state = capture_rng_state()
            self.load_b18_direct_challenger()
            if args.rerank_backend == "s_groupfirst_ensemble":
                self.load_s_groupfirst_ensemble_student()
            elif args.rerank_backend == "s_hybrid_groupfirst_combined":
                self.load_s_groupfirst_ensemble_student()
                self.load_s_combined_temporal_student()
                if getattr(args, "s_hybrid_policy_json", None):
                    self.s_hybrid_policy = json.loads(args.s_hybrid_policy_json)
                else:
                    self.s_hybrid_policy = {}
            else:
                self.load_s_combined_temporal_student()
            restore_rng_state(rng_state)
        if args.rerank_backend in self.top5_boundary_backends:
            rng_state = capture_rng_state()
            self.load_top5_boundary_gate()
            restore_rng_state(rng_state)
        self.top_m_recall_rows = []
        self.last_top_m_recall = None
        self.coarse_to_fine_rows = []
        self.student_rerank_debug_rows = []
        self.teacher_decision_records = []
        self.latent_rerank_distill_records = []
        self.latent_rerank_distill_completed_eval_indices = set()
        self.teacher_collection_v2_records = []
        self.fixed_candidate_records = self.load_fixed_candidate_records(args.fixed_candidate_actions_path)
        self.candidate_actions_export_records = []
        if args.export_latent_rerank_distill_dataset and args.latent_rerank_distill_resume:
            resume_path = self.latent_rerank_distill_output_path()
            if resume_path and os.path.isfile(resume_path):
                tmp_residual = resume_path + ".tmp"
                if os.path.isfile(tmp_residual):
                    os.remove(tmp_residual)
                existing_records = torch.load(resume_path, map_location="cpu", weights_only=False)
                if not isinstance(existing_records, list):
                    raise ValueError(f"Cannot resume from non-list latent rerank export: {resume_path}")
                self.latent_rerank_distill_records = existing_records
                self.latent_rerank_distill_completed_eval_indices = {
                    int(record["eval_index"])
                    for record in existing_records
                    if "eval_index" in record
                }
                print(
                    "Resuming latent rerank distillation export from "
                    f"{resume_path}; loaded {len(existing_records)} records, "
                    f"{len(self.latent_rerank_distill_completed_eval_indices)} completed eval_index values."
                )
        self.compare_rows = []
        self.compare_correlations = []
        self.compare_agreement_metrics = []
        self.planner_runtimes = {"fast_shortlist_time": 0.0, "rerank_time": 0.0}
        self.compare_runtimes = {
            "rgb_rollout_decode_time": 0.0,
            "lpips_score_time": 0.0,
            "latent_rollout_time": 0.0,
            "latent_l2_score_time": 0.0,
            "latent_cosine_score_time": 0.0,
            "goal_latent_encode_time": 0.0,
        }
        print(f"Planning score_mode: {self.score_mode}")
        print(
            f"planner_variant={args.planner_variant}, "
            f"fast_diffusion_steps={args.fast_diffusion_steps}, "
            f"fast_score_mode={args.fast_score_mode}, "
            f"rerank_diffusion_steps={args.rerank_diffusion_steps}, "
            f"rerank_top_m={args.rerank_top_m}, "
            f"rerank_score_mode={args.rerank_score_mode}, "
            f"adaptive_margin_threshold={args.adaptive_margin_threshold}, "
            f"adaptive_fallback_top_m={args.adaptive_fallback_top_m}, "
            f"adaptive_rerank_steps_by_margin={args.adaptive_rerank_steps_by_margin}, "
            f"adaptive_rerank_low_steps={args.adaptive_rerank_low_steps}, "
            f"adaptive_rerank_high_steps={args.adaptive_rerank_high_steps}, "
            f"adaptive_rerank_margin_threshold={args.adaptive_rerank_margin_threshold}, "
            f"diagnose_compare_rerank_steps={args.diagnose_compare_rerank_steps}, "
            f"adaptive_policy_path={args.adaptive_policy_path}, "
            f"export_latent_rerank_distill_dataset={args.export_latent_rerank_distill_dataset}, "
            f"export_teacher_collection_v2={args.export_teacher_collection_v2}, "
            f"latent_rerank_distill_teacher_steps={args.latent_rerank_distill_teacher_steps}, "
            f"rerank_backend={args.rerank_backend}, "
            f"student_rerank_checkpoint={args.student_rerank_checkpoint}"
        )
        print(f"yaw_penalty_weight={args.yaw_penalty_weight}")
        if self.args.compare_score_modes:
            print("Comparing score modes on the same sampled candidate actions; CEM updates use LPIPS ranking.")
        if self.score_mode != 'lpips':
            print("Using decoder-free latent-space scoring; VAE decode is only used when --decode_for_debug is set.")
        self.mode = 'cem' # assume CEM for planning
        self.num_samples = self.args.num_samples
        self.topk = self.args.topk
        self.opt_steps = self.args.opt_steps
        self.num_repeat_eval = self.args.num_repeat_eval
        self.action_dim = 3 # hardcoded (delta_x, delta_y, delta_yaw)

    def new_route2_timing(self):
        return {
            "candidate_generation_time": 0.0,
            "goal_encode_time": 0.0,
            "context_encode_time": 0.0,
            "coarse_rollout_time": 0.0,
            "coarse_scoring_time": 0.0,
            "top_m_selection_time": 0.0,
            "rerank_rollout_time": 0.0,
            "rerank_scoring_time": 0.0,
            "student_feature_time": 0.0,
            "student_inference_time": 0.0,
            "student_total_rerank_time": 0.0,
            "student_rerank_count": 0,
            "student_teacher_regret_sum": 0.0,
            "student_teacher_regret_count": 0,
            "student_teacher_agreement_count": 0,
            "student_teacher_compare_count": 0,
            "skipped_diffusion_rerank": False,
            "final_rollout_time": 0.0,
            "final_scoring_time": 0.0,
            "total_planning_time": 0.0,
        }

    def new_eval_timing(self):
        return {
            "rollout_time": 0.0,
            "decode_time": 0.0,
            "scoring_time": 0.0,
            "wrapper_setup_time": 0.0,
            "noise_initialization_time": 0.0,
            "ddim_loop_total_time": 0.0,
            "p_mean_variance_time": 0.0,
            "cdit_forward_time": 0.0,
            "timestep_tensor_setup_time": 0.0,
            "conditioning_setup_time": 0.0,
            "total_model_forward_wrapper_time": 0.0,
        }

    def new_route2_cache_stats(self):
        return {
            "goal_latent_cached_used": 0,
            "context_latent_cached_used": 0,
        }

    def new_adaptive_stats(self):
        return {
            "adaptive_early_exit_count": 0,
            "adaptive_mid_rerank_count": 0,
            "adaptive_high_fallback_count": 0,
            "adaptive_norm_margin_sum": 0.0,
            "adaptive_norm_margin_count": 0,
            "adaptive_rerank_norm_margin_sum": 0.0,
            "adaptive_rerank_norm_margin_count": 0,
            "adaptive_effective_step_equiv": 0.0,
        }

    def normalized_loss_margin(self, losses):
        sorted_loss = torch.sort(losses).values
        if sorted_loss.numel() < 2:
            return torch.tensor(float("inf"), device=losses.device, dtype=losses.dtype)
        raw_margin = sorted_loss[1] - sorted_loss[0]
        return raw_margin / (torch.std(losses) + 1e-8)

    def get_rerank_diffusion(self, steps):
        steps = int(steps)
        if steps not in self._diagnostic_diffusions:
            diffusion = create_diffusion(str(steps))
            diffusion.use_ddim = True
            self._diagnostic_diffusions[steps] = diffusion
        return self._diagnostic_diffusions[steps]

    def tensor_float_list(self, values):
        return [float(x) for x in values.detach().float().cpu().tolist()]

    def tensor_int_list(self, values):
        return [int(x) for x in values.detach().cpu().tolist()]

    def needs_teacher_candidate_fields(self):
        return (
            self.args.export_latent_rerank_distill_dataset
            or self.args.export_teacher_collection_v2
            or self.args.collect_teacher_decisions
            or self.args.rerank_backend in self.b18_challenger_backends
        )

    def needs_teacher_rollout_rng(self):
        return (
            self.args.export_latent_rerank_distill_dataset
            or self.args.export_teacher_collection_v2
        )

    def teacher_to_cpu(self, value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, np.ndarray):
            return torch.as_tensor(value).detach().cpu()
        if isinstance(value, (list, tuple)):
            return [self.teacher_to_cpu(x) for x in value]
        return value

    def teacher_latent_to_cpu_fp16(self, value):
        if value is None or not torch.is_tensor(value):
            return None
        return value.detach().to(device="cpu", dtype=torch.float16)

    def teacher_latent_fields(
        self,
        cur_goal_latent=None,
        cur_obs_latents=None,
        fast_pred_final_latents=None,
        shortlist_indices=None,
        rerank_pred_final_latents=None,
        fast_losses=None,
    ):
        if not self.args.teacher_record_latents:
            return {}
        goal_latent = None
        if cur_goal_latent is not None and cur_goal_latent.ndim == 4 and cur_goal_latent.shape[0] > 0:
            goal_latent = cur_goal_latent[0]

        context_latents = None
        context_latent_final = None
        if cur_obs_latents is not None and cur_obs_latents.ndim == 5 and cur_obs_latents.shape[0] > 0:
            context_latents = cur_obs_latents[0]
            context_latent_final = cur_obs_latents[0, -1]

        shortlist_fast_pred_final_latents = None
        if fast_pred_final_latents is not None and shortlist_indices is not None:
            shortlist_idx = shortlist_indices.to(fast_pred_final_latents.device)
            shortlist_fast_pred_final_latents = fast_pred_final_latents[shortlist_idx]

        return {
            "goal_latent": self.teacher_latent_to_cpu_fp16(goal_latent),
            "context_latent_final": self.teacher_latent_to_cpu_fp16(context_latent_final),
            "context_latents": self.teacher_latent_to_cpu_fp16(context_latents),
            "fast_pred_final_latents": self.teacher_latent_to_cpu_fp16(fast_pred_final_latents),
            "shortlist_fast_pred_final_latents": self.teacher_latent_to_cpu_fp16(shortlist_fast_pred_final_latents),
            "rerank_pred_final_latents": self.teacher_latent_to_cpu_fp16(rerank_pred_final_latents),
            "latent_cosine_fast_losses": fast_losses if self.args.fast_score_mode == "latent_cosine" else None,
        }

    def record_teacher_decision(self, record_dict):
        if not self.args.collect_teacher_decisions:
            return
        max_records = self.args.teacher_decision_max_records
        if max_records is not None and len(self.teacher_decision_records) >= max_records:
            return
        record_dict = dict(record_dict)
        record_dict["record_id"] = len(self.teacher_decision_records)
        self.teacher_decision_records.append({
            key: self.teacher_to_cpu(value)
            for key, value in record_dict.items()
        })

    def save_teacher_decisions(self):
        if not self.args.collect_teacher_decisions:
            return
        if dist.get_rank() != 0:
            return
        output_path = self.args.teacher_decision_output_path
        if output_path is None:
            output_path = os.path.join(
                "eval_output",
                "teacher_decisions",
                f"teacher_decisions_seed{self.args.seed}.pt",
            )
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save(self.teacher_decision_records, output_path)
        print(f"Saved teacher decision records to {output_path}")

        csv_path = os.path.splitext(output_path)[0] + ".csv"
        scalar_keys = [
            "record_id", "dataset_name", "batch_idx", "eval_index", "idx", "seed",
            "cem_step", "is_final_cem_step",
            "planner_variant", "score_mode", "fast_score_mode", "rerank_score_mode",
            "fast_diffusion_steps", "rerank_diffusion_steps", "rerank_top_m",
            "adaptive_rerank_steps_by_margin", "adaptive_rerank_low_steps",
            "adaptive_rerank_high_steps", "adaptive_rerank_margin_threshold",
            "selected_candidate_index", "selected_shortlist_rank", "adaptive_mode",
            "normalized_margin", "raw_margin", "effective_step_equiv",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=scalar_keys)
            writer.writeheader()
            for record in self.teacher_decision_records:
                writer.writerow({
                    key: record.get(key)
                    for key in scalar_keys
                    if not torch.is_tensor(record.get(key))
                    and not isinstance(record.get(key), (list, tuple, dict))
                })
        print(f"Saved teacher decision scalar summary to {csv_path}")

    def record_latent_rerank_distill_example(
        self,
        dataset_name,
        batch_idx,
        traj_id,
        cem_step,
        is_final_cem_step,
        deltas,
        cur_obs_image,
        cur_goal_image,
        cur_obs_latents,
        cur_goal_latent,
        fast_losses,
        fast_pred_final_latents,
        fast_sorted_indices,
        normalized_margin,
        raw_margin,
        active_selected_candidate_index,
        rollout_rng_state=None,
        planner_rerank_candidate_indices=None,
        planner_rerank_losses=None,
        planner_rerank_selected_local_index=None,
        planner_rerank_selected_original_index=None,
    ):
        if not self.args.export_latent_rerank_distill_dataset:
            return
        if not self.args.latent_rerank_distill_all_cem_steps and not is_final_cem_step:
            return
        max_records = self.args.latent_rerank_distill_max_records
        if max_records is not None and len(self.latent_rerank_distill_records) >= max_records:
            return
        if self.args.latent_rerank_distill_score_mode == "lpips":
            raise ValueError("Latent rerank distillation export does not support LPIPS/RGB decoding.")
        required = {
            "deltas": deltas,
            "cur_obs_latents": cur_obs_latents,
            "cur_goal_latent": cur_goal_latent,
            "fast_losses": fast_losses,
            "fast_pred_final_latents": fast_pred_final_latents,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "--export_latent_rerank_distill_dataset requires latent rerank candidate fields; "
                f"missing: {missing}. Use a latent rerank planner such as coarse_to_fine_latent."
            )
        if deltas.ndim != 3 or deltas.shape[-1] != 3:
            raise ValueError(f"candidate actions must have shape [K,T,3], got {tuple(deltas.shape)}")
        if fast_losses.ndim != 1 or fast_losses.shape[0] != deltas.shape[0]:
            raise ValueError(f"fast_losses must have shape [K], got {tuple(fast_losses.shape)}")
        if fast_pred_final_latents.ndim != 4 or fast_pred_final_latents.shape[0] != deltas.shape[0]:
            raise ValueError(
                "fast_pred_final_latents must have shape [K,C,H,W], "
                f"got {tuple(fast_pred_final_latents.shape)}"
            )

        teacher_losses = None
        teacher_final_latents = None
        teacher_selected_index = None
        if not self.args.latent_rerank_distill_skip_teacher:
            current_rng_state = self.capture_torch_rng_state()
            route2_cache_stats = dict(self.route2_cache_stats)
            if rollout_rng_state is not None:
                self.restore_torch_rng_state(rollout_rng_state)
            try:
                teacher_diffusion = self.get_rerank_diffusion(self.args.latent_rerank_distill_teacher_steps)
                teacher_losses, teacher_final_latents = self.score_candidate_deltas(
                    cur_obs_image,
                    cur_goal_image,
                    deltas,
                    self.args.latent_rerank_distill_score_mode,
                    teacher_diffusion,
                    timing=self.new_eval_timing(),
                    precomputed_goal_latent=cur_goal_latent,
                    precomputed_obs_latents=cur_obs_latents,
                )
            finally:
                self.restore_torch_rng_state(current_rng_state)
                self.route2_cache_stats = route2_cache_stats

            teacher_losses = teacher_losses.detach()
            teacher_final_latents = teacher_final_latents.detach()
            if teacher_losses.ndim != 1 or teacher_losses.shape[0] != deltas.shape[0]:
                raise ValueError(f"teacher_losses must have shape [K], got {tuple(teacher_losses.shape)}")
            if teacher_final_latents.ndim != 4 or teacher_final_latents.shape[0] != deltas.shape[0]:
                raise ValueError(
                    "teacher_final_latents must have shape [K,C,H,W], "
                    f"got {tuple(teacher_final_latents.shape)}"
                )
            teacher_selected_index = int(torch.argmin(teacher_losses).detach().cpu().item())
        record = {
            "record_id": len(self.latent_rerank_distill_records),
            "export_field_notes": (
                "teacher_selected_candidate_index is argmin(teacher_losses) over the exported "
                "independently recomputed fixed-step teacher rollout. planner_rerank_losses are "
                "the actual planner decision losses on planner_rerank_candidate_indices when available. "
                "active_planner_selected_candidate_index is diagnostic only: "
                "it is the candidate selected by the active planner loss path, which may be restricted "
                "to the fast top-m shortlist and should not be used as the distillation target unless "
                "that alignment has been explicitly verified."
            ),
            "dataset_name": dataset_name,
            "batch_idx": batch_idx,
            "eval_index": traj_id,
            "idx": traj_id,
            "seed": self.args.seed,
            "cem_step": int(cem_step),
            "is_final_cem_step": bool(is_final_cem_step),
            "planner_variant": self.args.planner_variant,
            "fast_diffusion_steps": self.args.fast_diffusion_steps,
            "teacher_diffusion_steps": int(self.args.latent_rerank_distill_teacher_steps),
            "teacher_score_mode": self.args.latent_rerank_distill_score_mode,
            "rerank_top_m": self.args.rerank_top_m,
            "candidate_actions": self.teacher_to_cpu(deltas),
            "context_latents": self.teacher_latent_to_cpu_fp16(cur_obs_latents[0]),
            "goal_latent": self.teacher_latent_to_cpu_fp16(cur_goal_latent[0]),
            "fast_losses": self.teacher_to_cpu(fast_losses),
            "fast_sorted_indices": self.teacher_to_cpu(fast_sorted_indices),
            "fast_pred_final_latents": self.teacher_latent_to_cpu_fp16(fast_pred_final_latents),
            "fast_rollout_trace": self.teacher_to_cpu(
                getattr(self, "last_fast_rollout_trace", None)
            ),
            "teacher_losses": self.teacher_to_cpu(teacher_losses),
            "teacher_final_latents": self.teacher_latent_to_cpu_fp16(teacher_final_latents),
            "teacher_selected_candidate_index": teacher_selected_index,
            "teacher_export_skipped": bool(self.args.latent_rerank_distill_skip_teacher),
            # Diagnostic only. For coarse_to_fine_latent this is the active planner's selected
            # candidate after fast top-m filtering/reranking, not the full-candidate teacher target.
            "active_planner_selected_candidate_index": int(active_selected_candidate_index),
            "normalized_margin": normalized_margin,
            "raw_margin": raw_margin,
        }
        if planner_rerank_candidate_indices is not None:
            record["planner_rerank_candidate_indices"] = self.teacher_to_cpu(planner_rerank_candidate_indices)
        if planner_rerank_losses is not None:
            record["planner_rerank_losses"] = self.teacher_to_cpu(planner_rerank_losses)
        if planner_rerank_selected_local_index is not None:
            record["planner_rerank_selected_local_index"] = int(planner_rerank_selected_local_index)
        if planner_rerank_selected_original_index is not None:
            record["planner_rerank_selected_original_index"] = int(planner_rerank_selected_original_index)
        self.latent_rerank_distill_records.append(record)
        self.latent_rerank_distill_completed_eval_indices.add(int(traj_id))
        if self.args.latent_rerank_distill_incremental_save:
            self.save_latent_rerank_distill_dataset()

    def latent_rerank_distill_output_path(self):
        output_path = self.args.latent_rerank_distill_output_path
        if output_path is None:
            output_dir = self.args.output_dir or "."
            output_path = os.path.join(output_dir, "latent_rerank_distill_dataset.pt")
        return output_path

    def save_latent_rerank_distill_dataset(self):
        if not self.args.export_latent_rerank_distill_dataset:
            return
        if dist.get_rank() != 0:
            return
        output_path = self.latent_rerank_distill_output_path()
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        tmp_path = output_path + ".tmp"
        torch.save(self.latent_rerank_distill_records, tmp_path)
        os.replace(tmp_path, output_path)
        print(f"Saved latent rerank distillation dataset to {output_path}")
        print(f"Latent rerank distillation records: {len(self.latent_rerank_distill_records)}")
        if self.latent_rerank_distill_records:
            sample = self.latent_rerank_distill_records[0]
            shape_keys = [
                "candidate_actions",
                "context_latents",
                "goal_latent",
                "fast_losses",
                "fast_pred_final_latents",
                "teacher_losses",
                "teacher_final_latents",
            ]
            print("Latent rerank distillation sample tensor shapes:")
            for key in shape_keys:
                value = sample.get(key)
                if torch.is_tensor(value):
                    print(f"  {key}: {tuple(value.shape)} {value.dtype}")

    def load_fixed_candidate_records(self, path):
        if not path:
            return {}
        records = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(records, list):
            raise ValueError(f"--fixed_candidate_actions_path must point to a list of records: {path}")
        by_key = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            if "candidate_actions" not in record or "eval_index" not in record:
                continue
            key = (int(record.get("seed", self.args.seed)), int(record["eval_index"]))
            by_key[key] = record
        print(f"Loaded fixed candidate records from {path}: {len(by_key)} keyed records")
        return by_key

    def fixed_candidate_record_for(self, traj_id):
        if not self.fixed_candidate_records:
            return None
        key = (int(self.args.seed), int(traj_id))
        record = self.fixed_candidate_records.get(key)
        if record is None and self.args.fixed_candidate_actions_strict:
            raise KeyError(f"Missing fixed candidate record for seed={self.args.seed} eval_index={traj_id}")
        return record

    def save_candidate_actions_export(self):
        output_path = self.args.export_candidate_actions_path
        if not output_path:
            return
        if dist.get_rank() != 0:
            return
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        tmp_path = output_path + ".tmp"
        torch.save(self.candidate_actions_export_records, tmp_path)
        os.replace(tmp_path, output_path)
        print(f"Saved candidate action export to {output_path}")
        print(f"Candidate action records: {len(self.candidate_actions_export_records)}")

    def teacher_collection_v2_output_path(self):
        output_path = self.args.teacher_collection_v2_output_path
        if output_path is None:
            output_path = os.path.join(
                "eval_output",
                "teacher_collection_v2",
                "teacher_collection_v2.pt",
            )
        return output_path

    def should_export_teacher_collection_v2_step(self, cem_step, is_final_cem_step):
        spec = str(self.args.teacher_collection_v2_cem_steps).strip().lower()
        if spec == "final":
            return bool(is_final_cem_step)
        if spec == "all":
            return True
        selected = {int(item.strip()) for item in spec.split(",") if item.strip()}
        return int(cem_step) in selected

    def selected_teacher_collection_v2_eval(self, traj_id):
        spec = self.args.teacher_collection_v2_selected_eval_indices
        if spec is None or str(spec).strip() == "":
            return True
        selected = {int(item.strip()) for item in str(spec).split(",") if item.strip()}
        traj_id = int(traj_id)
        local_eval_index = traj_id % 16
        return traj_id in selected or local_eval_index in selected

    def tensor_top_indices(self, losses, count):
        if losses is None or losses.numel() == 0:
            return []
        count = max(1, min(int(count), int(losses.numel())))
        return self.tensor_int_list(torch.argsort(losses)[:count])

    def loss_margin_value(self, losses, rank=2):
        if losses is None or losses.numel() < rank:
            return None
        sorted_loss = torch.sort(losses.detach().float()).values
        return float((sorted_loss[rank - 1] - sorted_loss[0]).detach().cpu().item())

    def record_teacher_collection_v2_example(
        self,
        dataset_name,
        batch_idx,
        traj_id,
        cem_step,
        is_final_cem_step,
        deltas,
        cur_obs_image,
        cur_goal_image,
        cur_obs_latents,
        cur_goal_latent,
        fast_losses,
        fast_pred_final_latents,
        fast_sorted_indices,
        normalized_margin,
        raw_margin,
        active_selected_candidate_index,
        rollout_rng_state=None,
        planner_rerank_candidate_indices=None,
        planner_rerank_losses=None,
        planner_rerank_selected_local_index=None,
        planner_rerank_selected_original_index=None,
    ):
        if not self.args.export_teacher_collection_v2:
            return
        if not self.should_export_teacher_collection_v2_step(cem_step, is_final_cem_step):
            return
        if not self.selected_teacher_collection_v2_eval(traj_id):
            return
        max_records = self.args.teacher_collection_v2_max_records
        if max_records is not None and len(self.teacher_collection_v2_records) >= max_records:
            return
        if self.args.latent_rerank_distill_score_mode == "lpips":
            raise ValueError("teacher_collection_v2 export does not support LPIPS/RGB teacher scoring.")
        required = {
            "deltas": deltas,
            "cur_obs_latents": cur_obs_latents,
            "cur_goal_latent": cur_goal_latent,
            "fast_losses": fast_losses,
            "fast_pred_final_latents": fast_pred_final_latents,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "--export_teacher_collection_v2 requires latent rerank candidate fields; "
                f"missing: {missing}. Use a latent rerank planner such as coarse_to_fine_latent."
            )
        if deltas.ndim != 3 or deltas.shape[-1] != 3:
            raise ValueError(f"candidate actions must have shape [K,T,3], got {tuple(deltas.shape)}")
        if fast_losses.ndim != 1 or fast_losses.shape[0] != deltas.shape[0]:
            raise ValueError(f"fast_losses must have shape [K], got {tuple(fast_losses.shape)}")

        current_rng_state = self.capture_torch_rng_state()
        route2_cache_stats = dict(self.route2_cache_stats)
        if rollout_rng_state is not None:
            self.restore_torch_rng_state(rollout_rng_state)
        try:
            teacher_diffusion = self.get_rerank_diffusion(self.args.latent_rerank_distill_teacher_steps)
            teacher_losses, teacher_final_latents = self.score_candidate_deltas(
                cur_obs_image,
                cur_goal_image,
                deltas,
                self.args.latent_rerank_distill_score_mode,
                teacher_diffusion,
                timing=self.new_eval_timing(),
                precomputed_goal_latent=cur_goal_latent,
                precomputed_obs_latents=cur_obs_latents,
            )
        finally:
            self.restore_torch_rng_state(current_rng_state)
            self.route2_cache_stats = route2_cache_stats

        teacher_losses = teacher_losses.detach()
        fast_losses = fast_losses.detach()
        candidate_hash = candidate_set_hash(deltas)
        action_hashes = candidate_action_hashes(deltas)
        if self.args.teacher_collection_v2_fail_on_missing_hash and not candidate_hash:
            raise ValueError("candidate_set_hash could not be computed.")

        n_candidates = int(deltas.shape[0])
        horizon = int(deltas.shape[1])
        action_dim = int(deltas.shape[2])
        global_eval_index = int(traj_id)
        local_eval_index = global_eval_index % 16
        chunk = f"{global_eval_index // 16:02d}"
        fast_top1 = int(torch.argmin(fast_losses).detach().cpu().item())
        teacher_top1 = int(torch.argmin(teacher_losses).detach().cpu().item())
        teacher_margin_12 = self.loss_margin_value(teacher_losses, rank=2)
        teacher_margin_15 = self.loss_margin_value(teacher_losses, rank=5)
        fast_margin_12 = self.loss_margin_value(fast_losses, rank=2)
        planner_indices_cpu = None
        planner_losses_cpu = None
        planner_rank_local = []
        planner_rank_global = []
        planner_reference_scope = None
        strict_candidate_parity = False
        if planner_rerank_candidate_indices is not None and planner_rerank_losses is not None:
            planner_indices_cpu = planner_rerank_candidate_indices.detach().cpu().long()
            planner_losses_cpu = planner_rerank_losses.detach().cpu().float()
            planner_order = torch.argsort(planner_losses_cpu)
            planner_rank_local = self.tensor_int_list(planner_order)
            planner_rank_global = self.tensor_int_list(planner_indices_cpu[planner_order])
            planner_reference_scope = (
                "all_n" if int(planner_indices_cpu.numel()) == n_candidates else "shortlist"
            )
            strict_candidate_parity = bool(
                torch.all((planner_indices_cpu >= 0) & (planner_indices_cpu < n_candidates)).item()
            )

        latent_summaries = {}
        if self.args.teacher_collection_v2_include_latent_summaries:
            latent_summaries = {
                "fast_pred_final_latents_summary": tensor_stats_dict(fast_pred_final_latents),
                "teacher_final_latents_summary": tensor_stats_dict(teacher_final_latents),
                "goal_latent_summary": tensor_stats_dict(cur_goal_latent),
                "context_latents_summary": tensor_stats_dict(cur_obs_latents),
            }

        outcome_nulls = {
            "fast_outcome_ATE": None,
            "fast_outcome_RPE": None,
            "fast_outcome_pos": None,
            "fast_outcome_yaw": None,
            "fast_outcome_time": None,
            "teacher_outcome_ATE": None,
            "teacher_outcome_RPE": None,
            "teacher_outcome_pos": None,
            "teacher_outcome_yaw": None,
            "teacher_outcome_time": None,
            "student_outcome_ATE": None,
            "student_outcome_RPE": None,
            "student_outcome_pos": None,
            "student_outcome_yaw": None,
            "student_outcome_time": None,
            "dATE_teacher_minus_fast": None,
            "dRPE_teacher_minus_fast": None,
            "dpos_teacher_minus_fast": None,
            "dyaw_teacher_minus_fast": None,
            "dtime_teacher_minus_fast": None,
            "severe_harm_teacher": None,
            "yaw_harm_teacher": None,
            "safe_use_teacher": None,
            "safe_use_student": None,
        }

        record = {
            "schema_version": "teacher_collection_v2",
            "record_id": len(self.teacher_collection_v2_records),
            "collection_id": self.args.teacher_collection_v2_collection_id,
            "route_id": None,
            "dataset_name": dataset_name,
            "chunk": chunk,
            "eval_index": local_eval_index,
            "global_eval_index": global_eval_index,
            "batch_idx": int(batch_idx),
            "seed": int(self.args.seed),
            "cem_step": int(cem_step),
            "is_final_cem_step": bool(is_final_cem_step),
            "N": n_candidates,
            "num_samples": n_candidates,
            "top_m": int(self.args.rerank_top_m),
            "rerank_top_m": int(self.args.rerank_top_m),
            "planner_variant": self.args.planner_variant,
            "score_mode": self.args.score_mode,
            "fast_score_mode": self.args.fast_score_mode,
            "teacher_score_mode": self.args.latent_rerank_distill_score_mode,
            "rerank_score_mode": self.args.rerank_score_mode,
            "fast_diffusion_steps": int(self.args.fast_diffusion_steps),
            "teacher_diffusion_steps": int(self.args.latent_rerank_distill_teacher_steps),
            "rerank_diffusion_steps": int(self.args.rerank_diffusion_steps),
            "checkpoint_path": f'{self.config["results_dir"]}/{self.config["run_name"]}/checkpoints/{self.args.ckp}.pth.tar',
            "config_path": self.args.exp,
            "split_path": self.args.navigation_eval_index,
            "dataset_recon_path": self.config["eval_datasets"][dataset_name].get("data_folder"),
            "timestamp": time.time(),
            "candidate_actions": self.teacher_to_cpu(deltas),
            "candidate_set_hash": candidate_hash,
            "candidate_action_hashes": action_hashes,
            "action_dim": action_dim,
            "horizon": horizon,
            "candidate_count": n_candidates,
            "fast_losses_all_n": self.teacher_to_cpu(fast_losses),
            "teacher_losses_all_n": self.teacher_to_cpu(teacher_losses),
            "teacher_rank_all_n": self.tensor_int_list(torch.argsort(teacher_losses)),
            "fast_rank_all_n": self.tensor_int_list(torch.argsort(fast_losses)),
            "teacher_top1_index": teacher_top1,
            "teacher_top3_indices": self.tensor_top_indices(teacher_losses, 3),
            "teacher_top5_indices": self.tensor_top_indices(teacher_losses, 5),
            "fast_top1_index": fast_top1,
            "fast_top3_indices": self.tensor_top_indices(fast_losses, 3),
            "fast_top5_indices": self.tensor_top_indices(fast_losses, 5),
            "top1_top2_margin_teacher": teacher_margin_12,
            "top1_top5_margin_teacher": teacher_margin_15,
            "top1_top2_margin_fast": fast_margin_12,
            "loss_std_teacher": float(torch.std(teacher_losses.float(), unbiased=False).detach().cpu().item()),
            "ambiguous_teacher_margin_lt_0_005": (
                teacher_margin_12 is not None and teacher_margin_12 < 0.005
            ),
            "ambiguous_teacher_margin_lt_0_01": (
                teacher_margin_12 is not None and teacher_margin_12 < 0.01
            ),
            "planner_rerank_candidate_indices": self.teacher_to_cpu(planner_indices_cpu),
            "planner_rerank_losses": self.teacher_to_cpu(planner_losses_cpu),
            "planner_rerank_rank": planner_rank_local,
            "planner_rerank_top1_index_global": planner_rank_global[0] if planner_rank_global else None,
            "planner_rerank_top3_indices_global": planner_rank_global[:3],
            "planner_rerank_top5_indices_global": planner_rank_global[:5],
            "planner_reference_scope": planner_reference_scope,
            "strict_candidate_parity": strict_candidate_parity,
            "planner_losses_all_n": self.teacher_to_cpu(planner_losses_cpu) if planner_reference_scope == "all_n" else None,
            "planner_rank_all_n": planner_rank_global if planner_reference_scope == "all_n" else None,
            "planner_all_n_available": planner_reference_scope == "all_n",
            "fast_selected_index": fast_top1,
            "teacher_selected_index": teacher_top1,
            "student_selected_index": None,
            "active_planner_selected_candidate_index": int(active_selected_candidate_index),
            "planner_rerank_selected_local_index": (
                int(planner_rerank_selected_local_index)
                if planner_rerank_selected_local_index is not None else None
            ),
            "planner_rerank_selected_original_index": (
                int(planner_rerank_selected_original_index)
                if planner_rerank_selected_original_index is not None else None
            ),
            "normalized_margin": normalized_margin,
            "raw_margin": raw_margin,
            "has_teacher_losses_all_n": True,
            "has_planner_reference": planner_reference_scope is not None,
            "has_planner_outcome": False,
            "has_candidate_set_hash": bool(candidate_hash),
            "strict_path_parity": False,
            "partial_parity": True,
            "export_notes": (
                "teacher_collection_v2 logs teacher_losses and fast_losses over the same full candidate set. "
                "planner_rerank_losses are the active planner reference and may be shortlist-only. "
                "Planner outcome labels are explicit nulls unless a future fixed-candidate outcome hook is added."
            ),
        }
        record.update(outcome_nulls)
        record.update(latent_summaries)
        self.teacher_collection_v2_records.append(record)

    def save_teacher_collection_v2_dataset(self):
        if not self.args.export_teacher_collection_v2:
            return
        if dist.get_rank() != 0:
            return
        output_path = self.teacher_collection_v2_output_path()
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        tmp_path = output_path + ".tmp"
        torch.save(self.teacher_collection_v2_records, tmp_path)
        os.replace(tmp_path, output_path)
        print(f"Saved teacher_collection_v2 dataset to {output_path}")
        print(f"teacher_collection_v2 records: {len(self.teacher_collection_v2_records)}")

    def json_float_list(self, values):
        return json.dumps(self.tensor_float_list(values))

    def json_int_list(self, values):
        return json.dumps(self.tensor_int_list(values))

    def tensor_nested_float_list(self, values):
        if values is None:
            return None
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().float().tolist()
        if isinstance(values, np.ndarray):
            return values.astype(float).tolist()
        return values

    def json_nested_float_list(self, values):
        nested = self.tensor_nested_float_list(values)
        return json.dumps(nested) if nested is not None else None

    def loss_margin(self, losses, top_k_last=None):
        sorted_loss = torch.sort(losses.detach()).values
        if sorted_loss.numel() < 2:
            top1_top2 = float("inf")
        else:
            top1_top2 = float((sorted_loss[1] - sorted_loss[0]).detach().cpu())
        if top_k_last is None or sorted_loss.numel() == 0:
            top1_topk_last = None
        else:
            last_idx = max(0, min(int(top_k_last), sorted_loss.numel()) - 1)
            top1_topk_last = float((sorted_loss[last_idx] - sorted_loss[0]).detach().cpu())
        return top1_top2, top1_topk_last

    def record_adaptive_decision(self, mode, norm_margin, rerank_norm_margin=None, effective_step_equiv=0.0):
        key = {
            "early_exit": "adaptive_early_exit_count",
            "mid_rerank": "adaptive_mid_rerank_count",
            "high_fallback": "adaptive_high_fallback_count",
        }[mode]
        self.adaptive_stats[key] += 1
        self.adaptive_stats["adaptive_norm_margin_sum"] += float(norm_margin.detach().cpu())
        self.adaptive_stats["adaptive_norm_margin_count"] += 1
        if rerank_norm_margin is not None:
            self.adaptive_stats["adaptive_rerank_norm_margin_sum"] += float(rerank_norm_margin.detach().cpu())
            self.adaptive_stats["adaptive_rerank_norm_margin_count"] += 1
        self.adaptive_stats["adaptive_effective_step_equiv"] += float(effective_step_equiv)
        if self.args.adaptive_debug:
            rerank_msg = "none" if rerank_norm_margin is None else f"{float(rerank_norm_margin.detach().cpu()):.4f}"
            print(
                f"adaptive_mode={mode} "
                f"norm_margin={float(norm_margin.detach().cpu()):.4f} "
                f"rerank_norm_margin={rerank_msg} "
                f"effective_step_equiv={float(effective_step_equiv):.1f}"
            )

    def adaptive_summary_stats(self):
        norm_count = self.adaptive_stats["adaptive_norm_margin_count"]
        rerank_count = self.adaptive_stats["adaptive_rerank_norm_margin_count"]
        return {
            "adaptive_early_exit_count": self.adaptive_stats["adaptive_early_exit_count"],
            "adaptive_mid_rerank_count": self.adaptive_stats["adaptive_mid_rerank_count"],
            "adaptive_high_fallback_count": self.adaptive_stats["adaptive_high_fallback_count"],
            "adaptive_avg_norm_margin": (
                self.adaptive_stats["adaptive_norm_margin_sum"] / norm_count
                if norm_count else None
            ),
            "adaptive_avg_rerank_norm_margin": (
                self.adaptive_stats["adaptive_rerank_norm_margin_sum"] / rerank_count
                if rerank_count else None
            ),
            "adaptive_effective_step_equiv": self.adaptive_stats["adaptive_effective_step_equiv"],
        }

    def new_adaptive_two_gate_stats(self):
        return {
            "adaptive_two_gate_early_exit_count": 0,
            "adaptive_two_gate_topm_fallback_count": 0,
            "adaptive_two_gate_full_fallback_count": 0,
            "adaptive_two_gate_exit_prob_sum": 0.0,
            "adaptive_two_gate_exit_prob_count": 0,
            "adaptive_two_gate_topm_prob_sum": 0.0,
            "adaptive_two_gate_topm_prob_count": 0,
            "adaptive_two_gate_effective_step_equiv": 0.0,
        }

    def adaptive_two_gate_summary_stats(self):
        exit_count = self.adaptive_two_gate_stats["adaptive_two_gate_exit_prob_count"]
        topm_count = self.adaptive_two_gate_stats["adaptive_two_gate_topm_prob_count"]
        return {
            "adaptive_two_gate_early_exit_count": self.adaptive_two_gate_stats["adaptive_two_gate_early_exit_count"],
            "adaptive_two_gate_topm_fallback_count": self.adaptive_two_gate_stats["adaptive_two_gate_topm_fallback_count"],
            "adaptive_two_gate_full_fallback_count": self.adaptive_two_gate_stats["adaptive_two_gate_full_fallback_count"],
            "adaptive_two_gate_avg_exit_prob": (
                self.adaptive_two_gate_stats["adaptive_two_gate_exit_prob_sum"] / exit_count
                if exit_count else None
            ),
            "adaptive_two_gate_avg_topm_prob": (
                self.adaptive_two_gate_stats["adaptive_two_gate_topm_prob_sum"] / topm_count
                if topm_count else None
            ),
            "adaptive_two_gate_effective_step_equiv": self.adaptive_two_gate_stats["adaptive_two_gate_effective_step_equiv"],
        }

    def load_adaptive_two_gate_policy(self):
        if self.args.adaptive_policy_path is None:
            raise ValueError("--adaptive_policy_path is required for planner_variant=adaptive_two_gate_low_high")
        with open(self.args.adaptive_policy_path, "rb") as f:
            policy = pickle.load(f)
        required_keys = [
            "exit_gate_model",
            "topm_gate_model",
            "feature_columns",
            "fallback_top_m",
            "tau_exit",
            "tau_topm",
        ]
        missing = [key for key in required_keys if key not in policy]
        if missing:
            raise ValueError(f"Adaptive policy artifact is missing keys: {missing}")
        self.adaptive_policy = policy
        print(
            "Loaded adaptive two-gate policy "
            f"path={self.args.adaptive_policy_path}, "
            f"fallback_top_m={policy['fallback_top_m']}, "
            f"tau_exit={policy['tau_exit']}, tau_topm={policy['tau_topm']}, "
            f"features={len(policy['feature_columns'])}"
        )

    def model_positive_probability(self, model, features):
        proba = model.predict_proba(features)
        if proba.shape[1] == 1:
            classifier = getattr(model, "named_steps", {}).get("classifier", model)
            classes = getattr(classifier, "classes_", [1])
            return float(proba[0, 0]) if int(classes[0]) == 1 else 0.0
        classifier = getattr(model, "named_steps", {}).get("classifier", model)
        classes = getattr(classifier, "classes_", [0, 1])
        positive_cols = np.where(np.asarray(classes) == 1)[0]
        positive_col = int(positive_cols[0]) if len(positive_cols) else proba.shape[1] - 1
        return float(proba[0, positive_col])

    def ranked_loss_value(self, losses, sorted_idx, rank):
        rank = min(rank, len(sorted_idx)) - 1
        return float(losses[sorted_idx[rank]].detach().cpu())

    def online_low_step_features(self, low_losses, deltas):
        sorted_idx = [int(x) for x in torch.argsort(low_losses).detach().cpu().tolist()]
        low_std_tensor = torch.std(low_losses, unbiased=False)
        low_std = float(low_std_tensor.detach().cpu())
        low_min = self.ranked_loss_value(low_losses, sorted_idx, 1)
        low_second = self.ranked_loss_value(low_losses, sorted_idx, 2)
        low_mean = float(low_losses.mean().detach().cpu())
        low_max = float(low_losses.max().detach().cpu())
        denom = low_std_tensor + 1e-8

        def gap(rank):
            return self.ranked_loss_value(low_losses, sorted_idx, rank) - low_min

        gap_12 = low_second - low_min
        gap_13 = gap(3)
        gap_14 = gap(4)
        gap_16 = gap(6)
        probs = torch.softmax(-low_losses.float(), dim=0)
        entropy = -torch.sum(probs * torch.log(probs.clamp_min(1e-12)))
        top2 = sorted_idx[: min(2, len(sorted_idx))]
        top4 = sorted_idx[: min(4, len(sorted_idx))]
        top6 = sorted_idx[: min(6, len(sorted_idx))]

        final_deltas = deltas[:, -1].detach().float()
        low_top1 = sorted_idx[0]
        low_top2 = sorted_idx[1] if len(sorted_idx) > 1 else sorted_idx[0]
        return {
            "low_loss_min": low_min,
            "low_loss_second": low_second,
            "low_loss_mean": low_mean,
            "low_loss_std": low_std,
            "low_loss_range": low_max - low_min,
            "low_gap_12": gap_12,
            "low_gap_13": gap_13,
            "low_gap_14": gap_14,
            "low_gap_16": gap_16,
            "low_norm_gap_12": float((torch.tensor(gap_12, device=low_losses.device) / denom).detach().cpu()),
            "low_norm_gap_13": float((torch.tensor(gap_13, device=low_losses.device) / denom).detach().cpu()),
            "low_norm_gap_14": float((torch.tensor(gap_14, device=low_losses.device) / denom).detach().cpu()),
            "low_norm_gap_16": float((torch.tensor(gap_16, device=low_losses.device) / denom).detach().cpu()),
            "low_softmax_entropy": float(entropy.detach().cpu()),
            "low_top1_prob": float(probs[sorted_idx[0]].detach().cpu()),
            "low_top2_prob": float(probs[top2].sum().detach().cpu()),
            "low_top4_prob_sum": float(probs[top4].sum().detach().cpu()),
            "low_rank_spread_top4": int(max(top4) - min(top4)) if top4 else 0,
            "low_rank_spread_top6": int(max(top6) - min(top6)) if top6 else 0,
            "candidate_delta_mean_x": float(final_deltas[:, 0].mean().detach().cpu()),
            "candidate_delta_std_x": float(final_deltas[:, 0].std(unbiased=False).detach().cpu()),
            "candidate_delta_mean_y": float(final_deltas[:, 1].mean().detach().cpu()),
            "candidate_delta_std_y": float(final_deltas[:, 1].std(unbiased=False).detach().cpu()),
            "candidate_delta_mean_yaw": float(final_deltas[:, 2].mean().detach().cpu()),
            "candidate_delta_std_yaw": float(final_deltas[:, 2].std(unbiased=False).detach().cpu()),
            "top1_delta_x": float(final_deltas[low_top1, 0].detach().cpu()),
            "top1_delta_y": float(final_deltas[low_top1, 1].detach().cpu()),
            "top1_delta_yaw": float(final_deltas[low_top1, 2].detach().cpu()),
            "top2_delta_x": float(final_deltas[low_top2, 0].detach().cpu()),
            "top2_delta_y": float(final_deltas[low_top2, 1].detach().cpu()),
            "top2_delta_yaw": float(final_deltas[low_top2, 2].detach().cpu()),
        }

    def adaptive_policy_feature_vector(self, low_losses, deltas):
        feature_map = self.online_low_step_features(low_losses, deltas)
        missing = [name for name in self.adaptive_policy["feature_columns"] if name not in feature_map]
        if missing:
            raise ValueError(f"Adaptive policy feature columns are not available online: {missing}")
        values = [feature_map[name] for name in self.adaptive_policy["feature_columns"]]
        return np.asarray(values, dtype=np.float32).reshape(1, -1)

    def record_adaptive_two_gate_decision(self, mode, exit_prob, topm_prob, effective_step_equiv):
        key = {
            "early_exit": "adaptive_two_gate_early_exit_count",
            "topm_fallback": "adaptive_two_gate_topm_fallback_count",
            "full_fallback": "adaptive_two_gate_full_fallback_count",
        }[mode]
        self.adaptive_two_gate_stats[key] += 1
        self.adaptive_two_gate_stats["adaptive_two_gate_exit_prob_sum"] += float(exit_prob)
        self.adaptive_two_gate_stats["adaptive_two_gate_exit_prob_count"] += 1
        if topm_prob is not None:
            self.adaptive_two_gate_stats["adaptive_two_gate_topm_prob_sum"] += float(topm_prob)
            self.adaptive_two_gate_stats["adaptive_two_gate_topm_prob_count"] += 1
        self.adaptive_two_gate_stats["adaptive_two_gate_effective_step_equiv"] += float(effective_step_equiv)
        if self.args.adaptive_policy_debug:
            print(
                f"adaptive_two_gate_mode={mode} "
                f"rng_isolation={not self.args.disable_adaptive_policy_rng_isolation} "
                f"exit_prob={float(exit_prob):.4f} "
                f"topm_prob={'none' if topm_prob is None else f'{float(topm_prob):.4f}'} "
                f"effective_step_equiv={float(effective_step_equiv):.1f}"
            )

    def add_route2_timing_delta(self, prefix, before_timing):
        self.route2_timing[f"{prefix}_rollout_time"] += (
            self.eval_timing["rollout_time"] - before_timing["rollout_time"]
        )
        self.route2_timing[f"{prefix}_scoring_time"] += (
            self.eval_timing["scoring_time"] - before_timing["scoring_time"]
        )

    def capture_torch_rng_state(self):
        state = {"cpu": torch.get_rng_state(), "cuda": None}
        if self.device.type == "cuda" and torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state(self.device)
        return state

    def restore_torch_rng_state(self, state):
        torch.set_rng_state(state["cpu"])
        if state.get("cuda") is not None and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.set_rng_state(state["cuda"], self.device)

    def load_latent_scorer(self):
        if self.args.latent_scorer_checkpoint is None:
            raise ValueError("--latent_scorer_checkpoint is required for score_mode=learned_spatial_scorer")

        scorer_device = self.args.latent_scorer_device
        if scorer_device is None:
            self.latent_scorer_device = self.device
        else:
            self.latent_scorer_device = torch.device(scorer_device)

        checkpoint = load_latent_scorer_checkpoint(self.args.latent_scorer_checkpoint)
        state_dict = checkpoint["model_state_dict"]
        saved_args = checkpoint.get("args", {})
        eval_metrics = checkpoint.get("eval_metrics", {})
        model_type = saved_args.get("model_type", eval_metrics.get("model_type", "spatial_cnn"))
        if model_type != "spatial_cnn":
            raise ValueError(f"learned_spatial_scorer requires a spatial_cnn checkpoint, got {model_type}")

        action_dim = saved_args.get("action_dim", eval_metrics.get("action_dim"))
        if action_dim is None:
            action_dim = state_dict["action_encoder.0.weight"].shape[1]
        cnn_channels = saved_args.get("cnn_channels", eval_metrics.get("cnn_channels"))
        if cnn_channels is None:
            cnn_channels = state_dict["latent_encoder.0.weight"].shape[0]
        action_embed_dim = saved_args.get("action_embed_dim", eval_metrics.get("action_embed_dim"))
        if action_embed_dim is None:
            action_embed_dim = state_dict["action_encoder.0.weight"].shape[0]
        dropout = saved_args.get("dropout", eval_metrics.get("dropout", 0.0))

        scorer = SpatialCNNScorer(
            action_dim=action_dim,
            cnn_channels=cnn_channels,
            action_embed_dim=action_embed_dim,
            dropout=dropout,
        )
        scorer.load_state_dict(state_dict, strict=True)
        scorer.eval().to(self.latent_scorer_device)
        self.latent_scorer = scorer

        print(f"Loaded learned spatial latent scorer from {self.args.latent_scorer_checkpoint}")
        print(f"Latent scorer model_type={model_type}")
        print(f"Latent scorer parameter_count={count_parameters(self.latent_scorer)}")
        print("learned_spatial_scorer uses decoder-free latent scoring: no RGB decode and no LPIPS scoring.")

    def load_student_reranker(self):
        if self.args.student_rerank_checkpoint is None:
            raise ValueError("--student_rerank_checkpoint is required for student rerank backend or debug compare")

        helper_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "teacher_baseline")
        if helper_dir not in sys.path:
            sys.path.insert(0, helper_dir)
        from train_latent_rerank_student import DeepSetsResidualReranker, StudentReranker

        checkpoint = torch.load(self.args.student_rerank_checkpoint, map_location="cpu", weights_only=False)
        if "model_state_dict" not in checkpoint:
            raise ValueError(f"student checkpoint missing model_state_dict: {self.args.student_rerank_checkpoint}")
        if "feature_mean" not in checkpoint or "feature_std" not in checkpoint:
            raise ValueError(
                "--rerank_backend student requires checkpoint feature_mean and feature_std "
                "from offline training"
            )

        saved_args = checkpoint.get("args", {})
        state_dict = checkpoint["model_state_dict"]
        feature_mean = torch.as_tensor(checkpoint["feature_mean"]).detach().float()
        feature_std = torch.as_tensor(checkpoint["feature_std"]).detach().float()
        feature_dim = int(checkpoint.get("feature_dim", feature_mean.numel()))
        dropout = float(saved_args.get("dropout", 0.0))
        self.student_rerank_arch = str(saved_args.get("student_arch", self.args.student_arch))
        hidden_dim = None
        depth = None
        if self.student_rerank_arch != "deepsets_residual":
            if "hidden_dim" in saved_args:
                hidden_dim = int(saved_args["hidden_dim"])
            elif "net.0.weight" in state_dict:
                hidden_dim = int(state_dict["net.0.weight"].shape[0])
            else:
                raise ValueError(
                    "MLP student checkpoint does not contain hidden_dim or net.0.weight; "
                    f"student_arch={self.student_rerank_arch}"
                )
            depth = int(saved_args.get("depth", 0))
        self.student_rerank_include_pooled_latents = bool(saved_args.get("include_pooled_latents", False))
        self.student_rerank_include_action_geometry = bool(saved_args.get("include_action_geometry", False))
        self.student_rerank_include_list_context_features = bool(
            saved_args.get("include_list_context_features", False)
        )
        self.student_rerank_list_context_std_floor = float(saved_args.get("list_context_std_floor", 1e-3))
        self.student_rerank_list_context_clip = float(saved_args.get("list_context_clip", 5.0))
        self.student_rerank_feature_clip_value = float(saved_args.get("feature_clip_value", 10.0))
        self.student_rerank_normalize_action_features = bool(saved_args.get("normalize_action_features", False))
        self.student_rerank_normalize_latent_summary_features = bool(
            saved_args.get("normalize_latent_summary_features", False)
        )
        self.student_rerank_normalize_base_features = bool(saved_args.get("normalize_base_features", False))
        self.student_rerank_action_feature_mode = str(saved_args.get("action_feature_mode", "raw"))
        self.student_rerank_action_yaw_clip = float(saved_args.get("action_yaw_clip", 3.14159265))
        self.student_rerank_action_feature_clip = float(saved_args.get("action_feature_clip", 10.0))
        self.student_rerank_model_type = str(saved_args.get("model_type", "mlp"))
        self.student_rerank_residual_clip = float(saved_args.get("residual_clip", 0.0))
        if self.args.student_residual_clip is not None:
            self.student_rerank_residual_clip = float(self.args.student_residual_clip)
        self.student_rerank_fast_prior_weight = float(saved_args.get("fast_prior_weight", 1.0))
        self.student_rerank_list_context_variant = str(saved_args.get("list_context_variant", "full"))
        if self.student_rerank_arch != "deepsets_residual" and depth <= 0:
            depth = sum(
                1
                for key in state_dict
                if key.startswith("net.") and key.endswith(".weight")
            ) - 1
        if self.student_rerank_arch == "deepsets_residual":
            model = DeepSetsResidualReranker(
                feature_dim=feature_dim,
                candidate_hidden_dim=int(saved_args.get("candidate_hidden_dim", 64)),
                set_hidden_dim=int(saved_args.get("set_hidden_dim", 64)),
                head_hidden_dim=int(saved_args.get("score_head_hidden_dim", 64)),
                confidence_hidden_dim=int(saved_args.get("confidence_head_hidden_dim", 64)),
                dropout=dropout,
            )
        else:
            model = StudentReranker(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                depth=depth,
                dropout=dropout,
            )
        model.load_state_dict(state_dict, strict=True)
        model.eval().to(self.student_rerank_device)
        self.student_rerank_model = model
        self.student_rerank_feature_mean = feature_mean.to(self.student_rerank_device)
        self.student_rerank_feature_std = feature_std.to(self.student_rerank_device)
        feature_indices = None
        for key in ("feature_indices", "kept_feature_indices", "feature_keep_indices"):
            if key in checkpoint and checkpoint[key] is not None:
                feature_indices = checkpoint[key]
                break
        self.student_rerank_feature_indices = (
            [int(x) for x in feature_indices]
            if feature_indices is not None else None
        )
        expected_dim = (
            len(self.student_rerank_feature_indices)
            if self.student_rerank_feature_indices is not None
            else self.student_rerank_feature_mean.numel()
        )
        if expected_dim != feature_dim:
            raise ValueError(
                f"student checkpoint feature_dim={feature_dim} but selected feature count is {expected_dim}"
            )
        if self.student_rerank_feature_mean.numel() != feature_dim:
            raise ValueError(
                f"student checkpoint feature_mean dim {self.student_rerank_feature_mean.numel()} "
                f"does not match model feature_dim {feature_dim}"
            )
        if self.student_rerank_feature_std.numel() != feature_dim:
            raise ValueError(
                f"student checkpoint feature_std dim {self.student_rerank_feature_std.numel()} "
                f"does not match model feature_dim {feature_dim}"
            )
        print(
            "Loaded student rerank backend "
            f"path={self.args.student_rerank_checkpoint}, "
            f"feature_dim={feature_dim}, hidden_dim={hidden_dim}, depth={depth}, dropout={dropout}, "
            f"include_pooled_latents={self.student_rerank_include_pooled_latents}, "
            f"include_action_geometry={self.student_rerank_include_action_geometry}, "
            f"include_list_context_features={self.student_rerank_include_list_context_features}, "
            f"list_context_std_floor={self.student_rerank_list_context_std_floor}, "
            f"list_context_clip={self.student_rerank_list_context_clip}, "
            f"feature_clip_value={self.student_rerank_feature_clip_value}, "
            f"normalize_action_features={self.student_rerank_normalize_action_features}, "
            f"normalize_latent_summary_features={self.student_rerank_normalize_latent_summary_features}, "
            f"normalize_base_features={self.student_rerank_normalize_base_features}, "
            f"action_feature_mode={self.student_rerank_action_feature_mode}, "
            f"action_yaw_clip={self.student_rerank_action_yaw_clip}, "
            f"action_feature_clip={self.student_rerank_action_feature_clip}, "
            f"model_type={self.student_rerank_model_type}, "
            f"student_arch={self.student_rerank_arch}, "
            f"residual_clip={self.student_rerank_residual_clip}, "
            f"fast_prior_weight={self.student_rerank_fast_prior_weight}, "
            f"list_context_variant={self.student_rerank_list_context_variant}"
        )

    def build_student_rerank_features(self, deltas, fast_loss, fast_pred_final_latents, goal_latent, rerank_idx_cpu):
        helper_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "teacher_baseline")
        if helper_dir not in sys.path:
            sys.path.insert(0, helper_dir)
        from latent_rerank_distill_dataset import build_candidate_features

        rerank_idx_device = rerank_idx_cpu.to(fast_loss.device)
        single_goal_latent = goal_latent[0] if goal_latent.ndim == 4 else goal_latent
        features = build_candidate_features(
            deltas[rerank_idx_cpu].detach().cpu(),
            fast_loss[rerank_idx_device].detach().cpu(),
            fast_pred_final_latents[rerank_idx_device].detach().cpu(),
            single_goal_latent.detach().cpu(),
            include_pooled_latents=self.student_rerank_include_pooled_latents,
            include_action_geometry=self.student_rerank_include_action_geometry,
            include_list_context_features=self.student_rerank_include_list_context_features,
            list_context_variant=self.student_rerank_list_context_variant,
            list_context_std_floor=self.student_rerank_list_context_std_floor,
            list_context_clip=self.student_rerank_list_context_clip,
            feature_clip_value=self.student_rerank_feature_clip_value,
            normalize_action_features=self.student_rerank_normalize_action_features,
            normalize_latent_summary_features=self.student_rerank_normalize_latent_summary_features,
            normalize_base_features=self.student_rerank_normalize_base_features,
            action_feature_mode=self.student_rerank_action_feature_mode,
            action_yaw_clip=self.student_rerank_action_yaw_clip,
            action_feature_clip=self.student_rerank_action_feature_clip,
        ).float()
        if self.student_rerank_feature_indices is not None:
            features = features[:, self.student_rerank_feature_indices]
        if features.shape[-1] != self.student_rerank_feature_mean.numel():
            raise ValueError(
                f"online student feature dim {features.shape[-1]} does not match checkpoint "
                f"feature dim {self.student_rerank_feature_mean.numel()}"
            )
        return features.to(self.student_rerank_device)

    def load_b18_direct_challenger(self):
        checkpoint_path = (
            self.args.b51_safegate_checkpoint
            if self.args.rerank_backend == "b51_safegate"
            else self.args.b18_challenger_checkpoint
        )
        if checkpoint_path is None:
            required = "--b51_safegate_checkpoint" if self.args.rerank_backend == "b51_safegate" else "--b18_challenger_checkpoint"
            raise ValueError(f"{required} is required for --rerank_backend {self.args.rerank_backend}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "state_dict" not in checkpoint and "ensemble_state_dicts" not in checkpoint:
            raise ValueError(f"B18 checkpoint missing state_dict: {self.args.b18_challenger_checkpoint}")
        config = checkpoint.get("config", {})
        hidden = int(config.get("hidden", 32))
        feature_dim = int(len(checkpoint.get("feature_cols", [])) or checkpoint["mu"].shape[0])

        class B18DirectFastTopKChallenger(torch.nn.Module):
            def __init__(self, in_dim, hidden_dim):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, 3),
                )

            def forward(self, x):
                y = self.net(x)
                return y[:, 0], y[:, 1], y[:, 2]

        ensemble_state_dicts = checkpoint.get("ensemble_state_dicts")
        if ensemble_state_dicts is not None:
            models = []
            for state_dict in ensemble_state_dicts:
                member = B18DirectFastTopKChallenger(feature_dim, hidden)
                member.load_state_dict(state_dict, strict=True)
                member.eval().to(self.device)
                models.append(member)
            model = torch.nn.ModuleList(models)
            self.b18_challenger_feature_mean = torch.as_tensor(checkpoint["mus"]).detach().float().to(self.device)
            self.b18_challenger_feature_std = torch.as_tensor(checkpoint["sds"]).detach().float().to(self.device)
        else:
            model = B18DirectFastTopKChallenger(feature_dim, hidden)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval().to(self.device)
            self.b18_challenger_feature_mean = torch.as_tensor(checkpoint["mu"]).detach().float().to(self.device)
            self.b18_challenger_feature_std = torch.as_tensor(checkpoint["sd"]).detach().float().to(self.device)
        self.b18_challenger_model = model
        self.b18_challenger_args = dict(config)
        if self.b18_challenger_feature_mean.shape[-1] != feature_dim:
            raise ValueError(
                f"B18 feature mean dim {self.b18_challenger_feature_mean.shape[-1]} does not match feature_dim={feature_dim}"
            )
        ensemble_size = int(len(ensemble_state_dicts)) if ensemble_state_dicts is not None else 1
        print(
            ("Loaded B51 safegate " if self.args.rerank_backend == "b51_safegate" else "Loaded B18 direct challenger ")
            + f"path={checkpoint_path}, "
            f"feature_dim={feature_dim}, hidden={hidden}, ensemble={ensemble_size}, "
            f"p_threshold={self.args.b18_challenger_p_threshold}, "
            f"tau={self.args.b18_challenger_tau}, "
            f"confidence_threshold={self.args.b18_challenger_confidence_threshold}, "
            f"min_rank={self.args.b18_challenger_min_fast_rank}, "
            f"max_rank={self.args.b18_challenger_max_fast_rank}"
        )

    def load_b48_outcome_student(self):
        if self.args.b48_outcome_checkpoint is None:
            raise ValueError("--b48_outcome_checkpoint is required for --rerank_backend b48_outcome_student")
        if str(self.args.b48_outcome_checkpoint).endswith(".json"):
            with open(self.args.b48_outcome_checkpoint, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
        else:
            try:
                import joblib
            except ImportError as exc:
                raise ImportError("B48 outcome student requires joblib/sklearn to load the checkpoint.") from exc
            checkpoint = joblib.load(self.args.b48_outcome_checkpoint)
        if "model" not in checkpoint or "feature_cols" not in checkpoint:
            raise ValueError(f"B48 checkpoint missing model/feature_cols: {self.args.b48_outcome_checkpoint}")
        model = checkpoint["model"]
        if model.get("kind") == "portable_linear":
            for key in ("scaler_mean", "scaler_scale", "reg_coef", "reg_intercept", "helpful_coef", "helpful_intercept", "harm_coef", "harm_intercept"):
                if key not in model:
                    raise ValueError(f"B48 portable checkpoint model missing {key}: {self.args.b48_outcome_checkpoint}")
        elif model.get("kind") == "portable_mlp":
            for key in ("feature_mean", "feature_scale", "layers"):
                if key not in model:
                    raise ValueError(f"B48 portable MLP checkpoint model missing {key}: {self.args.b48_outcome_checkpoint}")
        else:
            for key in ("reg", "helpful", "harm"):
                if key not in model:
                    raise ValueError(f"B48 checkpoint model missing {key}: {self.args.b48_outcome_checkpoint}")
        self.b48_outcome_student = model
        self.b48_outcome_feature_cols = list(checkpoint["feature_cols"])
        self.b48_outcome_policy = dict(checkpoint.get("policy", {}))
        if self.args.b48_outcome_dynamic_policy_json is not None:
            with open(self.args.b48_outcome_dynamic_policy_json, "r", encoding="utf-8") as f:
                self.b48_dynamic_policy = json.load(f)
            if "challengeable_model" not in self.b48_dynamic_policy or "gates" not in self.b48_dynamic_policy:
                raise ValueError(
                    f"B48 dynamic policy missing challengeable_model/gates: {self.args.b48_outcome_dynamic_policy_json}"
                )
        print(
            "Loaded B48 outcome student "
            f"path={self.args.b48_outcome_checkpoint}, "
            f"feature_dim={len(self.b48_outcome_feature_cols)}, "
            f"p_helpful_min={self.args.b48_outcome_p_helpful_min}, "
            f"p_harm_max={self.args.b48_outcome_p_harm_max}, "
            f"pred_date_max={self.args.b48_outcome_pred_date_max}, "
            f"rank_min={self.args.b48_outcome_min_fast_rank}, "
            f"rank_max={self.args.b48_outcome_max_fast_rank}"
        )
        if self.b48_dynamic_policy is not None:
            print(f"Loaded B48 dynamic policy path={self.args.b48_outcome_dynamic_policy_json}")

    def build_b18_candidate_features(self, deltas, fast_loss, fast_pred_final_latents, goal_latent, context_latents):
        actions = deltas.detach().float()
        fast = fast_loss.detach().float().to(actions.device)
        pred = fast_pred_final_latents.detach().float().to(actions.device)
        goal = goal_latent.detach().float().to(actions.device)
        context = context_latents.detach().float().to(actions.device) if context_latents is not None else None
        n = int(actions.shape[0])

        def zscore(values):
            values = values.float()
            scale = values.std(unbiased=False).clamp_min(1e-8)
            return (values - values.mean()) / scale

        order = torch.argsort(fast)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(n, device=actions.device, dtype=torch.float32)
        idx_norm = torch.arange(n, device=actions.device, dtype=torch.float32) / max(1, n - 1)
        flat_a = actions.reshape(n, -1)
        chunks = [
            zscore(fast).unsqueeze(1),
            (ranks / max(1, n - 1)).unsqueeze(1),
            idx_norm.unsqueeze(1),
            flat_a.mean(dim=1, keepdim=True),
            flat_a.std(dim=1, unbiased=False, keepdim=True),
            flat_a.min(dim=1, keepdim=True).values,
            flat_a.max(dim=1, keepdim=True).values,
            torch.linalg.norm(flat_a, dim=1, keepdim=True),
            torch.linalg.norm(actions[:, -1, :] - actions[:, 0, :], dim=1, keepdim=True),
            actions.abs().mean(dim=1),
            actions[:, -1, :],
        ]
        flat_p = pred.reshape(n, -1)
        chunks.extend([
            flat_p.mean(dim=1, keepdim=True),
            flat_p.std(dim=1, unbiased=False, keepdim=True),
            (torch.linalg.norm(flat_p, dim=1, keepdim=True) / math.sqrt(flat_p.shape[1])),
        ])
        g = goal.reshape(1, -1)
        diff = flat_p - g
        denom = torch.linalg.norm(flat_p, dim=1) * torch.linalg.norm(g.reshape(-1)) + 1e-8
        chunks.extend([
            (torch.linalg.norm(diff, dim=1, keepdim=True) / math.sqrt(flat_p.shape[1])),
            ((flat_p @ g.reshape(-1)) / denom).unsqueeze(1),
        ])
        gflat = goal.reshape(-1)
        chunks.extend([
            torch.full((n, 1), float(gflat.mean().detach().cpu()), device=actions.device),
            torch.full((n, 1), float(gflat.std(unbiased=False).detach().cpu()), device=actions.device),
        ])
        if context is not None:
            cflat = context.reshape(-1)
            context_mean = float(cflat.mean().detach().cpu())
            context_std = float(cflat.std(unbiased=False).detach().cpu())
        else:
            context_mean = 0.0
            context_std = 0.0
        chunks.extend([
            torch.full((n, 1), context_mean, device=actions.device),
            torch.full((n, 1), context_std, device=actions.device),
        ])
        features = torch.cat(chunks, dim=1)
        return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def score_b18_direct_challenger(
        self,
        deltas,
        fast_loss,
        fast_pred_final_latents,
        goal_latent,
        context_latents,
        rerank_idx_cpu,
    ):
        if self.b18_challenger_model is None:
            raise ValueError("B18 direct challenger backend is not loaded")
        self.sync_cuda_for_timing()
        feature_start = time.perf_counter()
        all_features = self.build_b18_candidate_features(
            deltas, fast_loss, fast_pred_final_latents, goal_latent, context_latents
        ).to(self.device)
        rerank_idx_device = rerank_idx_cpu.to(all_features.device)
        local_features = all_features[rerank_idx_device]
        k = int(local_features.shape[0])
        if k < 1:
            raise ValueError("B18 direct challenger requires at least one rerank candidate")
        fast_top1_local = 0
        candidate_rows = []
        candidate_locals = []
        min_rank = max(2, int(self.args.b18_challenger_min_fast_rank))
        max_rank = min(int(self.args.b18_challenger_max_fast_rank), k)
        local_actions = deltas[rerank_idx_cpu.to(deltas.device)].detach().float()
        fast_action = local_actions[fast_top1_local]
        local_fast_losses = fast_loss[rerank_idx_cpu.to(fast_loss.device)].detach().float()

        def add_action_feature_columns(row, candidate_action, baseline_action):
            cand = candidate_action.detach().float()
            base = baseline_action.detach().float()
            delta = cand - base
            cand_cpu = cand.detach().cpu()
            base_cpu = base.detach().cpu()
            delta_cpu = delta.detach().cpu()
            steps = int(cand_cpu.shape[0])
            dims = int(cand_cpu.shape[1])
            for t in range(steps):
                for d in range(dims):
                    row[f"candidate_action_t{t}_d{d}"] = float(cand_cpu[t, d])
                    row[f"fast_action_t{t}_d{d}"] = float(base_cpu[t, d])
                    row[f"action_delta_t{t}_d{d}"] = float(delta_cpu[t, d])
            row["candidate_action_norm"] = float(torch.linalg.vector_norm(cand).detach().cpu())
            row["fast_action_norm"] = float(torch.linalg.vector_norm(base).detach().cpu())
            row["action_delta_norm"] = float(torch.linalg.vector_norm(delta).detach().cpu())
            row["action_delta_xy_norm"] = float(torch.linalg.vector_norm(delta[:, :2]).detach().cpu())
            row["action_delta_yaw_norm"] = float(torch.linalg.vector_norm(delta[:, 2]).detach().cpu())
            row["action_delta_abs_mean"] = float(delta.abs().mean().detach().cpu())
            row["action_delta_abs_max"] = float(delta.abs().max().detach().cpu())
            row["action_delta_xy_abs_mean"] = float(delta[:, :2].abs().mean().detach().cpu())
            row["action_delta_yaw_abs_mean"] = float(delta[:, 2].abs().mean().detach().cpu())
            row["action_delta_yaw_abs_max"] = float(delta[:, 2].abs().max().detach().cpu())
            row["final_action_delta_norm"] = float(torch.linalg.vector_norm(delta[-1]).detach().cpu())
            row["final_xy_delta_norm"] = float(torch.linalg.vector_norm(delta[-1, :2]).detach().cpu())
            row["final_yaw_delta_abs"] = float(delta[-1, 2].abs().detach().cpu())
            row["candidate_action_smoothness"] = float(torch.linalg.vector_norm(cand[1:] - cand[:-1]).detach().cpu())
            row["fast_action_smoothness"] = float(torch.linalg.vector_norm(base[1:] - base[:-1]).detach().cpu())
            row["action_delta_smoothness"] = float(torch.linalg.vector_norm(delta[1:] - delta[:-1]).detach().cpu())
            row["candidate_yaw_abs_mean"] = float(cand[:, 2].abs().mean().detach().cpu())
            row["fast_yaw_abs_mean"] = float(base[:, 2].abs().mean().detach().cpu())
            row["candidate_xy_abs_mean"] = float(cand[:, :2].abs().mean().detach().cpu())
            row["fast_xy_abs_mean"] = float(base[:, :2].abs().mean().detach().cpu())
        action_distance_values = [float("nan")] * k
        final_action_distance_values = [float("nan")] * k
        yaw_distance_values = [float("nan")] * k
        fast_loss_margin_values = [float("nan")] * k
        action_distance_values[fast_top1_local] = 0.0
        final_action_distance_values[fast_top1_local] = 0.0
        yaw_distance_values[fast_top1_local] = 0.0
        fast_loss_margin_values[fast_top1_local] = 0.0
        for local in range(min_rank - 1, max_rank):
            pair = torch.cat([
                local_features[local],
                local_features[fast_top1_local],
                local_features[local] - local_features[fast_top1_local],
            ], dim=0)
            candidate_rows.append(pair)
            candidate_locals.append(local)
            action_delta = local_actions[local] - fast_action
            action_distance_values[local] = float(torch.linalg.vector_norm(action_delta).detach().cpu())
            final_action_distance_values[local] = float(torch.linalg.vector_norm(action_delta[-1]).detach().cpu())
            yaw_distance_values[local] = float(action_delta[:, -1].abs().mean().detach().cpu())
            fast_loss_margin_values[local] = float(
                (local_fast_losses[local] - local_fast_losses[fast_top1_local]).detach().cpu()
            )
        scores = local_fast_losses.to(self.device)
        selected_local = 0
        p_better_values = [float("nan")] * k
        pred_delta_values = [float("nan")] * k
        confidence_values = [float("nan")] * k
        pair_features_raw_for_debug = None
        pair_features_norm_for_debug = None
        if candidate_rows:
            pair_features_raw = torch.stack(candidate_rows, dim=0)
            if isinstance(self.b18_challenger_model, torch.nn.ModuleList):
                pair_features = (
                    pair_features_raw.unsqueeze(0) - self.b18_challenger_feature_mean.unsqueeze(1)
                ) / self.b18_challenger_feature_std.clamp_min(1e-6).unsqueeze(1)
                pair_features_for_debug = pair_features.mean(dim=0)
            else:
                pair_features = (
                    pair_features_raw - self.b18_challenger_feature_mean.unsqueeze(0)
                ) / self.b18_challenger_feature_std.clamp_min(1e-6).unsqueeze(0)
                pair_features_for_debug = pair_features
            pair_features_raw_for_debug = pair_features_raw.detach().cpu().float()
            pair_features_norm_for_debug = pair_features_for_debug.detach().cpu().float()
            self.sync_cuda_for_timing()
            feature_time = time.perf_counter() - feature_start
            self.sync_cuda_for_timing()
            inference_start = time.perf_counter()
            with torch.inference_mode():
                if isinstance(self.b18_challenger_model, torch.nn.ModuleList):
                    pred_parts = []
                    better_parts = []
                    conf_parts = []
                    for member_idx, member in enumerate(self.b18_challenger_model):
                        pred_i, better_i, conf_i = member(pair_features[member_idx])
                        pred_parts.append(pred_i)
                        better_parts.append(better_i)
                        conf_parts.append(conf_i)
                    pred_delta = torch.stack(pred_parts, dim=0).mean(dim=0)
                    better_logit = torch.stack(better_parts, dim=0).mean(dim=0)
                    conf_logit = torch.stack(conf_parts, dim=0).mean(dim=0)
                else:
                    pred_delta, better_logit, conf_logit = self.b18_challenger_model(pair_features)
                p_better = torch.sigmoid(better_logit)
                confidence = torch.sigmoid(conf_logit)
            self.sync_cuda_for_timing()
            inference_time = time.perf_counter() - inference_start
            if self.args.rerank_backend == "b51_safegate":
                passing = (
                    (p_better >= float(self.args.b51_helpful_threshold))
                    & (confidence <= float(self.args.b51_harm_risk_threshold))
                    & (pred_delta <= float(self.args.b51_pred_aate_threshold))
                )
            else:
                passing = (
                    (p_better >= float(self.args.b18_challenger_p_threshold))
                    & (pred_delta <= float(self.args.b18_challenger_tau))
                    & (confidence >= float(self.args.b18_challenger_confidence_threshold))
                )
            min_predicted_delta = self.args.b18_challenger_min_predicted_delta
            if min_predicted_delta is not None:
                passing = passing & (pred_delta >= float(min_predicted_delta))
            max_action_distance = self.args.b18_challenger_max_action_distance
            if max_action_distance is not None:
                action_guard = torch.tensor(
                    [action_distance_values[local] <= float(max_action_distance) for local in candidate_locals],
                    device=passing.device,
                    dtype=torch.bool,
                )
                passing = passing & action_guard
            min_action_distance = self.args.b18_challenger_min_action_distance
            if min_action_distance is not None:
                action_guard = torch.tensor(
                    [action_distance_values[local] >= float(min_action_distance) for local in candidate_locals],
                    device=passing.device,
                    dtype=torch.bool,
                )
                passing = passing & action_guard
            max_fast_loss_margin = self.args.b18_challenger_max_fast_loss_margin
            if max_fast_loss_margin is not None:
                fast_margin_guard = torch.tensor(
                    [fast_loss_margin_values[local] <= float(max_fast_loss_margin) for local in candidate_locals],
                    device=passing.device,
                    dtype=torch.bool,
                )
                passing = passing & fast_margin_guard
            max_yaw_distance = self.args.b18_challenger_max_yaw_distance
            if max_yaw_distance is not None:
                yaw_guard = torch.tensor(
                    [yaw_distance_values[local] <= float(max_yaw_distance) for local in candidate_locals],
                    device=passing.device,
                    dtype=torch.bool,
                )
                passing = passing & yaw_guard
            if bool(passing.any().detach().cpu().item()):
                pass_idx = torch.where(passing)[0]
                best_pass = pass_idx[torch.argmin(pred_delta[pass_idx])]
                selected_local = int(candidate_locals[int(best_pass.detach().cpu().item())])
                scores = scores + torch.arange(k, device=scores.device, dtype=scores.dtype) * 1e-6
                scores[selected_local] = scores[0] - scores.new_tensor(1e-4)
            for j, local in enumerate(candidate_locals):
                p_better_values[local] = float(p_better[j].detach().cpu())
                pred_delta_values[local] = float(pred_delta[j].detach().cpu())
                confidence_values[local] = float(confidence[j].detach().cpu())
        else:
            self.sync_cuda_for_timing()
            feature_time = time.perf_counter() - feature_start
            inference_time = 0.0
        self.last_student_rerank_score_debug = {
            "raw_scores": self.json_float_list(scores.detach().to(self.device)),
            "residual": pred_delta_values,
            "confidence": confidence_values,
            "b18_p_better": p_better_values,
            "b18_predicted_delta": pred_delta_values,
            "b18_action_distance": action_distance_values,
            "b18_final_action_distance": final_action_distance_values,
            "b18_yaw_distance": yaw_distance_values,
            "b18_fast_loss_margin": fast_loss_margin_values,
            "b18_candidate_features": self.tensor_nested_float_list(local_features.detach().cpu().float()),
            "b18_candidate_feature_original_indices": [
                int(rerank_idx_cpu[int(local)].detach().cpu().item()) for local in range(k)
            ],
            "b18_pair_features_raw": self.tensor_nested_float_list(pair_features_raw_for_debug),
            "b18_pair_features_norm": self.tensor_nested_float_list(pair_features_norm_for_debug),
            "b18_pair_feature_locals": [int(local) for local in candidate_locals],
            "b18_pair_feature_original_indices": [
                int(rerank_idx_cpu[int(local)].detach().cpu().item()) for local in candidate_locals
            ],
            "b18_selected_local": selected_local,
            "b18_promoted": bool(selected_local != 0),
            "b18_selected_p_better": p_better_values[selected_local],
            "b18_selected_predicted_delta": pred_delta_values[selected_local],
            "b18_selected_confidence": confidence_values[selected_local],
            "b18_selected_action_distance": action_distance_values[selected_local],
            "b18_selected_final_action_distance": final_action_distance_values[selected_local],
            "b18_selected_yaw_distance": yaw_distance_values[selected_local],
            "b18_selected_fast_loss_margin": fast_loss_margin_values[selected_local],
            "b51_helpful_score": p_better_values,
            "b51_predicted_AATE": pred_delta_values,
            "b51_harm_risk": confidence_values,
            "b51_selected_helpful_score": p_better_values[selected_local],
            "b51_selected_predicted_AATE": pred_delta_values[selected_local],
            "b51_selected_harm_risk": confidence_values[selected_local],
            "b51_helpful_threshold": float(self.args.b51_helpful_threshold),
            "b51_harm_risk_threshold": float(self.args.b51_harm_risk_threshold),
            "b51_pred_aate_threshold": float(self.args.b51_pred_aate_threshold),
            "b18_p_threshold": float(self.args.b18_challenger_p_threshold),
            "b18_tau": float(self.args.b18_challenger_tau),
            "b18_min_predicted_delta": self.args.b18_challenger_min_predicted_delta,
            "b18_confidence_threshold": float(self.args.b18_challenger_confidence_threshold),
            "b18_min_fast_rank": int(self.args.b18_challenger_min_fast_rank),
            "b18_min_action_distance": self.args.b18_challenger_min_action_distance,
            "b18_max_action_distance": self.args.b18_challenger_max_action_distance,
            "b18_max_fast_loss_margin": self.args.b18_challenger_max_fast_loss_margin,
            "b18_max_yaw_distance": self.args.b18_challenger_max_yaw_distance,
        }
        return scores.detach().to(self.device), feature_time, inference_time, selected_local

    def score_b48_outcome_student(
        self,
        deltas,
        fast_loss,
        fast_pred_final_latents,
        goal_latent,
        context_latents,
        rerank_idx_cpu,
    ):
        if self.b18_challenger_model is None:
            raise ValueError("B48 outcome student requires loaded B18 feature model")
        if self.b48_outcome_student is None:
            raise ValueError("B48 outcome student backend is not loaded")

        self.sync_cuda_for_timing()
        feature_start = time.perf_counter()
        all_features = self.build_b18_candidate_features(
            deltas, fast_loss, fast_pred_final_latents, goal_latent, context_latents
        ).to(self.device)
        rerank_idx_device = rerank_idx_cpu.to(all_features.device)
        local_features = all_features[rerank_idx_device]
        k = int(local_features.shape[0])
        if k < 1:
            raise ValueError("B48 outcome student requires at least one rerank candidate")

        fast_top1_local = 0
        max_rank = min(int(self.args.b48_outcome_max_fast_rank), k)
        min_rank = max(2, int(self.args.b48_outcome_min_fast_rank))
        local_actions = deltas[rerank_idx_cpu.to(deltas.device)].detach().float()
        fast_action = local_actions[fast_top1_local]
        local_fast_losses = fast_loss[rerank_idx_cpu.to(fast_loss.device)].detach().float()

        candidate_rows = []
        candidate_locals = []
        action_distance_values = [float("nan")] * k
        final_action_distance_values = [float("nan")] * k
        yaw_distance_values = [float("nan")] * k
        fast_loss_margin_values = [float("nan")] * k
        action_distance_values[fast_top1_local] = 0.0
        final_action_distance_values[fast_top1_local] = 0.0
        yaw_distance_values[fast_top1_local] = 0.0
        fast_loss_margin_values[fast_top1_local] = 0.0
        for local in range(1, max_rank):
            pair = torch.cat([
                local_features[local],
                local_features[fast_top1_local],
                local_features[local] - local_features[fast_top1_local],
            ], dim=0)
            candidate_rows.append(pair)
            candidate_locals.append(local)
            action_delta = local_actions[local] - fast_action
            action_distance_values[local] = float(torch.linalg.vector_norm(action_delta).detach().cpu())
            final_action_distance_values[local] = float(torch.linalg.vector_norm(action_delta[-1]).detach().cpu())
            yaw_distance_values[local] = float(action_delta[:, -1].abs().mean().detach().cpu())
            fast_loss_margin_values[local] = float(
                (local_fast_losses[local] - local_fast_losses[fast_top1_local]).detach().cpu()
            )

        scores = local_fast_losses.to(self.device)
        selected_local = 0
        b18_p_better_values = [float("nan")] * k
        b18_pred_delta_values = [float("nan")] * k
        b18_confidence_values = [float("nan")] * k
        b48_pred_date_values = [float("nan")] * k
        b48_p_helpful_values = [float("nan")] * k
        b48_p_harm_values = [float("nan")] * k
        b48_utility_values = [float("nan")] * k
        pair_features_raw_for_debug = None
        pair_features_norm_for_debug = None
        feature_time = 0.0
        inference_time = 0.0

        if candidate_rows:
            pair_features_raw = torch.stack(candidate_rows, dim=0)
            if isinstance(self.b18_challenger_model, torch.nn.ModuleList):
                pair_features = (
                    pair_features_raw.unsqueeze(0) - self.b18_challenger_feature_mean.unsqueeze(1)
                ) / self.b18_challenger_feature_std.clamp_min(1e-6).unsqueeze(1)
                pair_features_for_debug = pair_features.mean(dim=0)
            else:
                pair_features = (
                    pair_features_raw - self.b18_challenger_feature_mean.unsqueeze(0)
                ) / self.b18_challenger_feature_std.clamp_min(1e-6).unsqueeze(0)
                pair_features_for_debug = pair_features
            pair_features_raw_for_debug = pair_features_raw.detach().cpu().float()
            pair_features_norm_for_debug = pair_features_for_debug.detach().cpu().float()
            self.sync_cuda_for_timing()
            feature_time = time.perf_counter() - feature_start

            self.sync_cuda_for_timing()
            inference_start = time.perf_counter()
            with torch.inference_mode():
                if isinstance(self.b18_challenger_model, torch.nn.ModuleList):
                    pred_parts = []
                    better_parts = []
                    conf_parts = []
                    for member_idx, member in enumerate(self.b18_challenger_model):
                        pred_i, better_i, conf_i = member(pair_features[member_idx])
                        pred_parts.append(pred_i)
                        better_parts.append(better_i)
                        conf_parts.append(conf_i)
                    b18_pred_delta = torch.stack(pred_parts, dim=0).mean(dim=0)
                    b18_better_logit = torch.stack(better_parts, dim=0).mean(dim=0)
                    b18_conf_logit = torch.stack(conf_parts, dim=0).mean(dim=0)
                else:
                    b18_pred_delta, b18_better_logit, b18_conf_logit = self.b18_challenger_model(pair_features)
                b18_p_better = torch.sigmoid(b18_better_logit)
                b18_confidence = torch.sigmoid(b18_conf_logit)

            pair_np = pair_features_for_debug.detach().cpu().numpy()
            rows = []
            for j, local in enumerate(candidate_locals):
                fast_rank = local + 1
                p_better = float(b18_p_better[j].detach().cpu())
                predicted_delta = float(b18_pred_delta[j].detach().cpu())
                action_distance = action_distance_values[local]
                row = {
                    "candidate_local_rank": fast_rank,
                    "rank_minus_2": fast_rank - 2,
                    "rank_over_16": fast_rank / 16.0,
                    "b18_p_better": p_better,
                    "b18_predicted_delta": predicted_delta,
                    "abs_b18_predicted_delta": abs(predicted_delta),
                    "b18_action_distance": action_distance,
                    "b18_fast_loss_margin": fast_loss_margin_values[local],
                    "p_x_action": p_better * action_distance,
                    "pred_x_action": predicted_delta * action_distance,
                    "rank_x_action": fast_rank * action_distance,
                    "rank_x_pred": fast_rank * predicted_delta,
                }
                add_action_feature_columns(row, local_actions[local], fast_action)
                for idx, value in enumerate(pair_np[j]):
                    row[f"pair_norm_{idx:02d}"] = float(value)
                rows.append(row)

            import pandas as pd
            frame = pd.DataFrame(rows)
            group_base_cols = [
                "b18_p_better",
                "b18_predicted_delta",
                "b18_action_distance",
                "b18_fast_loss_margin",
            ]
            for col in group_base_cols:
                if col in frame.columns:
                    values = frame[col].astype(float)
                    g_mean = float(values.mean())
                    g_std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    g_min = float(values.min())
                    g_max = float(values.max())
                    frame[f"group_{col}_mean"] = g_mean
                    frame[f"group_{col}_std"] = g_std
                    frame[f"group_{col}_min"] = g_min
                    frame[f"group_{col}_max"] = g_max
                    frame[f"{col}_minus_group_mean"] = values - g_mean
                    frame[f"{col}_minus_group_min"] = values - g_min
                    frame[f"group_{col}_range"] = g_max - g_min
            frame["group_candidate_count"] = len(frame)
            frame["group_rank_min"] = float(frame["candidate_local_rank"].min()) if len(frame) else 0.0
            frame["group_rank_max"] = float(frame["candidate_local_rank"].max()) if len(frame) else 0.0
            missing = [col for col in self.b48_outcome_feature_cols if col not in frame.columns]
            if missing:
                raise ValueError(f"B48 online feature frame missing columns: {missing[:10]}")
            frame = frame[self.b48_outcome_feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            b48_feature_rows_for_debug = frame.copy()
            b48_feature_z_absmax_for_debug = None
            model = self.b48_outcome_student
            x = frame.values
            if model.get("kind") == "portable_linear":
                mean = np.asarray(model["scaler_mean"], dtype=np.float32)
                scale = np.asarray(model["scaler_scale"], dtype=np.float32)
                x = (x - mean.reshape(1, -1)) / np.maximum(scale.reshape(1, -1), 1e-6)
                b48_feature_z_absmax_for_debug = np.nanmax(np.abs(x), axis=0)
                b48_pred_date = x @ np.asarray(model["reg_coef"], dtype=np.float32) + float(model["reg_intercept"])
                helpful_logit = x @ np.asarray(model["helpful_coef"], dtype=np.float32) + float(model["helpful_intercept"])
                harm_logit = x @ np.asarray(model["harm_coef"], dtype=np.float32) + float(model["harm_intercept"])
                b48_p_helpful = 1.0 / (1.0 + np.exp(-helpful_logit))
                b48_p_harm = 1.0 / (1.0 + np.exp(-harm_logit))
            elif model.get("kind") == "portable_mlp":
                mean = np.asarray(model["feature_mean"], dtype=np.float32)
                scale = np.asarray(model["feature_scale"], dtype=np.float32)
                h = (x - mean.reshape(1, -1)) / np.maximum(scale.reshape(1, -1), 1e-6)
                feature_clip = model.get("feature_clip")
                if feature_clip is not None:
                    h = np.clip(h, -float(feature_clip), float(feature_clip))
                b48_feature_z_absmax_for_debug = np.nanmax(np.abs(h), axis=0)
                for layer in model["layers"]:
                    w = np.asarray(layer["weight"], dtype=np.float32)
                    b = np.asarray(layer["bias"], dtype=np.float32)
                    h = h @ w.T + b.reshape(1, -1)
                    if layer.get("activation") == "relu":
                        h = np.maximum(h, 0.0)
                if h.shape[1] < 3:
                    raise ValueError("B48 portable MLP must output [pred_dATE, helpful_logit, harm_logit]")
                date_target_mean = float(model.get("date_target_mean", 0.0))
                date_target_scale = float(model.get("date_target_scale", 1.0))
                b48_pred_date = h[:, 0] * date_target_scale + date_target_mean
                helpful_logit = h[:, 1]
                harm_logit = h[:, 2]
                b48_p_helpful = 1.0 / (1.0 + np.exp(-helpful_logit))
                b48_p_harm = 1.0 / (1.0 + np.exp(-harm_logit))
            else:
                if model.get("scaler") is not None:
                    x = model["scaler"].transform(x)
                b48_pred_date = model["reg"].predict(x)
                b48_p_helpful = model["helpful"].predict_proba(x)[:, 1]
                b48_p_harm = model["harm"].predict_proba(x)[:, 1]
            b48_utility = b48_pred_date + 0.05 * b48_p_harm - 0.01 * b48_p_helpful

            self.sync_cuda_for_timing()
            inference_time = time.perf_counter() - inference_start
            candidate_ranks = np.asarray([local + 1 for local in candidate_locals])
            fast_topk_spread = float((local_fast_losses[-1] - local_fast_losses[0]).detach().cpu())
            fast_top1_loss = float(local_fast_losses[0].detach().cpu())
            challengeable_prob = float("nan")
            dynamic_gate_name = None
            use_dynamic_gate = bool(self.args.b48_outcome_dynamic_gate)
            dynamic_challengeable = False
            if use_dynamic_gate:
                spread_min = self.args.b48_outcome_dynamic_spread_min
                top1_loss_min = self.args.b48_outcome_dynamic_top1_loss_min
                dynamic_challengeable = (
                    (spread_min is not None and fast_topk_spread >= float(spread_min))
                    or (top1_loss_min is not None and fast_top1_loss >= float(top1_loss_min))
                )
            if self.b48_dynamic_policy is not None:
                model_cfg = self.b48_dynamic_policy["challengeable_model"]
                feature_names = list(model_cfg["feature_names"])
                group_feature_values = {
                    "fast_topk_spread": fast_topk_spread,
                    "fast_top1_loss": fast_top1_loss,
                    "min_pred_dATE": float(np.nanmin(b48_pred_date)),
                    "mean_pred_dATE": float(np.nanmean(b48_pred_date)),
                    "max_p_helpful": float(np.nanmax(b48_p_helpful)),
                    "mean_p_helpful": float(np.nanmean(b48_p_helpful)),
                    "min_p_harm": float(np.nanmin(b48_p_harm)),
                    "mean_p_harm": float(np.nanmean(b48_p_harm)),
                    "min_utility": float(np.nanmin(b48_utility)),
                    "loose_pass_count": float(np.sum(
                        (candidate_ranks >= 3)
                        & (candidate_ranks <= 10)
                        & (b48_p_helpful >= 0.50)
                        & (b48_p_harm <= 0.60)
                        & (b48_pred_date <= -0.02)
                    )),
                    "strong_pass_count": float(np.sum(
                        (candidate_ranks >= 3)
                        & (candidate_ranks <= 8)
                        & (b48_p_helpful >= 0.58)
                        & (b48_p_harm <= 0.45)
                        & (b48_pred_date <= -0.08)
                    )),
                }
                raw_features = np.asarray([group_feature_values[name] for name in feature_names], dtype=np.float32)
                mean = np.asarray(model_cfg["scaler_mean"], dtype=np.float32)
                scale = np.asarray(model_cfg["scaler_scale"], dtype=np.float32)
                coef = np.asarray(model_cfg["coef"], dtype=np.float32)
                z = (raw_features - mean) / np.maximum(scale, 1e-6)
                logit = float(z @ coef + float(model_cfg["intercept"]))
                challengeable_prob = float(1.0 / (1.0 + np.exp(-logit)))
                gates = self.b48_dynamic_policy["gates"]
                high_min = float(gates.get("high_p_min", 0.70))
                low_max = float(gates.get("low_p_max", 0.35))
                if challengeable_prob >= high_min:
                    dynamic_gate_name = "high"
                elif challengeable_prob <= low_max:
                    dynamic_gate_name = "low"
                else:
                    dynamic_gate_name = "mid"
                gate_cfg = gates[dynamic_gate_name]
                dynamic_challengeable = dynamic_gate_name != "low"
                use_dynamic_gate = True
            gate_p_helpful_min = float(self.args.b48_outcome_p_helpful_min)
            gate_p_harm_max = float(self.args.b48_outcome_p_harm_max)
            gate_pred_date_max = float(self.args.b48_outcome_pred_date_max)
            gate_min_rank = min_rank
            gate_max_rank = max_rank
            if self.b48_dynamic_policy is not None:
                gate_p_helpful_min = float(gate_cfg["p_helpful_min"])
                gate_p_harm_max = float(gate_cfg["p_harm_max"])
                gate_pred_date_max = float(gate_cfg["pred_dATE_max"])
                gate_min_rank = max(2, int(gate_cfg["rank_min"]))
                gate_max_rank = min(k, int(gate_cfg["rank_max"]))
            if use_dynamic_gate and not dynamic_challengeable:
                if self.args.b48_outcome_conservative_p_helpful_min is not None:
                    gate_p_helpful_min = float(self.args.b48_outcome_conservative_p_helpful_min)
                if self.args.b48_outcome_conservative_p_harm_max is not None:
                    gate_p_harm_max = float(self.args.b48_outcome_conservative_p_harm_max)
                if self.args.b48_outcome_conservative_pred_date_max is not None:
                    gate_pred_date_max = float(self.args.b48_outcome_conservative_pred_date_max)
                if self.args.b48_outcome_conservative_min_fast_rank is not None:
                    gate_min_rank = max(2, int(self.args.b48_outcome_conservative_min_fast_rank))
                if self.args.b48_outcome_conservative_max_fast_rank is not None:
                    gate_max_rank = min(k, int(self.args.b48_outcome_conservative_max_fast_rank))
            groupfirst_candidate_mask = (
                (candidate_ranks >= gate_min_rank)
                & (candidate_ranks <= gate_max_rank)
            )
            passing = (
                groupfirst_candidate_mask
                & (b48_p_helpful >= gate_p_helpful_min)
                & (b48_p_harm <= gate_p_harm_max)
                & (b48_pred_date <= gate_pred_date_max)
            )
            min_delta = self.args.b48_outcome_min_pred_date
            if min_delta is not None:
                passing = passing & (b48_pred_date >= float(min_delta))
            max_action_distance = self.args.b48_outcome_max_action_distance
            if max_action_distance is not None:
                action_guard = np.asarray(
                    [action_distance_values[local] <= float(max_action_distance) for local in candidate_locals],
                    dtype=bool,
                )
                groupfirst_candidate_mask = groupfirst_candidate_mask & action_guard
                passing = passing & action_guard
            max_fast_loss_margin = self.args.b48_outcome_max_fast_loss_margin
            if max_fast_loss_margin is not None:
                fast_margin_guard = np.asarray(
                    [fast_loss_margin_values[local] <= float(max_fast_loss_margin) for local in candidate_locals],
                    dtype=bool,
                )
                groupfirst_candidate_mask = groupfirst_candidate_mask & fast_margin_guard
                passing = passing & fast_margin_guard
            max_yaw_distance = self.args.b48_outcome_max_yaw_distance
            if max_yaw_distance is not None:
                yaw_guard = np.asarray(
                    [yaw_distance_values[local] <= float(max_yaw_distance) for local in candidate_locals],
                    dtype=bool,
                )
                groupfirst_candidate_mask = groupfirst_candidate_mask & yaw_guard
                passing = passing & yaw_guard
            min_fast_top16_spread = self.args.b48_outcome_min_fast_top16_spread
            if min_fast_top16_spread is not None:
                spread_guard = fast_topk_spread >= float(min_fast_top16_spread)
                groupfirst_candidate_mask = groupfirst_candidate_mask & spread_guard
                passing = passing & spread_guard
            min_fast_top1_loss = self.args.b48_outcome_min_fast_top1_loss
            if min_fast_top1_loss is not None:
                top1_guard = fast_top1_loss >= float(min_fast_top1_loss)
                groupfirst_candidate_mask = groupfirst_candidate_mask & top1_guard
                passing = passing & top1_guard
            if bool(self.args.b48_outcome_groupfirst_gate) and bool(np.any(groupfirst_candidate_mask)):
                candidate_idx = np.where(groupfirst_candidate_mask)[0]
                best_pre_gate = int(candidate_idx[np.lexsort((b48_pred_date[candidate_idx], b48_utility[candidate_idx]))[0]])
                if bool(passing[best_pre_gate]):
                    selected_local = int(candidate_locals[best_pre_gate])
                    scores = scores + torch.arange(k, device=scores.device, dtype=scores.dtype) * 1e-6
                    scores[selected_local] = scores[0] - scores.new_tensor(1e-4)
            elif bool(np.any(passing)):
                pass_idx = np.where(passing)[0]
                best_pass = int(pass_idx[np.lexsort((b48_pred_date[pass_idx], b48_utility[pass_idx]))[0]])
                selected_local = int(candidate_locals[best_pass])
                scores = scores + torch.arange(k, device=scores.device, dtype=scores.dtype) * 1e-6
                scores[selected_local] = scores[0] - scores.new_tensor(1e-4)
            for j, local in enumerate(candidate_locals):
                b18_p_better_values[local] = float(b18_p_better[j].detach().cpu())
                b18_pred_delta_values[local] = float(b18_pred_delta[j].detach().cpu())
                b18_confidence_values[local] = float(b18_confidence[j].detach().cpu())
                b48_pred_date_values[local] = float(b48_pred_date[j])
                b48_p_helpful_values[local] = float(b48_p_helpful[j])
                b48_p_harm_values[local] = float(b48_p_harm[j])
                b48_utility_values[local] = float(b48_utility[j])
        else:
            self.sync_cuda_for_timing()
            feature_time = time.perf_counter() - feature_start

        self.last_student_rerank_score_debug = {
            "raw_scores": self.json_float_list(scores.detach().to(self.device)),
            "residual": b48_pred_date_values,
            "confidence": b48_p_helpful_values,
            "b18_p_better": b18_p_better_values,
            "b18_predicted_delta": b18_pred_delta_values,
            "b18_action_distance": action_distance_values,
            "b18_final_action_distance": final_action_distance_values,
            "b18_yaw_distance": yaw_distance_values,
            "b18_fast_loss_margin": fast_loss_margin_values,
            "b18_candidate_features": self.tensor_nested_float_list(local_features.detach().cpu().float()),
            "b18_candidate_feature_original_indices": [
                int(rerank_idx_cpu[int(local)].detach().cpu().item()) for local in range(k)
            ],
            "b18_pair_features_raw": self.tensor_nested_float_list(pair_features_raw_for_debug),
            "b18_pair_features_norm": self.tensor_nested_float_list(pair_features_norm_for_debug),
            "b18_pair_feature_locals": [int(local) for local in candidate_locals],
            "b18_pair_feature_original_indices": [
                int(rerank_idx_cpu[int(local)].detach().cpu().item()) for local in candidate_locals
            ],
            "b18_selected_local": selected_local,
            "b18_promoted": bool(selected_local != 0),
            "b18_selected_p_better": b18_p_better_values[selected_local],
            "b18_selected_predicted_delta": b18_pred_delta_values[selected_local],
            "b18_selected_confidence": b18_confidence_values[selected_local],
            "b18_selected_action_distance": action_distance_values[selected_local],
            "b18_selected_final_action_distance": final_action_distance_values[selected_local],
            "b18_selected_yaw_distance": yaw_distance_values[selected_local],
            "b18_selected_fast_loss_margin": fast_loss_margin_values[selected_local],
            "b48_pred_dATE": b48_pred_date_values,
            "b48_p_helpful": b48_p_helpful_values,
            "b48_p_harm": b48_p_harm_values,
            "b48_utility": b48_utility_values,
            "b48_feature_cols": list(self.b48_outcome_feature_cols),
            "b48_online_feature_rows": (
                b48_feature_rows_for_debug.astype(float).to_dict(orient="records")
                if "b48_feature_rows_for_debug" in locals() else None
            ),
            "b48_online_feature_z_absmax": (
                b48_feature_z_absmax_for_debug.astype(float).tolist()
                if b48_feature_z_absmax_for_debug is not None else None
            ),
            "b48_selected_pred_dATE": b48_pred_date_values[selected_local],
            "b48_selected_p_helpful": b48_p_helpful_values[selected_local],
            "b48_selected_p_harm": b48_p_harm_values[selected_local],
            "b48_selected_utility": b48_utility_values[selected_local],
            "b48_p_helpful_min": float(self.args.b48_outcome_p_helpful_min),
            "b48_p_harm_max": float(self.args.b48_outcome_p_harm_max),
            "b48_pred_date_max": float(self.args.b48_outcome_pred_date_max),
            "b48_dynamic_gate": bool(self.args.b48_outcome_dynamic_gate),
            "b48_dynamic_challengeable": bool(dynamic_challengeable),
            "b48_dynamic_spread_min": self.args.b48_outcome_dynamic_spread_min,
            "b48_dynamic_top1_loss_min": self.args.b48_outcome_dynamic_top1_loss_min,
            "b48_challengeable_prob": challengeable_prob,
            "b48_dynamic_gate_name": dynamic_gate_name,
            "b48_runtime_gate_p_helpful_min": gate_p_helpful_min,
            "b48_runtime_gate_p_harm_max": gate_p_harm_max,
            "b48_runtime_gate_pred_date_max": gate_pred_date_max,
            "b48_runtime_gate_min_fast_rank": int(gate_min_rank),
            "b48_runtime_gate_max_fast_rank": int(gate_max_rank),
            "b48_groupfirst_gate": bool(self.args.b48_outcome_groupfirst_gate),
            "b48_min_pred_date": self.args.b48_outcome_min_pred_date,
            "b48_min_fast_rank": int(self.args.b48_outcome_min_fast_rank),
            "b48_max_fast_rank": int(self.args.b48_outcome_max_fast_rank),
            "b48_max_action_distance": self.args.b48_outcome_max_action_distance,
            "b48_max_fast_loss_margin": self.args.b48_outcome_max_fast_loss_margin,
            "b48_max_yaw_distance": self.args.b48_outcome_max_yaw_distance,
            "b48_fast_top16_spread": fast_topk_spread if candidate_rows else None,
            "b48_fast_top1_loss": fast_top1_loss if candidate_rows else None,
            "b48_min_fast_top16_spread": self.args.b48_outcome_min_fast_top16_spread,
            "b48_min_fast_top1_loss": self.args.b48_outcome_min_fast_top1_loss,
        }
        return scores.detach().to(self.device), feature_time, inference_time, selected_local


    def load_s_combined_temporal_student(self):
        if self.args.s_combined_temporal_checkpoint is None:
            raise ValueError("--s_combined_temporal_checkpoint is required for --rerank_backend s_combined_temporal")
        helper_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics", "b48_unified_outcome_student")
        if helper_dir not in sys.path:
            sys.path.insert(0, helper_dir)
        from planner_s_combined_temporal_runtime import load_s_combined_temporal_checkpoint

        self.s_combined_temporal_state = load_s_combined_temporal_checkpoint(
            self.args.s_combined_temporal_checkpoint,
            self.device,
        )
        if not self.args.export_fast_rollout_trace:
            raise ValueError("s_combined_temporal requires --export_fast_rollout_trace")
        print(
            "Loaded experimental S combined temporal student "
            f"path={self.args.s_combined_temporal_checkpoint}, "
            f"rank_range={self.s_combined_temporal_state.rank_min}-{self.s_combined_temporal_state.rank_max}, "
            f"old_features={len(self.s_combined_temporal_state.old_feature_cols)}"
        )

    def load_s_groupfirst_ensemble_student(self):
        if self.args.s_groupfirst_ensemble_checkpoint is None:
            raise ValueError("--s_groupfirst_ensemble_checkpoint is required for --rerank_backend s_groupfirst_ensemble")
        helper_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics", "b48_unified_outcome_student")
        if helper_dir not in sys.path:
            sys.path.insert(0, helper_dir)
        from planner_s_groupfirst_ensemble_runtime import load_s_groupfirst_ensemble_checkpoint

        self.s_groupfirst_ensemble_state = load_s_groupfirst_ensemble_checkpoint(
            self.args.s_groupfirst_ensemble_checkpoint,
            self.device,
        )
        if getattr(self.args, "s_groupfirst_risk_veto_json", None):
            import json as _json
            risk_veto = _json.loads(self.args.s_groupfirst_risk_veto_json)
            self.s_groupfirst_ensemble_state.checkpoint.setdefault("policy", {})["risk_veto"] = risk_veto
        if not self.args.export_fast_rollout_trace:
            raise ValueError("s_groupfirst_ensemble requires --export_fast_rollout_trace")
        print(
            "Loaded experimental S group-first ensemble student "
            f"path={self.args.s_groupfirst_ensemble_checkpoint}, "
            f"rank_range={self.s_groupfirst_ensemble_state.rank_min}-{self.s_groupfirst_ensemble_state.rank_max}, "
            f"old_features={len(self.s_groupfirst_ensemble_state.old_feature_cols)}, "
            f"threshold={self.s_groupfirst_ensemble_state.threshold}"
        )

    def load_top5_boundary_gate(self):
        if self.args.top5_boundary_gate_checkpoint is None:
            raise ValueError("--top5_boundary_gate_checkpoint is required for --rerank_backend top5_boundary_gate")
        helper_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "teacher_baseline")
        if helper_dir not in sys.path:
            sys.path.insert(0, helper_dir)
        from train_top5_boundary_gate import Top5BoundaryGate

        checkpoint = torch.load(self.args.top5_boundary_gate_checkpoint, map_location="cpu", weights_only=False)
        saved_args = checkpoint.get("args", {})
        feature_dim = int(checkpoint["feature_dim"])
        model = Top5BoundaryGate(
            feature_dim=feature_dim,
            hidden_dim=int(saved_args.get("hidden_dim", 96)),
            dropout=float(saved_args.get("dropout", 0.05)),
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval().to(self.device)
        self.top5_boundary_gate_model = model
        self.top5_boundary_gate_feature_mean = torch.as_tensor(checkpoint["feature_mean"]).detach().float().to(self.device)
        self.top5_boundary_gate_feature_std = torch.as_tensor(checkpoint["feature_std"]).detach().float().to(self.device)
        self.top5_boundary_gate_args = dict(saved_args)
        print(
            "Loaded top5 boundary gate "
            f"path={self.args.top5_boundary_gate_checkpoint}, "
            f"feature_dim={feature_dim}, hidden_dim={saved_args.get('hidden_dim', 96)}, "
            f"threshold={self.args.top5_boundary_threshold}, "
            f"remove_rank={self.args.top5_boundary_remove_rank}, "
            f"use_remove_head={self.args.top5_boundary_use_remove_head}"
        )

    def build_top5_boundary_gate_features(self, deltas, fast_loss, fast_pred_final_latents, goal_latent, rerank_idx_cpu):
        helper_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "teacher_baseline")
        if helper_dir not in sys.path:
            sys.path.insert(0, helper_dir)
        from latent_rerank_distill_dataset import build_candidate_features

        rerank_idx_device = rerank_idx_cpu.to(fast_loss.device)
        single_goal_latent = goal_latent[0] if goal_latent.ndim == 4 else goal_latent
        gate_args = self.top5_boundary_gate_args
        features = build_candidate_features(
            deltas[rerank_idx_cpu].detach().cpu(),
            fast_loss[rerank_idx_device].detach().cpu(),
            fast_pred_final_latents[rerank_idx_device].detach().cpu(),
            single_goal_latent.detach().cpu(),
            include_pooled_latents=bool(gate_args.get("include_pooled_latents", False)),
            include_action_geometry=True,
            include_list_context_features=True,
            list_context_variant=str(gate_args.get("list_context_variant", "full")),
            list_context_std_floor=float(gate_args.get("list_context_std_floor", 1e-3)),
            list_context_clip=float(gate_args.get("list_context_clip", 5.0)),
            feature_clip_value=float(gate_args.get("feature_clip_value", 10.0)),
            normalize_action_features=bool(gate_args.get("normalize_action_features", False)),
            normalize_latent_summary_features=bool(gate_args.get("normalize_latent_summary_features", False)),
            normalize_base_features=bool(gate_args.get("normalize_base_features", False)),
            action_feature_mode=str(gate_args.get("action_feature_mode", "raw")),
            action_yaw_clip=float(gate_args.get("action_yaw_clip", 3.14159265)),
            action_feature_clip=float(gate_args.get("action_feature_clip", 10.0)),
        ).float().to(self.device)
        if features.shape[-1] != self.top5_boundary_gate_feature_mean.numel():
            raise ValueError(
                f"online top5 boundary feature dim {features.shape[-1]} does not match checkpoint "
                f"feature dim {self.top5_boundary_gate_feature_mean.numel()}"
            )
        return features

    def score_top5_boundary_gate(self, deltas, fast_loss, fast_pred_final_latents, goal_latent, rerank_idx_cpu):
        if self.top5_boundary_gate_model is None:
            raise ValueError("top5 boundary gate is not loaded")
        self.sync_cuda_for_timing()
        feature_start = time.perf_counter()
        features = self.build_top5_boundary_gate_features(
            deltas, fast_loss, fast_pred_final_latents, goal_latent, rerank_idx_cpu
        )
        features = (
            features - self.top5_boundary_gate_feature_mean.unsqueeze(0)
        ) / self.top5_boundary_gate_feature_std.clamp_min(1e-6).unsqueeze(0)
        self.sync_cuda_for_timing()
        feature_time = time.perf_counter() - feature_start

        self.sync_cuda_for_timing()
        inference_start = time.perf_counter()
        with torch.inference_mode():
            enter_logit, remove_logits = self.top5_boundary_gate_model(features.unsqueeze(0))
            enter_prob = torch.sigmoid(enter_logit).flatten()[0]
            remove_logits = remove_logits.squeeze(0)
            remove_rank = int(torch.argmax(remove_logits).detach().cpu().item())
        self.sync_cuda_for_timing()
        inference_time = time.perf_counter() - inference_start
        return float(enter_prob.detach().cpu()), remove_rank, feature_time, inference_time

    def normalize_student_fast_baseline(self, fast_loss_topm):
        values = fast_loss_topm.detach().float()
        centered = values - values.mean()
        scale = values.std(unbiased=False).clamp_min(1e-6)
        return centered / scale

    def clip_student_residual(self, residual):
        clip = float(self.student_rerank_residual_clip)
        if clip <= 0:
            return residual
        return clip * torch.tanh(residual / clip)

    def score_student_rerank(
        self,
        deltas,
        fast_loss,
        fast_pred_final_latents,
        goal_latent,
        rerank_idx_cpu,
        force_zero_residual=False,
    ):
        if self.student_rerank_model is None:
            raise ValueError("student rerank backend is not loaded")
        self.sync_cuda_for_timing()
        feature_start = time.perf_counter()
        features = self.build_student_rerank_features(
            deltas, fast_loss, fast_pred_final_latents, goal_latent, rerank_idx_cpu
        )
        features = (
            features - self.student_rerank_feature_mean.unsqueeze(0)
        ) / self.student_rerank_feature_std.clamp_min(1e-6).unsqueeze(0)
        self.sync_cuda_for_timing()
        feature_time = time.perf_counter() - feature_start

        self.sync_cuda_for_timing()
        inference_start = time.perf_counter()
        with torch.inference_mode():
            model_features = (
                features.unsqueeze(0)
                if self.student_rerank_arch == "deepsets_residual"
                else features
            )
            raw_scores = self.student_rerank_model(model_features)
            if raw_scores.ndim > 1:
                raw_scores = raw_scores.squeeze(0)
            if self.student_rerank_model_type in {"residual_fast", "residual_linear"}:
                clipped_residual = self.clip_student_residual(raw_scores)
                fast_loss_topm = fast_loss[rerank_idx_cpu.to(fast_loss.device)].detach().to(self.student_rerank_device)
                baseline = self.normalize_student_fast_baseline(fast_loss_topm)
                residual = torch.zeros_like(clipped_residual) if force_zero_residual else clipped_residual
                scores = float(self.student_rerank_fast_prior_weight) * baseline + residual
            else:
                residual = torch.zeros_like(raw_scores) if force_zero_residual else raw_scores
                scores = residual
            k = scores.numel()
            rank_norm = torch.arange(k, dtype=scores.dtype, device=scores.device) / max(1, k - 1)
            scores = scores + float(self.args.student_rank_prior_weight) * rank_norm
            confidence_debug = getattr(self.student_rerank_model, "last_confidence", None)
            if confidence_debug is None:
                confidence_debug = residual.new_full(residual.shape, float("nan"))
            elif confidence_debug.ndim > 1:
                confidence_debug = confidence_debug.squeeze(0)
            self.last_student_rerank_score_debug = {
                "raw_scores": self.json_float_list(raw_scores.detach().to(self.device)),
                "residual": self.json_float_list(residual.detach().to(self.device)),
                "residual_mean_abs": float(residual.detach().abs().mean().cpu()),
                "residual_max_abs": float(residual.detach().abs().max().cpu()),
                "confidence": self.json_float_list(confidence_debug.detach().to(self.device)),
                "rank_prior_weight": float(self.args.student_rank_prior_weight),
            }
        self.sync_cuda_for_timing()
        inference_time = time.perf_counter() - inference_start
        return scores.detach().to(self.device), feature_time, inference_time

    def apply_student_selection_guards(self, student_scores):
        guarded_scores = student_scores.clone()
        k = guarded_scores.numel()
        fallback_to_fast = False
        if self.args.student_max_local_rank is not None and self.args.student_max_local_rank >= 0:
            max_rank = int(self.args.student_max_local_rank)
            if max_rank + 1 < k:
                guarded_scores[max_rank + 1:] = float("inf")
        selected_local = int(torch.argmin(guarded_scores).detach().cpu().item())
        if self.args.student_select_margin_threshold > 0:
            best_score = float(student_scores[selected_local].detach().cpu())
            fast_top1_score = float(student_scores[0].detach().cpu())
            if best_score > fast_top1_score - float(self.args.student_select_margin_threshold):
                fallback_to_fast = True
                selected_local = 0
        if self.args.student_selective_gate:
            confidence = self.last_student_rerank_score_debug.get("confidence")
            residual = self.last_student_rerank_score_debug.get("residual")
            confidence_value = None
            residual_abs = None
            if isinstance(confidence, list) and selected_local < len(confidence):
                confidence_value = confidence[selected_local]
            if isinstance(residual, list) and selected_local < len(residual):
                residual_abs = abs(float(residual[selected_local]))
            ordered = torch.argsort(student_scores)
            student_margin = float(
                (student_scores[ordered[1]] - student_scores[ordered[0]]).abs().detach().cpu()
            ) if k > 1 else float("inf")
            gate_ok = True
            if confidence_value is None or not math.isfinite(float(confidence_value)):
                gate_ok = False
            elif float(confidence_value) <= float(self.args.student_confidence_threshold):
                gate_ok = False
            if student_margin <= float(self.args.student_margin_threshold):
                gate_ok = False
            if residual_abs is None or residual_abs >= float(self.args.student_residual_safety_threshold):
                gate_ok = False
            if self.args.student_max_local_rank is not None and self.args.student_max_local_rank >= 0:
                if selected_local > int(self.args.student_max_local_rank):
                    gate_ok = False
            if not gate_ok:
                fallback_to_fast = True
                selected_local = 0
        if selected_local == 0:
            guarded_scores[0] = torch.min(guarded_scores) - guarded_scores.new_tensor(1e-6)
        else:
            guarded_scores[selected_local] = torch.min(guarded_scores) - guarded_scores.new_tensor(1e-6)
        return guarded_scores, selected_local, fallback_to_fast

    def action_geometry_for_candidates(self, candidate_actions):
        actions = candidate_actions.detach().float()
        xy_norm = torch.linalg.norm(actions[..., :2], dim=-1)
        yaw = actions[..., 2] if actions.shape[-1] > 2 else actions.new_zeros(actions.shape[:-1])
        if yaw.shape[1] > 1:
            yaw_diff = wrap_to_pi(yaw[:, 1:] - yaw[:, :-1]).abs()
            yaw_smooth = yaw_diff.mean(dim=1)
        else:
            yaw_smooth = yaw.new_zeros(yaw.shape[0])
        return {
            "yaw_abs_max": yaw.abs().amax(dim=1),
            "yaw_abs_mean": yaw.abs().mean(dim=1),
            "yaw_smooth": yaw_smooth,
            "xy_norm_mean": xy_norm.mean(dim=1),
            "xy_norm_max": xy_norm.amax(dim=1),
        }

    def apply_student_cem_safety_gate(
        self,
        cem_scores,
        fast_loss_topm,
        candidate_actions_topm,
        current_top5_change_count,
    ):
        k = int(cem_scores.numel())
        topk = max(1, min(int(self.topk), k))
        fast_topk = set(range(topk))
        cem_order = torch.argsort(cem_scores).detach().cpu().tolist()
        cem_topk = set(int(x) for x in cem_order[:topk])
        entered = sorted(cem_topk - fast_topk)
        removed = sorted(fast_topk - cem_topk)
        debug = {
            "enabled": bool(self.args.student_cem_safety_gate),
            "accepted": True,
            "reason": "no_top5_change",
            "topk": topk,
            "entered_local": entered,
            "removed_local": removed,
            "change_count_before": int(current_top5_change_count),
            "promoted_geometry": {},
            "removed_geometry": {},
        }
        if not entered and not removed:
            return cem_scores, debug

        debug["reason"] = "accepted"
        max_changes = int(self.args.student_cem_max_top5_changes_per_sample)
        if max_changes >= 0 and int(current_top5_change_count) >= max_changes:
            debug["accepted"] = False
            debug["reason"] = "max_top5_changes"
            return fast_loss_topm.detach().clone(), debug

        if not entered or not removed:
            debug["accepted"] = False
            debug["reason"] = "non_pairwise_top5_change"
            return fast_loss_topm.detach().clone(), debug

        geom = self.action_geometry_for_candidates(candidate_actions_topm)
        entered_idx = torch.as_tensor(entered, device=candidate_actions_topm.device, dtype=torch.long)
        removed_idx = torch.as_tensor(removed, device=candidate_actions_topm.device, dtype=torch.long)
        checks = [
            ("yaw_abs_max", float(self.args.student_cem_yaw_abs_margin), None),
            ("yaw_abs_mean", float(self.args.student_cem_yaw_abs_margin), None),
            ("yaw_smooth", float(self.args.student_cem_yaw_smooth_margin), None),
            ("xy_norm_mean", float(self.args.student_cem_action_norm_margin), float(self.args.student_cem_action_norm_ratio)),
            ("xy_norm_max", float(self.args.student_cem_action_norm_margin), float(self.args.student_cem_action_norm_ratio)),
        ]
        for key, margin, ratio in checks:
            entered_value = geom[key][entered_idx].mean()
            removed_value = geom[key][removed_idx].mean()
            debug["promoted_geometry"][key] = float(entered_value.detach().cpu())
            debug["removed_geometry"][key] = float(removed_value.detach().cpu())
            limit = removed_value + entered_value.new_tensor(margin)
            if ratio is not None and ratio > 0:
                limit = torch.maximum(limit, removed_value * ratio + entered_value.new_tensor(margin))
            if bool((entered_value > limit).detach().cpu().item()):
                debug["accepted"] = False
                debug["reason"] = f"{key}_risk"
                return fast_loss_topm.detach().clone(), debug

        return cem_scores, debug

    def init_mu_sigma(self, obs_0, traj_len):
        n_evals = obs_0.shape[0]
        mu = torch.zeros(n_evals, self.action_dim) 
        mu[:, ] = torch.tensor(data_hyperparams[self.args.datasets]['mu'])
        sigma = torch.ones([n_evals, self.action_dim])
        sigma[:, ] = torch.tensor(data_hyperparams[self.args.datasets]['var_scale']) 
        return mu, sigma
        
    def generate_actions(self, dataset_save_output_dir, dataset_name, idxs, obs_image, goal_image, gt_actions, len_traj_pred, batch_idx=None):
        self.eval_timing = self.new_eval_timing()
        self.route2_timing = self.new_route2_timing()
        self.route2_cache_stats = self.new_route2_cache_stats()
        self.last_top_m_recall = None
        top_m_recall_start_idx = len(self.top_m_recall_rows)
        total_planning_start = time.perf_counter()
        debug_eval = os.environ.get("PLANNER_DEBUG_EVAL", "0") == "1"
        if debug_eval:
            print(
                f"[planner-debug] generate_actions start batch={batch_idx} "
                f"n_evals={obs_image.shape[0]} opt_steps={self.opt_steps}",
                flush=True,
            )
        idx_string = "_".join(map(str, idxs.flatten().int().tolist()))
        if dataset_save_output_dir is not None:
            image_plot_dir = os.path.join(dataset_save_output_dir, 'plots')
            os.makedirs(image_plot_dir, exist_ok=True)
        else:
            image_plot_dir = None
        
        n_evals = obs_image.shape[0]
        mu, sigma = self.init_mu_sigma(obs_image, len_traj_pred)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        student_cem_top5_change_counts = [0 for _ in range(n_evals)]
        cached_obs_latents = None
        cached_goal_latents = None
        use_latent_cache = (
            not self.args.compare_score_modes
            and (
                self.score_mode != "lpips"
                or self.args.planner_variant in [
                    "fast_latent",
                    "hybrid_latent_rerank",
                    "coarse_to_fine_latent",
                    "adaptive_latent_rerank",
                    "adaptive_two_gate_low_high",
                ]
            )
        )
        if use_latent_cache:
            self.sync_cuda_for_timing()
            start = time.perf_counter()
            cached_obs_latents = encode_rgb_to_latent(self.vae, obs_image, self.device, scale=True)
            self.sync_cuda_for_timing()
            self.route2_timing["context_encode_time"] += time.perf_counter() - start

            self.sync_cuda_for_timing()
            start = time.perf_counter()
            cached_goal_latents = encode_rgb_to_latent(self.vae, goal_image.squeeze(1), self.device, scale=True)
            self.sync_cuda_for_timing()
            self.route2_timing["goal_encode_time"] += time.perf_counter() - start
            if debug_eval:
                print("[planner-debug] latent cache ready", flush=True)

        for i in range(self.opt_steps):
            losses = []
            for traj in range(n_evals):
                traj_id = int(idxs.flatten()[traj].item())
                if debug_eval:
                    print(f"[planner-debug] opt={i} traj={traj} traj_id={traj_id} start", flush=True)
                self.sync_cuda_for_timing()
                candidate_generation_start = time.perf_counter()
                if self.args.per_eval_candidate_seed:
                    generator = torch.Generator(device=self.device)
                    candidate_seed = (
                        int(self.args.seed) * 1_000_003
                        + int(traj_id) * 9_176
                        + int(i) * 101
                    ) % (2**63 - 1)
                    generator.manual_seed(candidate_seed)
                    noise = torch.randn(
                        self.num_samples,
                        self.action_dim,
                        device=self.device,
                        generator=generator,
                    )
                else:
                    noise = torch.randn(self.num_samples, self.action_dim).to(self.device)
                sample = noise * sigma[traj] + mu[traj]
                single_delta = sample[:, :2]
                deltas = single_delta.unsqueeze(1).repeat(1, len_traj_pred, 1)
                # apply constraint by zeroing out specific action components
                constraint = getattr(self.args, 'constraint', 'none')
                if constraint == 'forward_first':
                    # first 5 steps: x only (zero y), last 3 steps: y only (zero x)
                    deltas[:, :5, 1] = 0
                    deltas[:, 5:, 0] = 0
                elif constraint == 'left_right_first':
                    # first 3 steps: y only (zero x/forward)
                    deltas[:, :3, 0] = 0
                elif constraint == 'straight_then_forward':
                    # first 3 steps: zero y (go straight)
                    deltas[:, :3, 1] = 0
                unnorm_deltas = unnormalize_data(deltas, ACTION_STATS_TORCH)
                delta_yaw = calculate_delta_yaw(unnorm_deltas)
                deltas = torch.cat((deltas, delta_yaw.to(deltas.device)), dim=-1)
                final_yaw_control = apply_planner_final_yaw_policy(
                    sample[:, -1] * np.pi,
                    self.args.planner_final_yaw_policy,
                )
                deltas[:, -1, -1] += final_yaw_control
                fixed_candidate_record = self.fixed_candidate_record_for(traj_id)
                fixed_fast_loss = None
                fixed_fast_sorted_indices = None
                if fixed_candidate_record is not None:
                    fixed_deltas = fixed_candidate_record["candidate_actions"].detach().float().to(self.device)
                    if fixed_deltas.ndim != 3 or fixed_deltas.shape[1:] != deltas.shape[1:]:
                        raise ValueError(
                            "Fixed candidate_actions must have shape [K,T,A] matching generated candidates; "
                            f"got {tuple(fixed_deltas.shape)}, expected [K,{deltas.shape[1]},{deltas.shape[2]}]"
                        )
                    if int(fixed_deltas.shape[0]) != int(self.num_samples):
                        raise ValueError(
                            f"Fixed candidate count {fixed_deltas.shape[0]} does not match num_samples={self.num_samples}"
                        )
                    deltas = fixed_deltas
                    if fixed_candidate_record.get("fast_losses") is not None:
                        fixed_fast_loss = fixed_candidate_record["fast_losses"].detach().float().to(self.device)
                        if fixed_fast_loss.ndim != 1 or fixed_fast_loss.shape[0] != deltas.shape[0]:
                            raise ValueError(
                                f"Fixed fast_losses must have shape [K], got {tuple(fixed_fast_loss.shape)}"
                            )
                        fixed_fast_sorted_indices = torch.argsort(fixed_fast_loss)
                self.sync_cuda_for_timing()
                self.route2_timing["candidate_generation_time"] += time.perf_counter() - candidate_generation_start

                cur_obs_image = obs_image[traj].unsqueeze(0).repeat(self.num_samples, 1, 1, 1, 1) 
                cur_goal_image = goal_image[traj].unsqueeze(0).repeat(self.args.num_samples, 1, 1, 1, 1).squeeze(1)
                cur_obs_latents = None
                cur_goal_latent = None
                teacher_fast_loss = None
                teacher_fast_sorted_indices = None
                teacher_shortlist_indices = None
                teacher_rerank_loss = None
                teacher_rerank_sorted_indices = None
                teacher_selected_shortlist_rank = None
                teacher_adaptive_mode = "unknown"
                teacher_normalized_margin = None
                teacher_raw_margin = None
                teacher_effective_step_equiv = None
                teacher_fast_pred_final_latents = None
                self.last_fast_rollout_trace = None
                teacher_rerank_pred_final_latents = None
                teacher_candidate_rng_state = None
                if cached_obs_latents is not None:
                    cur_obs_latents = cached_obs_latents[traj:traj + 1].repeat(self.num_samples, 1, 1, 1, 1)
                if cached_goal_latents is not None:
                    cur_goal_latent = cached_goal_latents[traj:traj + 1].repeat(self.num_samples, 1, 1, 1)
                elif self.score_mode != 'lpips':
                    cur_goal_latent = encode_rgb_to_latent(self.vae, cur_goal_image, self.device, scale=True)
                
                # WM is stochastic, so we can repeat the evaluation of each trajectory and average to reduce variance
                if self.args.compare_score_modes:
                    scores = self.compare_candidate_scores(dataset_name, traj_id, i, cur_obs_image, cur_goal_image, deltas)
                    loss = scores["lpips"]
                    preds = scores["preds_rgb"]
                elif self.args.planner_variant == "fast_latent":
                    self.sync_cuda_for_timing()
                    start = time.perf_counter()
                    before_timing = dict(self.eval_timing)
                    teacher_candidate_rng_state = (
                        self.capture_torch_rng_state()
                        if self.needs_teacher_rollout_rng() else None
                    )
                    loss, preds = self.score_candidate_deltas(
                        cur_obs_image, cur_goal_image, deltas, self.args.fast_score_mode, self.fast_diffusion,
                        precomputed_goal_latent=cur_goal_latent,
                        precomputed_obs_latents=cur_obs_latents,
                    )
                    if self.needs_teacher_candidate_fields():
                        teacher_fast_loss = loss
                        teacher_fast_pred_final_latents = preds
                        teacher_fast_sorted_indices = torch.argsort(loss)
                        teacher_raw_margin, _ = self.loss_margin(loss)
                        teacher_normalized_margin = float(self.normalized_loss_margin(loss).detach().cpu())
                    self.add_route2_timing_delta("coarse", before_timing)
                    self.sync_cuda_for_timing()
                    self.planner_runtimes["fast_shortlist_time"] += time.perf_counter() - start
                    if self.args.decode_for_debug:
                        preds = self.decode_latents_for_debug(preds)
                elif self.args.planner_variant == "hybrid_latent_rerank":
                    self.sync_cuda_for_timing()
                    start = time.perf_counter()
                    before_timing = dict(self.eval_timing)
                    teacher_candidate_rng_state = (
                        self.capture_torch_rng_state()
                        if self.needs_teacher_rollout_rng() else None
                    )
                    fast_loss, teacher_fast_pred_final_latents = self.score_candidate_deltas(
                        cur_obs_image, cur_goal_image, deltas, self.args.fast_score_mode, self.fast_diffusion,
                        precomputed_goal_latent=cur_goal_latent,
                        precomputed_obs_latents=cur_obs_latents,
                    )
                    if self.needs_teacher_candidate_fields():
                        teacher_fast_loss = fast_loss
                        teacher_fast_sorted_indices = torch.argsort(fast_loss)
                        teacher_raw_margin, _ = self.loss_margin(fast_loss)
                        teacher_normalized_margin = float(self.normalized_loss_margin(fast_loss).detach().cpu())
                    self.add_route2_timing_delta("coarse", before_timing)
                    self.sync_cuda_for_timing()
                    self.planner_runtimes["fast_shortlist_time"] += time.perf_counter() - start

                    self.sync_cuda_for_timing()
                    top_m_selection_start = time.perf_counter()
                    configured_rerank_top_m = (
                        self.args.student_rerank_top_m
                        if (
                            self.args.rerank_backend in self.learned_student_rerank_backends
                            or self.args.student_rerank_debug_compare_teacher
                        )
                        else self.args.rerank_top_m
                    )
                    rerank_count = max(1, min(configured_rerank_top_m, self.num_samples))
                    rerank_idx = torch.argsort(fast_loss)[:rerank_count]
                    # rerank_idx must be on CPU because candidate tensors are CPU-indexed here.
                    rerank_idx_cpu = rerank_idx.detach().cpu()
                    self.sync_cuda_for_timing()
                    self.route2_timing["top_m_selection_time"] += time.perf_counter() - top_m_selection_start
                    if self.args.compute_top_m_recall:
                        oracle_timing = self.new_eval_timing()
                        oracle_loss, _ = self.score_candidate_deltas(
                            cur_obs_image, cur_goal_image, deltas,
                            self.args.rerank_score_mode, self.rerank_diffusion,
                            timing=oracle_timing,
                            precomputed_goal_latent=cur_goal_latent,
                            precomputed_obs_latents=cur_obs_latents,
                        )
                        oracle_best_idx = int(torch.argmin(oracle_loss).detach().cpu().item())
                        coarse_top_m = [int(x) for x in rerank_idx.detach().cpu().tolist()]
                        recall_hit = oracle_best_idx in coarse_top_m
                        self.top_m_recall_rows.append({
                            "dataset": dataset_name,
                            "eval_index": traj_id,
                            "plan_iter": i,
                            "fast_score_mode": self.args.fast_score_mode,
                            "rerank_score_mode": self.args.rerank_score_mode,
                            "fast_diffusion_steps": self.args.fast_diffusion_steps,
                            "rerank_diffusion_steps": self.args.rerank_diffusion_steps,
                            "rerank_top_m": rerank_count,
                            "oracle_best_idx": oracle_best_idx,
                            "coarse_top_m": coarse_top_m,
                            "top_m_recall_full_best": recall_hit,
                        })
                    self.sync_cuda_for_timing()
                    start = time.perf_counter()
                    before_timing = dict(self.eval_timing)
                    rerank_idx_latent = rerank_idx.to(self.device)
                    rerank_goal_latent = cur_goal_latent[rerank_idx_latent] if cur_goal_latent is not None else None
                    rerank_obs_latents = cur_obs_latents[rerank_idx_latent] if cur_obs_latents is not None else None
                    rerank_loss, preds = self.score_candidate_deltas(
                        cur_obs_image[rerank_idx_cpu], cur_goal_image[rerank_idx_cpu], deltas[rerank_idx_cpu],
                        self.args.rerank_score_mode, self.rerank_diffusion,
                        precomputed_goal_latent=rerank_goal_latent,
                        precomputed_obs_latents=rerank_obs_latents,
                    )
                    teacher_rerank_pred_final_latents = preds
                    self.add_route2_timing_delta("rerank", before_timing)
                    self.sync_cuda_for_timing()
                    self.planner_runtimes["rerank_time"] += time.perf_counter() - start

                    loss = torch.full_like(fast_loss, float("inf"))
                    # Match dtype/device before writing rerank losses back into the full loss tensor.
                    rerank_loss = rerank_loss.to(device=loss.device, dtype=loss.dtype)
                    rerank_idx_for_loss = rerank_idx.to(loss.device)
                    loss[rerank_idx_for_loss] = rerank_loss
                    if self.args.rerank_score_mode != "lpips" and self.args.decode_for_debug:
                        preds = self.decode_latents_for_debug(preds)
                elif self.args.planner_variant == "coarse_to_fine_latent":
                    if debug_eval:
                        print(f"[planner-debug] opt={i} traj={traj} before coarse score", flush=True)
                    self.sync_cuda_for_timing()
                    ctf_total_start = time.perf_counter()
                    start = time.perf_counter()
                    before_timing = dict(self.eval_timing)
                    coarse_rng_state = self.capture_torch_rng_state()
                    teacher_candidate_rng_state = coarse_rng_state
                    if fixed_fast_loss is not None and self.args.fixed_candidate_actions_skip_fast_rollout:
                        fast_loss = fixed_fast_loss.to(self.device)
                        teacher_fast_pred_final_latents = cur_goal_latent.detach().clone()
                    else:
                        fast_loss, teacher_fast_pred_final_latents = self.score_candidate_deltas(
                            cur_obs_image, cur_goal_image, deltas, self.args.fast_score_mode, self.fast_diffusion,
                            precomputed_goal_latent=cur_goal_latent,
                            precomputed_obs_latents=cur_obs_latents,
                        )
                        if debug_eval:
                            print(f"[planner-debug] opt={i} traj={traj} after coarse score", flush=True)
                        if fixed_fast_loss is not None:
                            fast_loss = fixed_fast_loss.to(device=fast_loss.device, dtype=fast_loss.dtype)
                    self.add_route2_timing_delta("coarse", before_timing)
                    self.sync_cuda_for_timing()
                    fast_rollout_elapsed = time.perf_counter() - start
                    self.planner_runtimes["fast_shortlist_time"] += fast_rollout_elapsed

                    self.sync_cuda_for_timing()
                    top_m_selection_start = time.perf_counter()
                    rerank_count = max(1, min(self.args.rerank_top_m, self.num_samples))
                    # latent_cosine and the other latent modes are losses; lower is better.
                    # The low-step pass is only a filter. Final CEM ranking below uses only high-step rerank losses.
                    rerank_idx = torch.argsort(fast_loss)[:rerank_count]
                    rerank_idx_cpu = rerank_idx.detach().cpu()
                    fast_topk_indices = [int(x) for x in rerank_idx_cpu.tolist()]
                    fast_top1_index = fast_topk_indices[0]
                    fast_rank_indices = torch.argsort(fast_loss)
                    fast_rank_indices_list = [int(x) for x in fast_rank_indices.detach().cpu().tolist()]
                    fast_topk_losses = fast_loss[rerank_idx.to(fast_loss.device)]
                    if self.args.export_candidate_actions_path and i == self.opt_steps - 1:
                        self.candidate_actions_export_records.append({
                            "schema_version": "fixed_candidate_actions_v1",
                            "dataset_name": dataset_name,
                            "eval_index": int(traj_id),
                            "idx": int(traj_id),
                            "seed": int(self.args.seed),
                            "planner_variant": self.args.planner_variant,
                            "fast_diffusion_steps": int(self.args.fast_diffusion_steps),
                            "fast_score_mode": self.args.fast_score_mode,
                            "num_samples": int(self.num_samples),
                            "candidate_actions": deltas.detach().cpu(),
                            "candidate_set_hash": candidate_set_hash(deltas.detach().cpu()),
                            "fast_losses": fast_loss.detach().cpu(),
                            "fast_sorted_indices": fast_rank_indices.detach().cpu(),
                            "fast_rank_indices": fast_rank_indices.detach().cpu(),
                            "fast_top1_index": int(fast_top1_index),
                        })
                    fast_margin_top1_top2, fast_margin_top1_topk_last = self.loss_margin(
                        fast_loss, top_k_last=rerank_count
                    )
                    normalized_fast_margin_top1_top2 = float(
                        self.normalized_loss_margin(fast_loss).detach().cpu()
                    )
                    self.sync_cuda_for_timing()
                    self.route2_timing["top_m_selection_time"] += time.perf_counter() - top_m_selection_start

                    rerank_idx_latent = rerank_idx.to(self.device)
                    rerank_goal_latent = cur_goal_latent[rerank_idx_latent] if cur_goal_latent is not None else None
                    rerank_obs_latents = cur_obs_latents[rerank_idx_latent] if cur_obs_latents is not None else None
                    rerank_steps_for_case = self.args.rerank_diffusion_steps
                    adaptive_rerank_reason = "disabled"
                    adaptive_margin_value = None
                    if self.args.adaptive_rerank_steps_by_margin:
                        adaptive_margin_value = normalized_fast_margin_top1_top2
                        if normalized_fast_margin_top1_top2 >= self.args.adaptive_rerank_margin_threshold:
                            rerank_steps_for_case = self.args.adaptive_rerank_low_steps
                            adaptive_rerank_reason = "fast_margin_high_use_low_steps"
                        else:
                            rerank_steps_for_case = self.args.adaptive_rerank_high_steps
                            adaptive_rerank_reason = "fast_margin_low_use_high_steps"
                    rerank_diffusion_for_case = self.get_rerank_diffusion(rerank_steps_for_case)

                    student_scores = None
                    student_selected_original_idx = None
                    teacher_selected_original_idx = None
                    student_teacher_agreement = None
                    student_teacher_regret = None
                    teacher_compare_rerank_loss = None
                    student_fallback_to_fast = False
                    student_raw_selected_local = None
                    student_selected_local_guarded = None
                    student_feature_time = 0.0
                    student_inference_time = 0.0
                    student_total_rerank_time = 0.0
                    top5_boundary_prob = None
                    top5_boundary_pred_remove = None
                    top5_boundary_accepted = False
                    force_rank_xl_rerank_loss = None
                    skipped_diffusion_rerank = self.args.rerank_backend in self.shortcut_rerank_backends
                    if self.args.rerank_backend in self.learned_student_rerank_backends:
                        self.sync_cuda_for_timing()
                        student_total_start = time.perf_counter()
                        student_score_rng_state = capture_rng_state()
                        student_scores, student_feature_time, student_inference_time = self.score_student_rerank(
                            deltas,
                            fast_loss,
                            teacher_fast_pred_final_latents,
                            cur_goal_latent[0],
                            rerank_idx_cpu,
                            force_zero_residual=self.args.rerank_backend == "zero_residual_student",
                        )
                        restore_rng_state(student_score_rng_state)
                        self.sync_cuda_for_timing()
                        student_total_rerank_time = time.perf_counter() - student_total_start
                        self.route2_timing["student_feature_time"] += student_feature_time
                        self.route2_timing["student_inference_time"] += student_inference_time
                        self.route2_timing["student_total_rerank_time"] += student_total_rerank_time
                        self.route2_timing["student_rerank_count"] += 1
                        self.route2_timing["skipped_diffusion_rerank"] = True
                        student_raw_selected_local = int(torch.argmin(student_scores).detach().cpu().item())
                        rerank_loss, student_selected_local_guarded, student_fallback_to_fast = self.apply_student_selection_guards(
                            student_scores
                        )
                        preds = teacher_fast_pred_final_latents[rerank_idx.to(teacher_fast_pred_final_latents.device)]
                        if student_fallback_to_fast and self.args.student_gate_fallback == "teacher_diffusion":
                            self.restore_torch_rng_state(coarse_rng_state)
                            self.sync_cuda_for_timing()
                            start = time.perf_counter()
                            before_timing = dict(self.eval_timing)
                            rerank_loss, preds = self.score_candidate_deltas(
                                cur_obs_image[rerank_idx_cpu], cur_goal_image[rerank_idx_cpu], deltas[rerank_idx_cpu],
                                self.args.rerank_score_mode, rerank_diffusion_for_case,
                                precomputed_goal_latent=rerank_goal_latent,
                                precomputed_obs_latents=rerank_obs_latents,
                            )
                            self.add_route2_timing_delta("rerank", before_timing)
                            self.sync_cuda_for_timing()
                            rerank_rollout_elapsed = time.perf_counter() - start
                            self.planner_runtimes["rerank_time"] += rerank_rollout_elapsed
                            skipped_diffusion_rerank = False
                            self.route2_timing["skipped_diffusion_rerank"] = False
                    elif self.args.rerank_backend == "identity_fast_loss":
                        self.sync_cuda_for_timing()
                        identity_start = time.perf_counter()
                        rerank_loss = fast_loss[rerank_idx.to(fast_loss.device)].detach().clone()
                        preds = teacher_fast_pred_final_latents[rerank_idx.to(teacher_fast_pred_final_latents.device)]
                        self.sync_cuda_for_timing()
                        student_total_rerank_time = time.perf_counter() - identity_start
                        self.route2_timing["student_total_rerank_time"] += student_total_rerank_time
                        self.route2_timing["student_rerank_count"] += 1
                        self.route2_timing["skipped_diffusion_rerank"] = True
                        student_scores = rerank_loss.detach().to(self.device)
                        student_raw_selected_local = int(torch.argmin(student_scores).detach().cpu().item())
                        student_selected_local_guarded = student_raw_selected_local
                    elif self.args.rerank_backend == "force_fast_rank":
                        self.sync_cuda_for_timing()
                        force_start = time.perf_counter()
                        rerank_loss = fast_loss[rerank_idx.to(fast_loss.device)].detach().clone()
                        preds = teacher_fast_pred_final_latents[rerank_idx.to(teacher_fast_pred_final_latents.device)]
                        force_rank = max(1, int(self.args.force_fast_rank))
                        force_local = min(force_rank - 1, int(rerank_loss.numel()) - 1)
                        if self.args.force_fast_rank_score_with_rerank:
                            force_original = int(rerank_idx_cpu[force_local].item())
                            aux_rng_state = self.capture_torch_rng_state()
                            self.restore_torch_rng_state(coarse_rng_state)
                            try:
                                before_timing = dict(self.eval_timing)
                                forced_loss, _ = self.score_candidate_deltas(
                                    cur_obs_image[force_original:force_original + 1],
                                    cur_goal_image[force_original:force_original + 1],
                                    deltas[force_original:force_original + 1],
                                    self.args.rerank_score_mode,
                                    rerank_diffusion_for_case,
                                    precomputed_goal_latent=(
                                        cur_goal_latent[force_original:force_original + 1]
                                        if cur_goal_latent is not None else None
                                    ),
                                    precomputed_obs_latents=(
                                        cur_obs_latents[force_original:force_original + 1]
                                        if cur_obs_latents is not None else None
                                    ),
                                )
                                self.add_route2_timing_delta("rerank", before_timing)
                                force_rank_xl_rerank_loss = float(forced_loss[0].detach().cpu())
                            finally:
                                self.restore_torch_rng_state(aux_rng_state)
                        if force_local > 0:
                            rerank_loss[force_local] = rerank_loss[0] - rerank_loss.new_tensor(1e-4)
                        self.sync_cuda_for_timing()
                        student_total_rerank_time = time.perf_counter() - force_start
                        self.route2_timing["student_total_rerank_time"] += student_total_rerank_time
                        self.route2_timing["student_rerank_count"] += 1
                        self.route2_timing["skipped_diffusion_rerank"] = True
                        student_scores = rerank_loss.detach().to(self.device)
                        student_raw_selected_local = int(torch.argmin(student_scores).detach().cpu().item())
                        student_selected_local_guarded = student_raw_selected_local
                    elif self.args.rerank_backend in self.b18_challenger_backends:
                        self.sync_cuda_for_timing()
                        student_total_start = time.perf_counter()
                        b18_rng_state = capture_rng_state()
                        student_scores, student_feature_time, student_inference_time, b18_selected_local = self.score_b18_direct_challenger(
                            deltas,
                            fast_loss,
                            teacher_fast_pred_final_latents,
                            cur_goal_latent[0],
                            cur_obs_latents,
                            rerank_idx_cpu,
                        )
                        restore_rng_state(b18_rng_state)
                        rerank_loss = student_scores.detach().clone()
                        preds = teacher_fast_pred_final_latents[rerank_idx.to(teacher_fast_pred_final_latents.device)]
                        self.sync_cuda_for_timing()
                        student_total_rerank_time = time.perf_counter() - student_total_start
                        self.route2_timing["student_feature_time"] += student_feature_time
                        self.route2_timing["student_inference_time"] += student_inference_time
                        self.route2_timing["student_total_rerank_time"] += student_total_rerank_time
                        self.route2_timing["student_rerank_count"] += 1
                        self.route2_timing["skipped_diffusion_rerank"] = True
                        student_raw_selected_local = int(torch.argmin(student_scores).detach().cpu().item())
                        student_selected_local_guarded = int(b18_selected_local)
                        student_fallback_to_fast = int(b18_selected_local) == 0
                    elif self.args.rerank_backend == "b48_outcome_student":
                        self.sync_cuda_for_timing()
                        student_total_start = time.perf_counter()
                        b48_rng_state = capture_rng_state()
                        student_scores, student_feature_time, student_inference_time, b48_selected_local = self.score_b48_outcome_student(
                            deltas,
                            fast_loss,
                            teacher_fast_pred_final_latents,
                            cur_goal_latent[0],
                            cur_obs_latents,
                            rerank_idx_cpu,
                        )
                        restore_rng_state(b48_rng_state)
                        rerank_loss = student_scores.detach().clone()
                        preds = teacher_fast_pred_final_latents[rerank_idx.to(teacher_fast_pred_final_latents.device)]
                        self.sync_cuda_for_timing()
                        student_total_rerank_time = time.perf_counter() - student_total_start
                        self.route2_timing["student_feature_time"] += student_feature_time
                        self.route2_timing["student_inference_time"] += student_inference_time
                        self.route2_timing["student_total_rerank_time"] += student_total_rerank_time
                        self.route2_timing["student_rerank_count"] += 1
                        self.route2_timing["skipped_diffusion_rerank"] = True
                        student_raw_selected_local = int(torch.argmin(student_scores).detach().cpu().item())
                        student_selected_local_guarded = int(b48_selected_local)
                        student_fallback_to_fast = int(b48_selected_local) == 0
                    elif self.args.rerank_backend == "s_combined_temporal":
                        self.sync_cuda_for_timing()
                        student_total_start = time.perf_counter()
                        s_combined_rng_state = capture_rng_state()
                        helper_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics", "b48_unified_outcome_student")
                        if helper_dir not in sys.path:
                            sys.path.insert(0, helper_dir)
                        from planner_s_combined_temporal_runtime import score_s_combined_temporal
                        student_scores, student_feature_time, student_inference_time, s_combined_selected_local, s_combined_debug = score_s_combined_temporal(
                            self,
                            self.s_combined_temporal_state,
                            deltas,
                            fast_loss,
                            teacher_fast_pred_final_latents,
                            cur_goal_latent[0],
                            cur_obs_latents,
                            rerank_idx_cpu,
                        )
                        restore_rng_state(s_combined_rng_state)
                        rerank_loss = student_scores.detach().clone()
                        preds = teacher_fast_pred_final_latents[rerank_idx.to(teacher_fast_pred_final_latents.device)]
                        self.sync_cuda_for_timing()
                        student_total_rerank_time = time.perf_counter() - student_total_start
                        self.route2_timing["student_feature_time"] += student_feature_time
                        self.route2_timing["student_inference_time"] += student_inference_time
                        self.route2_timing["student_total_rerank_time"] += student_total_rerank_time
                        self.route2_timing["student_rerank_count"] += 1
                        self.route2_timing["skipped_diffusion_rerank"] = True
                        student_raw_selected_local = int(torch.argmin(student_scores).detach().cpu().item())
                        student_selected_local_guarded = int(s_combined_selected_local)
                        student_fallback_to_fast = int(s_combined_selected_local) == 0
                        self.last_student_rerank_score_debug = s_combined_debug
                    elif self.args.rerank_backend == "s_groupfirst_ensemble":
                        self.sync_cuda_for_timing()
                        student_total_start = time.perf_counter()
                        s_groupfirst_rng_state = capture_rng_state()
                        helper_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics", "b48_unified_outcome_student")
                        if helper_dir not in sys.path:
                            sys.path.insert(0, helper_dir)
                        from planner_s_groupfirst_ensemble_runtime import score_s_groupfirst_ensemble
                        student_scores, student_feature_time, student_inference_time, s_groupfirst_selected_local, s_groupfirst_debug = score_s_groupfirst_ensemble(
                            self,
                            self.s_groupfirst_ensemble_state,
                            deltas,
                            fast_loss,
                            teacher_fast_pred_final_latents,
                            cur_goal_latent[0],
                            cur_obs_latents,
                            rerank_idx_cpu,
                        )
                        restore_rng_state(s_groupfirst_rng_state)
                        rerank_loss = student_scores.detach().clone()
                        preds = teacher_fast_pred_final_latents[rerank_idx.to(teacher_fast_pred_final_latents.device)]
                        self.sync_cuda_for_timing()
                        student_total_rerank_time = time.perf_counter() - student_total_start
                        self.route2_timing["student_feature_time"] += student_feature_time
                        self.route2_timing["student_inference_time"] += student_inference_time
                        self.route2_timing["student_total_rerank_time"] += student_total_rerank_time
                        self.route2_timing["student_rerank_count"] += 1
                        self.route2_timing["skipped_diffusion_rerank"] = True
                        student_raw_selected_local = int(torch.argmin(student_scores).detach().cpu().item())
                        student_selected_local_guarded = int(s_groupfirst_selected_local)
                        student_fallback_to_fast = int(s_groupfirst_selected_local) == 0
                        self.last_student_rerank_score_debug = s_groupfirst_debug
                    elif self.args.rerank_backend == "s_hybrid_groupfirst_combined":
                        self.sync_cuda_for_timing()
                        student_total_start = time.perf_counter()
                        s_hybrid_rng_state = capture_rng_state()
                        helper_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics", "b48_unified_outcome_student")
                        if helper_dir not in sys.path:
                            sys.path.insert(0, helper_dir)
                        from planner_s_hybrid_groupfirst_combined_runtime import score_s_hybrid_groupfirst_combined
                        student_scores, student_feature_time, student_inference_time, s_hybrid_selected_local, s_hybrid_debug = score_s_hybrid_groupfirst_combined(
                            self,
                            self.s_groupfirst_ensemble_state,
                            self.s_combined_temporal_state,
                            self.s_hybrid_policy,
                            deltas,
                            fast_loss,
                            teacher_fast_pred_final_latents,
                            cur_goal_latent[0],
                            cur_obs_latents,
                            rerank_idx_cpu,
                        )
                        restore_rng_state(s_hybrid_rng_state)
                        rerank_loss = student_scores.detach().clone()
                        preds = teacher_fast_pred_final_latents[rerank_idx.to(teacher_fast_pred_final_latents.device)]
                        self.sync_cuda_for_timing()
                        student_total_rerank_time = time.perf_counter() - student_total_start
                        self.route2_timing["student_feature_time"] += student_feature_time
                        self.route2_timing["student_inference_time"] += student_inference_time
                        self.route2_timing["student_total_rerank_time"] += student_total_rerank_time
                        self.route2_timing["student_rerank_count"] += 1
                        self.route2_timing["skipped_diffusion_rerank"] = True
                        student_raw_selected_local = int(torch.argmin(student_scores).detach().cpu().item())
                        student_selected_local_guarded = int(s_hybrid_selected_local)
                        student_fallback_to_fast = int(s_hybrid_selected_local) == 0
                        self.last_student_rerank_score_debug = s_hybrid_debug
                    elif self.args.rerank_backend == "top5_boundary_gate":
                        self.sync_cuda_for_timing()
                        student_total_start = time.perf_counter()
                        top5_rng_state = capture_rng_state()
                        top5_boundary_prob, top5_boundary_pred_remove, student_feature_time, student_inference_time = self.score_top5_boundary_gate(
                            deltas,
                            fast_loss,
                            teacher_fast_pred_final_latents,
                            cur_goal_latent[0],
                            rerank_idx_cpu,
                        )
                        restore_rng_state(top5_rng_state)
                        rerank_loss = fast_loss[rerank_idx.to(fast_loss.device)].detach().clone()
                        preds = teacher_fast_pred_final_latents[rerank_idx.to(teacher_fast_pred_final_latents.device)]
                        self.sync_cuda_for_timing()
                        student_total_rerank_time = time.perf_counter() - student_total_start
                        self.route2_timing["student_feature_time"] += student_feature_time
                        self.route2_timing["student_inference_time"] += student_inference_time
                        self.route2_timing["student_total_rerank_time"] += student_total_rerank_time
                        self.route2_timing["student_rerank_count"] += 1
                        self.route2_timing["skipped_diffusion_rerank"] = True
                        student_scores = rerank_loss.detach().to(self.device)
                        student_raw_selected_local = int(torch.argmin(student_scores).detach().cpu().item())
                        student_selected_local_guarded = student_raw_selected_local
                    else:
                        self.restore_torch_rng_state(coarse_rng_state)
                        self.sync_cuda_for_timing()
                        start = time.perf_counter()
                        before_timing = dict(self.eval_timing)
                        rerank_loss, preds = self.score_candidate_deltas(
                            cur_obs_image[rerank_idx_cpu], cur_goal_image[rerank_idx_cpu], deltas[rerank_idx_cpu],
                            self.args.rerank_score_mode, rerank_diffusion_for_case,
                            precomputed_goal_latent=rerank_goal_latent,
                            precomputed_obs_latents=rerank_obs_latents,
                        )
                        teacher_rerank_pred_final_latents = preds
                        self.add_route2_timing_delta("rerank", before_timing)
                        self.sync_cuda_for_timing()
                        rerank_rollout_elapsed = time.perf_counter() - start
                        self.planner_runtimes["rerank_time"] += rerank_rollout_elapsed

                    selection_rerank_loss = rerank_loss
                    cem_rerank_loss = rerank_loss
                    cem_update_mode = "rerank_loss"
                    student_cem_gate_debug = {
                        "enabled": False,
                        "accepted": True,
                        "reason": "disabled",
                        "topk": int(self.topk),
                        "entered_local": [],
                        "removed_local": [],
                        "change_count_before": 0,
                        "promoted_geometry": {},
                        "removed_geometry": {},
                    }
                    if (
                        self.args.rerank_backend in self.learned_student_rerank_backends
                        and skipped_diffusion_rerank
                    ):
                        cem_update_mode = self.args.student_cem_update_mode
                        if cem_update_mode == "fast_loss":
                            cem_rerank_loss = fast_loss[rerank_idx.to(fast_loss.device)].detach().clone()
                        elif cem_update_mode == "student_score":
                            cem_rerank_loss = student_scores.detach().clone()
                        elif cem_update_mode == "guarded_student_score":
                            cem_rerank_loss = selection_rerank_loss.detach().clone()
                        else:
                            raise ValueError(f"unsupported student_cem_update_mode={cem_update_mode}")
                        if (
                            self.args.student_cem_safety_gate
                            and cem_update_mode == "student_score"
                        ):
                            fast_loss_topm = fast_loss[rerank_idx.to(fast_loss.device)].detach().clone()
                            candidate_actions_topm = deltas[rerank_idx.to(deltas.device)]
                            cem_rerank_loss, student_cem_gate_debug = self.apply_student_cem_safety_gate(
                                cem_rerank_loss,
                                fast_loss_topm,
                                candidate_actions_topm,
                                student_cem_top5_change_counts[traj],
                            )
                            if not student_cem_gate_debug["accepted"]:
                                cem_update_mode = f"{cem_update_mode}_safety_fallback_fast_loss"
                            elif student_cem_gate_debug["entered_local"] or student_cem_gate_debug["removed_local"]:
                                student_cem_top5_change_counts[traj] += 1
                    elif self.args.rerank_backend == "top5_boundary_gate" and skipped_diffusion_rerank:
                        fast_loss_topm = fast_loss[rerank_idx.to(fast_loss.device)].detach().clone()
                        cem_rerank_loss = fast_loss_topm.clone()
                        cem_update_mode = "top5_boundary_gate_fallback_fast_loss"
                        topk = max(1, min(int(self.topk), int(cem_rerank_loss.numel())))
                        promote_local = topk
                        remove_local = (
                            int(top5_boundary_pred_remove)
                            if self.args.top5_boundary_use_remove_head and top5_boundary_pred_remove is not None
                            else int(self.args.top5_boundary_remove_rank)
                        )
                        remove_local = max(0, min(topk - 1, remove_local))
                        student_cem_gate_debug = {
                            "enabled": True,
                            "accepted": False,
                            "reason": "prob_below_threshold",
                            "topk": int(topk),
                            "entered_local": [],
                            "removed_local": [],
                            "change_count_before": int(student_cem_top5_change_counts[traj]),
                            "promoted_geometry": {},
                            "removed_geometry": {},
                            "top5_boundary_prob": top5_boundary_prob,
                            "top5_boundary_pred_remove": top5_boundary_pred_remove,
                            "top5_boundary_remove_local": remove_local,
                        }
                        blocked_pred_remove = set()
                        if self.args.top5_boundary_disallow_pred_remove:
                            blocked_pred_remove = {
                                int(x.strip())
                                for x in str(self.args.top5_boundary_disallow_pred_remove).split(",")
                                if x.strip() != ""
                            }
                        if int(i) < int(self.args.top5_boundary_min_plan_iter):
                            student_cem_gate_debug["reason"] = "before_min_plan_iter"
                        elif (
                            top5_boundary_pred_remove is not None
                            and int(top5_boundary_pred_remove) in blocked_pred_remove
                        ):
                            student_cem_gate_debug["reason"] = "pred_remove_blocked"
                        elif (
                            self.args.top5_boundary_norm_margin_min is not None
                            and float(normalized_fast_margin_top1_top2) < float(self.args.top5_boundary_norm_margin_min)
                        ):
                            student_cem_gate_debug["reason"] = "norm_margin_below_min"
                        elif (
                            self.args.top5_boundary_norm_margin_max is not None
                            and float(normalized_fast_margin_top1_top2) > float(self.args.top5_boundary_norm_margin_max)
                        ):
                            student_cem_gate_debug["reason"] = "norm_margin_above_max"
                        elif top5_boundary_prob is not None and float(top5_boundary_prob) >= float(self.args.top5_boundary_threshold):
                            if promote_local >= int(cem_rerank_loss.numel()):
                                student_cem_gate_debug["reason"] = "not_enough_candidates"
                            else:
                                proposed_cem_loss = cem_rerank_loss.clone()
                                original_promote_score = proposed_cem_loss[promote_local].clone()
                                proposed_cem_loss[promote_local] = proposed_cem_loss[remove_local] - 1e-4
                                proposed_cem_loss[remove_local] = original_promote_score + 1e-4
                                student_cem_gate_debug.update({
                                    "accepted": True,
                                    "reason": "accepted",
                                    "entered_local": [int(promote_local)],
                                    "removed_local": [int(remove_local)],
                                })
                                if self.args.student_cem_safety_gate:
                                    candidate_actions_topm = deltas[rerank_idx.to(deltas.device)]
                                    proposed_cem_loss, student_cem_gate_debug = self.apply_student_cem_safety_gate(
                                        proposed_cem_loss,
                                        fast_loss_topm,
                                        candidate_actions_topm,
                                        student_cem_top5_change_counts[traj],
                                    )
                                    student_cem_gate_debug["top5_boundary_prob"] = top5_boundary_prob
                                    student_cem_gate_debug["top5_boundary_pred_remove"] = top5_boundary_pred_remove
                                    student_cem_gate_debug["top5_boundary_remove_local"] = remove_local
                                if student_cem_gate_debug["accepted"]:
                                    cem_rerank_loss = proposed_cem_loss
                                    cem_update_mode = "top5_boundary_gate"
                                    top5_boundary_accepted = True
                                    if student_cem_gate_debug["entered_local"] or student_cem_gate_debug["removed_local"]:
                                        student_cem_top5_change_counts[traj] += 1
                                else:
                                    cem_update_mode = "top5_boundary_gate_safety_fallback_fast_loss"

                    if self.args.student_rerank_debug_compare_teacher:
                        debug_rng_state = self.capture_torch_rng_state()
                        self.restore_torch_rng_state(coarse_rng_state)
                        debug_timing = self.new_eval_timing()
                        teacher_compare_rerank_loss, _ = self.score_candidate_deltas(
                            cur_obs_image[rerank_idx_cpu], cur_goal_image[rerank_idx_cpu], deltas[rerank_idx_cpu],
                            self.args.rerank_score_mode, rerank_diffusion_for_case,
                            timing=debug_timing,
                            precomputed_goal_latent=rerank_goal_latent,
                            precomputed_obs_latents=rerank_obs_latents,
                        )
                        self.restore_torch_rng_state(debug_rng_state)
                        if student_scores is None:
                            self.sync_cuda_for_timing()
                            student_total_start = time.perf_counter()
                            if self.args.rerank_backend == "identity_fast_loss":
                                student_scores = rerank_loss.detach().to(self.device)
                                student_feature_time = 0.0
                                student_inference_time = 0.0
                            else:
                                student_scores, student_feature_time, student_inference_time = self.score_student_rerank(
                                    deltas,
                                    fast_loss,
                                    teacher_fast_pred_final_latents,
                                    cur_goal_latent[0],
                                    rerank_idx_cpu,
                                    force_zero_residual=self.args.rerank_backend == "zero_residual_student",
                                )
                                self.route2_timing["student_rerank_count"] += 1
                            self.sync_cuda_for_timing()
                            student_total_rerank_time = time.perf_counter() - student_total_start
                            self.route2_timing["student_feature_time"] += student_feature_time
                            self.route2_timing["student_inference_time"] += student_inference_time
                            self.route2_timing["student_total_rerank_time"] += student_total_rerank_time
                        if student_raw_selected_local is None:
                            student_raw_selected_local = int(torch.argmin(student_scores).detach().cpu().item())
                        if student_selected_local_guarded is None:
                            _, student_selected_local_guarded, student_fallback_to_fast = self.apply_student_selection_guards(
                                student_scores
                            )
                        student_selected_local = int(student_selected_local_guarded)
                        teacher_selected_local_for_compare = int(torch.argmin(teacher_compare_rerank_loss).detach().cpu().item())
                        student_selected_original_idx = int(rerank_idx_cpu[student_selected_local].item())
                        teacher_selected_original_idx = int(rerank_idx_cpu[teacher_selected_local_for_compare].item())
                        student_teacher_agreement = student_selected_original_idx == teacher_selected_original_idx
                        student_teacher_regret = float(
                            (
                                teacher_compare_rerank_loss[student_selected_local]
                                - teacher_compare_rerank_loss[teacher_selected_local_for_compare]
                            ).detach().cpu()
                        )
                        self.route2_timing["student_teacher_compare_count"] += 1
                        self.route2_timing["student_teacher_agreement_count"] += int(student_teacher_agreement)
                        self.route2_timing["student_teacher_regret_sum"] += student_teacher_regret
                        self.route2_timing["student_teacher_regret_count"] += 1
                        self.student_rerank_debug_rows.append({
                            "dataset": dataset_name,
                            "eval_index": traj_id,
                            "plan_iter": i,
                            "seed": self.args.seed,
                            "rerank_backend": self.args.rerank_backend,
                            "fast_top_m_indices": json.dumps(fast_topk_indices),
                            "student_selected_index": student_selected_original_idx,
                            "teacher_selected_index": teacher_selected_original_idx,
                            "student_teacher_agreement": student_teacher_agreement,
                            "student_teacher_regret": student_teacher_regret,
                            "student_scores": self.json_float_list(student_scores),
                            "student_raw_scores": self.last_student_rerank_score_debug.get("raw_scores"),
                            "student_residual": self.last_student_rerank_score_debug.get("residual"),
                            "student_confidence": self.last_student_rerank_score_debug.get("confidence"),
                            "student_residual_mean_abs": self.last_student_rerank_score_debug.get("residual_mean_abs"),
                            "student_residual_max_abs": self.last_student_rerank_score_debug.get("residual_max_abs"),
                            "b18_p_better": self.last_student_rerank_score_debug.get("b18_p_better"),
                            "b18_predicted_delta": self.last_student_rerank_score_debug.get("b18_predicted_delta"),
                            "b18_promoted": self.last_student_rerank_score_debug.get("b18_promoted"),
                            "student_raw_selected_local_rank": student_raw_selected_local,
                            "student_selected_local_rank": student_selected_local,
                            "student_fallback_to_fast": student_fallback_to_fast,
                            "student_rank_prior_weight": self.args.student_rank_prior_weight,
                            "student_select_margin_threshold": self.args.student_select_margin_threshold,
                            "student_max_local_rank": self.args.student_max_local_rank,
                            "teacher_rerank_losses": self.json_float_list(teacher_compare_rerank_loss),
                        })

                    if (
                        self.args.rerank_backend in self.shortcut_rerank_backends
                        and not (
                            student_fallback_to_fast
                            and self.args.student_gate_fallback == "teacher_diffusion"
                        )
                    ):
                        rerank_rollout_elapsed = 0.0
                        self.planner_runtimes["rerank_time"] += student_total_rerank_time

                    loss = torch.full_like(fast_loss, float("inf"))
                    rerank_loss = selection_rerank_loss.to(device=loss.device, dtype=loss.dtype)
                    cem_rerank_loss = cem_rerank_loss.to(device=loss.device, dtype=loss.dtype)
                    rerank_idx_for_loss = rerank_idx.to(loss.device)
                    loss[rerank_idx_for_loss] = cem_rerank_loss
                    selected_local = int(torch.argmin(rerank_loss).detach().cpu().item())
                    selected_original_idx = int(rerank_idx_cpu[selected_local].item())
                    final_score = float(rerank_loss[selected_local].detach().cpu())
                    fast_score_of_selected = float(fast_loss[selected_original_idx].detach().cpu())
                    rerank_order_local = torch.argsort(rerank_loss)
                    rerank_rank_original_indices = [
                        int(rerank_idx_cpu[int(local_idx)].item())
                        for local_idx in rerank_order_local.detach().cpu().tolist()
                    ]
                    selected_fast_rank = fast_rank_indices_list.index(selected_original_idx) + 1
                    rerank_margin_top1_top2, _ = self.loss_margin(rerank_loss)
                    normalized_rerank_margin_top1_top2 = float(
                        self.normalized_loss_margin(rerank_loss).detach().cpu()
                    )
                    cem_update_order_local = torch.argsort(cem_rerank_loss)
                    cem_update_rank_original_indices = [
                        int(rerank_idx_cpu[int(local_idx)].item())
                        for local_idx in cem_update_order_local.detach().cpu().tolist()
                    ]
                    teacher_fast_loss = fast_loss
                    teacher_fast_sorted_indices = fast_rank_indices
                    teacher_shortlist_indices = rerank_idx_cpu
                    teacher_rerank_loss = teacher_compare_rerank_loss if teacher_compare_rerank_loss is not None else (
                        rerank_loss if self.args.rerank_backend in self.diffusion_rerank_backends else None
                    )
                    teacher_rerank_sorted_indices = torch.as_tensor(rerank_rank_original_indices)
                    teacher_selected_shortlist_rank = selected_local + 1
                    teacher_normalized_margin = normalized_fast_margin_top1_top2
                    teacher_raw_margin = fast_margin_top1_top2
                    teacher_effective_step_equiv = (
                        self.num_samples * self.args.fast_diffusion_steps
                        + rerank_count * rerank_steps_for_case
                    )
                    if not self.args.adaptive_rerank_steps_by_margin:
                        teacher_adaptive_mode = "fixed_rerank"
                    elif rerank_steps_for_case == self.args.adaptive_rerank_high_steps:
                        teacher_adaptive_mode = "high_fallback"
                    else:
                        teacher_adaptive_mode = "mid_rerank"

                    oracle_best_idx = None
                    oracle_hit = None
                    oracle_score_gap = None
                    full_high_losses_json = None
                    full_high_rank_indices_json = None
                    full_high_best_fast_rank = None
                    full_high_best_rerank_rank_if_in_topk = None
                    selected_matches_full_high_best = None
                    spearman_fast_vs_full_high = None
                    spearman_rerank_vs_full_high_on_topk = None
                    if self.args.diagnose_full_rerank_oracle:
                        post_rerank_rng_state = self.capture_torch_rng_state()
                        self.restore_torch_rng_state(coarse_rng_state)
                        oracle_timing = self.new_eval_timing()
                        oracle_loss, _ = self.score_candidate_deltas(
                            cur_obs_image, cur_goal_image, deltas,
                            self.args.rerank_score_mode, rerank_diffusion_for_case,
                            timing=oracle_timing,
                            precomputed_goal_latent=cur_goal_latent,
                            precomputed_obs_latents=cur_obs_latents,
                        )
                        oracle_best_idx = int(torch.argmin(oracle_loss).detach().cpu().item())
                        oracle_hit = oracle_best_idx in fast_topk_indices
                        full_high_rank_indices = torch.argsort(oracle_loss)
                        full_high_losses_json = self.json_float_list(oracle_loss)
                        full_high_rank_indices_json = self.json_int_list(full_high_rank_indices)
                        full_high_best_fast_rank = fast_rank_indices_list.index(oracle_best_idx) + 1
                        if oracle_hit:
                            full_high_best_rerank_rank_if_in_topk = (
                                rerank_rank_original_indices.index(oracle_best_idx) + 1
                            )
                        oracle_best_loss = float(oracle_loss[oracle_best_idx].detach().cpu())
                        oracle_score_gap = final_score - oracle_best_loss
                        selected_matches_full_high_best = selected_original_idx == oracle_best_idx
                        spearman_fast_vs_full_high = self.spearman_from_ranks(
                            self.candidate_ranks(fast_loss), self.candidate_ranks(oracle_loss)
                        )
                        full_high_on_topk = oracle_loss[rerank_idx.to(oracle_loss.device)]
                        spearman_rerank_vs_full_high_on_topk = self.spearman_from_ranks(
                            self.candidate_ranks(rerank_loss), self.candidate_ranks(full_high_on_topk)
                        )
                        self.restore_torch_rng_state(post_rerank_rng_state)

                    compare_steps = []
                    if self.args.diagnose_compare_rerank_steps:
                        compare_steps = [
                            int(x.strip())
                            for x in self.args.diagnose_compare_rerank_steps.split(",")
                            if x.strip()
                        ]
                    compare_selected_indices = {}
                    compare_losses = {}
                    compare_low_steps = None
                    compare_high_steps = None
                    compare_selected_index_low_steps = None
                    compare_selected_index_high_steps = None
                    compare_selected_match = None
                    compare_rank_low_selected_under_high = None
                    compare_rank_high_selected_under_low = None
                    compare_high_score_gap_low_selected_vs_high_selected = None
                    if compare_steps:
                        post_compare_rng_state = self.capture_torch_rng_state()
                        for compare_step in compare_steps:
                            self.restore_torch_rng_state(coarse_rng_state)
                            compare_loss, _ = self.score_candidate_deltas(
                                cur_obs_image[rerank_idx_cpu],
                                cur_goal_image[rerank_idx_cpu],
                                deltas[rerank_idx_cpu],
                                self.args.rerank_score_mode,
                                self.get_rerank_diffusion(compare_step),
                                timing=self.new_eval_timing(),
                                precomputed_goal_latent=rerank_goal_latent,
                                precomputed_obs_latents=rerank_obs_latents,
                            )
                            compare_loss = compare_loss.detach().to(device=rerank_loss.device, dtype=rerank_loss.dtype)
                            compare_losses[compare_step] = compare_loss
                            compare_selected_local = int(torch.argmin(compare_loss).detach().cpu().item())
                            compare_selected_indices[compare_step] = int(
                                rerank_idx_cpu[compare_selected_local].item()
                            )
                        if len(compare_steps) >= 2:
                            low_step = compare_steps[0]
                            high_step = compare_steps[-1]
                            compare_low_steps = low_step
                            compare_high_steps = high_step
                            low_selected_idx = compare_selected_indices[low_step]
                            high_selected_idx = compare_selected_indices[high_step]
                            compare_selected_index_low_steps = low_selected_idx
                            compare_selected_index_high_steps = high_selected_idx
                            compare_selected_match = low_selected_idx == high_selected_idx
                            low_losses = compare_losses[low_step]
                            high_losses = compare_losses[high_step]
                            low_rank_original = [
                                int(rerank_idx_cpu[int(local_idx)].item())
                                for local_idx in torch.argsort(low_losses).detach().cpu().tolist()
                            ]
                            high_rank_original = [
                                int(rerank_idx_cpu[int(local_idx)].item())
                                for local_idx in torch.argsort(high_losses).detach().cpu().tolist()
                            ]
                            compare_rank_low_selected_under_high = high_rank_original.index(low_selected_idx) + 1
                            compare_rank_high_selected_under_low = low_rank_original.index(high_selected_idx) + 1
                            low_selected_local = fast_topk_indices.index(low_selected_idx)
                            high_selected_local = fast_topk_indices.index(high_selected_idx)
                            compare_high_score_gap_low_selected_vs_high_selected = float(
                                (
                                    high_losses[low_selected_local] - high_losses[high_selected_local]
                                ).detach().cpu()
                            )
                        self.restore_torch_rng_state(post_compare_rng_state)

                    try:
                        coarse_candidate_set_hash = candidate_set_hash(deltas.detach().cpu())
                        coarse_candidate_action_hashes = candidate_action_hashes(deltas.detach().cpu())
                    except Exception as exc:
                        coarse_candidate_set_hash = None
                        coarse_candidate_action_hashes = None
                        print(f"Warning: failed to compute coarse candidate hashes for eval_index={traj_id}: {exc}")

                    self.coarse_to_fine_rows.append({
                        "dataset": dataset_name,
                        "scene_id": traj_id,
                        "sample_id": traj_id,
                        "plan_iter": i,
                        "seed": self.args.seed,
                        "num_samples": self.num_samples,
                        "fast_diffusion_steps": self.args.fast_diffusion_steps,
                        "rerank_diffusion_steps": rerank_steps_for_case,
                        "rerank_top_k": rerank_count,
                        "rerank_backend": self.args.rerank_backend,
                        "candidate_set_hash": coarse_candidate_set_hash,
                        "candidate_action_hashes_json": (
                            json.dumps(coarse_candidate_action_hashes)
                            if coarse_candidate_action_hashes is not None else None
                        ),
                        "candidate_actions_shape_json": json.dumps(list(deltas.shape)),
                        "student_rerank_checkpoint": self.args.student_rerank_checkpoint,
                        "student_rerank_top_m": self.args.student_rerank_top_m,
                        "fast_top1_index": fast_top1_index,
                        "fast_losses_json": self.json_float_list(fast_loss),
                        "fast_rank_indices_json": json.dumps(fast_rank_indices_list),
                        "rerank_selected_original_index": selected_original_idx,
                        "fast_topk_indices": fast_topk_indices,
                        "fast_topk_losses_json": self.json_float_list(fast_topk_losses),
                        "rerank_losses_json": self.json_float_list(rerank_loss),
                        "rerank_rank_original_indices_json": json.dumps(rerank_rank_original_indices),
                        "student_cem_update_mode": cem_update_mode,
                        "cem_update_losses_json": self.json_float_list(cem_rerank_loss),
                        "cem_update_rank_original_indices_json": json.dumps(cem_update_rank_original_indices),
                        "student_cem_safety_gate_enabled": student_cem_gate_debug["enabled"],
                        "student_cem_safety_gate_accepted": student_cem_gate_debug["accepted"],
                        "student_cem_safety_gate_reason": student_cem_gate_debug["reason"],
                        "student_cem_safety_gate_entered_local": json.dumps(student_cem_gate_debug["entered_local"]),
                        "student_cem_safety_gate_removed_local": json.dumps(student_cem_gate_debug["removed_local"]),
                        "student_cem_safety_gate_change_count_before": student_cem_gate_debug["change_count_before"],
                        "student_cem_safety_gate_promoted_geometry": json.dumps(student_cem_gate_debug["promoted_geometry"]),
                        "student_cem_safety_gate_removed_geometry": json.dumps(student_cem_gate_debug["removed_geometry"]),
                        "top5_boundary_prob": top5_boundary_prob,
                        "top5_boundary_pred_remove": top5_boundary_pred_remove,
                        "top5_boundary_accepted": top5_boundary_accepted,
                        "selected_original_idx": selected_original_idx,
                        "selected_local_idx": selected_local,
                        "selected_fast_rank": selected_fast_rank,
                        "force_rank_xl_rerank_loss": force_rank_xl_rerank_loss,
                        "selected_rerank_loss": final_score,
                        "selected_fast_loss": fast_score_of_selected,
                        "fast_margin_top1_top2": fast_margin_top1_top2,
                        "fast_margin_top1_topk_last": fast_margin_top1_topk_last,
                        "rerank_margin_top1_top2": rerank_margin_top1_top2,
                        "normalized_fast_margin_top1_top2": normalized_fast_margin_top1_top2,
                        "normalized_rerank_margin_top1_top2": normalized_rerank_margin_top1_top2,
                        "effective_step_units": (
                            self.num_samples * self.args.fast_diffusion_steps
                            + rerank_count * rerank_steps_for_case
                        ),
                        "adaptive_rerank_used_steps": rerank_steps_for_case,
                        "adaptive_rerank_reason": adaptive_rerank_reason,
                        "adaptive_margin_value": adaptive_margin_value,
                        "final_score": final_score,
                        "fast_score_of_selected": fast_score_of_selected,
                        "wall_clock_time_total": time.perf_counter() - ctf_total_start,
                        "wall_clock_time_fast_rollout": fast_rollout_elapsed,
                        "wall_clock_time_rerank_rollout": rerank_rollout_elapsed,
                        "student_feature_time": student_feature_time,
                        "student_inference_time": student_inference_time,
                        "student_total_rerank_time": student_total_rerank_time,
                        "skipped_diffusion_rerank": skipped_diffusion_rerank,
                        "student_selected_original_idx": student_selected_original_idx,
                        "teacher_selected_original_idx": teacher_selected_original_idx,
                        "student_teacher_agreement": student_teacher_agreement,
                        "student_teacher_regret": student_teacher_regret,
            "student_scores_json": self.json_float_list(student_scores) if student_scores is not None else None,
            "b18_p_better_json": (
                json.dumps(self.last_student_rerank_score_debug.get("b18_p_better"))
                if self.last_student_rerank_score_debug.get("b18_p_better") is not None else None
            ),
            "b18_predicted_delta_json": (
                json.dumps(self.last_student_rerank_score_debug.get("b18_predicted_delta"))
                if self.last_student_rerank_score_debug.get("b18_predicted_delta") is not None else None
            ),
            "b18_action_distance_json": (
                json.dumps(self.last_student_rerank_score_debug.get("b18_action_distance"))
                if self.last_student_rerank_score_debug.get("b18_action_distance") is not None else None
            ),
            "b18_final_action_distance_json": (
                json.dumps(self.last_student_rerank_score_debug.get("b18_final_action_distance"))
                if self.last_student_rerank_score_debug.get("b18_final_action_distance") is not None else None
            ),
            "b18_yaw_distance_json": (
                json.dumps(self.last_student_rerank_score_debug.get("b18_yaw_distance"))
                if self.last_student_rerank_score_debug.get("b18_yaw_distance") is not None else None
            ),
            "b18_fast_loss_margin_json": (
                json.dumps(self.last_student_rerank_score_debug.get("b18_fast_loss_margin"))
                if self.last_student_rerank_score_debug.get("b18_fast_loss_margin") is not None else None
            ),
            "b18_candidate_features_json": (
                json.dumps(self.last_student_rerank_score_debug.get("b18_candidate_features"))
                if self.last_student_rerank_score_debug.get("b18_candidate_features") is not None else None
            ),
            "b18_candidate_feature_original_indices_json": (
                json.dumps(self.last_student_rerank_score_debug.get("b18_candidate_feature_original_indices"))
                if self.last_student_rerank_score_debug.get("b18_candidate_feature_original_indices") is not None else None
            ),
            "b18_pair_features_raw_json": (
                json.dumps(self.last_student_rerank_score_debug.get("b18_pair_features_raw"))
                if self.last_student_rerank_score_debug.get("b18_pair_features_raw") is not None else None
            ),
            "b18_pair_features_norm_json": (
                json.dumps(self.last_student_rerank_score_debug.get("b18_pair_features_norm"))
                if self.last_student_rerank_score_debug.get("b18_pair_features_norm") is not None else None
            ),
            "b18_pair_feature_locals_json": (
                json.dumps(self.last_student_rerank_score_debug.get("b18_pair_feature_locals"))
                if self.last_student_rerank_score_debug.get("b18_pair_feature_locals") is not None else None
            ),
            "b18_pair_feature_original_indices_json": (
                json.dumps(self.last_student_rerank_score_debug.get("b18_pair_feature_original_indices"))
                if self.last_student_rerank_score_debug.get("b18_pair_feature_original_indices") is not None else None
            ),
            "b18_promoted": self.last_student_rerank_score_debug.get("b18_promoted"),
            "b18_selected_p_better": self.last_student_rerank_score_debug.get("b18_selected_p_better"),
            "b18_selected_predicted_delta": self.last_student_rerank_score_debug.get("b18_selected_predicted_delta"),
            "b18_selected_action_distance": self.last_student_rerank_score_debug.get("b18_selected_action_distance"),
            "b18_selected_final_action_distance": self.last_student_rerank_score_debug.get("b18_selected_final_action_distance"),
            "b18_selected_yaw_distance": self.last_student_rerank_score_debug.get("b18_selected_yaw_distance"),
            "b18_selected_fast_loss_margin": self.last_student_rerank_score_debug.get("b18_selected_fast_loss_margin"),
            "b48_pred_dATE_json": (
                json.dumps(self.last_student_rerank_score_debug.get("b48_pred_dATE"))
                if self.last_student_rerank_score_debug.get("b48_pred_dATE") is not None else None
            ),
            "b48_p_helpful_json": (
                json.dumps(self.last_student_rerank_score_debug.get("b48_p_helpful"))
                if self.last_student_rerank_score_debug.get("b48_p_helpful") is not None else None
            ),
            "b48_p_harm_json": (
                json.dumps(self.last_student_rerank_score_debug.get("b48_p_harm"))
                if self.last_student_rerank_score_debug.get("b48_p_harm") is not None else None
            ),
            "b48_utility_json": (
                json.dumps(self.last_student_rerank_score_debug.get("b48_utility"))
                if self.last_student_rerank_score_debug.get("b48_utility") is not None else None
            ),
            "b48_feature_cols_json": (
                json.dumps(self.last_student_rerank_score_debug.get("b48_feature_cols"))
                if self.last_student_rerank_score_debug.get("b48_feature_cols") is not None else None
            ),
            "b48_online_feature_rows_json": (
                json.dumps(self.last_student_rerank_score_debug.get("b48_online_feature_rows"))
                if self.last_student_rerank_score_debug.get("b48_online_feature_rows") is not None else None
            ),
            "b48_online_feature_z_absmax_json": (
                json.dumps(self.last_student_rerank_score_debug.get("b48_online_feature_z_absmax"))
                if self.last_student_rerank_score_debug.get("b48_online_feature_z_absmax") is not None else None
            ),
            "b48_selected_pred_dATE": self.last_student_rerank_score_debug.get("b48_selected_pred_dATE"),
            "b48_selected_p_helpful": self.last_student_rerank_score_debug.get("b48_selected_p_helpful"),
            "b48_selected_p_harm": self.last_student_rerank_score_debug.get("b48_selected_p_harm"),
            "b48_selected_utility": self.last_student_rerank_score_debug.get("b48_selected_utility"),
            "s_combined_pred_dATE_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_pred_dATE"))
                if self.last_student_rerank_score_debug.get("s_combined_pred_dATE") is not None else None
            ),
            "s_combined_pred_dRPE_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_pred_dRPE"))
                if self.last_student_rerank_score_debug.get("s_combined_pred_dRPE") is not None else None
            ),
            "s_combined_pred_dpos_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_pred_dpos"))
                if self.last_student_rerank_score_debug.get("s_combined_pred_dpos") is not None else None
            ),
            "s_combined_pred_dyaw_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_pred_dyaw"))
                if self.last_student_rerank_score_debug.get("s_combined_pred_dyaw") is not None else None
            ),
            "s_combined_p_helpful_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_p_helpful"))
                if self.last_student_rerank_score_debug.get("s_combined_p_helpful") is not None else None
            ),
            "s_combined_p_harm_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_p_harm"))
                if self.last_student_rerank_score_debug.get("s_combined_p_harm") is not None else None
            ),
            "s_combined_pred_dATE_std_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_pred_dATE_std"))
                if self.last_student_rerank_score_debug.get("s_combined_pred_dATE_std") is not None else None
            ),
            "s_combined_pred_dRPE_std_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_pred_dRPE_std"))
                if self.last_student_rerank_score_debug.get("s_combined_pred_dRPE_std") is not None else None
            ),
            "s_combined_pred_dpos_std_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_pred_dpos_std"))
                if self.last_student_rerank_score_debug.get("s_combined_pred_dpos_std") is not None else None
            ),
            "s_combined_pred_dyaw_std_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_pred_dyaw_std"))
                if self.last_student_rerank_score_debug.get("s_combined_pred_dyaw_std") is not None else None
            ),
            "s_combined_p_helpful_std_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_p_helpful_std"))
                if self.last_student_rerank_score_debug.get("s_combined_p_helpful_std") is not None else None
            ),
            "s_combined_p_harm_std_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_p_harm_std"))
                if self.last_student_rerank_score_debug.get("s_combined_p_harm_std") is not None else None
            ),
            "s_combined_p_select_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_p_select"))
                if self.last_student_rerank_score_debug.get("s_combined_p_select") is not None else None
            ),
            "s_combined_select_margin_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_select_margin"))
                if self.last_student_rerank_score_debug.get("s_combined_select_margin") is not None else None
            ),
            "s_combined_utility_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_utility"))
                if self.last_student_rerank_score_debug.get("s_combined_utility") is not None else None
            ),
            "s_combined_utility_std_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_utility_std"))
                if self.last_student_rerank_score_debug.get("s_combined_utility_std") is not None else None
            ),
            "s_combined_candidate_local_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_candidate_local"))
                if self.last_student_rerank_score_debug.get("s_combined_candidate_local") is not None else None
            ),
            "s_combined_candidate_rank_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_candidate_rank"))
                if self.last_student_rerank_score_debug.get("s_combined_candidate_rank") is not None else None
            ),
            "s_combined_candidate_fast_loss_vs_top1_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_candidate_fast_loss_vs_top1"))
                if self.last_student_rerank_score_debug.get("s_combined_candidate_fast_loss_vs_top1") is not None else None
            ),
            "s_combined_group_fast_top1_loss_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_group_fast_top1_loss"))
                if self.last_student_rerank_score_debug.get("s_combined_group_fast_top1_loss") is not None else None
            ),
            "s_combined_group_fast_top2_loss_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_group_fast_top2_loss"))
                if self.last_student_rerank_score_debug.get("s_combined_group_fast_top2_loss") is not None else None
            ),
            "s_combined_fast_margin_top1_top2_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_fast_margin_top1_top2"))
                if self.last_student_rerank_score_debug.get("s_combined_fast_margin_top1_top2") is not None else None
            ),
            "s_combined_normalized_fast_margin_top1_top2_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_normalized_fast_margin_top1_top2"))
                if self.last_student_rerank_score_debug.get("s_combined_normalized_fast_margin_top1_top2") is not None else None
            ),
            "s_combined_b18_p_better_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_b18_p_better"))
                if self.last_student_rerank_score_debug.get("s_combined_b18_p_better") is not None else None
            ),
            "s_combined_b18_pred_delta_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_b18_pred_delta"))
                if self.last_student_rerank_score_debug.get("s_combined_b18_pred_delta") is not None else None
            ),
            "s_combined_b18_action_distance_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_b18_action_distance"))
                if self.last_student_rerank_score_debug.get("s_combined_b18_action_distance") is not None else None
            ),
            "s_combined_b18_final_action_distance_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_b18_final_action_distance"))
                if self.last_student_rerank_score_debug.get("s_combined_b18_final_action_distance") is not None else None
            ),
            "s_combined_b18_yaw_distance_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_b18_yaw_distance"))
                if self.last_student_rerank_score_debug.get("s_combined_b18_yaw_distance") is not None else None
            ),
            "s_combined_b18_fast_loss_margin_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_b18_fast_loss_margin"))
                if self.last_student_rerank_score_debug.get("s_combined_b18_fast_loss_margin") is not None else None
            ),
            "s_combined_selected_local": self.last_student_rerank_score_debug.get("s_combined_selected_local"),
            "s_combined_gate_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_gate"))
                if self.last_student_rerank_score_debug.get("s_combined_gate") is not None else None
            ),
            "s_combined_p_fallback": self.last_student_rerank_score_debug.get("s_combined_p_fallback"),
            "s_combined_p_challenge": self.last_student_rerank_score_debug.get("s_combined_p_challenge"),
            "s_combined_guard_pred_dATE_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_guard_pred_dATE"))
                if self.last_student_rerank_score_debug.get("s_combined_guard_pred_dATE") is not None else None
            ),
            "s_combined_guard_pred_dRPE_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_guard_pred_dRPE"))
                if self.last_student_rerank_score_debug.get("s_combined_guard_pred_dRPE") is not None else None
            ),
            "s_combined_guard_pred_dpos_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_guard_pred_dpos"))
                if self.last_student_rerank_score_debug.get("s_combined_guard_pred_dpos") is not None else None
            ),
            "s_combined_guard_pred_dyaw_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_guard_pred_dyaw"))
                if self.last_student_rerank_score_debug.get("s_combined_guard_pred_dyaw") is not None else None
            ),
            "s_combined_guard_p_safe_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_guard_p_safe"))
                if self.last_student_rerank_score_debug.get("s_combined_guard_p_safe") is not None else None
            ),
            "s_combined_guard_p_harm_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_guard_p_harm"))
                if self.last_student_rerank_score_debug.get("s_combined_guard_p_harm") is not None else None
            ),
            "s_combined_guard_p_strong_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_guard_p_strong"))
                if self.last_student_rerank_score_debug.get("s_combined_guard_p_strong") is not None else None
            ),
            "s_combined_guard_score_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_guard_score"))
                if self.last_student_rerank_score_debug.get("s_combined_guard_score") is not None else None
            ),
            "teacher_debug_rerank_losses_json": (
                            self.json_float_list(teacher_compare_rerank_loss)
                            if teacher_compare_rerank_loss is not None else None
                        ),
                        "full_high_losses_json": full_high_losses_json,
                        "full_high_rank_indices_json": full_high_rank_indices_json,
                        "full_high_best_index": oracle_best_idx,
                        "full_high_best_in_fast_topk": oracle_hit,
                        "full_high_best_fast_rank": full_high_best_fast_rank,
                        "full_high_best_rerank_rank_if_in_topk": full_high_best_rerank_rank_if_in_topk,
                        "selected_vs_full_high_score_gap": oracle_score_gap,
                        "selected_matches_full_high_best": selected_matches_full_high_best,
                        "spearman_fast_vs_full_high": spearman_fast_vs_full_high,
                        "spearman_rerank_vs_full_high_on_topk": spearman_rerank_vs_full_high_on_topk,
                        "full_high_score_gap": oracle_score_gap,
                        "compare_rerank_steps": self.args.diagnose_compare_rerank_steps or None,
                        "compare_selected_indices_json": (
                            json.dumps({str(k): v for k, v in compare_selected_indices.items()})
                            if compare_selected_indices else None
                        ),
                        "compare_low_steps": compare_low_steps,
                        "compare_high_steps": compare_high_steps,
                        "compare_selected_index_low_steps": compare_selected_index_low_steps,
                        "compare_selected_index_high_steps": compare_selected_index_high_steps,
                        "compare_selected_indices_match": compare_selected_match,
                        "compare_rank_low_selected_under_high": compare_rank_low_selected_under_high,
                        "compare_rank_high_selected_under_low": compare_rank_high_selected_under_low,
                        "compare_high_score_gap_low_selected_vs_high_selected": (
                            compare_high_score_gap_low_selected_vs_high_selected
                        ),
                    })
                    if self.args.rerank_score_mode != "lpips" and self.args.decode_for_debug:
                        preds = self.decode_latents_for_debug(preds)
                elif self.args.planner_variant == "adaptive_latent_rerank":
                    self.sync_cuda_for_timing()
                    start = time.perf_counter()
                    before_timing = dict(self.eval_timing)
                    teacher_candidate_rng_state = (
                        self.capture_torch_rng_state()
                        if self.needs_teacher_rollout_rng() else None
                    )
                    fast_loss, preds = self.score_candidate_deltas(
                        cur_obs_image, cur_goal_image, deltas, self.args.fast_score_mode, self.fast_diffusion,
                        precomputed_goal_latent=cur_goal_latent,
                        precomputed_obs_latents=cur_obs_latents,
                    )
                    if self.needs_teacher_candidate_fields():
                        teacher_fast_loss = fast_loss
                        teacher_fast_pred_final_latents = preds
                        teacher_fast_sorted_indices = torch.argsort(fast_loss)
                        teacher_raw_margin, _ = self.loss_margin(fast_loss)
                        teacher_normalized_margin = float(self.normalized_loss_margin(fast_loss).detach().cpu())
                    self.add_route2_timing_delta("coarse", before_timing)
                    self.sync_cuda_for_timing()
                    self.planner_runtimes["fast_shortlist_time"] += time.perf_counter() - start

                    norm_margin = self.normalized_loss_margin(fast_loss)
                    rerank_count = max(1, min(self.args.rerank_top_m, self.num_samples))
                    fallback_count = max(1, min(self.args.adaptive_fallback_top_m, rerank_count))
                    coarse_cost = self.num_samples * self.args.fast_diffusion_steps
                    rerank_cost = rerank_count * self.args.rerank_diffusion_steps
                    fallback_cost = fallback_count * self.args.diffusion_steps

                    if norm_margin >= self.args.adaptive_margin_threshold:
                        loss = fast_loss
                        self.record_adaptive_decision(
                            "early_exit", norm_margin,
                            effective_step_equiv=coarse_cost,
                        )
                        if self.args.decode_for_debug:
                            preds = self.decode_latents_for_debug(preds)
                    else:
                        self.sync_cuda_for_timing()
                        top_m_selection_start = time.perf_counter()
                        rerank_idx = torch.argsort(fast_loss)[:rerank_count]
                        rerank_idx_cpu = rerank_idx.detach().cpu()
                        self.sync_cuda_for_timing()
                        self.route2_timing["top_m_selection_time"] += time.perf_counter() - top_m_selection_start

                        self.sync_cuda_for_timing()
                        start = time.perf_counter()
                        before_timing = dict(self.eval_timing)
                        rerank_idx_latent = rerank_idx.to(self.device)
                        rerank_goal_latent = cur_goal_latent[rerank_idx_latent] if cur_goal_latent is not None else None
                        rerank_obs_latents = cur_obs_latents[rerank_idx_latent] if cur_obs_latents is not None else None
                        rerank_loss, preds = self.score_candidate_deltas(
                            cur_obs_image[rerank_idx_cpu], cur_goal_image[rerank_idx_cpu], deltas[rerank_idx_cpu],
                            self.args.rerank_score_mode, self.rerank_diffusion,
                            precomputed_goal_latent=rerank_goal_latent,
                            precomputed_obs_latents=rerank_obs_latents,
                        )
                        self.add_route2_timing_delta("rerank", before_timing)
                        self.sync_cuda_for_timing()
                        self.planner_runtimes["rerank_time"] += time.perf_counter() - start

                        rerank_norm_margin = self.normalized_loss_margin(rerank_loss)
                        if rerank_norm_margin >= self.args.adaptive_rerank_margin_threshold:
                            loss = torch.full_like(fast_loss, float("inf"))
                            rerank_loss = rerank_loss.to(device=loss.device, dtype=loss.dtype)
                            loss[rerank_idx.to(loss.device)] = rerank_loss
                            self.record_adaptive_decision(
                                "mid_rerank", norm_margin, rerank_norm_margin,
                                effective_step_equiv=coarse_cost + rerank_cost,
                            )
                            if self.args.rerank_score_mode != "lpips" and self.args.decode_for_debug:
                                preds = self.decode_latents_for_debug(preds)
                        else:
                            fallback_order = torch.argsort(rerank_loss)[:fallback_count]
                            fallback_idx = rerank_idx[fallback_order]
                            fallback_idx_cpu = fallback_idx.detach().cpu()
                            fallback_idx_latent = fallback_idx.to(self.device)
                            fallback_goal_latent = cur_goal_latent[fallback_idx_latent] if cur_goal_latent is not None else None
                            fallback_obs_latents = cur_obs_latents[fallback_idx_latent] if cur_obs_latents is not None else None

                            self.sync_cuda_for_timing()
                            start = time.perf_counter()
                            before_timing = dict(self.eval_timing)
                            fallback_loss, preds = self.score_candidate_deltas(
                                cur_obs_image[fallback_idx_cpu], cur_goal_image[fallback_idx_cpu], deltas[fallback_idx_cpu],
                                self.args.rerank_score_mode, self.diffusion,
                                precomputed_goal_latent=fallback_goal_latent,
                                precomputed_obs_latents=fallback_obs_latents,
                            )
                            self.add_route2_timing_delta("final", before_timing)
                            self.sync_cuda_for_timing()
                            self.planner_runtimes["rerank_time"] += time.perf_counter() - start

                            loss = torch.full_like(fast_loss, float("inf"))
                            fallback_loss = fallback_loss.to(device=loss.device, dtype=loss.dtype)
                            loss[fallback_idx.to(loss.device)] = fallback_loss
                            self.record_adaptive_decision(
                                "high_fallback", norm_margin, rerank_norm_margin,
                                effective_step_equiv=coarse_cost + rerank_cost + fallback_cost,
                            )
                            if self.args.rerank_score_mode != "lpips" and self.args.decode_for_debug:
                                preds = self.decode_latents_for_debug(preds)
                elif self.args.planner_variant == "adaptive_two_gate_low_high":
                    self.sync_cuda_for_timing()
                    start = time.perf_counter()
                    before_timing = dict(self.eval_timing)
                    isolate_probe_rng = not self.args.disable_adaptive_policy_rng_isolation
                    probe_rng_state = self.capture_torch_rng_state() if isolate_probe_rng else None
                    teacher_candidate_rng_state = (
                        self.capture_torch_rng_state()
                        if self.needs_teacher_rollout_rng() else None
                    )
                    fast_loss, preds = self.score_candidate_deltas(
                        cur_obs_image, cur_goal_image, deltas, self.args.fast_score_mode, self.fast_diffusion,
                        precomputed_goal_latent=cur_goal_latent,
                        precomputed_obs_latents=cur_obs_latents,
                    )
                    if self.needs_teacher_candidate_fields():
                        teacher_fast_loss = fast_loss
                        teacher_fast_pred_final_latents = preds
                        teacher_fast_sorted_indices = torch.argsort(fast_loss)
                        teacher_raw_margin, _ = self.loss_margin(fast_loss)
                        teacher_normalized_margin = float(self.normalized_loss_margin(fast_loss).detach().cpu())
                    self.add_route2_timing_delta("coarse", before_timing)
                    self.sync_cuda_for_timing()
                    self.planner_runtimes["fast_shortlist_time"] += time.perf_counter() - start

                    feature_vector = self.adaptive_policy_feature_vector(fast_loss, deltas)
                    exit_prob = self.model_positive_probability(self.adaptive_policy["exit_gate_model"], feature_vector)
                    fallback_top_m = max(1, min(int(self.adaptive_policy["fallback_top_m"]), self.num_samples))
                    coarse_cost = self.num_samples * self.args.fast_diffusion_steps
                    topm_cost = fallback_top_m * self.args.diffusion_steps
                    full_cost = self.num_samples * self.args.diffusion_steps

                    if exit_prob >= float(self.adaptive_policy["tau_exit"]):
                        loss = fast_loss
                        self.record_adaptive_two_gate_decision(
                            "early_exit", exit_prob, None,
                            effective_step_equiv=coarse_cost,
                        )
                        if self.args.decode_for_debug:
                            preds = self.decode_latents_for_debug(preds)
                    else:
                        topm_prob = self.model_positive_probability(self.adaptive_policy["topm_gate_model"], feature_vector)
                        if topm_prob >= float(self.adaptive_policy["tau_topm"]):
                            self.sync_cuda_for_timing()
                            top_m_selection_start = time.perf_counter()
                            fallback_idx = torch.argsort(fast_loss)[:fallback_top_m]
                            fallback_idx_cpu = fallback_idx.detach().cpu()
                            self.sync_cuda_for_timing()
                            self.route2_timing["top_m_selection_time"] += time.perf_counter() - top_m_selection_start

                            fallback_idx_latent = fallback_idx.to(self.device)
                            fallback_goal_latent = cur_goal_latent[fallback_idx_latent] if cur_goal_latent is not None else None
                            fallback_obs_latents = cur_obs_latents[fallback_idx_latent] if cur_obs_latents is not None else None
                            if probe_rng_state is not None:
                                self.restore_torch_rng_state(probe_rng_state)
                            self.sync_cuda_for_timing()
                            start = time.perf_counter()
                            before_timing = dict(self.eval_timing)
                            fallback_loss, preds = self.score_candidate_deltas(
                                cur_obs_image[fallback_idx_cpu], cur_goal_image[fallback_idx_cpu], deltas[fallback_idx_cpu],
                                self.args.rerank_score_mode, self.diffusion,
                                precomputed_goal_latent=fallback_goal_latent,
                                precomputed_obs_latents=fallback_obs_latents,
                            )
                            self.add_route2_timing_delta("final", before_timing)
                            self.sync_cuda_for_timing()
                            self.planner_runtimes["rerank_time"] += time.perf_counter() - start

                            loss = torch.full_like(fast_loss, float("inf"))
                            fallback_loss = fallback_loss.to(device=loss.device, dtype=loss.dtype)
                            loss[fallback_idx.to(loss.device)] = fallback_loss
                            self.record_adaptive_two_gate_decision(
                                "topm_fallback", exit_prob, topm_prob,
                                effective_step_equiv=coarse_cost + topm_cost,
                            )
                            if self.args.rerank_score_mode != "lpips" and self.args.decode_for_debug:
                                preds = self.decode_latents_for_debug(preds)
                        else:
                            if probe_rng_state is not None:
                                self.restore_torch_rng_state(probe_rng_state)
                            self.sync_cuda_for_timing()
                            start = time.perf_counter()
                            before_timing = dict(self.eval_timing)
                            loss, preds = self.score_candidate_deltas(
                                cur_obs_image, cur_goal_image, deltas,
                                self.args.rerank_score_mode, self.diffusion,
                                precomputed_goal_latent=cur_goal_latent,
                                precomputed_obs_latents=cur_obs_latents,
                            )
                            self.add_route2_timing_delta("final", before_timing)
                            self.sync_cuda_for_timing()
                            self.planner_runtimes["rerank_time"] += time.perf_counter() - start
                            self.record_adaptive_two_gate_decision(
                                "full_fallback", exit_prob, topm_prob,
                                effective_step_equiv=coarse_cost + full_cost,
                            )
                            if self.args.rerank_score_mode != "lpips" and self.args.decode_for_debug:
                                preds = self.decode_latents_for_debug(preds)
                elif (
                    self.args.candidate_batch_size is not None
                    or self.num_repeat_eval * self.num_samples > 120
                ):
                    cur_losses = []
                    cur_yaw_debugs = []
                    for r in range(self.num_repeat_eval):
                        if self.score_mode == 'lpips':
                            candidate_batch_size = self.args.candidate_batch_size or self.num_samples
                            if candidate_batch_size < 1:
                                raise ValueError("--candidate_batch_size must be >= 1")
                            pred_chunks = []
                            loss_chunks = []
                            for chunk_start in range(0, self.num_samples, candidate_batch_size):
                                chunk_end = min(chunk_start + candidate_batch_size, self.num_samples)
                                chunk_preds = self.autoregressive_rollout(
                                    cur_obs_image[chunk_start:chunk_end],
                                    deltas[chunk_start:chunk_end],
                                    self.args.rollout_stride,
                                    timing=self.eval_timing,
                                )
                                chunk_preds = chunk_preds[:, -1] # take the last predicted image
                                self.sync_cuda_for_timing()
                                start = time.perf_counter()
                                chunk_loss = self.loss_fn(
                                    chunk_preds.to(self.device),
                                    cur_goal_image[chunk_start:chunk_end].to(self.device),
                                ).flatten(0)
                                self.add_scoring_time(start)
                                pred_chunks.append(chunk_preds)
                                loss_chunks.append(chunk_loss)
                            preds = torch.cat(pred_chunks, dim=0)
                            loss = torch.cat(loss_chunks, dim=0)
                        else:
                            pred_latents = self.autoregressive_rollout_latent(
                                cur_obs_image, deltas, self.args.rollout_stride,
                                timing=self.eval_timing,
                                precomputed_obs_latents=cur_obs_latents,
                            )
                            preds = pred_latents[:, -1]
                            self.sync_cuda_for_timing()
                            start = time.perf_counter()
                            loss = self.score_latents(preds, cur_goal_latent, deltas)
                            self.add_scoring_time(start)
                            if self.args.debug_yaw_penalty and self.score_mode == "latent_l2_yaw" and self._last_yaw_debug is not None:
                                cur_yaw_debugs.append(self._last_yaw_debug)
                        cur_losses.append(loss)

                    loss = torch.stack(cur_losses).mean(dim=0)
                    if cur_yaw_debugs:
                        self._last_yaw_debug = self.average_yaw_debug(cur_yaw_debugs)
                    if self.score_mode != 'lpips' and self.args.decode_for_debug:
                        preds = self.decode_latents_for_debug(preds)
                else:
                    expanded_deltas = deltas.repeat(self.num_repeat_eval, 1, 1) 
                    expanded_obs_image = cur_obs_image.repeat(self.num_repeat_eval, 1, 1, 1, 1) 
                    expanded_goal_image = cur_goal_image.repeat(self.num_repeat_eval, 1, 1, 1) 

                    if self.score_mode == 'lpips':
                        preds = self.autoregressive_rollout(expanded_obs_image, expanded_deltas, self.args.rollout_stride, timing=self.eval_timing)
                        preds = preds[:, -1]

                        self.sync_cuda_for_timing()
                        start = time.perf_counter()
                        loss = self.loss_fn(preds.to(self.device), expanded_goal_image.to(self.device)).flatten(0)
                        self.add_scoring_time(start)
                    else:
                        expanded_goal_latent = cur_goal_latent.repeat(self.num_repeat_eval, 1, 1, 1)
                        expanded_obs_latents = (
                            cur_obs_latents.repeat(self.num_repeat_eval, 1, 1, 1, 1)
                            if cur_obs_latents is not None else None
                        )
                        preds = self.autoregressive_rollout_latent(
                            expanded_obs_image, expanded_deltas, self.args.rollout_stride,
                            timing=self.eval_timing,
                            precomputed_obs_latents=expanded_obs_latents,
                        )
                        preds = preds[:, -1]

                        self.sync_cuda_for_timing()
                        start = time.perf_counter()
                        loss = self.score_latents(preds, expanded_goal_latent, expanded_deltas)
                        self.add_scoring_time(start)

                    loss = loss.view(self.num_repeat_eval, -1)
                    loss = loss.mean(dim=0)
                    if self.args.debug_yaw_penalty and self.score_mode == "latent_l2_yaw" and self._last_yaw_debug is not None:
                        self._last_yaw_debug = self.average_yaw_debug([self._last_yaw_debug], num_repeat=self.num_repeat_eval)

                    preds = preds[:self.args.num_samples]
                    if self.score_mode != 'lpips' and self.args.decode_for_debug:
                        preds = self.decode_latents_for_debug(preds)

                sorted_idx = torch.argsort(loss)
                effective_topk = self.topk
                if self.args.planner_variant in [
                    "hybrid_latent_rerank",
                    "coarse_to_fine_latent",
                    "adaptive_latent_rerank",
                    "adaptive_two_gate_low_high",
                ]:
                    finite_count = int(torch.isfinite(loss).sum().detach().cpu().item())
                    effective_topk = max(1, min(self.topk, finite_count, self.num_samples))
                topk_idx = sorted_idx[:effective_topk]
                topk_action = deltas[topk_idx][:, -1]
                topk_control = sample[topk_idx]
                selected_candidate_index = int(topk_idx[0].detach().cpu().item())
                if self.args.debug_cem_action_stats:
                    legacy_mu_update = topk_action.mean(dim=0)
                    legacy_sigma_update = topk_action.std(dim=0)
                    raw_mu_update = topk_control.mean(dim=0)
                    raw_sigma_update = topk_control.std(dim=0)
                    chosen_mu_update = raw_mu_update if self.args.cem_update_space == "raw_sample" else legacy_mu_update
                    chosen_sigma_update = raw_sigma_update if self.args.cem_update_space == "raw_sample" else legacy_sigma_update
                    debug_row = {
                        "debug": "cem_action_stats",
                        "dataset": dataset_name,
                        "eval_index": traj_id,
                        "cem_step": int(i),
                        "cem_update_space": self.args.cem_update_space,
                        "planner_final_yaw_policy": self.args.planner_final_yaw_policy,
                        "mu_shape": list(mu.shape),
                        "sigma_shape": list(sigma.shape),
                        "sample_shape": list(sample.shape),
                        "deltas_shape": list(deltas.shape),
                        "topk_idx_shape": list(topk_idx.shape),
                        "topk_action_shape": list(topk_action.shape),
                        "topk_control_shape": list(topk_control.shape),
                        "mu_yaw_before_update": tensor_stats_dict(mu[traj, 2:3]),
                        "sigma_yaw_before_update": tensor_stats_dict(sigma[traj, 2:3]),
                        "sampled_yaw_control_before_pi": tensor_stats_dict(sample[:, -1]),
                        "final_yaw_control_after_policy": tensor_stats_dict(final_yaw_control),
                        "final_timestep_yaw_after_pi": tensor_stats_dict(deltas[:, -1, 2]),
                        "candidate_yaw_t0_to_t6": tensor_stats_dict(deltas[:, :-1, 2]),
                        "candidate_yaw_t7": tensor_stats_dict(deltas[:, -1, 2]),
                        "topk_action_yaw_t7": tensor_stats_dict(topk_action[:, 2]),
                        "topk_control_yaw": tensor_stats_dict(topk_control[:, 2]),
                        "legacy_updated_mu_yaw": tensor_stats_dict(legacy_mu_update[2:3]),
                        "legacy_updated_sigma_yaw": tensor_stats_dict(legacy_sigma_update[2:3]),
                        "raw_sample_updated_mu_yaw": tensor_stats_dict(raw_mu_update[2:3]),
                        "raw_sample_updated_sigma_yaw": tensor_stats_dict(raw_sigma_update[2:3]),
                        "chosen_updated_mu_yaw": tensor_stats_dict(chosen_mu_update[2:3]),
                        "chosen_updated_sigma_yaw": tensor_stats_dict(chosen_sigma_update[2:3]),
                    }
                    print(json.dumps(debug_row, sort_keys=True))
                if self.args.export_latent_rerank_distill_dataset:
                    self.record_latent_rerank_distill_example(
                        dataset_name=dataset_name,
                        batch_idx=batch_idx,
                        traj_id=traj_id,
                        cem_step=i,
                        is_final_cem_step=i == self.opt_steps - 1,
                        deltas=deltas,
                        cur_obs_image=cur_obs_image,
                        cur_goal_image=cur_goal_image,
                        cur_obs_latents=cur_obs_latents,
                        cur_goal_latent=cur_goal_latent,
                        fast_losses=teacher_fast_loss,
                        fast_pred_final_latents=teacher_fast_pred_final_latents,
                        fast_sorted_indices=teacher_fast_sorted_indices,
                        normalized_margin=teacher_normalized_margin,
                        raw_margin=teacher_raw_margin,
                        active_selected_candidate_index=selected_candidate_index,
                        rollout_rng_state=teacher_candidate_rng_state,
                        planner_rerank_candidate_indices=teacher_shortlist_indices,
                        planner_rerank_losses=teacher_rerank_loss,
                        planner_rerank_selected_local_index=(
                            teacher_selected_shortlist_rank - 1
                            if teacher_selected_shortlist_rank is not None else None
                        ),
                        planner_rerank_selected_original_index=(
                            selected_original_idx
                            if self.args.planner_variant == "coarse_to_fine_latent" else None
                        ),
                    )
                if self.args.export_teacher_collection_v2:
                    self.record_teacher_collection_v2_example(
                        dataset_name=dataset_name,
                        batch_idx=batch_idx,
                        traj_id=traj_id,
                        cem_step=i,
                        is_final_cem_step=i == self.opt_steps - 1,
                        deltas=deltas,
                        cur_obs_image=cur_obs_image,
                        cur_goal_image=cur_goal_image,
                        cur_obs_latents=cur_obs_latents,
                        cur_goal_latent=cur_goal_latent,
                        fast_losses=teacher_fast_loss,
                        fast_pred_final_latents=teacher_fast_pred_final_latents,
                        fast_sorted_indices=teacher_fast_sorted_indices,
                        normalized_margin=teacher_normalized_margin,
                        raw_margin=teacher_raw_margin,
                        active_selected_candidate_index=selected_candidate_index,
                        rollout_rng_state=teacher_candidate_rng_state,
                        planner_rerank_candidate_indices=teacher_shortlist_indices,
                        planner_rerank_losses=teacher_rerank_loss,
                        planner_rerank_selected_local_index=(
                            teacher_selected_shortlist_rank - 1
                            if teacher_selected_shortlist_rank is not None else None
                        ),
                        planner_rerank_selected_original_index=(
                            selected_original_idx
                            if self.args.planner_variant == "coarse_to_fine_latent" else None
                        ),
                    )
                teacher_record = {
                    "dataset_name": dataset_name,
                    "batch_idx": batch_idx,
                    "eval_index": traj_id,
                    "idx": traj_id,
                    "seed": self.args.seed,
                    "cem_step": int(i),
                    "is_final_cem_step": i == self.opt_steps - 1,
                    "planner_variant": self.args.planner_variant,
                    "score_mode": self.score_mode,
                    "fast_score_mode": self.args.fast_score_mode,
                    "rerank_score_mode": self.args.rerank_score_mode,
                    "fast_diffusion_steps": self.args.fast_diffusion_steps,
                    "rerank_diffusion_steps": self.args.rerank_diffusion_steps,
                    "rerank_top_m": self.args.rerank_top_m,
                    "adaptive_rerank_steps_by_margin": self.args.adaptive_rerank_steps_by_margin,
                    "adaptive_rerank_low_steps": self.args.adaptive_rerank_low_steps,
                    "adaptive_rerank_high_steps": self.args.adaptive_rerank_high_steps,
                    "adaptive_rerank_margin_threshold": self.args.adaptive_rerank_margin_threshold,
                    "selected_candidate_index": selected_candidate_index,
                    "selected_shortlist_rank": teacher_selected_shortlist_rank,
                    "adaptive_mode": teacher_adaptive_mode,
                    "normalized_margin": teacher_normalized_margin,
                    "raw_margin": teacher_raw_margin,
                    "effective_step_equiv": teacher_effective_step_equiv,
                    "candidate_actions": deltas,
                    "fast_losses": teacher_fast_loss,
                    "fast_sorted_indices": teacher_fast_sorted_indices,
                    "shortlist_indices": teacher_shortlist_indices,
                    "rerank_losses": teacher_rerank_loss,
                    "rerank_sorted_indices": teacher_rerank_sorted_indices,
                    "selected_action": deltas[selected_candidate_index, -1],
                }
                teacher_record.update(self.teacher_latent_fields(
                    cur_goal_latent=cur_goal_latent,
                    cur_obs_latents=cur_obs_latents,
                    fast_pred_final_latents=teacher_fast_pred_final_latents,
                    shortlist_indices=teacher_shortlist_indices,
                    rerank_pred_final_latents=teacher_rerank_pred_final_latents,
                    fast_losses=teacher_fast_loss,
                ))
                self.record_teacher_decision(teacher_record)
                losses.append(loss[topk_idx[0]].item())   
                if self.args.cem_update_space == "raw_sample":
                    mu[traj] = topk_control.mean(dim=0)
                    sigma[traj] = topk_control.std(dim=0)
                else:
                    mu[traj] = topk_action.mean(dim=0)
                    sigma[traj] = topk_action.std(dim=0)
                self.record_score_row(dataset_name, traj_id, i, loss)

                can_plot_iter = (
                    self.args.planner_variant == "standard" and self.score_mode == 'lpips'
                ) or self.args.decode_for_debug
                if self.args.plot and image_plot_dir is not None and self.args.planner_variant != "hybrid_latent_rerank" and can_plot_iter:
                    self.visualize_trajectories(dataset_name, gt_actions, image_plot_dir, i, traj, traj_id, deltas, cur_obs_image, cur_goal_image, preds, loss, topk_idx)                    
        
        # Final rollout
        deltas = mu[:, :2]
        deltas = deltas.unsqueeze(1).repeat(1, len_traj_pred, 1)
        constraint = getattr(self.args, 'constraint', 'none')
        if constraint == 'forward_first':
            deltas[:, :5, 1] = 0
            deltas[:, 5:, 0] = 0
        elif constraint == 'left_right_first':
            deltas[:, :3, 0] = 0
        elif constraint == 'straight_then_forward':
            deltas[:, :3, 1] = 0

        # Calculate yaws
        unnorm_deltas = unnormalize_data(deltas, ACTION_STATS_TORCH)
        delta_yaw = calculate_delta_yaw(unnorm_deltas)
        deltas = torch.cat((deltas, delta_yaw.to(deltas.device)), dim=-1)
        final_yaw_control = apply_planner_final_yaw_policy(
            mu[:, -1] * np.pi,
            self.args.planner_final_yaw_policy,
        )
        deltas[:, -1, -1] += final_yaw_control

        before_final_timing = dict(self.eval_timing)
        if self.args.planner_variant == "fast_latent":
            loss, preds = self.score_candidate_deltas(
                obs_image, goal_image.squeeze(1), deltas, self.args.fast_score_mode, self.fast_diffusion,
                precomputed_goal_latent=cached_goal_latents,
                precomputed_obs_latents=cached_obs_latents,
            )
            if self.args.decode_for_debug:
                preds = self.decode_latents_for_debug(preds)
        elif (
            self.args.planner_variant == "coarse_to_fine_latent"
            and self.args.rerank_backend in self.shortcut_rerank_backends
        ):
            loss = torch.zeros(obs_image.shape[0], device=self.device)
            preds = obs_image[:, -1].to(self.device)
        elif self.args.planner_variant in ["hybrid_latent_rerank", "coarse_to_fine_latent"]:
            loss, preds = self.score_candidate_deltas(
                obs_image, goal_image.squeeze(1), deltas, self.args.rerank_score_mode, self.rerank_diffusion,
                precomputed_goal_latent=cached_goal_latents,
                precomputed_obs_latents=cached_obs_latents,
            )
            if self.args.rerank_score_mode != "lpips" and self.args.decode_for_debug:
                preds = self.decode_latents_for_debug(preds)
        elif self.score_mode == 'lpips':
            preds = self.autoregressive_rollout(obs_image, deltas, self.args.rollout_stride, timing=self.eval_timing)
            preds = preds[:, -1] # take the last predicted image

            self.sync_cuda_for_timing()
            start = time.perf_counter()
            loss = self.loss_fn(preds.to(self.device), goal_image.squeeze(1).to(self.device)).flatten(0)
            self.add_scoring_time(start)
        else:
            if cached_goal_latents is None:
                goal_latent = encode_rgb_to_latent(self.vae, goal_image.squeeze(1), self.device, scale=True)
            else:
                goal_latent = cached_goal_latents
                self.route2_cache_stats["goal_latent_cached_used"] += 1
            preds = self.autoregressive_rollout_latent(
                obs_image, deltas, self.args.rollout_stride, timing=self.eval_timing,
                precomputed_obs_latents=cached_obs_latents,
            )
            preds = preds[:, -1]

            self.sync_cuda_for_timing()
            start = time.perf_counter()
            loss = self.score_latents(preds, goal_latent, deltas)
            self.add_scoring_time(start)
            if self.args.decode_for_debug:
                preds = self.decode_latents_for_debug(preds)

        self.add_route2_timing_delta("final", before_final_timing)

        if self.args.save_preds:
            save_planning_pred(dataset_save_output_dir, n_evals, idxs, obs_image, goal_image, preds, deltas, loss, gt_actions)
        
        can_plot_final = (
            self.args.planner_variant == "standard" and self.score_mode == 'lpips'
        ) or (
            self.args.planner_variant == "hybrid_latent_rerank" and self.args.rerank_score_mode == "lpips"
        ) or self.args.decode_for_debug
        if self.args.plot and can_plot_final:
            img_name = os.path.join(image_plot_dir, f'FINAL_{idx_string}.png')
            plot_batch_final(obs_image[:, -1].to(self.device), preds, goal_image.squeeze(1).to(self.device), idxs, losses, save_path=img_name)

        pred_actions = get_action_torch(deltas[:, :, :2], ACTION_STATS_TORCH)
        pred_yaw = deltas[:, :, -1].sum(1)
        self.route2_timing["total_planning_time"] = time.perf_counter() - total_planning_start
        new_top_m_recall_rows = self.top_m_recall_rows[top_m_recall_start_idx:]
        if new_top_m_recall_rows:
            hits = [float(row["top_m_recall_full_best"]) for row in new_top_m_recall_rows]
            self.last_top_m_recall = float(np.mean(hits))
        return pred_actions, pred_yaw

    def visualize_trajectories(self, dataset_name, gt_actions, image_plot_dir, i, traj, traj_id, deltas, cur_obs_image, cur_goal_image, preds, loss, topk_idx):
        img_for_plotting = torch.cat([cur_goal_image[0:1].to(self.device), preds])
        loss_for_plotting = torch.cat((torch.tensor([0]).to(self.device), loss))
        img_name = os.path.join(image_plot_dir, f'idx{traj_id}_iter{i}.png')
        plot_images_with_losses(img_for_plotting, loss_for_plotting, save_path=img_name)
        plot_name = os.path.join(image_plot_dir, f'idx{traj_id}_iter{i}_trajs.png')
        num_plot = self.args.num_samples
        log_viz_single(
                        dataset_name, 
                        cur_obs_image[0], 
                        cur_goal_image[0], 
                        preds[:num_plot], 
                        deltas[:num_plot], 
                        loss[:num_plot], 
                        topk_idx[0:1], 
                        gt_actions[traj], 
                        ACTION_STATS_TORCH, 
                        plan_iter=i, 
                        output_dir=plot_name
                    )
    
    def autoregressive_rollout(self, obs_image, deltas, rollout_stride, diffusion=None, timing=None):
        deltas = deltas.unflatten(1, (-1, rollout_stride)).sum(2)
        preds = []
        curr_obs = obs_image.clone().to(self.device)
        diffusion = diffusion or self.diffusion
        
        for i in range(deltas.shape[1]):
            curr_delta = deltas[:, i:i+1]
            all_models = self.model, diffusion, self.vae
            x_pred_pixels = model_forward_wrapper(all_models, curr_obs, curr_delta, self.args.rollout_stride, self.latent_size, num_cond=self.num_cond, device=self.device, timing=timing)
            x_pred_pixels = x_pred_pixels.unsqueeze(1)
            
            curr_obs = torch.cat((curr_obs, x_pred_pixels), dim=1) # append current prediction
            curr_obs = curr_obs[:, 1:] # remove first observation
            preds.append(x_pred_pixels)
        
        preds = torch.cat(preds, 1)
        return preds

    def autoregressive_rollout_latent(
        self, obs_image, deltas, rollout_stride, diffusion=None, timing=None,
        precomputed_obs_latents=None
    ):
        deltas = deltas.unflatten(1, (-1, rollout_stride)).sum(2)
        preds = []
        if precomputed_obs_latents is None:
            curr_latents = encode_rgb_to_latent(self.vae, obs_image, self.device, scale=True)
        else:
            curr_latents = precomputed_obs_latents.clone().to(self.device)
            self.route2_cache_stats["context_latent_cached_used"] += 1
        diffusion = diffusion or self.diffusion

        for i in range(deltas.shape[1]):
            curr_delta = deltas[:, i:i+1]
            all_models = self.model, diffusion, self.vae
            pred_latent = model_forward_wrapper_from_latents(
                all_models, curr_latents, curr_delta, rollout_stride, self.latent_size,
                num_cond=self.num_cond, device=self.device, timing=timing
            )
            pred_latent = pred_latent.unsqueeze(1)

            curr_latents = torch.cat((curr_latents, pred_latent), dim=1) # append current latent prediction
            curr_latents = curr_latents[:, 1:] # remove first latent observation
            preds.append(pred_latent)

        preds = torch.cat(preds, 1)
        return preds

    def score_latents(self, pred_latent, goal_latent, deltas=None):
        return self.score_latents_with_mode(pred_latent, goal_latent, self.score_mode, deltas=deltas)

    def score_latents_with_mode(self, pred_latent, goal_latent, score_mode, deltas=None):
        # Decoder-free latent scoring only receives VAE latents; RGB decode and LPIPS live in the lpips branch.
        assert pred_latent.ndim == 4 and pred_latent.shape[1] == 4, f"expected pred latents [B,4,H,W], got {tuple(pred_latent.shape)}"
        assert goal_latent.ndim == 4 and goal_latent.shape[1] == 4, f"expected goal latents [B,4,H,W], got {tuple(goal_latent.shape)}"
        if score_mode in ['latent_l2', 'latent_l2_yaw']:
            latent_loss = ((pred_latent - goal_latent) ** 2).flatten(1).mean(dim=1)
            if score_mode == 'latent_l2_yaw':
                if deltas is None:
                    raise ValueError("score_mode='latent_l2_yaw' requires candidate_deltas")
                yaw_penalty = compute_action_yaw_smoothness_penalty(deltas.to(latent_loss.device))
                yaw_penalty = yaw_penalty.to(device=latent_loss.device, dtype=latent_loss.dtype)
                final_score = latent_loss + self.args.yaw_penalty_weight * yaw_penalty
                self._last_yaw_debug = {
                    "latent_loss": latent_loss.detach().float().cpu().tolist(),
                    "yaw_penalty": yaw_penalty.detach().float().cpu().tolist(),
                    "final_score": final_score.detach().float().cpu().tolist(),
                    "yaw_penalty_weight": self.args.yaw_penalty_weight,
                }
                return final_score
            return latent_loss
        if score_mode == 'latent_cosine':
            pred_flat = pred_latent.flatten(1)
            goal_flat = goal_latent.flatten(1)
            return 1 - F.cosine_similarity(pred_flat, goal_flat, dim=1)
        if score_mode == "learned_spatial_scorer":
            if deltas is None:
                raise ValueError("score_mode='learned_spatial_scorer' requires candidate_deltas")
            if self.latent_scorer is None:
                raise ValueError("Latent scorer is not loaded")
            with torch.inference_mode():
                scores = self.latent_scorer(
                    pred_latent.to(self.latent_scorer_device),
                    goal_latent.to(self.latent_scorer_device),
                    deltas.to(self.latent_scorer_device),
                )
            return scores.to(pred_latent.device)
        raise ValueError(f"Unsupported latent score mode: {score_mode}")

    def score_candidate_deltas(
        self, obs_image, goal_image, deltas, score_mode, diffusion, timing=None,
        precomputed_goal_latent=None, precomputed_obs_latents=None
    ):
        timing = self.eval_timing if timing is None else timing
        if score_mode == "lpips":
            preds = self.autoregressive_rollout(obs_image, deltas, self.args.rollout_stride, diffusion=diffusion, timing=timing)
            preds = preds[:, -1]
            self.sync_cuda_for_timing()
            start = time.perf_counter()
            loss = self.loss_fn(preds.to(self.device), goal_image.to(self.device)).flatten(0)
            self.add_scoring_time(start, timing=timing)
            return loss, preds

        # Decoder-free latent scoring: no VAE decode and no LPIPS loss in this branch.
        assert score_mode in ["latent_l2", "latent_cosine", "latent_l2_yaw", "learned_spatial_scorer"]
        if precomputed_goal_latent is None:
            goal_latent = encode_rgb_to_latent(self.vae, goal_image, self.device, scale=True)
        else:
            goal_latent = precomputed_goal_latent.to(self.device)
            self.route2_cache_stats["goal_latent_cached_used"] += 1
        preds = self.autoregressive_rollout_latent(
            obs_image, deltas, self.args.rollout_stride, diffusion=diffusion, timing=timing,
            precomputed_obs_latents=precomputed_obs_latents
        )
        if self.args.export_fast_rollout_trace and diffusion is self.fast_diffusion:
            # These summaries reuse the already-computed fast autoregressive
            # rollout. They add no CDiT calls and remain runtime-safe.
            goal_steps = goal_latent.unsqueeze(1)
            residual = preds - goal_steps
            residual_l2 = residual.square().flatten(2).mean(dim=2)
            pred_flat = preds.flatten(2)
            goal_flat = goal_steps.expand_as(preds).flatten(2)
            residual_cosine = 1.0 - F.cosine_similarity(pred_flat, goal_flat, dim=2)
            step_delta = torch.zeros_like(residual_l2)
            if preds.shape[1] > 1:
                step_delta[:, 1:] = (
                    preds[:, 1:] - preds[:, :-1]
                ).square().flatten(2).mean(dim=2).sqrt()
            channel_rms = residual.square().mean(dim=(-1, -2)).sqrt()
            pooled_residual = F.adaptive_avg_pool2d(
                residual.flatten(0, 1), (2, 2)
            ).unflatten(0, (residual.shape[0], residual.shape[1]))
            self.last_fast_rollout_trace = {
                "goal_l2_by_horizon": self.teacher_latent_to_cpu_fp16(residual_l2),
                "goal_cosine_by_horizon": self.teacher_latent_to_cpu_fp16(residual_cosine),
                "step_delta_rms_by_horizon": self.teacher_latent_to_cpu_fp16(step_delta),
                "goal_channel_rms_by_horizon": self.teacher_latent_to_cpu_fp16(channel_rms),
                "goal_residual_pool2_by_horizon": self.teacher_latent_to_cpu_fp16(pooled_residual),
            }
        preds = preds[:, -1]
        assert preds.ndim == 4 and preds.shape[1] == 4, f"expected latent predictions [B,4,H,W], got {tuple(preds.shape)}"
        self.sync_cuda_for_timing()
        start = time.perf_counter()
        loss = self.score_latents_with_mode(preds, goal_latent, score_mode, deltas=deltas)
        self.add_scoring_time(start, timing=timing)
        return loss, preds

    def average_yaw_debug(self, debug_rows, num_repeat=None):
        averaged = {"yaw_penalty_weight": self.args.yaw_penalty_weight}
        for key in ["latent_loss", "yaw_penalty", "final_score"]:
            values = [torch.tensor(row[key], dtype=torch.float32) for row in debug_rows]
            if len(values) == 1 and num_repeat is not None:
                values = [values[0].view(num_repeat, -1).mean(dim=0)]
            else:
                values = [torch.stack(values, dim=0).mean(dim=0)]
            averaged[key] = values[0].tolist()
        return averaged

    def decode_latents_for_debug(self, latents):
        images = self.vae.decode(latents / 0.18215).sample
        return torch.clip(images, -1., 1.)

    def sync_cuda_for_timing(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def add_scoring_time(self, start, timing=None):
        self.sync_cuda_for_timing()
        timing = self.eval_timing if timing is None else timing
        timing["scoring_time"] += time.perf_counter() - start

    def candidate_ranks(self, loss):
        order = torch.argsort(loss)
        ranks = torch.empty_like(order, dtype=torch.long)
        ranks[order] = torch.arange(1, loss.numel() + 1, device=loss.device)
        return ranks

    def spearman_from_ranks(self, rank_a, rank_b):
        a = rank_a.detach().float().cpu()
        b = rank_b.detach().float().cpu()
        a = a - a.mean()
        b = b - b.mean()
        denom = torch.sqrt((a * a).sum() * (b * b).sum())
        if denom.item() == 0:
            return 0.0
        return ((a * b).sum() / denom).item()

    def topk_overlap_ratio(self, rank_a, rank_b, k=3):
        k = min(k, rank_a.numel(), rank_b.numel())
        top_a = set(torch.argsort(rank_a)[:k].cpu().tolist())
        top_b = set(torch.argsort(rank_b)[:k].cpu().tolist())
        return len(top_a & top_b) / float(k)

    def compare_candidate_scores(self, dataset_name, traj_id, opt_iter, cur_obs_image, cur_goal_image, deltas):
        chunk_size = max(1, self.args.compare_chunk_size)
        lpips_chunks = []
        latent_l2_chunks = []
        latent_cosine_chunks = []
        preds_rgb_preview = []

        for start_idx in range(0, deltas.shape[0], chunk_size):
            end_idx = min(start_idx + chunk_size, deltas.shape[0])
            chunk_deltas = deltas[start_idx:end_idx]
            chunk_obs = cur_obs_image[start_idx:end_idx]
            chunk_goal = cur_goal_image[start_idx:end_idx]

            self.sync_cuda_for_timing()
            start = time.perf_counter()
            preds_rgb = self.autoregressive_rollout(chunk_obs, chunk_deltas, self.args.rollout_stride)
            preds_rgb = preds_rgb[:, -1]
            self.sync_cuda_for_timing()
            self.compare_runtimes["rgb_rollout_decode_time"] += time.perf_counter() - start

            self.sync_cuda_for_timing()
            start = time.perf_counter()
            lpips_loss_chunk = self.loss_fn(preds_rgb.to(self.device), chunk_goal.to(self.device)).flatten(0)
            self.sync_cuda_for_timing()
            self.compare_runtimes["lpips_score_time"] += time.perf_counter() - start
            lpips_chunks.append(lpips_loss_chunk.detach().float().cpu())
            preds_rgb_preview.append(preds_rgb.detach().cpu())

            self.sync_cuda_for_timing()
            start = time.perf_counter()
            goal_latent = encode_rgb_to_latent(self.vae, chunk_goal, self.device, scale=True)
            self.sync_cuda_for_timing()
            self.compare_runtimes["goal_latent_encode_time"] += time.perf_counter() - start

            self.sync_cuda_for_timing()
            start = time.perf_counter()
            pred_latents = self.autoregressive_rollout_latent(chunk_obs, chunk_deltas, self.args.rollout_stride)
            pred_latents = pred_latents[:, -1]
            self.sync_cuda_for_timing()
            self.compare_runtimes["latent_rollout_time"] += time.perf_counter() - start

            self.sync_cuda_for_timing()
            start = time.perf_counter()
            latent_l2_loss_chunk = ((pred_latents - goal_latent) ** 2).flatten(1).mean(dim=1)
            self.sync_cuda_for_timing()
            self.compare_runtimes["latent_l2_score_time"] += time.perf_counter() - start
            latent_l2_chunks.append(latent_l2_loss_chunk.detach().float().cpu())

            self.sync_cuda_for_timing()
            start = time.perf_counter()
            latent_cosine_loss_chunk = 1 - F.cosine_similarity(pred_latents.flatten(1), goal_latent.flatten(1), dim=1)
            self.sync_cuda_for_timing()
            self.compare_runtimes["latent_cosine_score_time"] += time.perf_counter() - start
            latent_cosine_chunks.append(latent_cosine_loss_chunk.detach().float().cpu())

            del preds_rgb, goal_latent, pred_latents
            del lpips_loss_chunk, latent_l2_loss_chunk, latent_cosine_loss_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        lpips_loss = torch.cat(lpips_chunks, dim=0)
        latent_l2_loss = torch.cat(latent_l2_chunks, dim=0)
        latent_cosine_loss = torch.cat(latent_cosine_chunks, dim=0)
        preds_rgb = torch.cat(preds_rgb_preview, dim=0).to(self.device)

        rank_lpips = self.candidate_ranks(lpips_loss)
        rank_latent_l2 = self.candidate_ranks(latent_l2_loss)
        rank_latent_cosine = self.candidate_ranks(latent_cosine_loss)

        corr_l2 = self.spearman_from_ranks(rank_lpips, rank_latent_l2)
        corr_cosine = self.spearman_from_ranks(rank_lpips, rank_latent_cosine)
        lpips_top1 = int(torch.argmin(rank_lpips).item())
        latent_l2_top1 = int(torch.argmin(rank_latent_l2).item())
        latent_cosine_top1 = int(torch.argmin(rank_latent_cosine).item())
        agreement_metrics = {
            "dataset": dataset_name,
            "traj_id": traj_id,
            "opt_iter": opt_iter,
            "top1_agreement_latent_l2": latent_l2_top1 == lpips_top1,
            "top1_agreement_latent_cosine": latent_cosine_top1 == lpips_top1,
            "top3_overlap_latent_l2": self.topk_overlap_ratio(rank_lpips, rank_latent_l2, k=3),
            "top3_overlap_latent_cosine": self.topk_overlap_ratio(rank_lpips, rank_latent_cosine, k=3),
            "lpips_rank_of_latent_l2_top1": int(rank_lpips[latent_l2_top1].item()),
            "lpips_rank_of_latent_cosine_top1": int(rank_lpips[latent_cosine_top1].item()),
        }
        self.compare_agreement_metrics.append(agreement_metrics)
        self.compare_correlations.append({
            "dataset": dataset_name,
            "traj_id": traj_id,
            "opt_iter": opt_iter,
            "spearman_lpips_latent_l2": corr_l2,
            "spearman_lpips_latent_cosine": corr_cosine,
            **agreement_metrics,
        })

        for candidate_id in range(self.num_samples):
            self.compare_rows.append({
                "dataset": dataset_name,
                "traj_id": traj_id,
                "opt_iter": opt_iter,
                "candidate_id": candidate_id,
                "lpips_loss": float(lpips_loss[candidate_id].detach().cpu()),
                "latent_l2_loss": float(latent_l2_loss[candidate_id].detach().cpu()),
                "latent_cosine_loss": float(latent_cosine_loss[candidate_id].detach().cpu()),
                "rank_lpips": int(rank_lpips[candidate_id].detach().cpu()),
                "rank_latent_l2": int(rank_latent_l2[candidate_id].detach().cpu()),
                "rank_latent_cosine": int(rank_latent_cosine[candidate_id].detach().cpu()),
            })

        return {"lpips": lpips_loss, "preds_rgb": preds_rgb}

    def record_score_row(self, dataset_name, traj_id, opt_iter, loss):
        if not self.args.save_score_table:
            return
        row = {
            "dataset": dataset_name,
            "traj_id": traj_id,
            "opt_iter": opt_iter,
            "score_mode": self.score_mode,
            "losses": loss.detach().float().cpu().tolist(),
        }
        if self.args.debug_yaw_penalty and self.score_mode == "latent_l2_yaw" and self._last_yaw_debug is not None:
            yaw_row = {
                "dataset": dataset_name,
                "traj_id": traj_id,
                "opt_iter": opt_iter,
                **self._last_yaw_debug,
            }
            row["yaw_debug"] = self._last_yaw_debug
            self.yaw_debug_rows.append(yaw_row)
        self.score_rows.append(row)

    def save_score_table(self):
        if not self.args.save_score_table or self.args.compare_score_modes:
            return
        output_path = self.args.score_table_path
        if output_path is None:
            output_dir = self.args.output_dir or "."
            output_path = os.path.join(output_dir, f"score_table_{self.eval_name}_{self.score_mode}.json")
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w") as f:
            if self.args.debug_yaw_penalty and self.score_mode == "latent_l2_yaw":
                json.dump({"rows": self.score_rows, "yaw_penalty_debug": self.yaw_debug_rows}, f, indent=2)
            else:
                json.dump(self.score_rows, f, indent=2)
        print(f"Saved score table to {output_path}")

    def save_compare_score_modes(self):
        if not self.args.compare_score_modes:
            return
        output_path = self.args.score_table_path
        if output_path is None:
            output_dir = self.args.output_dir or "."
            output_path = os.path.join(output_dir, f"score_mode_comparison_{self.eval_name}.json")
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        l2_corrs = [x["spearman_lpips_latent_l2"] for x in self.compare_correlations]
        cosine_corrs = [x["spearman_lpips_latent_cosine"] for x in self.compare_correlations]
        summary = {
            "spearman_lpips_latent_l2_mean": float(np.mean(l2_corrs)) if l2_corrs else None,
            "spearman_lpips_latent_cosine_mean": float(np.mean(cosine_corrs)) if cosine_corrs else None,
            "correlations_by_iteration": self.compare_correlations,
        }
        metric_keys = [
            "top1_agreement_latent_l2",
            "top1_agreement_latent_cosine",
            "top3_overlap_latent_l2",
            "top3_overlap_latent_cosine",
            "lpips_rank_of_latent_l2_top1",
            "lpips_rank_of_latent_cosine_top1",
        ]
        summary_metrics = {
            key: float(np.mean([float(x[key]) for x in self.compare_agreement_metrics])) if self.compare_agreement_metrics else None
            for key in metric_keys
        }
        peak_cuda_memory = None
        if torch.cuda.is_available():
            peak_cuda_memory = int(torch.cuda.max_memory_allocated(self.device))

        with open(output_path, "w") as f:
            json.dump({
                "rows": self.compare_rows,
                "summary_correlations": summary,
                "trajectory_agreement_metrics": self.compare_agreement_metrics,
                "summary_metrics": summary_metrics,
                "runtime_seconds": self.compare_runtimes,
                "peak_cuda_memory_bytes": peak_cuda_memory,
            }, f, indent=2)
        print(f"Saved score mode comparison to {output_path}")

    def save_planner_runtime(self):
        output_dir = self.args.output_dir or "."
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"planner_runtime_{self.eval_name}_{self.args.planner_variant}.json")
        student_count = int(self.route2_timing.get("student_rerank_count", 0))
        compare_count = int(self.route2_timing.get("student_teacher_compare_count", 0))
        regret_count = int(self.route2_timing.get("student_teacher_regret_count", 0))
        with open(output_path, "w") as f:
            json.dump({
                "planner_variant": self.args.planner_variant,
                "fast_diffusion_steps": self.args.fast_diffusion_steps,
                "fast_score_mode": self.args.fast_score_mode,
                "rerank_diffusion_steps": self.args.rerank_diffusion_steps,
                "rerank_top_m": self.args.rerank_top_m,
                "rerank_score_mode": self.args.rerank_score_mode,
                "rerank_backend": self.args.rerank_backend,
                "student_rerank_checkpoint": self.args.student_rerank_checkpoint,
                "student_rerank_top_m": self.args.student_rerank_top_m,
                "student_rerank_debug_compare_teacher": self.args.student_rerank_debug_compare_teacher,
                "average_student_feature_time": (
                    self.route2_timing["student_feature_time"] / student_count
                    if student_count > 0 else None
                ),
                "average_student_inference_time": (
                    self.route2_timing["student_inference_time"] / student_count
                    if student_count > 0 else None
                ),
                "average_student_total_rerank_time": (
                    self.route2_timing["student_total_rerank_time"] / student_count
                    if student_count > 0 else None
                ),
                "average_rerank_rollout_time": (
                    self.route2_timing["rerank_rollout_time"] / max(1, len(self.coarse_to_fine_rows))
                    if self.coarse_to_fine_rows else None
                ),
                "student_teacher_agreement": (
                    self.route2_timing["student_teacher_agreement_count"] / compare_count
                    if compare_count > 0 else None
                ),
                "student_teacher_regret": (
                    self.route2_timing["student_teacher_regret_sum"] / regret_count
                    if regret_count > 0 else None
                ),
                "runtime_seconds": self.planner_runtimes,
                "diffusion_rollout_timing_seconds": self.eval_timing,
                "route2_timing_seconds": self.route2_timing,
                "route2_cache_stats": self.route2_cache_stats,
                "adaptive_stats": self.adaptive_summary_stats(),
                "adaptive_two_gate_stats": self.adaptive_two_gate_summary_stats(),
                "top_m_recall_full_best": self.last_top_m_recall,
                "top_m_recall_rows": self.top_m_recall_rows,
                "coarse_to_fine_rows": self.coarse_to_fine_rows,
            }, f, indent=2)
        print(f"Saved planner runtime breakdown to {output_path}")

    def save_student_rerank_debug_csv(self):
        if not self.args.student_rerank_debug_compare_teacher or not self.student_rerank_debug_rows:
            return
        output_path = self.args.student_rerank_save_debug_csv
        if output_path is None:
            output_dir = self.args.output_dir or "."
            output_path = os.path.join(output_dir, f"student_rerank_debug_{self.eval_name}.csv")
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fieldnames = [
            "dataset", "eval_index", "plan_iter", "seed", "rerank_backend",
            "fast_top_m_indices", "student_selected_index", "teacher_selected_index",
            "student_teacher_agreement", "student_teacher_regret",
            "student_scores", "student_raw_scores", "student_residual", "student_confidence",
            "student_residual_mean_abs", "student_residual_max_abs",
            "b18_p_better", "b18_predicted_delta", "b18_promoted",
            "student_raw_selected_local_rank", "student_selected_local_rank",
            "student_fallback_to_fast", "student_rank_prior_weight",
            "student_select_margin_threshold", "student_max_local_rank",
            "teacher_rerank_losses",
        ]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.student_rerank_debug_rows)
        print(f"Saved student rerank debug CSV to {output_path}")

    def save_summary_csv(self):
        if not self.summary_rows:
            return
        output_path = self.args.summary_csv_path
        if output_path is None:
            output_dir = self.args.output_dir or "."
            output_path = os.path.join(output_dir, f"planner_summary_{self.eval_name}_{self.score_mode}_{self.args.diffusion_steps}.csv")
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fieldnames = [
            "planner_variant", "score_mode", "fast_score_mode", "rerank_score_mode",
            "rerank_backend", "student_rerank_checkpoint",
            "diffusion_steps", "fast_diffusion_steps", "rerank_diffusion_steps", "rerank_top_m",
            "seed", "eval_index",
            "ATE", "RPE_trans", "pos_diff", "yaw_diff", "time_per_iter",
            "peak_cuda_memory_bytes", "decode_time", "rollout_time", "scoring_time",
            "wrapper_setup_time", "noise_initialization_time", "ddim_loop_total_time",
            "p_mean_variance_time", "cdit_forward_time", "timestep_tensor_setup_time",
            "conditioning_setup_time", "total_model_forward_wrapper_time",
            "candidate_generation_time", "goal_encode_time", "context_encode_time",
            "coarse_rollout_time", "coarse_scoring_time",
            "top_m_selection_time", "rerank_rollout_time", "rerank_scoring_time",
            "student_feature_time", "student_inference_time", "student_total_rerank_time",
            "skipped_diffusion_rerank",
            "final_rollout_time", "final_scoring_time",
            "total_planning_time", "goal_latent_cached_used", "context_latent_cached_used",
            "student_teacher_agreement", "student_teacher_regret",
            "adaptive_early_exit_count", "adaptive_mid_rerank_count",
            "adaptive_high_fallback_count", "adaptive_avg_norm_margin",
            "adaptive_avg_rerank_norm_margin", "adaptive_effective_step_equiv",
            "adaptive_two_gate_early_exit_count", "adaptive_two_gate_topm_fallback_count",
            "adaptive_two_gate_full_fallback_count", "adaptive_two_gate_avg_exit_prob",
            "adaptive_two_gate_avg_topm_prob", "adaptive_two_gate_effective_step_equiv",
            "top_m_recall_full_best",
        ]
        for row in self.summary_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.summary_rows)
        print(f"Saved planner summary CSV to {output_path}")
    
    def get_eval_name(self):
        # Get evaluation name for logging. Should overwrite for specific experiments
        self.eval_name = f'CEM_N{self.args.num_samples}_K{self.args.topk}_RS{self.args.rollout_stride}_rep{self.args.num_repeat_eval}_OPT{self.args.opt_steps}'
        
    def actions_to_traj(self, actions):
        positions_xyz = torch.zeros((actions.shape[0], 3))
        positions_xyz[:, :2] = actions
        orientations_quat_wxyz = torch.zeros((actions.shape[0], 4)) # Define identity quaternion
        orientations_quat_wxyz[:, -1] = 1 # Define identity quaternion
        timestamps = torch.arange(actions.shape[0], dtype=torch.float64)
        traj = PoseTrajectory3D(positions_xyz=positions_xyz, orientations_quat_wxyz=orientations_quat_wxyz, timestamps=timestamps)
        return traj
    
    @torch.no_grad
    def evaluate(self):
        
        for dataset_name in self.dataset_names:
            metric_logger = dist.MetricLogger(delimiter="  ")
            header = 'Test:'
            eval_save_output_dir = None
            
            if self.args.save_preds:
                dataset_save_output_dir = os.path.join(self.args.save_output_dir, dataset_name)
                os.makedirs(dataset_save_output_dir, exist_ok=True)
                eval_save_output_dir = os.path.join(dataset_save_output_dir, self.eval_name)
                os.makedirs(eval_save_output_dir, exist_ok=True)
            
            # Evaluate the current dataset loader.
            curr_data_loader = self.datasets[dataset_name]
            debug_eval = os.environ.get("PLANNER_DEBUG_EVAL", "0") == "1"
            if debug_eval:
                print(f"[planner-debug] starting eval dataset={dataset_name}", flush=True)
            for batch_idx, (idxs, obs_image, goal_image, gt_actions, goal_pos) in enumerate(metric_logger.log_every(curr_data_loader, 1, header)):
                if self.args.max_eval_batches is not None and batch_idx >= self.args.max_eval_batches:
                    break
                if self.args.eval_index_offset:
                    idxs = idxs + int(self.args.eval_index_offset)
                if debug_eval:
                    print(
                        f"[planner-debug] batch={batch_idx} idxs={idxs.flatten().int().tolist()} "
                        f"obs_shape={tuple(obs_image.shape)} goal_shape={tuple(goal_image.shape)}",
                        flush=True,
                    )
                if (
                    self.args.export_latent_rerank_distill_dataset
                    and self.args.latent_rerank_distill_resume
                    and all(
                        int(idx) in self.latent_rerank_distill_completed_eval_indices
                        for idx in idxs.flatten().int().tolist()
                    )
                ):
                    print(f"Skipping completed latent rerank distill batch idxs={idxs.flatten().int().tolist()}")
                    continue
                obs_image = obs_image[:, -self.num_cond:]
                self.sync_cuda_for_timing()
                batch_start = time.perf_counter()
                if debug_eval:
                    print(f"[planner-debug] before generate_actions batch={batch_idx}", flush=True)
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    pred_actions, pred_yaw = self.generate_actions(
                        eval_save_output_dir,
                        dataset_name,
                        idxs,
                        obs_image,
                        goal_image,
                        gt_actions,
                        self.config["trajectory_eval_len_traj_pred"],
                        batch_idx=batch_idx,
                    )
                if debug_eval:
                    print(f"[planner-debug] after generate_actions batch={batch_idx}", flush=True)
                self.sync_cuda_for_timing()
                batch_elapsed = time.perf_counter() - batch_start
                time_per_iter = batch_elapsed / max(1, self.opt_steps)
                peak_cuda_memory = int(torch.cuda.max_memory_allocated(self.device)) if torch.cuda.is_available() else None
                for i in range(len(obs_image)):
                    pred_traj_i = self.actions_to_traj(pred_actions[i, :, :2])
                    gt_traj_i = self.actions_to_traj(gt_actions[i, :, :2])
                    
                    ate, rpe_trans, _ = self.eval_metrics(gt_traj_i, pred_traj_i)

                    pred_final_pos = pred_actions[i, -1, :2].to('cpu') # (2,)
                    pred_final_yaw = pred_yaw[i].to('cpu') # 
                    goal_final_pos = goal_pos[i, 0, :2] # (2,)
                    goal_final_yaw = goal_pos[i, 0, -1] # (B,)
                    pos_diff_norm = torch.norm(pred_final_pos - goal_final_pos)
                    yaw_diff = pred_final_yaw - goal_final_yaw  # 
                    yaw_diff_norm = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff)).abs()
                    
                    metric_logger.meters['{}_ate'.format(dataset_name)].update(ate, n=1)
                    metric_logger.meters['{}_rpe_trans'.format(dataset_name)].update(rpe_trans, n=1)
                    metric_logger.meters['{}_pos_diff_norm'.format(dataset_name)].update(pos_diff_norm, n=1)   
                    metric_logger.meters['{}_yaw_diff_norm'.format(dataset_name)].update(yaw_diff_norm, n=1)   
                    adaptive_summary = self.adaptive_summary_stats()
                    adaptive_two_gate_summary = self.adaptive_two_gate_summary_stats()
                    student_compare_count = int(self.route2_timing.get("student_teacher_compare_count", 0))
                    student_regret_count = int(self.route2_timing.get("student_teacher_regret_count", 0))
                    self.summary_rows.append({
                        "planner_variant": self.args.planner_variant,
                        "score_mode": self.score_mode,
                        "fast_score_mode": self.args.fast_score_mode,
                        "rerank_score_mode": self.args.rerank_score_mode,
                        "rerank_backend": self.args.rerank_backend,
                        "student_rerank_checkpoint": self.args.student_rerank_checkpoint,
                        "diffusion_steps": self.args.diffusion_steps,
                        "fast_diffusion_steps": self.args.fast_diffusion_steps,
                        "rerank_diffusion_steps": self.args.rerank_diffusion_steps,
                        "rerank_top_m": self.args.rerank_top_m,
                        "seed": self.args.seed,
                        "eval_index": int(idxs.flatten()[i].item()),
                        "ATE": float(ate),
                        "RPE_trans": float(rpe_trans),
                        "pos_diff": float(pos_diff_norm.detach().cpu()),
                        "yaw_diff": float(yaw_diff_norm.detach().cpu()),
                        "time_per_iter": float(time_per_iter),
                        "peak_cuda_memory_bytes": peak_cuda_memory,
                        "decode_time": float(self.eval_timing["decode_time"]),
                        "rollout_time": float(self.eval_timing["rollout_time"]),
                        "scoring_time": float(self.eval_timing["scoring_time"]),
                        "wrapper_setup_time": float(self.eval_timing["wrapper_setup_time"]),
                        "noise_initialization_time": float(self.eval_timing["noise_initialization_time"]),
                        "ddim_loop_total_time": float(self.eval_timing["ddim_loop_total_time"]),
                        "p_mean_variance_time": float(self.eval_timing["p_mean_variance_time"]),
                        "cdit_forward_time": float(self.eval_timing["cdit_forward_time"]),
                        "timestep_tensor_setup_time": float(self.eval_timing["timestep_tensor_setup_time"]),
                        "conditioning_setup_time": float(self.eval_timing["conditioning_setup_time"]),
                        "total_model_forward_wrapper_time": float(self.eval_timing["total_model_forward_wrapper_time"]),
                        "candidate_generation_time": float(self.route2_timing["candidate_generation_time"]),
                        "goal_encode_time": float(self.route2_timing["goal_encode_time"]),
                        "context_encode_time": float(self.route2_timing["context_encode_time"]),
                        "coarse_rollout_time": float(self.route2_timing["coarse_rollout_time"]),
                        "coarse_scoring_time": float(self.route2_timing["coarse_scoring_time"]),
                        "top_m_selection_time": float(self.route2_timing["top_m_selection_time"]),
                        "rerank_rollout_time": float(self.route2_timing["rerank_rollout_time"]),
                        "rerank_scoring_time": float(self.route2_timing["rerank_scoring_time"]),
                        "student_feature_time": float(self.route2_timing["student_feature_time"]),
                        "student_inference_time": float(self.route2_timing["student_inference_time"]),
                        "student_total_rerank_time": float(self.route2_timing["student_total_rerank_time"]),
                        "skipped_diffusion_rerank": bool(self.route2_timing["skipped_diffusion_rerank"]),
                        "final_rollout_time": float(self.route2_timing["final_rollout_time"]),
                        "final_scoring_time": float(self.route2_timing["final_scoring_time"]),
                        "total_planning_time": float(self.route2_timing["total_planning_time"]),
                        "goal_latent_cached_used": int(self.route2_cache_stats["goal_latent_cached_used"]),
                        "context_latent_cached_used": int(self.route2_cache_stats["context_latent_cached_used"]),
                        "student_teacher_agreement": (
                            self.route2_timing["student_teacher_agreement_count"] / student_compare_count
                            if student_compare_count > 0 else None
                        ),
                        "student_teacher_regret": (
                            self.route2_timing["student_teacher_regret_sum"] / student_regret_count
                            if student_regret_count > 0 else None
                        ),
                        "adaptive_early_exit_count": adaptive_summary["adaptive_early_exit_count"],
                        "adaptive_mid_rerank_count": adaptive_summary["adaptive_mid_rerank_count"],
                        "adaptive_high_fallback_count": adaptive_summary["adaptive_high_fallback_count"],
                        "adaptive_avg_norm_margin": adaptive_summary["adaptive_avg_norm_margin"],
                        "adaptive_avg_rerank_norm_margin": adaptive_summary["adaptive_avg_rerank_norm_margin"],
                        "adaptive_effective_step_equiv": adaptive_summary["adaptive_effective_step_equiv"],
                        "adaptive_two_gate_early_exit_count": adaptive_two_gate_summary["adaptive_two_gate_early_exit_count"],
                        "adaptive_two_gate_topm_fallback_count": adaptive_two_gate_summary["adaptive_two_gate_topm_fallback_count"],
                        "adaptive_two_gate_full_fallback_count": adaptive_two_gate_summary["adaptive_two_gate_full_fallback_count"],
                        "adaptive_two_gate_avg_exit_prob": adaptive_two_gate_summary["adaptive_two_gate_avg_exit_prob"],
                        "adaptive_two_gate_avg_topm_prob": adaptive_two_gate_summary["adaptive_two_gate_avg_topm_prob"],
                        "adaptive_two_gate_effective_step_equiv": adaptive_two_gate_summary["adaptive_two_gate_effective_step_equiv"],
                        "s_combined_pred_dATE_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_pred_dATE"))
                            if self.last_student_rerank_score_debug.get("s_combined_pred_dATE") is not None else None
                        ),
                        "s_combined_pred_dRPE_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_pred_dRPE"))
                            if self.last_student_rerank_score_debug.get("s_combined_pred_dRPE") is not None else None
                        ),
                        "s_combined_pred_dpos_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_pred_dpos"))
                            if self.last_student_rerank_score_debug.get("s_combined_pred_dpos") is not None else None
                        ),
                        "s_combined_pred_dyaw_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_pred_dyaw"))
                            if self.last_student_rerank_score_debug.get("s_combined_pred_dyaw") is not None else None
                        ),
                        "s_combined_p_helpful_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_p_helpful"))
                            if self.last_student_rerank_score_debug.get("s_combined_p_helpful") is not None else None
                        ),
                        "s_combined_p_harm_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_p_harm"))
                            if self.last_student_rerank_score_debug.get("s_combined_p_harm") is not None else None
                        ),
                        "s_combined_pred_dATE_std_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_pred_dATE_std"))
                            if self.last_student_rerank_score_debug.get("s_combined_pred_dATE_std") is not None else None
                        ),
                        "s_combined_pred_dRPE_std_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_pred_dRPE_std"))
                            if self.last_student_rerank_score_debug.get("s_combined_pred_dRPE_std") is not None else None
                        ),
                        "s_combined_pred_dpos_std_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_pred_dpos_std"))
                            if self.last_student_rerank_score_debug.get("s_combined_pred_dpos_std") is not None else None
                        ),
                        "s_combined_pred_dyaw_std_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_pred_dyaw_std"))
                            if self.last_student_rerank_score_debug.get("s_combined_pred_dyaw_std") is not None else None
                        ),
                        "s_combined_p_helpful_std_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_p_helpful_std"))
                            if self.last_student_rerank_score_debug.get("s_combined_p_helpful_std") is not None else None
                        ),
                        "s_combined_p_harm_std_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_p_harm_std"))
                            if self.last_student_rerank_score_debug.get("s_combined_p_harm_std") is not None else None
                        ),
                        "s_combined_p_select_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_p_select"))
                            if self.last_student_rerank_score_debug.get("s_combined_p_select") is not None else None
                        ),
                        "s_combined_select_margin_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_select_margin"))
                            if self.last_student_rerank_score_debug.get("s_combined_select_margin") is not None else None
                        ),
                        "s_combined_utility_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_utility"))
                            if self.last_student_rerank_score_debug.get("s_combined_utility") is not None else None
                        ),
                        "s_combined_utility_std_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_utility_std"))
                            if self.last_student_rerank_score_debug.get("s_combined_utility_std") is not None else None
                        ),
                        "s_combined_candidate_local_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_candidate_local"))
                            if self.last_student_rerank_score_debug.get("s_combined_candidate_local") is not None else None
                        ),
                        "s_combined_candidate_rank_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_candidate_rank"))
                            if self.last_student_rerank_score_debug.get("s_combined_candidate_rank") is not None else None
                        ),
                        "s_combined_candidate_fast_loss_vs_top1_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_candidate_fast_loss_vs_top1"))
                            if self.last_student_rerank_score_debug.get("s_combined_candidate_fast_loss_vs_top1") is not None else None
                        ),
                        "s_combined_group_fast_top1_loss_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_group_fast_top1_loss"))
                            if self.last_student_rerank_score_debug.get("s_combined_group_fast_top1_loss") is not None else None
                        ),
                        "s_combined_group_fast_top2_loss_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_group_fast_top2_loss"))
                            if self.last_student_rerank_score_debug.get("s_combined_group_fast_top2_loss") is not None else None
                        ),
                        "s_combined_fast_margin_top1_top2_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_fast_margin_top1_top2"))
                            if self.last_student_rerank_score_debug.get("s_combined_fast_margin_top1_top2") is not None else None
                        ),
                        "s_combined_normalized_fast_margin_top1_top2_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_normalized_fast_margin_top1_top2"))
                            if self.last_student_rerank_score_debug.get("s_combined_normalized_fast_margin_top1_top2") is not None else None
                        ),
                        "s_combined_b18_p_better_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_b18_p_better"))
                            if self.last_student_rerank_score_debug.get("s_combined_b18_p_better") is not None else None
                        ),
                        "s_combined_b18_pred_delta_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_b18_pred_delta"))
                            if self.last_student_rerank_score_debug.get("s_combined_b18_pred_delta") is not None else None
                        ),
                        "s_combined_b18_action_distance_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_b18_action_distance"))
                            if self.last_student_rerank_score_debug.get("s_combined_b18_action_distance") is not None else None
                        ),
                        "s_combined_b18_final_action_distance_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_b18_final_action_distance"))
                            if self.last_student_rerank_score_debug.get("s_combined_b18_final_action_distance") is not None else None
                        ),
                        "s_combined_b18_yaw_distance_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_b18_yaw_distance"))
                            if self.last_student_rerank_score_debug.get("s_combined_b18_yaw_distance") is not None else None
                        ),
                        "s_combined_b18_fast_loss_margin_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_b18_fast_loss_margin"))
                            if self.last_student_rerank_score_debug.get("s_combined_b18_fast_loss_margin") is not None else None
                        ),
                        "s_combined_selected_local": self.last_student_rerank_score_debug.get("s_combined_selected_local"),
                        "s_combined_gate_json": (
                            json.dumps(self.last_student_rerank_score_debug.get("s_combined_gate"))
                            if self.last_student_rerank_score_debug.get("s_combined_gate") is not None else None
                        ),
                        "s_combined_p_fallback": self.last_student_rerank_score_debug.get("s_combined_p_fallback"),
                        "s_combined_p_challenge": self.last_student_rerank_score_debug.get("s_combined_p_challenge"),
            "s_combined_guard_pred_dATE_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_guard_pred_dATE"))
                if self.last_student_rerank_score_debug.get("s_combined_guard_pred_dATE") is not None else None
            ),
            "s_combined_guard_pred_dRPE_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_guard_pred_dRPE"))
                if self.last_student_rerank_score_debug.get("s_combined_guard_pred_dRPE") is not None else None
            ),
            "s_combined_guard_pred_dpos_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_guard_pred_dpos"))
                if self.last_student_rerank_score_debug.get("s_combined_guard_pred_dpos") is not None else None
            ),
            "s_combined_guard_pred_dyaw_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_guard_pred_dyaw"))
                if self.last_student_rerank_score_debug.get("s_combined_guard_pred_dyaw") is not None else None
            ),
            "s_combined_guard_p_safe_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_guard_p_safe"))
                if self.last_student_rerank_score_debug.get("s_combined_guard_p_safe") is not None else None
            ),
            "s_combined_guard_p_harm_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_guard_p_harm"))
                if self.last_student_rerank_score_debug.get("s_combined_guard_p_harm") is not None else None
            ),
            "s_combined_guard_p_strong_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_guard_p_strong"))
                if self.last_student_rerank_score_debug.get("s_combined_guard_p_strong") is not None else None
            ),
            "s_combined_guard_score_json": (
                json.dumps(self.last_student_rerank_score_debug.get("s_combined_guard_score"))
                if self.last_student_rerank_score_debug.get("s_combined_guard_score") is not None else None
            ),
                        "top_m_recall_full_best": self.last_top_m_recall,
                    })
            if hasattr(self.args, 'save_output_dir') and self.args.save_output_dir is not None:
                output_fn = os.path.join(self.args.save_output_dir, f'{dataset_name}_{self.eval_name}.json')
                save_metric_to_disk(metric_logger, output_fn)

        self.save_score_table()
        self.save_compare_score_modes()
        self.save_planner_runtime()
        self.save_student_rerank_debug_csv()
        self.save_summary_csv()
        self.save_teacher_decisions()
        self.save_latent_rerank_distill_dataset()
        self.save_teacher_collection_v2_dataset()
        self.save_candidate_actions_export()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
            
    def eval_metrics(self, traj_ref, traj_pred):
        traj_ref, traj_pred = sync.associate_trajectories(traj_ref, traj_pred)
        
        result = main_ape.ape(traj_ref, traj_pred, est_name='traj',
            pose_relation=PoseRelation.translation_part, align=False, correct_scale=False)
        ate = result.stats['rmse']

        result = main_rpe.rpe(traj_ref, traj_pred, est_name='traj',
            pose_relation=PoseRelation.rotation_angle_deg, align=False, correct_scale=False,
            delta=1.0, delta_unit=metrics.Unit.frames, rel_delta_tol=0.1)
        rpe_rot = result.stats['rmse']

        result = main_rpe.rpe(traj_ref, traj_pred, est_name='traj',
            pose_relation=PoseRelation.translation_part, align=False, correct_scale=False,
            delta=1.0, delta_unit=metrics.Unit.frames, rel_delta_tol=0.1)
        rpe_trans = result.stats['rmse']

        return ate, rpe_trans, rpe_rot
    
if __name__ == "__main__":
    # Learned scorer smoke test:
    # python planning_eval.py ^
    #   --exp config/nwm_cdit_s.yaml ^
    #   --ckp 0100000 ^
    #   --datasets recon ^
    #   --output_dir eval_output\planner_learned_spatial_smoke ^
    #   --score_mode learned_spatial_scorer ^
    #   --latent_scorer_checkpoint eval_output\latent_scorer_spatial_cnn_seed2_val\scorer_best.pt ^
    #   --diffusion_steps 100 ^
    #   --planner_variant standard ^
    #   --seed 42 ^
    #   --num_samples 2 ^
    #   --batch_size 1 ^
    #   --max_eval_batches 1 ^
    #   --summary_csv_path eval_output\summary_results\planner_learned_spatial_smoke.csv
    parser = argparse.ArgumentParser()
    
    # Default Args
    parser.add_argument("--exp", type=str, default=None, help="experiment name")
    parser.add_argument("--ckp", type=str, default='0100000', help="experiment name")

    parser.add_argument("--datasets", type=str, default=None, help="dataset name")
    parser.add_argument("--output_dir", type=str, default=None, help="output dir to save model predictions")
    parser.add_argument('--save_preds', action='store_true', default=False, help='whether to save prediction tensors or not')
    parser.add_argument("--num_workers", type=int, default=8, help="num workers")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--diffusion_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--navigation_eval_index", type=str, default=None)
    parser.add_argument("--eval_index_offset", type=int, default=0)
    parser.add_argument(
        "--trajectory_eval_horizon",
        type=int,
        default=None,
        help="Override trajectory_eval_len_traj_pred from config/eval_config.yaml.",
    )
    parser.add_argument(
        "--eval_data_folder",
        type=str,
        default=None,
        help="Override eval_datasets[*].data_folder for local evaluation.",
    )
    
    # Planning Specific Args
    parser.add_argument("--num_samples", type=int, default=10, help="num nomad samples to predict")
    parser.add_argument("--rollout_stride", type=int, default=1, help="rollout stride")
    parser.add_argument("--topk", type=int, default=5, help="top k samples to take mean and var for CEM")
    parser.add_argument("--opt_steps", type=int, default=15, help="num iterations for CEM")
    parser.add_argument("--num_repeat_eval", type=int, default=1, help="number of evals for one action")
    parser.add_argument(
        "--candidate_batch_size",
        type=int,
        default=None,
        help="Evaluate planning candidates in equivalent chunks to limit peak CUDA memory.",
    )
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--constraint', type=str, default='none',
                        choices=['none', 'forward_first', 'left_right_first', 'straight_then_forward'])
    parser.add_argument("--score_mode", type=str, default="lpips",
                        choices=["lpips", "latent_l2", "latent_cosine", "latent_l2_yaw", "learned_spatial_scorer"])
    parser.add_argument("--latent_scorer_checkpoint", type=str, default=None)
    parser.add_argument("--latent_scorer_device", type=str, default=None)
    parser.add_argument("--yaw_penalty_weight", type=float, default=0.0)
    parser.add_argument("--debug_yaw_penalty", action="store_true")
    parser.add_argument("--debug_cem_action_stats", action="store_true")
    parser.add_argument(
        "--cem_update_space",
        type=str,
        default="legacy_final_delta",
        choices=["legacy_final_delta", "raw_sample"],
    )
    parser.add_argument(
        "--planner_final_yaw_policy",
        type=str,
        default="legacy",
        choices=["legacy", "clip", "wrap", "zero"],
    )
    parser.add_argument("--decode_for_debug", action="store_true")
    parser.add_argument("--save_score_table", action="store_true")
    parser.add_argument("--score_table_path", type=str, default=None)
    parser.add_argument("--summary_csv_path", type=str, default=None)
    parser.add_argument("--compare_score_modes", action="store_true")
    parser.add_argument("--compare_chunk_size", type=int, default=1)
    parser.add_argument("--collect_teacher_decisions", action="store_true", default=False)
    parser.add_argument("--teacher_decision_output_path", type=str, default=None)
    parser.add_argument("--teacher_decision_max_records", type=int, default=None)
    parser.add_argument("--teacher_record_latents", action="store_true", default=False)
    parser.add_argument("--export_candidate_actions_path", type=str, default=None)
    parser.add_argument("--fixed_candidate_actions_path", type=str, default=None)
    parser.add_argument("--fixed_candidate_actions_strict", action="store_true", default=False)
    parser.add_argument("--fixed_candidate_actions_skip_fast_rollout", action="store_true", default=False)
    parser.add_argument("--export_latent_rerank_distill_dataset", action="store_true", default=False)
    parser.add_argument("--export_fast_rollout_trace", action="store_true", default=False)
    parser.add_argument("--latent_rerank_distill_output_path", type=str, default=None)
    parser.add_argument("--latent_rerank_distill_max_records", type=int, default=None)
    parser.add_argument("--latent_rerank_distill_teacher_steps", type=int, default=100)
    parser.add_argument("--latent_rerank_distill_skip_teacher", action="store_true", default=False)
    parser.add_argument("--latent_rerank_distill_all_cem_steps", action="store_true", default=False)
    parser.add_argument("--latent_rerank_distill_incremental_save", action="store_true", default=False)
    parser.add_argument("--latent_rerank_distill_resume", action="store_true", default=False)
    parser.add_argument(
        "--latent_rerank_distill_score_mode",
        type=str,
        default="latent_cosine",
        choices=["latent_l2", "latent_cosine", "latent_l2_yaw"],
    )
    parser.add_argument("--export_teacher_collection_v2", action="store_true", default=False)
    parser.add_argument("--teacher_collection_v2_output_path", type=str, default=None)
    parser.add_argument("--teacher_collection_v2_collection_id", type=str, default=None)
    parser.add_argument("--teacher_collection_v2_cem_steps", type=str, default="final")
    parser.add_argument(
        "--teacher_collection_v2_log_outcomes",
        type=str,
        default="none",
        choices=["none", "selected_only", "fast_teacher", "fast_teacher_student"],
    )
    parser.add_argument(
        "--teacher_collection_v2_planner_reference_scope",
        type=str,
        default="shortlist",
        choices=["shortlist", "all_n_if_feasible"],
    )
    parser.add_argument("--teacher_collection_v2_include_latent_summaries", action="store_true", default=False)
    parser.add_argument("--teacher_collection_v2_max_records", type=int, default=None)
    parser.add_argument("--teacher_collection_v2_selected_eval_indices", type=str, default=None)
    parser.add_argument("--teacher_collection_v2_fail_on_missing_hash", action="store_true", default=False)
    parser.add_argument("--planner_variant", type=str, default="standard",
                        choices=[
                            "standard",
                            "fast_latent",
                            "hybrid_latent_rerank",
                            "coarse_to_fine_latent",
                            "adaptive_latent_rerank",
                            "adaptive_two_gate_low_high",
                        ])
    parser.add_argument("--fast_diffusion_steps", type=int, default=100)
    parser.add_argument(
        "--per_eval_candidate_seed",
        action="store_true",
        default=False,
        help=(
            "Seed candidate action noise from seed/eval_index/opt_step so the same eval case "
            "gets the same candidate pool independent of split order."
        ),
    )
    parser.add_argument("--fast_score_mode", type=str, default="latent_l2",
                        choices=["latent_l2", "latent_cosine", "latent_l2_yaw"])
    parser.add_argument("--rerank_diffusion_steps", type=int, default=250)
    parser.add_argument("--rerank_top_m", "--rerank_top_k", dest="rerank_top_m", type=int, default=4)
    parser.add_argument("--rerank_score_mode", type=str, default="lpips",
                        choices=["lpips", "latent_l2", "latent_cosine", "latent_l2_yaw"])
    parser.add_argument(
        "--rerank_backend",
        type=str,
        default="diffusion",
        choices=[
            "diffusion",
            "teacher_diffusion_rerank",
            "student",
            "B3_student",
            "zero_residual_student",
            "b18_direct_challenger",
            "b51_safegate",
            "b48_outcome_student",
            "s_combined_temporal",
            "s_groupfirst_ensemble",
            "s_hybrid_groupfirst_combined",
            "identity_fast_loss",
            "force_fast_rank",
            "top5_boundary_gate",
        ],
    )
    parser.add_argument("--force_fast_rank", type=int, default=1)
    parser.add_argument("--force_fast_rank_score_with_rerank", action="store_true")
    parser.add_argument("--b18_challenger_checkpoint", type=str, default=None)
    parser.add_argument("--b18_challenger_p_threshold", type=float, default=0.60)
    parser.add_argument("--b18_challenger_tau", type=float, default=-0.01)
    parser.add_argument("--b18_challenger_min_predicted_delta", type=float, default=None)
    parser.add_argument("--b18_challenger_confidence_threshold", type=float, default=0.0)
    parser.add_argument("--b18_challenger_min_fast_rank", type=int, default=2)
    parser.add_argument("--b18_challenger_max_fast_rank", type=int, default=5)
    parser.add_argument("--b18_challenger_min_action_distance", type=float, default=None)
    parser.add_argument("--b18_challenger_max_action_distance", type=float, default=None)
    parser.add_argument("--b18_challenger_max_fast_loss_margin", type=float, default=None)
    parser.add_argument("--b18_challenger_max_yaw_distance", type=float, default=None)
    parser.add_argument("--b51_safegate_checkpoint", type=str, default=None)
    parser.add_argument("--b51_helpful_threshold", type=float, default=0.58)
    parser.add_argument("--b51_harm_risk_threshold", type=float, default=0.78)
    parser.add_argument("--b51_pred_aate_threshold", type=float, default=-0.005)
    parser.add_argument("--b48_outcome_checkpoint", type=str, default=None)
    parser.add_argument("--b48_outcome_p_helpful_min", type=float, default=0.60)
    parser.add_argument("--b48_outcome_p_harm_max", type=float, default=0.75)
    parser.add_argument("--b48_outcome_pred_date_max", type=float, default=-0.005)
    parser.add_argument("--b48_outcome_min_pred_date", type=float, default=None)
    parser.add_argument("--b48_outcome_groupfirst_gate", action="store_true")
    parser.add_argument("--b48_outcome_min_fast_rank", type=int, default=4)
    parser.add_argument("--b48_outcome_max_fast_rank", type=int, default=10)
    parser.add_argument("--b48_outcome_max_action_distance", type=float, default=None)
    parser.add_argument("--b48_outcome_max_fast_loss_margin", type=float, default=None)
    parser.add_argument("--b48_outcome_max_yaw_distance", type=float, default=None)
    parser.add_argument("--b48_outcome_min_fast_top16_spread", type=float, default=None)
    parser.add_argument("--b48_outcome_min_fast_top1_loss", type=float, default=None)
    parser.add_argument("--b48_outcome_dynamic_gate", action="store_true")
    parser.add_argument("--b48_outcome_dynamic_policy_json", type=str, default=None)
    parser.add_argument("--b48_outcome_dynamic_spread_min", type=float, default=None)
    parser.add_argument("--b48_outcome_dynamic_top1_loss_min", type=float, default=None)
    parser.add_argument("--b48_outcome_conservative_p_helpful_min", type=float, default=None)
    parser.add_argument("--b48_outcome_conservative_p_harm_max", type=float, default=None)
    parser.add_argument("--b48_outcome_conservative_pred_date_max", type=float, default=None)
    parser.add_argument("--b48_outcome_conservative_min_fast_rank", type=int, default=None)
    parser.add_argument("--b48_outcome_conservative_max_fast_rank", type=int, default=None)
    parser.add_argument("--s_combined_temporal_checkpoint", type=str, default=None)
    parser.add_argument("--s_groupfirst_ensemble_checkpoint", type=str, default=None)
    parser.add_argument("--s_groupfirst_risk_veto_json", type=str, default=None)
    parser.add_argument("--s_hybrid_policy_json", type=str, default=None)
    parser.add_argument("--student_rerank_checkpoint", type=str, default=None)
    parser.add_argument("--student_rerank_top_m", type=int, default=None)
    parser.add_argument("--student_rerank_debug_compare_teacher", action="store_true")
    parser.add_argument("--student_rerank_save_debug_csv", type=str, default=None)
    parser.add_argument("--student_arch", default="mlp_residual", choices=["mlp_residual", "deepsets_residual"])
    parser.add_argument("--student_selective_gate", action="store_true")
    parser.add_argument("--student_confidence_threshold", type=float, default=0.5)
    parser.add_argument("--student_margin_threshold", type=float, default=0.0)
    parser.add_argument("--student_residual_safety_threshold", type=float, default=float("inf"))
    parser.add_argument("--student_gate_fallback", choices=["fast", "teacher_diffusion"], default="fast")
    parser.add_argument("--student_residual_clip", type=float, default=None)
    parser.add_argument("--student_rank_prior_weight", "--student_rerank_rank_prior_weight", type=float, default=0.0)
    parser.add_argument("--student_select_margin_threshold", type=float, default=0.0)
    parser.add_argument("--student_max_local_rank", type=int, default=-1)
    parser.add_argument(
        "--student_cem_update_mode",
        choices=["fast_loss", "student_score", "guarded_student_score"],
        default="fast_loss",
        help=(
            "For student rerank backends, choose which scores update the next CEM step. "
            "fast_loss isolates selection from CEM updates; student_score uses raw unguarded "
            "student scores; guarded_student_score preserves the old guard-as-CEM-loss behavior."
        ),
    )
    parser.add_argument("--student_cem_safety_gate", action="store_true")
    parser.add_argument("--student_cem_max_top5_changes_per_sample", type=int, default=1)
    parser.add_argument("--student_cem_yaw_abs_margin", type=float, default=0.05)
    parser.add_argument("--student_cem_yaw_smooth_margin", type=float, default=0.05)
    parser.add_argument("--student_cem_action_norm_margin", type=float, default=0.05)
    parser.add_argument("--student_cem_action_norm_ratio", type=float, default=1.10)
    parser.add_argument("--top5_boundary_gate_checkpoint", type=str, default=None)
    parser.add_argument("--top5_boundary_threshold", type=float, default=0.9)
    parser.add_argument("--top5_boundary_remove_rank", type=int, default=3)
    parser.add_argument("--top5_boundary_min_plan_iter", type=int, default=0)
    parser.add_argument("--top5_boundary_use_remove_head", action="store_true")
    parser.add_argument("--top5_boundary_disallow_pred_remove", type=str, default="")
    parser.add_argument("--top5_boundary_norm_margin_min", type=float, default=None)
    parser.add_argument("--top5_boundary_norm_margin_max", type=float, default=None)
    parser.add_argument("--compute_top_m_recall", action="store_true")
    parser.add_argument("--diagnose_full_rerank_oracle", action="store_true")
    parser.add_argument("--diagnose_compare_rerank_steps", type=str, default="")
    parser.add_argument("--adaptive_margin_threshold", type=float, default=1.5)
    parser.add_argument("--adaptive_rerank_margin_threshold", type=float, default=1.0)
    parser.add_argument("--adaptive_rerank_steps_by_margin", action="store_true")
    parser.add_argument("--adaptive_rerank_low_steps", type=int, default=80)
    parser.add_argument("--adaptive_rerank_high_steps", type=int, default=100)
    parser.add_argument("--adaptive_fallback_top_m", type=int, default=2)
    parser.add_argument("--adaptive_debug", action="store_true")
    parser.add_argument("--adaptive_policy_path", type=str, default=None)
    parser.add_argument("--adaptive_policy_debug", action="store_true")
    parser.add_argument("--disable_adaptive_policy_rng_isolation", action="store_true")
    cli_args = sys.argv[1:]
    args = parser.parse_args()

    if args.latent_rerank_distill_resume and not args.latent_rerank_distill_incremental_save:
        raise ValueError("--latent_rerank_distill_resume requires --latent_rerank_distill_incremental_save")

    if args.latent_rerank_distill_teacher_steps <= 0:
        raise ValueError("--latent_rerank_distill_teacher_steps must be a positive integer")

    if args.export_teacher_collection_v2:
        if args.teacher_collection_v2_collection_id is None:
            args.teacher_collection_v2_collection_id = f"teacher_collection_v2_seed{args.seed}"
        cem_spec = str(args.teacher_collection_v2_cem_steps).strip().lower()
        if cem_spec not in {"final", "all"}:
            try:
                [int(item.strip()) for item in cem_spec.split(",") if item.strip()]
            except ValueError as exc:
                raise ValueError(
                    "--teacher_collection_v2_cem_steps must be final, all, or comma-separated integers"
                ) from exc
        if args.teacher_collection_v2_log_outcomes != "none":
            print(
                "Warning: teacher_collection_v2 outcome logging is requested, but this patch only "
                "exports explicit null outcome placeholders. See outcome_logging_limitation.md."
            )

    if args.planner_variant == "coarse_to_fine_latent":
        # Conservative decoder-free coarse-to-fine defaults for the new variant only.
        # Existing variants keep their historical argparse defaults.
        if "--score_mode" not in cli_args and args.score_mode == "lpips":
            args.score_mode = "latent_cosine"
        if "--fast_score_mode" not in cli_args and args.fast_score_mode == "latent_l2":
            args.fast_score_mode = "latent_cosine"
        if "--rerank_score_mode" not in cli_args and args.rerank_score_mode == "lpips":
            args.rerank_score_mode = "latent_cosine"
        if "--fast_diffusion_steps" not in cli_args and args.fast_diffusion_steps == 100:
            args.fast_diffusion_steps = 20
        if "--rerank_diffusion_steps" not in cli_args and args.rerank_diffusion_steps == 250:
            args.rerank_diffusion_steps = 100
        if "--rerank_top_m" not in cli_args and "--rerank_top_k" not in cli_args and args.rerank_top_m == 4:
            args.rerank_top_m = 6
        if args.score_mode != "latent_cosine" or args.fast_score_mode != "latent_cosine" or args.rerank_score_mode != "latent_cosine":
            raise ValueError("coarse_to_fine_latent is intentionally limited to latent_cosine scoring.")

    if args.student_rerank_top_m is None:
        args.student_rerank_top_m = args.rerank_top_m
    if args.student_rerank_top_m <= 0:
        raise ValueError("--student_rerank_top_m must be a positive integer")
    learned_student_rerank_backends = {"student", "B3_student", "zero_residual_student"}
    b18_challenger_backends = {"b18_direct_challenger", "b51_safegate"}
    b48_outcome_backends = {"b48_outcome_student"}
    top5_boundary_backends = {"top5_boundary_gate"}
    force_fast_rank_backends = {"force_fast_rank"}
    shortcut_rerank_backends = (
        learned_student_rerank_backends
        | b18_challenger_backends
        | b48_outcome_backends
        | top5_boundary_backends
        | force_fast_rank_backends
        | {"identity_fast_loss"}
    )
    if args.rerank_backend in learned_student_rerank_backends and args.student_rerank_checkpoint is None:
        raise ValueError(f"--student_rerank_checkpoint is required when --rerank_backend {args.rerank_backend}")
    if args.rerank_backend == "s_combined_temporal" and args.s_combined_temporal_checkpoint is None:
        raise ValueError("--s_combined_temporal_checkpoint is required when --rerank_backend s_combined_temporal")
    if args.rerank_backend == "s_groupfirst_ensemble" and args.s_groupfirst_ensemble_checkpoint is None:
        raise ValueError("--s_groupfirst_ensemble_checkpoint is required when --rerank_backend s_groupfirst_ensemble")
    if args.rerank_backend == "s_hybrid_groupfirst_combined":
        if args.s_groupfirst_ensemble_checkpoint is None or args.s_combined_temporal_checkpoint is None:
            raise ValueError("--s_groupfirst_ensemble_checkpoint and --s_combined_temporal_checkpoint are required when --rerank_backend s_hybrid_groupfirst_combined")
    if args.rerank_backend in {"s_combined_temporal", "s_groupfirst_ensemble", "s_hybrid_groupfirst_combined"} and not args.export_fast_rollout_trace:
        raise ValueError("--export_fast_rollout_trace is required when --rerank_backend is an S temporal runtime backend")
    if args.rerank_backend == "b18_direct_challenger" and args.b18_challenger_checkpoint is None:
        raise ValueError("--b18_challenger_checkpoint is required when --rerank_backend b18_direct_challenger")
    if args.rerank_backend == "b51_safegate" and args.b51_safegate_checkpoint is None:
        raise ValueError("--b51_safegate_checkpoint is required when --rerank_backend b51_safegate")
    if args.rerank_backend in b48_outcome_backends:
        if args.b18_challenger_checkpoint is None:
            raise ValueError("--b18_challenger_checkpoint is required when --rerank_backend b48_outcome_student")
        if args.b48_outcome_checkpoint is None:
            raise ValueError("--b48_outcome_checkpoint is required when --rerank_backend b48_outcome_student")
    if args.rerank_backend in top5_boundary_backends and args.top5_boundary_gate_checkpoint is None:
        raise ValueError("--top5_boundary_gate_checkpoint is required when --rerank_backend top5_boundary_gate")
    if (
        args.student_rerank_debug_compare_teacher
        and args.rerank_backend != "identity_fast_loss"
        and args.student_rerank_checkpoint is None
    ):
        raise ValueError("--student_rerank_checkpoint is required with --student_rerank_debug_compare_teacher")
    if (
        args.rerank_backend in shortcut_rerank_backends
        or args.student_rerank_debug_compare_teacher
    ) and args.planner_variant != "coarse_to_fine_latent":
        raise ValueError("shortcut/student rerank backends are only supported with --planner_variant coarse_to_fine_latent")
    print(f"Using seed={args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    evaluator = WM_Planning_Evaluator(args)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    gpu_id = torch.cuda.current_device()  # Or args.gpu if explicitly set
    evaluator.evaluate()
