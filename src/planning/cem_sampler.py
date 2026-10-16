"""Multi-modal, archive-preserving CEM for H4 action trajectory refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from scoring.trajectory_quality_scorer_v2 import TrajectoryQualityEnsemble


HORIZON = 4
ACTION_DIM = 3


@dataclass(frozen=True)
class CEMConfig:
    modes: int = 4
    population_size: int = 64
    elites_per_mode: int = 4
    iterations: int = 4
    mean_update_rate: float = 0.50
    std_update_rate: float = 0.50
    std_floor: float = 0.02
    std_ceiling: float = 0.50
    temporal_correlation: float = 0.60
    kmedoids_iterations: int = 8
    seed: int = 1701

    @property
    def samples_per_mode(self) -> int:
        return self.population_size // self.modes

    @property
    def total_evaluation_budget(self) -> int:
        return self.population_size * self.iterations

    def validate(self) -> None:
        if self.modes <= 0 or self.population_size <= 0:
            raise ValueError("modes and population_size must be positive")
        if self.population_size % self.modes:
            raise ValueError("population_size must be divisible by modes")
        if not 0 < self.elites_per_mode <= self.samples_per_mode:
            raise ValueError("elites_per_mode must be in [1,samples_per_mode]")
        if self.iterations <= 0 or self.kmedoids_iterations <= 0:
            raise ValueError("iteration counts must be positive")
        if not 0 < self.mean_update_rate <= 1:
            raise ValueError("mean_update_rate must be in (0,1]")
        if not 0 < self.std_update_rate <= 1:
            raise ValueError("std_update_rate must be in (0,1]")
        if not 0 < self.std_floor <= self.std_ceiling:
            raise ValueError("std floor/ceiling are invalid")
        if not 0 <= self.temporal_correlation < 1:
            raise ValueError("temporal_correlation must be in [0,1)")


@dataclass(frozen=True)
class CEMState:
    means: torch.Tensor
    stds: torch.Tensor
    iteration: int


@dataclass(frozen=True)
class ConservativeJointCost:
    cost: torch.Tensor
    utility: torch.Tensor
    lcb_ate_gain: torch.Tensor
    lcb_rtp_gain: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class CEMResult:
    selected_actions: torch.Tensor
    selected_costs: torch.Tensor
    selected_archive_indices: torch.Tensor
    archive_actions: torch.Tensor
    archive_costs: torch.Tensor
    archive_valid: torch.Tensor
    archive_is_parent: torch.Tensor
    final_state: CEMState
    generated_evaluations: int


CostFunction = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
ConstraintFunction = Callable[[torch.Tensor], torch.Tensor]


def conservative_joint_cost(
    candidate: TrajectoryQualityEnsemble,
    reference: TrajectoryQualityEnsemble,
    *,
    ate_scale: float,
    rtp_scale: float,
    uncertainty_beta: float = 1.0,
    valid_mask: torch.Tensor | None = None,
    support_mask: torch.Tensor | None = None,
) -> ConservativeJointCost:
    """Convert two absolute metrics into a conservative Pareto utility.

    Positive utility means the candidate conservatively improves both ATE and
    RTP relative to the supplied controller reference.  Lower cost is better.
    """
    if ate_scale <= 0 or rtp_scale <= 0:
        raise ValueError("metric scales must be positive")
    if uncertainty_beta < 0:
        raise ValueError("uncertainty_beta must be non-negative")
    shape = candidate.ate_mean.shape
    tensors = (
        candidate.ate_std,
        candidate.rtp_mean,
        candidate.rtp_std,
        reference.ate_mean,
        reference.ate_std,
        reference.rtp_mean,
        reference.rtp_std,
    )
    if any(tensor.shape != shape for tensor in tensors):
        raise ValueError("candidate/reference ensemble shapes differ")
    ate_uncertainty = torch.sqrt(
        candidate.ate_std.square() + reference.ate_std.square()
    )
    rtp_uncertainty = torch.sqrt(
        candidate.rtp_std.square() + reference.rtp_std.square()
    )
    lcb_ate = (
        reference.ate_mean
        - candidate.ate_mean
        - uncertainty_beta * ate_uncertainty
    )
    lcb_rtp = (
        candidate.rtp_mean
        - reference.rtp_mean
        - uncertainty_beta * rtp_uncertainty
    )
    utility = torch.minimum(lcb_ate / ate_scale, lcb_rtp / rtp_scale)
    valid = torch.isfinite(utility)
    if valid_mask is not None:
        if valid_mask.shape != shape or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool with candidate shape")
        valid &= valid_mask
    if support_mask is not None:
        if support_mask.shape != shape or support_mask.dtype != torch.bool:
            raise ValueError("support_mask must be bool with candidate shape")
        valid &= support_mask
    cost = torch.where(valid, -utility, torch.full_like(utility, torch.inf))
    return ConservativeJointCost(
        cost=cost,
        utility=utility,
        lcb_ate_gain=lcb_ate,
        lcb_rtp_gain=lcb_rtp,
        valid=valid,
    )


def paired_conservative_joint_cost(
    ate_by_seed: torch.Tensor,
    rtp_by_seed: torch.Tensor,
    reference_ate_by_seed: torch.Tensor,
    reference_rtp_by_seed: torch.Tensor,
    *,
    ate_scale: float,
    rtp_scale: float,
    uncertainty_beta: float = 1.0,
    valid_mask: torch.Tensor | None = None,
) -> ConservativeJointCost:
    """Conservative cost from paired seed-wise candidate/reference gains.

    Pairing preserves covariance between each scorer seed and its reference.
    In particular, the reference compared with itself has exactly zero gain and
    zero uncertainty instead of a spurious independent-uncertainty penalty.
    """
    if ate_by_seed.ndim < 2 or rtp_by_seed.shape != ate_by_seed.shape:
        raise ValueError("expected matching [S,...] ATE/RTP tensors")
    if reference_ate_by_seed.shape != reference_rtp_by_seed.shape:
        raise ValueError("reference ATE/RTP shapes differ")
    if reference_ate_by_seed.shape[0] != ate_by_seed.shape[0]:
        raise ValueError("candidate/reference seed counts differ")
    try:
        ate_gain = reference_ate_by_seed - ate_by_seed
        rtp_gain = rtp_by_seed - reference_rtp_by_seed
    except RuntimeError as error:
        raise ValueError("candidate/reference shapes are not broadcastable") from error
    if ate_scale <= 0 or rtp_scale <= 0:
        raise ValueError("metric scales must be positive")
    if uncertainty_beta < 0:
        raise ValueError("uncertainty_beta must be non-negative")
    lcb_ate = ate_gain.mean(dim=0) - uncertainty_beta * ate_gain.std(
        dim=0, unbiased=False
    )
    lcb_rtp = rtp_gain.mean(dim=0) - uncertainty_beta * rtp_gain.std(
        dim=0, unbiased=False
    )
    utility = torch.minimum(lcb_ate / ate_scale, lcb_rtp / rtp_scale)
    valid = torch.isfinite(utility)
    if valid_mask is not None:
        if valid_mask.shape != utility.shape or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool with candidate shape")
        valid &= valid_mask
    cost = torch.where(valid, -utility, torch.full_like(utility, torch.inf))
    return ConservativeJointCost(
        cost=cost,
        utility=utility,
        lcb_ate_gain=lcb_ate,
        lcb_rtp_gain=lcb_rtp,
        valid=valid,
    )


def conservative_replacement_mask(
    cost: ConservativeJointCost,
    *,
    ate_margin: float = 0.0,
    rtp_margin: float = 0.0,
) -> torch.Tensor:
    """Allow replacement only when both conservative metric gains pass."""
    if ate_margin < 0 or rtp_margin < 0:
        raise ValueError("replacement margins must be non-negative")
    return (
        cost.valid
        & (cost.lcb_ate_gain > ate_margin)
        & (cost.lcb_rtp_gain > rtp_margin)
    )


def _kmedoids_single(
    actions: torch.Tensor,
    modes: int,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Small deterministic PAM-style k-medoids for one K16 parent pool."""
    count = actions.shape[0]
    if modes > count:
        raise ValueError("number of modes exceeds parent count")
    flat = actions.flatten(1)
    distance = torch.cdist(flat, flat)
    medoids = [int(distance.sum(dim=1).argmin().item())]
    while len(medoids) < modes:
        nearest = distance[:, medoids].amin(dim=1)
        nearest[torch.as_tensor(medoids, device=actions.device)] = -1
        medoids.append(int(nearest.argmax().item()))
    medoid_tensor = torch.as_tensor(medoids, device=actions.device)

    for _ in range(iterations):
        assignment = distance[:, medoid_tensor].argmin(dim=1)
        updated = medoid_tensor.clone()
        for mode in range(modes):
            members = torch.nonzero(assignment == mode, as_tuple=False).flatten()
            if members.numel() == 0:
                continue
            within = distance[members][:, members].sum(dim=1)
            updated[mode] = members[within.argmin()]
        if torch.equal(updated, medoid_tensor):
            break
        medoid_tensor = updated
    assignment = distance[:, medoid_tensor].argmin(dim=1)
    return medoid_tensor, assignment


class MultiModalCEM:
    """Independent per-mode CEM that never averages distinct action modes."""

    def __init__(self, config: CEMConfig | None = None) -> None:
        self.config = config or CEMConfig()
        self.config.validate()

    def initialize(self, parent_actions: torch.Tensor) -> CEMState:
        if parent_actions.ndim != 4 or tuple(parent_actions.shape[-2:]) != (
            HORIZON,
            ACTION_DIM,
        ):
            raise ValueError("expected parent_actions [B,K,4,3]")
        if parent_actions.shape[1] < self.config.modes:
            raise ValueError("parent pool is smaller than the number of modes")
        if not torch.isfinite(parent_actions).all():
            raise ValueError("parent_actions contains NaN or Inf")
        means = []
        stds = []
        for batch_index in range(parent_actions.shape[0]):
            parents = parent_actions[batch_index]
            medoids, assignment = _kmedoids_single(
                parents,
                self.config.modes,
                self.config.kmedoids_iterations,
            )
            global_std = parents.std(dim=0, unbiased=False).clamp_min(
                self.config.std_floor
            )
            mode_stds = []
            for mode in range(self.config.modes):
                members = parents[assignment == mode]
                if members.shape[0] >= 2:
                    value = members.std(dim=0, unbiased=False)
                else:
                    value = global_std
                mode_stds.append(
                    value.clamp(
                        min=self.config.std_floor,
                        max=self.config.std_ceiling,
                    )
                )
            means.append(parents[medoids])
            stds.append(torch.stack(mode_stds))
        return CEMState(
            means=torch.stack(means),
            stds=torch.stack(stds),
            iteration=0,
        )

    def sample(self, state: CEMState) -> torch.Tensor:
        expected_tail = (self.config.modes, HORIZON, ACTION_DIM)
        if tuple(state.means.shape[1:]) != expected_tail:
            raise ValueError("CEM state mean shape is invalid")
        if state.stds.shape != state.means.shape:
            raise ValueError("CEM state std shape differs from means")
        generator = torch.Generator(device=state.means.device)
        generator.manual_seed(self.config.seed + state.iteration)
        shape = (
            state.means.shape[0],
            self.config.modes,
            self.config.samples_per_mode,
            HORIZON,
            ACTION_DIM,
        )
        white = torch.randn(
            shape,
            generator=generator,
            device=state.means.device,
            dtype=state.means.dtype,
        )
        rho = self.config.temporal_correlation
        correlated = white.clone()
        innovation = (1.0 - rho * rho) ** 0.5
        for horizon_index in range(1, HORIZON):
            correlated[..., horizon_index, :] = (
                rho * correlated[..., horizon_index - 1, :]
                + innovation * white[..., horizon_index, :]
            )
        # Deliberately no clipping: out-of-support samples must be rejected.
        return (
            state.means[:, :, None]
            + state.stds[:, :, None] * correlated
        )

    def update(
        self,
        state: CEMState,
        samples: torch.Tensor,
        costs: torch.Tensor,
        valid: torch.Tensor,
    ) -> CEMState:
        expected = (
            state.means.shape[0],
            self.config.modes,
            self.config.samples_per_mode,
            HORIZON,
            ACTION_DIM,
        )
        if tuple(samples.shape) != expected:
            raise ValueError("sample shape is invalid")
        if costs.shape != expected[:3] or valid.shape != expected[:3]:
            raise ValueError("cost/valid shape is invalid")
        if valid.dtype != torch.bool:
            raise ValueError("valid must be boolean")
        new_means = state.means.clone()
        new_stds = state.stds.clone()
        for batch_index in range(samples.shape[0]):
            for mode in range(self.config.modes):
                usable = valid[batch_index, mode] & torch.isfinite(
                    costs[batch_index, mode]
                )
                indices = torch.nonzero(usable, as_tuple=False).flatten()
                if indices.numel() == 0:
                    continue
                usable_costs = costs[batch_index, mode, indices]
                elite_count = min(self.config.elites_per_mode, indices.numel())
                elite_indices = indices[
                    torch.topk(
                        usable_costs,
                        k=elite_count,
                        largest=False,
                        sorted=True,
                    ).indices
                ]
                elites = samples[batch_index, mode, elite_indices]
                elite_mean = elites.mean(dim=0)
                elite_std = elites.std(dim=0, unbiased=False)
                new_means[batch_index, mode] = torch.lerp(
                    state.means[batch_index, mode],
                    elite_mean,
                    self.config.mean_update_rate,
                )
                updated_std = torch.lerp(
                    state.stds[batch_index, mode],
                    elite_std,
                    self.config.std_update_rate,
                )
                new_stds[batch_index, mode] = updated_std.clamp(
                    min=self.config.std_floor,
                    max=self.config.std_ceiling,
                )
        return CEMState(
            means=new_means,
            stds=new_stds,
            iteration=state.iteration + 1,
        )

    def refine(
        self,
        parent_actions: torch.Tensor,
        cost_function: CostFunction,
        *,
        constraint_function: ConstraintFunction | None = None,
        fixed_winner_indices: torch.Tensor | None = None,
    ) -> CEMResult:
        """Run CEM while preserving the complete fixed pool and every sample."""
        state = self.initialize(parent_actions)
        batch, parent_count = parent_actions.shape[:2]
        if fixed_winner_indices is None:
            fixed_winner_indices = torch.zeros(
                batch,
                dtype=torch.long,
                device=parent_actions.device,
            )
        if fixed_winner_indices.shape != (batch,):
            raise ValueError("fixed_winner_indices must have shape [B]")
        if bool(
            ((fixed_winner_indices < 0) | (fixed_winner_indices >= parent_count)).any()
        ):
            raise ValueError("fixed_winner_indices is out of range")

        parent_costs, parent_valid = cost_function(parent_actions)
        if parent_costs.shape != (batch, parent_count):
            raise ValueError("cost_function returned wrong parent cost shape")
        if parent_valid.shape != parent_costs.shape or parent_valid.dtype != torch.bool:
            raise ValueError("cost_function returned invalid parent mask")
        archive_actions = [parent_actions]
        archive_costs = [parent_costs]
        archive_valid = [parent_valid & torch.isfinite(parent_costs)]
        archive_is_parent = [
            torch.ones_like(parent_valid, dtype=torch.bool)
        ]

        for _ in range(self.config.iterations):
            samples = self.sample(state)
            flat = samples.reshape(batch, self.config.population_size, HORIZON, ACTION_DIM)
            costs, valid = cost_function(flat)
            if costs.shape != (batch, self.config.population_size):
                raise ValueError("cost_function returned wrong generated cost shape")
            if valid.shape != costs.shape or valid.dtype != torch.bool:
                raise ValueError("cost_function returned invalid generated mask")
            if constraint_function is not None:
                constraint_valid = constraint_function(flat)
                if (
                    constraint_valid.shape != costs.shape
                    or constraint_valid.dtype != torch.bool
                ):
                    raise ValueError("constraint_function must return bool [B,K]")
                valid &= constraint_valid
            valid &= torch.isfinite(costs)
            costs = torch.where(valid, costs, torch.full_like(costs, torch.inf))
            state = self.update(
                state,
                samples,
                costs.reshape(
                    batch,
                    self.config.modes,
                    self.config.samples_per_mode,
                ),
                valid.reshape(
                    batch,
                    self.config.modes,
                    self.config.samples_per_mode,
                ),
            )
            archive_actions.append(flat)
            archive_costs.append(costs)
            archive_valid.append(valid)
            archive_is_parent.append(torch.zeros_like(valid))

        all_actions = torch.cat(archive_actions, dim=1)
        all_costs = torch.cat(archive_costs, dim=1)
        all_valid = torch.cat(archive_valid, dim=1)
        all_is_parent = torch.cat(archive_is_parent, dim=1)
        masked_costs = torch.where(
            all_valid,
            all_costs,
            torch.full_like(all_costs, torch.inf),
        )
        selected_indices = masked_costs.argmin(dim=1)
        no_valid = ~all_valid.any(dim=1)
        selected_indices = torch.where(
            no_valid,
            fixed_winner_indices,
            selected_indices,
        )
        row = torch.arange(batch, device=parent_actions.device)
        return CEMResult(
            selected_actions=all_actions[row, selected_indices],
            selected_costs=all_costs[row, selected_indices],
            selected_archive_indices=selected_indices,
            archive_actions=all_actions,
            archive_costs=all_costs,
            archive_valid=all_valid,
            archive_is_parent=all_is_parent,
            final_state=state,
            generated_evaluations=self.config.total_evaluation_budget,
        )


__all__ = [
    "CEMConfig",
    "CEMResult",
    "CEMState",
    "ConservativeJointCost",
    "MultiModalCEM",
    "conservative_joint_cost",
    "paired_conservative_joint_cost",
    "conservative_replacement_mask",
]
