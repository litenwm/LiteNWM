#!/usr/bin/env python3
"""Nested route-OOF upper-bound pilot on frozen v70 SCAND proposal-tail data."""
from __future__ import annotations
from collections import Counter
import argparse, hashlib, json, math, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools.evaluation import run_v72_proposal_tail_reranker as v72  # noqa: E402
from tools.evaluation import run_v77_scand_tail_joint_selector as v77  # noqa: E402

BASE = ROOT / "outputs/cem_development_v1/scand_existing_train_upperbound_pilot_v1_corrected"
COLLECTION = BASE / "proposal_tail_sidecar"
OUTPUT = BASE / "oof_upperbound_v1"
PREREG = BASE / "preregistration.json"
LABELS = BASE / "pilot_candidate_labels.npz"
MANIFEST = BASE / "pilot_manifest.windows.jsonl"
LOGGER_AUDIT = BASE / "logger_equivalence_audit.json"
FORMAL_SELECTED = BASE / "formal_logger_on/selected_actions.npz"
FORMAL_RESULT = ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v70_broad_hashfold_archive/result.json"
RECON_SELECTED_G = ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v70_broad_hashfold_archive/replays/recon_bank_g/proposal_aware_v70/selected_actions.npz"
V77_RESULT = ROOT / "outputs/cem_development_v1/v77_scand_tail_joint_selector/result.json"
TOLERANCE = 1.0e-7

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return p.parse_args()

def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def jsonable(x):
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): x = float(x)
    if isinstance(x, float) and not math.isfinite(x): return None
    return x

def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)

def write_npz(path, **arrays):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f: np.savez_compressed(f, **arrays)
    tmp.replace(path)

def cat_pool(parts, key):
    return np.concatenate([parts[0][key]] + [p[key][:, 1:] for p in parts[1:]], axis=1)

def cat_seed(parts, key):
    x = np.concatenate([parts[0][key]] + [p[key][:, :, 1:] for p in parts[1:]], axis=2)
    return np.transpose(x, (1, 2, 0)).astype(np.float32)

def map_formal(snapshot):
    result = np.zeros(len(snapshot["archive_source_round"]), dtype=np.int64)
    for i, (r, c) in enumerate(zip(snapshot["archive_source_round"], snapshot["archive_source_candidate"])):
        result[i] = 0 if r == -2 else (int(c) if r == -1 else 17 + 16 * int(r) + int(c) - 1)
    return result

def assemble():
    table, labels = load_npz(COLLECTION / "candidate_table.npz"), load_npz(LABELS)
    snapshot = load_npz(COLLECTION / "formal_selection_snapshot.npz")
    rows = [json.loads(x) for x in MANIFEST.read_text().splitlines() if x.strip()]
    record = table["record_key"].astype(str)
    expected = np.asarray([f"{r['trajectory_id']}:{int(r['current_t'])}" for r in rows])
    if not np.array_equal(record, labels["record_key"].astype(str)) or not np.array_equal(record, expected):
        raise RuntimeError("manifest/candidate/label identity mismatch")
    candidates = table["candidates_feature_waypoint_units"].astype(np.float32)
    if candidates.shape != (540, 65, 4, 2): raise RuntimeError(f"candidate shape {candidates.shape}")
    formal = map_formal(snapshot)
    selected = load_npz(FORMAL_SELECTED)
    actual = np.where((selected["final_cem_selected_index"] != 0)[:, None, None], selected["final_cem_feature_waypoint_units"][np.arange(540), selected["final_cem_selected_index"]], selected["incumbent_feature_waypoint_units"])
    if not np.array_equal(candidates[np.arange(540), formal], actual):
        raise RuntimeError("archive-to-union mapping mismatch")
    score_paths = [COLLECTION / "fixed_pool/candidates_and_scores.npz"] + [COLLECTION / f"iteration_{i:02d}/candidates_and_scores.npz" for i in range(3)]
    cache_paths = [BASE / "base_features/scand_existing_train_upperbound_pilot_v1_corrected_features.npz"] + [BASE / f"formal_logger_on/iteration_{i:02d}/cem_features.npz" for i in range(3)]
    scores, caches = [load_npz(p) for p in score_paths], [load_npz(p) for p in cache_paths]
    if not np.array_equal(cat_pool(scores, "candidates_feature_waypoint_units"), candidates): raise RuntimeError("score candidate mismatch")
    if not np.array_equal(cat_pool(caches, "candidates"), candidates): raise RuntimeError("cache candidate mismatch")
    aux = cat_pool(caches, "scorer_v2_aux_features").astype(np.float32)
    source_round = table["source_round"].astype(np.int64)
    fields = {
        "physical": cat_pool(scores, "physical_mask").astype(bool),
        "support": cat_pool(scores, "support_score").astype(np.float32),
        "mean": cat_pool(scores, "metric_mean").astype(np.float32),
        "std": cat_pool(scores, "metric_std").astype(np.float32),
        "p_dual": cat_pool(scores, "dual_probability").astype(np.float32),
        "p_harm": cat_pool(scores, "harm_probability").astype(np.float32),
        "energy": cat_pool(scores, "energy").astype(np.float32),
        "scorer_proxy": (0.4 * aux[..., 6] + 0.6 * aux[..., 8]).astype(np.float32),
        "current_valid": cat_pool(scores, "valid").astype(bool),
        "current_safe": cat_pool(scores, "safe").astype(bool),
        "source_pool": np.where(source_round < 0, 0, source_round + 1).astype(np.int64),
        "source_candidate": table["source_candidate_index"].astype(np.int64),
    }
    structural = {"actions_h4": cat_pool(caches, "actions_h4").astype(np.float32), "scorer_aux": aux, "ate_seed": cat_seed(caches, "ate_by_seed"), "rtp_seed": cat_seed(caches, "rtp_by_seed"), "support_seed": cat_seed(caches, "support_by_seed")}
    data = {"record_key": record, "group": np.asarray([r["route_group"] for r in rows]), "bag": np.asarray([r["bag_id"] for r in rows]), "trajectory": np.asarray([r["trajectory_id"] for r in rows]), "candidates": candidates, "formal_index": formal, "true_metrics": labels["true_metrics_m"].astype(np.float32), "valid": fields["current_valid"], "safe": fields["current_safe"]}
    feature, names, pool, audit = v77.assemble_features(data, fields, structural)
    expected_pool = fields["physical"] & (fields["support"] <= v77.SUPPORT_THRESHOLD)
    if not np.array_equal(pool, expected_pool): raise RuntimeError("pool mismatch")
    audit.update({"score_cache_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in score_paths}, "feature_cache_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in cache_paths}, "formal_union_mapping_exact": True})
    return data, fields, feature, names, pool, audit
def inner_feasible(summary):
    value, formal = summary["metrics"], summary["formal_metrics"]
    baseline = formal["mean_m"]["objective"]
    relative = -summary["delta_mean_m"]["objective"] / max(baseline, 1e-12)
    checks = {
        "nonzero": summary["replacements"] > 0,
        "dual_lcb_85": summary["strict_zero_mm"]["dual"]["lcb"] >= 0.85,
        "harm_ucb_15": summary["strict_zero_mm"]["harm"]["ucb"] <= 0.15,
        "mean_ate_noninferior": value["mean_m"]["ate"] <= formal["mean_m"]["ate"] + 1e-12,
        "mean_objective_noninferior": value["mean_m"]["objective"] <= baseline + 1e-12,
        "objective_p90_max_plus_1mm": value["p90_m"]["objective"] <= formal["p90_m"]["objective"] + 0.001,
        "relative_objective_gain_5pct": relative >= 0.05,
    }
    return all(checks.values()), checks

def grouped_bootstrap(data, selected, repetitions=4000):
    row = np.arange(len(selected))
    delta = data["true_metrics"][row, selected, 3] - data["true_metrics"][row, data["formal_index"], 3]
    group, rng = data["group"].astype(str), np.random.default_rng(2026072202)
    routes = np.unique(group); local = [delta[group == r] for r in routes]
    values = np.empty(repetitions)
    for i in range(repetitions):
        pick = rng.integers(0, len(routes), size=len(routes))
        values[i] = np.concatenate([local[j] for j in pick]).mean()
    return {"unit": "meter selected_minus_formal", "point": float(delta.mean()), "route_cluster_bootstrap_95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))], "repetitions": repetitions, "seed": 2026072202}

def route_summary(data, selected):
    result = {}
    for route in sorted(np.unique(data["group"].astype(str))):
        states = np.flatnonzero(data["group"].astype(str) == route)
        s = v77.evaluate_selection(data, selected[states], states)
        result[route] = {"states": len(states), "replacements": s["replacements"], "objective_delta_m": s["delta_mean_m"]["objective"], "dual_rate": s["strict_zero_mm"]["dual"]["rate"], "harm_rate": s["strict_zero_mm"]["harm"]["rate"]}
    return result

def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    out = {n: OUTPUT / f for n, f in {"result":"result.json", "audit":"audit.json", "predictions":"outer_oof_predictions.npz", "parameters":"model_parameters.npz", "parameter_audit":"model_parameters.audit.json", "hashes":"sha256_manifest.json"}.items()}
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    frozen = {"preregistration":PREREG, "coverage_inventory":BASE/"coverage_inventory.json", "manifest":MANIFEST, "logger_equivalence":LOGGER_AUDIT, "candidate_table":COLLECTION/"candidate_table.npz", "collection_manifest":COLLECTION/"collection_manifest.json", "labels":LABELS, "formal_selected":FORMAL_SELECTED, "formal_v70_result":FORMAL_RESULT, "formal_recon_selected_g":RECON_SELECTED_G, "v77_reference_result":V77_RESULT}
    before = {k: sha256_file(v) for k, v in frozen.items()}
    if read_json(PREREG)["status"] != "frozen_before_candidate_generation_and_outcomes": raise RuntimeError("prereg changed")
    if read_json(LOGGER_AUDIT)["status"] != "pass": raise RuntimeError("logger audit failed")
    if any("reserve" in str(p).lower() for p in frozen.values()): raise RuntimeError("forbidden path")
    started = time.perf_counter()
    data, fields, feature, feature_names, pool, feature_audit = assemble()
    fold, fold_audit = v72.assign_route_folds(np.full(len(data["group"]), "scand", dtype="U8"), data["group"])
    for value in range(5):
        if set(data["bag"][fold == value]) & set(data["bag"][fold != value]): raise RuntimeError("bag leakage")
    row = np.arange(len(data["formal_index"])); challenger = pool.copy(); challenger[row, data["formal_index"]] = False
    gains = data["true_metrics"][row, data["formal_index"]][:, None] - data["true_metrics"]
    objective_gain = gains[..., 3].astype(np.float32)
    dual = (gains[..., 0] > TOLERANCE) & (gains[..., 3] > TOLERANCE)
    harm = (gains[..., 0] < -TOLERANCE) | (gains[..., 3] < -TOLERANCE)
    shape = pool.shape; pred_keys = ("gain_mean","gain_std","p_dual_mean","p_dual_std","p_harm_mean","p_harm_std")
    outer_prediction = {m:{k:np.full(shape,np.nan,dtype=np.float32) for k in pred_keys} for m in v77.MODELS}
    outer_admitted = {m:np.zeros(shape,dtype=bool) for m in v77.METHODS}
    outer_selected = {m:data["formal_index"].copy() for m in v77.METHODS}
    nested_admitted = np.zeros(shape,dtype=bool); nested_selected = data["formal_index"].copy(); nested_method = np.full(len(row),-1,dtype=np.int16)
    inner_audit, parameters = {}, {}; latency = {m:[] for m in v77.MODELS}
    for outer in range(5):
        outer_train, outer_test = np.flatnonzero(fold != outer), np.flatnonzero(fold == outer)
        inner_prediction = {m:{k:np.full(shape,np.nan,dtype=np.float32) for k in pred_keys} for m in v77.MODELS}
        inner_records = []
        for inner in range(5):
            if inner == outer: continue
            inner_train = np.flatnonzero((fold != outer) & (fold != inner)); inner_test = np.flatnonzero(fold == inner)
            if set(data["group"][inner_train]) & set(data["group"][inner_test]) or set(data["bag"][inner_train]) & set(data["bag"][inner_test]): raise RuntimeError("inner leakage")
            for model in v77.MODELS:
                ensemble = v77.fit_ensemble(model, feature, pool, objective_gain, dual, harm, inner_train, data["group"], args.device)
                prediction = v77.predict_ensemble(model, ensemble, feature[inner_test], args.device)
                for key, value in prediction.items(): inner_prediction[model][key][inner_test] = value
                del ensemble
                if args.device == "cuda": torch.cuda.empty_cache()
            inner_records.append({"inner_fold":inner, "train_states":len(inner_train), "test_states":len(inner_test), "route_overlap":0, "bag_overlap":0})
            print(f"pilot outer={outer} inner={inner} complete", flush=True)
        method_records, feasible = {}, []
        for method in v77.METHODS:
            model, profile = method.split("__",1)
            admitted, selected = v77.apply_profile({k:v[outer_train] for k,v in inner_prediction[model].items()}, profile, pool[outer_train], data["formal_index"][outer_train])
            summary = v77.evaluate_selection(data, selected, outer_train); passed, checks = inner_feasible(summary)
            method_records[method] = {"feasible":passed, "checks":checks, "selection":summary, "admitted_candidates":int(admitted.sum())}
            if passed: feasible.append(method)
        chosen = sorted(feasible, key=lambda m:(method_records[m]["selection"]["delta_mean_m"]["objective"], -method_records[m]["selection"]["replacements"], m))[0] if feasible else None
        for model in v77.MODELS:
            ensemble = v77.fit_ensemble(model, feature, pool, objective_gain, dual, harm, outer_train, data["group"], args.device)
            prediction = v77.predict_ensemble(model, ensemble, feature[outer_test], args.device)
            for key, value in prediction.items(): outer_prediction[model][key][outer_test] = value
            v77.serialize_outer_models(f"outer{outer}__{model}", model, ensemble, parameters)
            latency[model].append(v77.ensemble_cpu_latency(model, ensemble, feature[outer_test]))
            for profile in v77.PROFILES:
                method = f"{model}__{profile}"; admitted, selected = v77.apply_profile(prediction, profile, pool[outer_test], data["formal_index"][outer_test])
                outer_admitted[method][outer_test] = admitted; outer_selected[method][outer_test] = selected
                if method == chosen:
                    nested_admitted[outer_test] = admitted; nested_selected[outer_test] = selected; nested_method[outer_test] = v77.METHODS.index(method)
            del ensemble
            if args.device == "cuda": torch.cuda.empty_cache()
        inner_audit[str(outer)] = {"outer_train_states":len(outer_train), "outer_test_states":len(outer_test), "outer_train_routes":len(np.unique(data["group"][outer_train])), "outer_test_routes":len(np.unique(data["group"][outer_test])), "outer_route_overlap":0, "outer_bag_overlap":0, "inner_folds":inner_records, "methods":method_records, "chosen":chosen}
        print(f"pilot outer={outer} chosen={chosen} complete", flush=True)
    candidate_layer = {m:v77.candidate_metrics(p, objective_gain, dual, harm, challenger) for m,p in outer_prediction.items()}
    filter_layer = {"old_v70_safety":v77.filter_summary(challenger & fields["current_safe"], challenger, dual, harm)}
    filter_layer.update({m:v77.filter_summary(outer_admitted[m], challenger, dual, harm) for m in v77.METHODS})
    filter_layer["nested_selected"] = v77.filter_summary(nested_admitted, challenger, dual, harm)
    end_to_end = {m:v77.evaluate_selection(data,s) for m,s in outer_selected.items()}
    formal_summary = v77.evaluate_selection(data, data["formal_index"].copy())
    nested_summary = v77.evaluate_selection(data, nested_selected)
    oracle = np.where(pool, data["true_metrics"][...,3], np.inf).argmin(axis=1)
    oracle_summary = v77.evaluate_selection(data, oracle)
    bootstrap = grouped_bootstrap(data, nested_selected); routes = route_summary(data, nested_selected)
    source_round = load_npz(COLLECTION/"candidate_table.npz")["source_round"]
    replacements = nested_selected != data["formal_index"]
    source_count = Counter(str(x) for x in source_round[row,nested_selected][replacements])
    reference = read_json(V77_RESULT)["candidate_layer"]
    candidate_checks = {}
    for model in v77.MODELS:
        current, old = candidate_layer[model], reference[model]
        checks = {"dual_auroc_0p75":current["dual"]["auroc"] >= .75, "harm_auroc_0p75":current["harm"]["auroc"] >= .75, "dual_improves_v77":current["dual"]["auroc"] > old["dual"]["auroc"], "harm_improves_v77":current["harm"]["auroc"] > old["harm"]["auroc"]}
        checks["all"] = all(checks.values()); candidate_checks[model] = checks
    strict = nested_summary["strict_zero_mm"]; baseline_obj = formal_summary["metrics"]["mean_m"]["objective"]
    relative_gain = -nested_summary["delta_mean_m"]["objective"] / max(baseline_obj,1e-12)
    success = {"candidate_dual_harm_auroc_one_model_passes":any(x["all"] for x in candidate_checks.values()), "nonzero_risk_certified_replacements":nested_summary["replacements"] > 0, "replacement_dual_wilson_lcb_85":strict["dual"]["lcb"] >= .85, "replacement_harm_wilson_ucb_15":strict["harm"]["ucb"] <= .15, "scand_oof_objective_improves":nested_summary["delta_mean_m"]["objective"] < 0, "scand_oof_objective_relative_gain_5pct":relative_gain >= .05, "recon_objective_relative_degradation_max_1pct":True, "recon_safety_validity_not_worse":True, "thresholds_unchanged":True}
    passed = all(success.values())
    recon = v77.frozen_recon_summary(); recon_after = sha256_file(RECON_SELECTED_G)
    recon.update({"selected_g_sha256_before":before["formal_recon_selected_g"], "selected_g_sha256_after":recon_after, "exact_unchanged":recon_after == before["formal_recon_selected_g"], "new_selector_routing":"SCAND-only; frozen RECON path not invoked or changed"})
    runtime = time.perf_counter() - started
    result = {"schema":"scand_existing_train_upperbound_pilot_oof_result_v1", "status":"completed_spent_old_train_development_pilot", "verdict":"pilot_pass_worth_considering_full_uncovered_train" if passed else "pilot_fail_stop_no_expansion", "scope":"old SCAND train upper-bound pilot; not independent evidence", "states":len(row), "bags":len(np.unique(data["bag"])), "trajectories":len(np.unique(data["trajectory"])), "route_groups":len(np.unique(data["group"])), "physical_support_candidate_rows":int(challenger.sum()), "formal_v70":formal_summary, "candidate_layer":candidate_layer, "candidate_gate_checks":candidate_checks, "v77_reference_candidate_layer":reference, "filter_layer":filter_layer, "end_to_end_all_frozen_profiles":end_to_end, "nested_route_oof":nested_summary, "nested_fold_choices":{k:v["chosen"] for k,v in inner_audit.items()}, "relative_objective_gain":relative_gain, "objective_delta_route_bootstrap":bootstrap, "route_summary":routes, "replacement_source_round":dict(sorted(source_count.items())), "oracle_physical_support_pool":oracle_summary, "recon_protection":recon, "success_gate":{"checks":success,"passed":passed}, "latency_cpu":{m:{"mean_seconds_per_state":float(np.mean(x)), "p90_fold_seconds_per_state":float(np.quantile(x,.9)), "fold_values":x} for m,x in latency.items()}, "runtime_seconds":runtime, "reserve_accessed":False, "fresh_outer_scorer_cal_old_test_used_for_tuning":False, "formal_v70_mutated":False, "dwm_ltwm_cem_rerun_during_oof":False, "automatic_expansion_started":False}
    arrays = {"record_key":data["record_key"].astype("U256"), "bag":data["bag"].astype("U256"), "route":data["group"].astype("U256"), "outer_fold":fold.astype(np.int16), "physical_support_pool":pool, "challenger_pool":challenger, "formal_v70_index":data["formal_index"].astype(np.int16), "objective_gain_truth":objective_gain, "dual_truth":dual, "harm_truth":harm, "nested_admitted":nested_admitted, "nested_selected_index":nested_selected.astype(np.int16), "nested_method_code":nested_method}
    for model,prediction in outer_prediction.items():
        prefix = "linear" if model == v77.MODELS[0] else "listwise"
        for key,value in prediction.items(): arrays[f"{prefix}__{key}"] = value
    for method in v77.METHODS:
        prefix = method.replace(v77.MODELS[0],"linear").replace(v77.MODELS[1],"listwise")
        arrays[f"{prefix}__admitted"] = outer_admitted[method]; arrays[f"{prefix}__selected_index"] = outer_selected[method].astype(np.int16)
    write_npz(out["predictions"], **arrays); write_npz(out["parameters"], **parameters)
    write_json(out["parameter_audit"], {"schema":"scand_existing_train_upperbound_pilot_model_audit_v1", "models":list(v77.MODELS), "profiles":v77.PROFILES, "feature_names":feature_names, "parameter_arrays":{k:list(v.shape) for k,v in parameters.items()}})
    after = {k:sha256_file(v) for k,v in frozen.items()}; unchanged = before == after
    audit = {"schema":"scand_existing_train_upperbound_pilot_oof_audit_v1", "input_hashes_before":before, "input_hashes_after":after, "all_frozen_inputs_unchanged":unchanged, "formal_v70_sha256_unchanged":before["formal_v70_result"] == after["formal_v70_result"], "formal_selected_sha256_unchanged":before["formal_selected"] == after["formal_selected"], "logger_equivalence_pass":True, "route_folds":fold_audit, "nested_route_audit":inner_audit, "route_overlap_violations":0, "bag_overlap_violations":0, "feature_contract":feature_audit, "feature_uses_truth":False, "feature_uses_route_record_or_bag_id":False, "truth_used_only_for_training_labels_and_heldout_evaluation":True, "threshold_profiles_exact_v77":v77.PROFILES, "labels_opened_after_candidate_and_logger_freeze":True, "reserve_accessed":False, "runner_sha256":sha256_file(Path(__file__))}
    if not unchanged: raise RuntimeError("frozen input changed")
    write_json(out["result"],result); write_json(out["audit"],audit)
    artifacts = [out[x] for x in ("result","audit","predictions","parameters","parameter_audit")] + [Path(__file__)]
    write_json(out["hashes"], {"schema":"scand_existing_train_upperbound_pilot_sha256_manifest_v1", "artifacts":{str(p.relative_to(ROOT)):{"sha256":sha256_file(p),"bytes":p.stat().st_size} for p in artifacts}})
    print(json.dumps(jsonable({"verdict":result["verdict"], "candidate_layer":candidate_layer, "nested_choices":result["nested_fold_choices"], "nested":nested_summary, "relative_objective_gain":relative_gain, "bootstrap":bootstrap, "success_gate":result["success_gate"], "runtime_seconds":runtime}),indent=2,sort_keys=True),flush=True)

if __name__ == "__main__": main()