"""Layer 5 Full-Scale Simulation Tests.

Usage:
    python run_l5_simulations.py              # Full suite (resumes from checkpoint)
    python run_l5_simulations.py --test T5.1  # Single test
    python run_l5_simulations.py --quick      # Smoke test (~2 min)
    python run_l5_simulations.py --force      # Ignore checkpoints, re-run all
    python run_l5_simulations.py --clean      # Clear old results, then run full

Checkpoint: each test saves results on completion. Re-running skips completed tests.
"""

import argparse
import json
import shutil
import time
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

import numpy as np

# Default: use max 25% of CPU cores to avoid saturating the system
DEFAULT_WORKERS = min(6, max(1, cpu_count() // 4))
ACTIVE_WORKERS = DEFAULT_WORKERS

from sim_utils import (
    RESULTS_DIR, get_all_patient_names, simulate_patient, simulate_closed_loop,
    is_completed, save_results, PatientAction,
)
from aegis_testing.aegis_core.safety_supervisor import (
    SafetyAction, SafetySupervisor, SafetyTier,
)

# ── Scenario constants ───────────────────────────────────────
MEALS = {
    "standard":  [(420, 45), (720, 70), (1080, 80), (1320, 15)],
    "high_carb": [(420, 80), (720, 120), (1080, 130), (1320, 30)],
    "low_carb":  [(420, 20), (720, 30), (1080, 30)],
    "skipped":   [(720, 70), (1080, 80)],
    "irregular": [(540, 60), (840, 40), (1260, 100)],
}
ACTIVITIES = {"sedentary": 1.0, "mild": 1.15, "intense": 1.50}
NOISES = {"ideal": 0.0, "low": 5.0, "typical": 10.0, "high": 20.0}


# ── Pretty output helpers ────────────────────────────────────

def header(test_id, name, runs, est_time=""):
    """Print a formatted test header."""
    est = f" | Est: {est_time}" if est_time else ""
    print(f"""
┌─────────────────────────────────────────────────────────┐
│  {test_id} — {name:<46s}│
│  Runs: {runs:<10}{est:>40s} │
└─────────────────────────────────────────────────────────┘""")


def progress(current, total, start_time):
    """Print a clean progress bar."""
    pct = current / max(1, total) * 100
    elapsed = time.time() - start_time
    eta = elapsed / max(1, current) * (total - current) if current > 0 else 0
    bar_len = 30
    filled = int(bar_len * current / max(1, total))
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {current:>6}/{total} ({pct:5.1f}%)  "
          f"Elapsed: {timedelta(seconds=int(elapsed))}  "
          f"ETA: {timedelta(seconds=int(eta))}  ", end="", flush=True)
    if current >= total:
        print()


def result_line(label, value, unit="", pass_crit="", passed=None):
    """Print a formatted result line."""
    status = ""
    if passed is not None:
        status = " ✅" if passed else " ❌"
    crit = f"  (need: {pass_crit})" if pass_crit else ""
    print(f"  │ {label:<30s} {value:>10s} {unit:<6s}{crit}{status}")


def verdict(passed):
    """Print pass/fail verdict."""
    if passed:
        print("  ╰→ ✅ PASSED")
    else:
        print("  ╰→ ❌ FAILED")


# ── T5.1: Tier Priority ──────────────────────────────────────

def run_t5_1(quick=False):
    n_seeds = 5 if quick else 50
    sup = SafetySupervisor(hysteresis_hold_steps=1)

    cases = []
    for g in [20, 30, 40, 50, 53]:
        for d in [0, 2, 5, 10, 15, 20]: cases.append((g, d, "suspend"))
    for g in [54, 58, 62, 66, 69]:
        for d in [1, 5, 10]:             cases.append((g, d, "block"))
    for g in [100, 150, 200]:
        for d in [15.1, 16, 20, 25]:     cases.append((g, d, "reduce"))  # 15.1 is boundary
    for g in [120, 150, 170]:
        for d in [0, 0.5, 15.0]:         cases.append((g, d, "allow"))   # 15.0 is NOT reduce

    total = len(cases) * n_seeds
    header("T5.1", "Exhaustive Tier Priority", total, "~10s")

    results, correct = [], 0
    t0 = time.time()

    for i, (g, d, expected) in enumerate(cases):
        for seed in range(n_seeds):
            sup.reset_hysteresis()
            traj = np.full(36, float(g)) + np.random.RandomState(seed).randn(36) * 3
            r = sup.verify(g, traj, d, prediction_std=15.0, n_observations=200)
            t1 = r.tier_results[SafetyTier.TIER_1_REFLEX]
            ok = {"suspend": t1.action == SafetyAction.SUSPEND,
                  "block":   t1.action == SafetyAction.BLOCK,
                  "reduce":  t1.action == SafetyAction.REDUCE,
                  "allow":   t1.action == SafetyAction.ALLOW}[expected]
            if ok: correct += 1
            results.append({"g": g, "d": d, "expected": expected, "ok": ok})
        progress((i + 1) * n_seeds, total, t0)

    acc = correct / (len(cases) * n_seeds) * 100
    elapsed = time.time() - t0
    result_line("Accuracy", f"{acc:.1f}", "%", "100%", acc == 100)
    result_line("Time", f"{elapsed:.1f}", "sec")
    passed = acc == 100.0
    verdict(passed)
    save_results("T5.1", results, {"test_id": "T5.1", "accuracy": acc,
                 "pass": passed, "seconds": elapsed, "runs": total})
    return passed, elapsed


# ── T5.2: Reflex Boundary ────────────────────────────────────

def run_t5_2(quick=False):
    sup = SafetySupervisor(hysteresis_hold_steps=1)
    glucose_vals = list(range(20, 301)) + [
        53.5, 53.9, 53.99, 54.0, 54.01, 54.1,
        69.5, 69.9, 69.99, 70.0, 70.01, 70.1,
        179.9, 180.0, 180.1, 249.9, 250.0, 250.1,
    ]
    doses = [0, 1, 5, 10, 15, 20, 25]
    total = len(glucose_vals) * len(doses)
    header("T5.2", "Reflex Boundary (int + float edges)", total, "~5s")

    results, correct = [], 0
    t0 = time.time()

    for i, g in enumerate(glucose_vals):
        for d in doses:
            sup.reset_hysteresis()
            r = sup.verify(float(g), np.full(36, float(g)), float(d))
            t1 = r.tier_results[SafetyTier.TIER_1_REFLEX]
            if g < 54:                exp = SafetyAction.SUSPEND
            elif g < 70 and d > 0:    exp = SafetyAction.BLOCK
            elif d > 15:              exp = SafetyAction.REDUCE
            else:                     exp = SafetyAction.ALLOW
            ok = t1.action == exp
            if ok: correct += 1
            results.append({"g": g, "d": d, "ok": ok})
        if (i + 1) % 30 == 0: progress((i + 1) * len(doses), total, t0)
    progress(total, total, t0)

    acc = correct / total * 100
    elapsed = time.time() - t0
    result_line("Integer boundaries", f"{len(range(20,301))}", "values")
    result_line("Float boundaries", "18", "values")
    result_line("Accuracy", f"{acc:.1f}", "%", "100%", acc == 100)
    result_line("Time", f"{elapsed:.1f}", "sec")
    passed = acc == 100.0
    verdict(passed)
    save_results("T5.2", results, {"test_id": "T5.2", "accuracy": acc,
                 "pass": passed, "seconds": elapsed, "runs": total})
    return passed, elapsed


# ── T5.3: Closed-Loop STL Satisfaction ────────────────────────
#    L5 actively controls doses in a simglucose closed-loop.

def _closed_loop_worker(args):
    """Worker: closed-loop sim where L5 modifies each dose step-by-step."""
    patient, meal_name, meals, noise_std, seed = args
    t0 = time.time()
    try:
        # 24-hour sim with L5 in-the-loop
        result = simulate_closed_loop(
            patient_name=patient,
            duration_hours=24,
            meals=meals,
            noise_std=noise_std,
            seed=seed,
        )
        stl = result["stl"]
        g = result["glucose"]
        m = result["metrics"]
        elapsed = time.time() - t0
        return {
            "patient": patient, "meal": meal_name, "noise": noise_std,
            "seed": seed, "ok": True,
            "phi1": stl["no_severe_hypo"]["satisfied"],
            "phi2": stl["no_extreme_hyper"]["satisfied"],
            "phi3": stl["hypo_recovery"]["satisfied"],
            "rob": stl["no_severe_hypo"]["robustness"],
            "min_g": float(np.min(g)), "max_g": float(np.max(g)),
            "tir": m["tir"], "tbr_54": m["tbr_54"],
            "doses_blocked": int(np.sum(result["safe_doses"] < result["proposed_doses"])),
            "sim_time": elapsed,
        }
    except Exception as e:
        return {"patient": patient, "meal": meal_name, "seed": seed,
                "ok": False, "err": str(e), "sim_time": time.time() - t0}


def run_t5_3(quick=False):
    patients = get_all_patient_names()
    n_seeds = 2 if quick else 20
    if quick:
        patients = patients[:3]
        meals_cfg = dict(list(MEALS.items())[:2])
        noises = {"ideal": 0.0}
    else:
        meals_cfg = MEALS
        noises = NOISES

    # Build scenarios: patient × meal × noise × seed
    scenarios = [(p, mn, m, n, s)
                 for p in patients for mn, m in meals_cfg.items()
                 for _, n in noises.items()
                 for s in range(n_seeds)]
    total = len(scenarios)
    workers = ACTIVE_WORKERS
    header("T5.3", "Closed-Loop STL (L5 in-the-loop)", total,
           "~8-12 hrs" if not quick else "~5 min")
    print(f"  │ CLOSED-LOOP: controller proposes → L5 verifies → safe dose fed to patient")
    print(f"  │ Patients: {len(patients)} | Meals: {len(meals_cfg)} | "
          f"Noise: {len(noises)} | Seeds: {n_seeds}")
    print(f"  │ Workers: {workers}")
    print(f"  │")
    print(f"  │ {'#':>6}  {'Patient':<14} {'Meal':<10} {'φ₁':>3} {'φ₂':>3} {'φ₃':>3}  "
          f"{'Min BG':>7} {'Max BG':>7} {'TIR':>5} {'Blkd':>5}  {'Time':>5}  {'ETA':>10}")
    print(f"  │ {'─'*95}")

    t0 = time.time()
    results = []
    phi1_ok, phi2_ok, phi3_ok, n_ok, n_fail = 0, 0, 0, 0, 0

    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_closed_loop_worker, scenarios)):
            results.append(r)
            done = i + 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            eta_str = str(timedelta(seconds=int(eta)))

            if r.get("ok"):
                n_ok += 1
                if r["phi1"]: phi1_ok += 1
                if r["phi2"]: phi2_ok += 1
                if r["phi3"]: phi3_ok += 1
                p1 = "✅" if r["phi1"] else "❌"
                p2 = "✅" if r["phi2"] else "❌"
                p3 = "✅" if r["phi3"] else "❌"
                print(f"  │ {done:>6}  {r['patient']:<14} {r['meal']:<10} "
                      f"{p1:>3} {p2:>3} {p3:>3}  "
                      f"{r['min_g']:>7.1f} {r['max_g']:>7.1f} "
                      f"{r['tir']:>4.0f}% {r['doses_blocked']:>5}  "
                      f"{r['sim_time']:>4.1f}s  {eta_str:>10}")
            else:
                n_fail += 1
                print(f"  │ {done:>6}  {r['patient']:<14} {'FAILED':<10} "
                      f"{'':>3} {'':>3} {'':>3}  "
                      f"{'':>7} {'':>7} {'':>5} {'':>5}  "
                      f"{r['sim_time']:>4.1f}s  {eta_str:>10}  ⚠ {r.get('err','')[:30]}")

            if done % 20 == 0 or done == total:
                rate_1 = phi1_ok / max(1, n_ok) * 100
                rate_2 = phi2_ok / max(1, n_ok) * 100
                rate_3 = phi3_ok / max(1, n_ok) * 100
                speed = done / max(0.1, elapsed)
                print(f"  │ ── {done}/{total} ({done/total*100:.1f}%) ── "
                      f"φ₁={rate_1:.1f}% φ₂={rate_2:.1f}% φ₃={rate_3:.1f}% "
                      f"│ {speed:.1f} sim/s │ "
                      f"Elapsed: {timedelta(seconds=int(elapsed))} "
                      f"ETA: {eta_str} ──")

    elapsed = time.time() - t0
    ok = [r for r in results if r.get("ok")]
    phi1 = phi2 = phi3 = 0
    if ok:
        phi1 = sum(r["phi1"] for r in ok) / len(ok) * 100
        phi2 = sum(r["phi2"] for r in ok) / len(ok) * 100
        phi3 = sum(r["phi3"] for r in ok) / len(ok) * 100
        avg_tir = np.mean([r["tir"] for r in ok])
        avg_tbr = np.mean([r["tbr_54"] for r in ok])
        avg_blk = np.mean([r["doses_blocked"] for r in ok])
        result_line("φ₁ no severe hypo", f"{phi1:.1f}", "%", "≥99%", phi1 >= 99)
        result_line("φ₂ no extreme hyper", f"{phi2:.1f}", "%", "≥99%", phi2 >= 99)
        result_line("φ₃ hypo recovery", f"{phi3:.1f}", "%")
        result_line("Avg TIR (70-180)", f"{avg_tir:.1f}", "%")
        result_line("Avg TBR (<54)", f"{avg_tbr:.1f}", "%", "≤1%", avg_tbr <= 1)
        result_line("Avg doses blocked/day", f"{avg_blk:.0f}", "")
        result_line("Sim failures", f"{len(results)-len(ok)}", "")
        passed = phi1 >= 99.0
    else:
        print("  │ ❌ All simulations failed")
        passed = False
    result_line("Time", f"{elapsed:.0f}", "sec")
    verdict(passed)
    save_results("T5.3", results, {"test_id": "T5.3", "runs": total,
                 "phi1": phi1, "phi2": phi2, "phi3": phi3,
                 "pass": passed, "seconds": elapsed})
    return passed, elapsed


# ── T5.4: Seldonian Constraint ────────────────────────────────

def run_t5_4(quick=False):
    sizes = [10, 50, 100, 500, 1000]
    n_mc = 20 if quick else 1000
    total = len(sizes) * n_mc
    header("T5.4", "Seldonian Constraint (high-confidence)", total, "~30s")

    results = []
    t0 = time.time()
    count = 0
    per_size = {}

    for n_obs in sizes:
        violations = 0
        for mc in range(n_mc):
            rng = np.random.RandomState(mc)
            g, d, std = rng.uniform(60, 180), rng.uniform(0, 10), rng.uniform(5, 30)
            sup = SafetySupervisor(seldonian_delta=0.01, seldonian_alpha=0.05,
                                   hysteresis_hold_steps=1)
            traj = np.full(36, g) + rng.randn(36) * std * 0.3
            r = sup.verify(g, traj, d, prediction_std=std, n_observations=n_obs)
            actual = g - r.safe_dose * 25.0 + rng.randn() * std
            viol = actual < 54 and r.overall_safe
            if viol: violations += 1
            results.append({"n": n_obs, "violation": viol})
            count += 1
            if count % 200 == 0: progress(count, total, t0)
        rate = violations / n_mc * 100
        per_size[n_obs] = rate
        result_line(f"n={n_obs}", f"{rate:.2f}", "%", "≤5%", rate <= 5)
    progress(total, total, t0)

    elapsed = time.time() - t0
    passed = all(v <= 5.0 for v in per_size.values())
    result_line("Time", f"{elapsed:.1f}", "sec")
    verdict(passed)
    save_results("T5.4", results, {"test_id": "T5.4", "rates": per_size,
                 "pass": passed, "seconds": elapsed, "runs": total})
    return passed, elapsed


# ── T5.5: Cold Start ──────────────────────────────────────────

def run_t5_5(quick=False):
    n_mc = 20 if quick else 500
    total = n_mc * 35
    header("T5.5", "Cold Start Relaxation Schedule", total, "~5s")

    results = []
    t0 = time.time()
    for mc in range(n_mc):
        for day in range(35):
            sup = SafetySupervisor(cold_start_days=30)
            sup.set_day(day)
            actual = sup._get_cold_start_relaxation()
            expected = 1.0 if day >= 30 else 5.0 - 4.0 * (day / 30.0)
            results.append({"day": day, "error": abs(actual - expected)})
        if (mc + 1) % 50 == 0: progress((mc + 1) * 35, total, t0)
    progress(total, total, t0)

    max_err = max(r["error"] for r in results)
    elapsed = time.time() - t0
    passed = max_err <= 0.005
    result_line("Max deviation", f"{max_err:.6f}", "", "≤0.005", passed)
    result_line("Checkpoints", "35", "days")
    result_line("Time", f"{elapsed:.1f}", "sec")
    verdict(passed)
    save_results("T5.5", results, {"test_id": "T5.5", "max_error": max_err,
                 "pass": passed, "seconds": elapsed, "runs": total})
    return passed, elapsed


# ── T5.6: Adversarial ─────────────────────────────────────────

def run_t5_6(quick=False):
    n_per = 20 if quick else 500
    patterns = {
        "staircase":   lambda rng: (rng.choice([20, 40, 53, 54, 69, 70, 100, 250, 400]),
                                     rng.uniform(0, 25)),
        "oscillation": lambda rng: (70.0 + rng.choice([-2, -1, 0, 1, 2]), rng.uniform(0, 5)),
        "max_dose":    lambda rng: (rng.uniform(80, 200), 15.0 + rng.uniform(0.01, 10)),
        "negative_bg": lambda rng: (rng.uniform(-100, 0), rng.uniform(0, 10)),
        "extreme":     lambda rng: (rng.choice([0.001, 999, float('inf')]),
                                     rng.choice([0, 100, float('inf')])),
    }
    total = len(patterns) * n_per
    header("T5.6", "Adversarial Input Testing", total, "~5s")

    results, unsafe, crashes = [], 0, 0
    t0 = time.time()
    for name, gen in patterns.items():
        p_unsafe = 0
        for i in range(n_per):
            rng = np.random.RandomState(i)
            g, d = gen(rng)
            try:
                sup = SafetySupervisor(hysteresis_hold_steps=1)
                traj = np.full(36, max(0, g) if np.isfinite(g) else 100.0)
                r = sup.verify(g, traj, d)
                bad = np.isfinite(g) and g < 54 and r.safe_dose > 0
                if bad: unsafe += 1; p_unsafe += 1
                results.append({"p": name, "ok": not bad})
            except Exception as e:
                crashes += 1
                results.append({"p": name, "crash": str(e)})
        result_line(f"{name}", f"{p_unsafe}", "unsafe", "0", p_unsafe == 0)
    progress(total, total, t0)

    elapsed = time.time() - t0
    passed = unsafe == 0 and crashes == 0
    result_line("Total unsafe", f"{unsafe}", "", "0", unsafe == 0)
    result_line("Crashes", f"{crashes}", "", "0", crashes == 0)
    result_line("Time", f"{elapsed:.1f}", "sec")
    verdict(passed)
    save_results("T5.6", results, {"test_id": "T5.6", "unsafe": unsafe,
                 "crashes": crashes, "pass": passed, "seconds": elapsed, "runs": total})
    return passed, elapsed


# ── T5.8: Cascading Failure ───────────────────────────────────

def run_t5_8(quick=False):
    n_per = 20 if quick else 500
    modes = [("L2_high", "L2 predicts +50%"), ("L2_low", "L2 predicts -50%"),
             ("L4_dangerous", "L4 proposes danger"), ("multi", "Multiple failures")]
    total = len(modes) * n_per
    header("T5.8", "Cascading Failure Resilience", total, "~10s")

    results, violations = [], 0
    t0 = time.time()
    for mid, mname in modes:
        mv = 0
        for i in range(n_per):
            rng = np.random.RandomState(i)
            g = rng.uniform(40, 200)
            sup = SafetySupervisor(hysteresis_hold_steps=1)
            if mid == "L2_high":     traj, d = np.full(36, g*1.5), rng.uniform(0, 10)
            elif mid == "L2_low":    traj, d = np.full(36, g*0.5), rng.uniform(0, 10)
            elif mid == "L4_dangerous": traj, d = np.full(36, g), rng.uniform(15, 30)
            else:                    traj, d = np.full(36, g*rng.uniform(0.3, 2)), rng.uniform(10, 25)
            r = sup.verify(g, traj, d)
            bad = g < 54 and r.safe_dose > 0
            if bad: violations += 1; mv += 1
            results.append({"mode": mid, "g": g, "bad": bad})
        result_line(f"{mname}", f"{mv}", "violations", "0", mv == 0)

    elapsed = time.time() - t0
    passed = violations == 0
    result_line("Total violations", f"{violations}", "", "0", passed)
    result_line("Time", f"{elapsed:.1f}", "sec")
    verdict(passed)
    save_results("T5.8", results, {"test_id": "T5.8", "violations": violations,
                 "pass": passed, "seconds": elapsed, "runs": total})
    return passed, elapsed


# ── T5.9: Threshold Oscillation ───────────────────────────────

def run_t5_9(quick=False):
    thresholds = [54, 70, 180, 250]
    amps = [1, 2, 5, 10] if not quick else [1, 5]
    n_seeds = 10 if quick else 500
    total = len(thresholds) * len(amps) * n_seeds
    header("T5.9", "Threshold Oscillation / Hysteresis", total, "~2 min")

    results = []
    t0 = time.time()
    count = 0
    for th in thresholds:
        for amp in amps:
            trans_list = []
            for seed in range(n_seeds):
                sup = SafetySupervisor(hysteresis_hold_steps=3)
                rng = np.random.RandomState(seed)
                transitions, last = 0, None
                for i in range(24):
                    g = th + amp * np.sin(2 * np.pi * i / 6) + rng.randn() * 0.5
                    r = sup.verify(g, np.full(36, g), 2.0)
                    if last is not None and r.active_tier != last: transitions += 1
                    last = r.active_tier
                tph = transitions / 2.0
                trans_list.append(tph)
                results.append({"th": th, "amp": amp, "tph": tph})
                count += 1
                if count % 100 == 0: progress(count, total, t0)
            result_line(f"th={th} amp=±{amp}", f"{np.mean(trans_list):.1f}",
                       "/hr", "≤4" if amp <= 2 else "",
                       np.max(trans_list) <= 4 if amp <= 2 else None)
    progress(total, total, t0)

    small = [r["tph"] for r in results if r["amp"] <= 2]
    max_small = max(small) if small else 0
    elapsed = time.time() - t0
    passed = max_small <= 4
    result_line("Max trans/hr (amp≤2)", f"{max_small:.1f}", "/hr", "≤4", passed)
    result_line("Time", f"{elapsed:.1f}", "sec")
    verdict(passed)
    save_results("T5.9", results, {"test_id": "T5.9", "max_tph": max_small,
                 "pass": passed, "seconds": elapsed, "runs": total})
    return passed, elapsed


# ── Master runner ─────────────────────────────────────────────

ALL_TESTS = [
    ("T5.1", "Exhaustive Tier Priority",        run_t5_1,  "5,000"),
    ("T5.2", "Reflex Boundary Testing",         run_t5_2,  "2,065"),
    ("T5.3", "STL Satisfaction (simglucose)",    run_t5_3,  "90,000"),
    ("T5.4", "Seldonian Constraint",             run_t5_4,  "5,000"),
    ("T5.5", "Cold Start Relaxation",            run_t5_5,  "17,500"),
    ("T5.6", "Adversarial Input Testing",        run_t5_6,  "2,500"),
    ("T5.8", "Cascading Failure Resilience",     run_t5_8,  "2,000"),
    ("T5.9", "Threshold Oscillation/Hysteresis", run_t5_9,  "8,000"),
]


def main():
    parser = argparse.ArgumentParser(description="AEGIS L5 Simulation Tests")
    parser.add_argument("--test", help="Run single test, e.g. T5.1")
    parser.add_argument("--quick", action="store_true", help="Smoke test")
    parser.add_argument("--force", action="store_true", help="Re-run completed")
    parser.add_argument("--clean", action="store_true", help="Clear old results first")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel workers for T5.3 (default: {DEFAULT_WORKERS}, max: {cpu_count()})")
    args = parser.parse_args()

    if args.clean and RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        print("  🗑  Cleared old results")

    # Set global worker count
    global ACTIVE_WORKERS
    ACTIVE_WORKERS = min(args.workers, cpu_count())

    mode = "QUICK (smoke)" if args.quick else "FULL (publication)"
    print(f"""
╔═════════════════════════════════════════════════════════╗
║        AEGIS 3.0 — Layer 5 Safety Supervisor           ║
║        Full-Scale Simulation Test Suite                 ║
╠═════════════════════════════════════════════════════════╣
║  Mode:    {mode:<46s}║
║  Time:    {datetime.now():%Y-%m-%d %H:%M:%S}{' ':>35s}║
║  CPUs:    {cpu_count()} total, {ACTIVE_WORKERS} workers{' ':>{44-len(str(cpu_count()))-len(str(ACTIVE_WORKERS))}}║
║  Output:  {str(RESULTS_DIR):<46s}║
╚═════════════════════════════════════════════════════════╝""")

    # Show test plan
    tests = ALL_TESTS
    if args.test:
        tests = [(t, n, f, r) for t, n, f, r in ALL_TESTS if t == args.test]
        if not tests:
            print(f"  Unknown: {args.test}. Options: {[t[0] for t in ALL_TESTS]}")
            return

    print("\n  Test Plan:")
    for tid, name, _, runs in tests:
        status = "✓ done" if (not args.force and is_completed(tid)) else "pending"
        print(f"    {tid}  {name:<38s} {runs:>7s} runs  [{status}]")
    print()

    t0 = time.time()
    scoreboard = {}

    for tid, name, fn, runs in tests:
        if not args.force and is_completed(tid):
            # Load existing result
            with open(RESULTS_DIR / f"{tid}_summary.json") as f:
                old = json.load(f)
            passed = old.get("pass", False)
            secs = old.get("seconds", 0)
            scoreboard[tid] = (passed, secs, True)
            print(f"\n  ⏭  {tid} — already completed "
                  f"({'✅' if passed else '❌'}, {secs:.0f}s) — skipping")
            continue

        try:
            passed, secs = fn(quick=args.quick)
            scoreboard[tid] = (passed, secs, False)
        except Exception as e:
            print(f"\n  💥 {tid} CRASHED: {e}")
            scoreboard[tid] = (False, 0, False)

    # ── Final scoreboard ──
    total_time = time.time() - t0
    n_pass = sum(1 for p, _, _ in scoreboard.values() if p)
    n_total = len(scoreboard)

    print(f"""
╔═════════════════════════════════════════════════════════╗
║                    FINAL SCOREBOARD                    ║
╠══════╤═══════════════════════════════╤════════╤════════╣
║ Test │ Name                          │ Result │  Time  ║
╠══════╪═══════════════════════════════╪════════╪════════╣""")
    for tid, name, _, _ in tests:
        if tid in scoreboard:
            passed, secs, skipped = scoreboard[tid]
            status = "✅ PASS" if passed else "❌ FAIL"
            skip = " ⏭" if skipped else ""
            print(f"║ {tid} │ {name:<29s} │ {status} │ {secs:>5.0f}s{skip} ║")
    print(f"""╠══════╧═══════════════════════════════╧════════╧════════╣
║  Result: {n_pass}/{n_total} passed | Total: {timedelta(seconds=int(total_time))!s:<25s}║
╚═════════════════════════════════════════════════════════╝""")

    overall = {"passed": n_pass, "failed": n_total - n_pass,
               "total_seconds": total_time, "tests": {
                   tid: {"pass": bool(p), "seconds": float(s)}
                   for tid, (p, s, _) in scoreboard.items()}}
    with open(RESULTS_DIR / "L5_overall.json", "w") as f:
        json.dump(overall, f, indent=2, default=str)


if __name__ == "__main__":
    main()
