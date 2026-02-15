"""Shared utilities for Layer 5 full-scale simulation tests.

Provides: patient loading, glucose simulation, clinical metrics,
checkpointing (resume support), and result saving.
"""

import json
import sys
import time
from collections import namedtuple
from datetime import timedelta
from pathlib import Path

import numpy as np

# simglucose
from simglucose.patient.t1dpatient import T1DPatient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from aegis_testing.aegis_core.safety_supervisor import SafetySupervisor

PatientAction = namedtuple("patient_action", ["CHO", "insulin"])

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "layer5"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_all_patient_names():
    """Get all 30 simglucose patient names."""
    import pandas as pd
    import pkg_resources
    params = pkg_resources.resource_filename("simglucose", "params/vpatient_params.csv")
    return list(pd.read_csv(params)["Name"])


def simulate_patient(patient_name, duration_hours=24, basal_rate=1.0,
                     meals=None, seed=42):
    """Run simglucose simulation, return glucose trace (1-min resolution)."""
    np.random.seed(seed)
    patient = T1DPatient.withName(patient_name)
    patient.reset()

    n_steps = int(duration_hours * 60)
    glucose = np.zeros(n_steps)
    meal_map = {int(t): c for t, c in (meals or [])}

    for step in range(n_steps):
        action = PatientAction(CHO=meal_map.get(step, 0.0), insulin=basal_rate)
        patient.step(action)
        glucose[step] = max(0.0, patient.observation.Gsub)

    return glucose


def simple_controller(glucose, target=120.0, basal=0.03, gain=0.0002,
                      max_correction=0.02):
    """Simple proportional controller for dose proposals.

    Scaled for simglucose 1-minute timesteps where basal ≈ 0.03 U/min
    is the normal insulin delivery rate (empirically determined from
    the simglucose virtual patient cohort).

    Dose scale reference (at basal=0.03):
      - glucose=120: 0.030 U/min (basal only)
      - glucose=200: 0.046 U/min (+53% correction)
      - glucose=300: 0.050 U/min (max correction capped)
      - glucose< 70: 0.000 U/min (suspended)

    Mimics what L4 (Decision Engine) would propose:
    - Hypo (<54): suspend all insulin
    - Low (<70): suspend insulin
    - Low-normal (70-90): reduce basal to 50-80%
    - Normal (90-target): full basal only
    - Above target: basal + small proportional correction
    """
    # === Hypo / low glucose: reduce or suspend ===
    if glucose < 54:
        return 0.0   # Severe hypo: suspend everything
    if glucose < 70:
        return 0.0   # Hypo: suspend
    if glucose < 80:
        return basal * 0.5   # Low warning: half basal
    if glucose < 90:
        return basal * 0.8   # Low-normal: 80% basal

    # === Normal / high glucose: basal + proportional correction ===
    error = glucose - target  # positive = hyperglycemia
    correction = max(0.0, error * gain)
    dose = basal + min(correction, max_correction)
    return dose


# --- Per-patient basal calibration ---

_BASAL_CACHE: dict = {}  # patient_name -> calibrated basal rate


def calibrate_patient_basal(patient_name, duration_hours=6, target_tir=100.0):
    """Find the basal rate that keeps this patient closest to 70-180 mg/dL.

    Runs a quick open-loop sim at several candidate rates and picks the one
    with the best Time-in-Range. Results are cached per patient.

    Args:
        patient_name: simglucose patient name
        duration_hours: calibration sim length (6h is enough)
        target_tir: ideal TIR to aim for

    Returns:
        float: optimal basal rate (U/min)
    """
    if patient_name in _BASAL_CACHE:
        return _BASAL_CACHE[patient_name]

    candidates = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    best_rate, best_score = 0.03, -1

    for rate in candidates:
        try:
            glucose = simulate_patient(patient_name, duration_hours=duration_hours,
                                       basal_rate=rate, meals=None, seed=0)
            # Score = TIR (higher is better) with penalty for hypo
            tir = float(np.mean((glucose >= 70) & (glucose <= 180)) * 100)
            tbr = float(np.mean(glucose < 54) * 100)  # severe hypo penalty
            score = tir - tbr * 5  # Heavy penalty for hypo
            if score > best_score:
                best_score = score
                best_rate = rate
        except Exception:
            continue

    _BASAL_CACHE[patient_name] = best_rate
    return best_rate


def simulate_closed_loop(patient_name, duration_hours=24, meals=None,
                         noise_std=0.0, seed=42, supervisor_kwargs=None,
                         basal=None):
    """Run closed-loop simulation with L5 actively controlling doses.

    The loop:
      1. Read glucose from simglucose patient
      2. Simple controller proposes a dose
      3. L5 Safety Supervisor verifies/modifies the dose
      4. The SAFE dose (L5's output) is fed to the patient
      5. Repeat

    Returns:
        dict with 'glucose', 'proposed_doses', 'safe_doses', 'actions',
        'metrics' (clinical), 'stl' (STL satisfaction on final trace)
    """
    np.random.seed(seed)
    patient = T1DPatient.withName(patient_name)
    patient.reset()

    sup_kwargs = supervisor_kwargs or {}
    sup = SafetySupervisor(**sup_kwargs)

    n_steps = int(duration_hours * 60)
    glucose_trace = np.zeros(n_steps)
    proposed_doses = np.zeros(n_steps)
    safe_doses = np.zeros(n_steps)
    actions = []

    meal_map = {int(t): c for t, c in (meals or [])}
    rng = np.random.RandomState(seed + 999)

    # Warm-up trajectory buffer (last 36 readings at 5-min intervals)
    traj_buffer = np.full(36, 120.0)

    for step in range(n_steps):
        # 1. Read current glucose (with optional sensor noise)
        raw_g = max(0.0, patient.observation.Gsub)
        glucose = raw_g + rng.randn() * noise_std if noise_std > 0 else raw_g
        glucose = max(0.0, glucose)
        glucose_trace[step] = glucose

        # Update trajectory buffer every 5 min
        if step % 5 == 0 and step > 0:
            traj_buffer = np.roll(traj_buffer, -1)
            traj_buffer[-1] = glucose

        # 2. Controller proposes a dose
        effective_basal = basal if basal is not None else 0.03
        proposed = simple_controller(glucose, basal=effective_basal)
        proposed_doses[step] = proposed

        # 3. L5 verifies and modifies the dose
        # n_observations represents accumulated calibration data — a real
        # deployment would have thousands of historical observations.
        # Starting at 500 gives a tight Hoeffding bound (UCB ≈ 0.03).
        n_obs = 500 + step  # Grows as system collects more data
        pred_std = max(3.0, noise_std) if noise_std > 0 else 3.0
        result = sup.verify(
            glucose=glucose,
            predicted_trajectory=traj_buffer.copy(),
            recommended_dose=proposed,
            prediction_std=pred_std,
            n_observations=n_obs,
        )
        safe_dose = result.safe_dose
        safe_doses[step] = safe_dose
        actions.append(int(result.active_tier))

        # 4. Feed the SAFE dose (L5's output) back to the patient
        cho = meal_map.get(step, 0.0)
        patient.step(PatientAction(CHO=cho, insulin=safe_dose))

    # Compute metrics on the resulting trace
    stl = sup.check_stl_satisfaction(glucose_trace, timestep_minutes=1.0)
    metrics = clinical_metrics(glucose_trace)

    return {
        "glucose": glucose_trace,
        "proposed_doses": proposed_doses,
        "safe_doses": safe_doses,
        "actions": actions,
        "metrics": metrics,
        "stl": stl,
    }


def clinical_metrics(glucose):
    """Compute standard clinical metrics from a glucose trace."""
    g = glucose
    mean = float(np.mean(g))
    return {
        "mean": mean,
        "std": float(np.std(g)),
        "cv": float(np.std(g) / max(1e-6, mean) * 100),
        "tir": float(np.mean((g >= 70) & (g <= 180)) * 100),
        "tbr_70": float(np.mean(g < 70) * 100),
        "tbr_54": float(np.mean(g < 54) * 100),
        "tar_180": float(np.mean(g > 180) * 100),
        "min": float(np.min(g)),
        "max": float(np.max(g)),
    }


# --- Checkpointing ---

def is_completed(test_id):
    """Check if a test already has saved results (for resume)."""
    summary_path = RESULTS_DIR / f"{test_id}_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                data = json.load(f)
            return "pass" in data
        except Exception:
            return False
    return False


def save_results(test_id, results, summary):
    """Save raw results + summary JSON."""
    with open(RESULTS_DIR / f"{test_id}_raw.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(RESULTS_DIR / f"{test_id}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


# --- Progress ---

def progress(current, total, prefix="", start_time=None):
    """Print simple progress line."""
    pct = current / max(1, total) * 100
    eta = ""
    if start_time and current > 0:
        elapsed = time.time() - start_time
        remaining = elapsed / current * (total - current)
        eta = f" | ETA: {timedelta(seconds=int(remaining))}"
    print(f"\r  {prefix} {current}/{total} ({pct:.0f}%){eta}    ", end="", flush=True)
    if current >= total:
        print()
