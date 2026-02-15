"""AEGIS 3.0 — Layer 5: Safety Supervisor (standalone module).

Extracted from the notebook prototype into a testable, importable Python module.
Implements the three-tier hierarchical safety architecture:
  Tier 1 — Reflex Controller   (model-free threshold logic)
  Tier 2 — STL Monitor         (Signal Temporal Logic on predicted trajectory)
  Tier 3 — Seldonian Constraint (high-confidence probabilistic bound)

Plus: hysteresis, cold-start relaxation, and insulin-on-board tracking.
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ── Enums ─────────────────────────────────────────────────────

class SafetyAction(enum.IntEnum):
    SUSPEND = 0   # All insulin halted
    BLOCK   = 1   # Bolus blocked, basal may continue
    REDUCE  = 2   # Dose capped to max_bolus
    ALLOW   = 3   # No modification


class SafetyTier(enum.IntEnum):
    TIER_1_REFLEX    = 1
    TIER_2_STL       = 2
    TIER_3_SELDONIAN = 3
    TIER_4_NOMINAL   = 4


# ── Data classes ──────────────────────────────────────────────

@dataclass
class SafetyThresholds:
    glucose_danger_low: float = 54.0     # mg/dL — severe hypo
    glucose_warning_low: float = 70.0    # mg/dL — mild hypo
    glucose_target: float = 110.0
    glucose_warning_high: float = 180.0
    glucose_danger_high: float = 250.0
    glucose_extreme_high: float = 400.0
    max_bolus: float = 15.0              # U
    max_iob: float = 25.0               # U


@dataclass
class TierResult:
    tier: SafetyTier
    safe: bool
    action: SafetyAction
    dose: float
    reason: str
    p_harm: Optional[float] = None
    robustness: Optional[float] = None


@dataclass
class SafetyResult:
    overall_safe: bool
    active_tier: SafetyTier
    safe_dose: float
    original_dose: float
    tier_results: Dict[SafetyTier, TierResult] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    iob: float = 0.0


# ── Safety Supervisor ────────────────────────────────────────

class SafetySupervisor:
    """Hierarchical three-tier safety supervisor.

    Parameters
    ----------
    thresholds : SafetyThresholds, optional
    seldonian_delta : float
        Maximum tolerated probability of constraint violation.
    seldonian_alpha : float
        Significance level for the upper confidence bound.
    hysteresis_buffer : float
        Dead-band width (mg/dL) around thresholds to prevent chattering.
    hysteresis_hold_steps : int
        Number of consecutive readings before a tier transition is accepted.
    cold_start_days : int
        Duration of the cold-start relaxation schedule.
    insulin_sensitivity : float
        Estimated mg/dL drop per unit of insulin (for outcome model).
    """

    def __init__(
        self,
        thresholds: Optional[SafetyThresholds] = None,
        seldonian_delta: float = 0.01,
        seldonian_alpha: float = 0.05,
        hysteresis_buffer: float = 2.0,
        hysteresis_hold_steps: int = 3,
        cold_start_days: int = 30,
        insulin_sensitivity: float = 25.0,
    ):
        self.th = thresholds or SafetyThresholds()
        self.seldonian_delta = seldonian_delta
        self.seldonian_alpha = seldonian_alpha
        self.hysteresis_buffer = hysteresis_buffer
        self.hysteresis_hold_steps = hysteresis_hold_steps
        self.cold_start_days = cold_start_days
        self.insulin_sensitivity = insulin_sensitivity

        # Hysteresis state
        self._last_tier: Optional[SafetyTier] = None
        self._hold_counter: int = 0
        self._candidate_tier: Optional[SafetyTier] = None

        # Cold-start state
        self._current_day: int = 0

        # IOB tracking
        self._iob: float = 0.0
        self._iob_history: List[float] = []

    # ── Public API ────────────────────────────────────────────

    def verify(
        self,
        glucose: float,
        predicted_trajectory: Optional[np.ndarray],
        recommended_dose: float,
        prediction_std: float = 15.0,
        n_observations: int = 100,
    ) -> SafetyResult:
        """Run all three safety tiers and return a composite result."""
        t0 = time.perf_counter()

        # Sanitise inputs
        glucose = self._sanitise_glucose(glucose)
        recommended_dose = max(0.0, recommended_dose) if np.isfinite(recommended_dose) else 0.0
        if predicted_trajectory is None or len(predicted_trajectory) == 0:
            predicted_trajectory = np.full(36, glucose)

        # Tier 1: Reflex (model-free)
        t1 = self._tier1_reflex(glucose, recommended_dose)

        # Tier 2: STL (trajectory-based)
        t2 = self._tier2_stl(glucose, predicted_trajectory, recommended_dose)

        # Tier 3: Seldonian (probabilistic)
        t3 = self._tier3_seldonian(glucose, recommended_dose, prediction_std, n_observations)

        # Combine: highest-priority (lowest number) failing tier wins
        tier_results = {
            SafetyTier.TIER_1_REFLEX: t1,
            SafetyTier.TIER_2_STL: t2,
            SafetyTier.TIER_3_SELDONIAN: t3,
        }

        # Determine safe dose (strictest tier wins)
        safe_dose = recommended_dose
        active_tier = SafetyTier.TIER_4_NOMINAL
        overall_safe = True

        for tier in [SafetyTier.TIER_1_REFLEX, SafetyTier.TIER_2_STL, SafetyTier.TIER_3_SELDONIAN]:
            tr = tier_results[tier]
            if not tr.safe:
                safe_dose = min(safe_dose, tr.dose)
                if tier < active_tier:
                    active_tier = tier
                overall_safe = False

        # Apply hysteresis
        active_tier = self._apply_hysteresis(active_tier)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return SafetyResult(
            overall_safe=overall_safe,
            active_tier=active_tier,
            safe_dose=safe_dose,
            original_dose=recommended_dose,
            tier_results=tier_results,
            processing_time_ms=elapsed_ms,
            iob=self._iob,
        )

    def check_stl_satisfaction(
        self, glucose_trace: np.ndarray, dt_minutes: float = 5.0,
    ) -> dict:
        """Evaluate STL specifications on a completed glucose trace.

        Returns dict with keys: no_severe_hypo, no_extreme_hyper, hypo_recovery.
        """
        trace = np.asarray(glucose_trace, dtype=float)
        min_g = np.min(trace) if len(trace) > 0 else 0.0
        max_g = np.max(trace) if len(trace) > 0 else 0.0

        # φ₁: □[0,T](G ≥ 54)
        phi1_sat = bool(min_g >= self.th.glucose_danger_low)
        phi1_rob = float(min_g - self.th.glucose_danger_low)

        # φ₂: □[0,T](G ≤ 400)
        phi2_sat = bool(max_g <= self.th.glucose_extreme_high)
        phi2_rob = float(self.th.glucose_extreme_high - max_g)

        # φ₃: G < 70 → ◇[0,30min](G ≥ 80)
        recovery_window = int(30.0 / dt_minutes)
        phi3_sat = True
        for i in range(len(trace)):
            if trace[i] < self.th.glucose_warning_low:
                end = min(i + recovery_window + 1, len(trace))
                if not np.any(trace[i:end] >= 80.0):
                    phi3_sat = False
                    break

        return {
            "no_severe_hypo": {"satisfied": phi1_sat, "robustness": phi1_rob},
            "no_extreme_hyper": {"satisfied": phi2_sat, "robustness": phi2_rob},
            "hypo_recovery": {"satisfied": phi3_sat, "robustness": 0.0},
        }

    # ── IOB tracking ──────────────────────────────────────────

    def record_dose(self, dose: float, duration_minutes: float = 300.0):
        """Record an administered dose for IOB tracking."""
        self._iob_history.append((dose, duration_minutes, 0.0))

    def tick_iob(self, dt_minutes: float = 5.0):
        """Advance IOB decay by one timestep."""
        remaining = []
        self._iob = 0.0
        for dose, dur, elapsed in self._iob_history:
            new_elapsed = elapsed + dt_minutes
            if new_elapsed < dur:
                frac_remaining = 1.0 - new_elapsed / dur
                self._iob += dose * frac_remaining
                remaining.append((dose, dur, new_elapsed))
        self._iob_history = remaining

    def get_iob(self) -> float:
        return self._iob

    # ── Hysteresis ────────────────────────────────────────────

    def reset_hysteresis(self):
        self._last_tier = None
        self._hold_counter = 0
        self._candidate_tier = None

    def _apply_hysteresis(self, raw_tier: SafetyTier) -> SafetyTier:
        if self._last_tier is None:
            self._last_tier = raw_tier
            return raw_tier

        # Emergency always passes through immediately
        if raw_tier == SafetyTier.TIER_1_REFLEX:
            self._last_tier = raw_tier
            self._hold_counter = 0
            self._candidate_tier = None
            return raw_tier

        if raw_tier != self._last_tier:
            if raw_tier == self._candidate_tier:
                self._hold_counter += 1
            else:
                self._candidate_tier = raw_tier
                self._hold_counter = 1

            if self._hold_counter >= self.hysteresis_hold_steps:
                self._last_tier = raw_tier
                self._candidate_tier = None
                self._hold_counter = 0
                return raw_tier
            else:
                return self._last_tier
        else:
            self._candidate_tier = None
            self._hold_counter = 0
            return raw_tier

    # ── Cold start ────────────────────────────────────────────

    def set_day(self, day: int):
        self._current_day = day

    def _get_cold_start_relaxation(self) -> float:
        """Return the relaxation multiplier: 5.0 at day 0 → 1.0 at cold_start_days."""
        if self._current_day >= self.cold_start_days:
            return 1.0
        return 5.0 - 4.0 * (self._current_day / self.cold_start_days)

    # ── Tier implementations ──────────────────────────────────

    def _sanitise_glucose(self, g: float) -> float:
        if not np.isfinite(g):
            return 0.0  # Treat as worst-case
        return g

    def _tier1_reflex(self, glucose: float, dose: float) -> TierResult:
        """Model-free threshold logic."""
        if glucose < self.th.glucose_danger_low:
            return TierResult(
                SafetyTier.TIER_1_REFLEX, False, SafetyAction.SUSPEND, 0.0,
                f"SUSPEND: glucose {glucose:.1f} < {self.th.glucose_danger_low}")

        if glucose < self.th.glucose_warning_low and dose > 0:
            return TierResult(
                SafetyTier.TIER_1_REFLEX, False, SafetyAction.BLOCK, 0.0,
                f"BLOCK: glucose {glucose:.1f} < {self.th.glucose_warning_low} with dose > 0")

        if dose > self.th.max_bolus:
            return TierResult(
                SafetyTier.TIER_1_REFLEX, False, SafetyAction.REDUCE, self.th.max_bolus,
                f"REDUCE: dose {dose:.1f} > max {self.th.max_bolus}")

        return TierResult(
            SafetyTier.TIER_1_REFLEX, True, SafetyAction.ALLOW, dose,
            "Tier 1 PASS")

    def _tier2_stl(
        self, glucose: float, trajectory: np.ndarray, dose: float,
    ) -> TierResult:
        """Signal Temporal Logic on predicted trajectory."""
        traj = np.asarray(trajectory, dtype=float)

        # φ₁ check on trajectory
        effect_per_step = dose * self.insulin_sensitivity / max(1, len(traj))
        adjusted = traj.copy()
        for i in range(len(adjusted)):
            frac = min(1.0, i / max(1, len(adjusted) * 0.5))
            adjusted[i] -= dose * self.insulin_sensitivity * frac

        min_predicted = float(np.min(adjusted))
        max_predicted = float(np.max(traj))

        if min_predicted < self.th.glucose_danger_low:
            safe_dose = max(0.0, dose * 0.5)
            return TierResult(
                SafetyTier.TIER_2_STL, False, SafetyAction.REDUCE, safe_dose,
                f"STL: predicted min {min_predicted:.1f} < {self.th.glucose_danger_low}")

        if max_predicted > self.th.glucose_extreme_high:
            return TierResult(
                SafetyTier.TIER_2_STL, False, SafetyAction.ALLOW, dose,
                f"STL: predicted max {max_predicted:.1f} > {self.th.glucose_extreme_high}")

        return TierResult(
            SafetyTier.TIER_2_STL, True, SafetyAction.ALLOW, dose,
            "Tier 2 PASS")

    def _tier3_seldonian(
        self, glucose: float, dose: float,
        prediction_std: float, n_observations: int,
    ) -> TierResult:
        """Seldonian high-confidence probabilistic bound.

        Uses Hoeffding's inequality with cold-start relaxation.
        """
        relaxation = self._get_cold_start_relaxation()
        effective_delta = self.seldonian_delta * relaxation

        # Predicted post-dose glucose (simple pharmacokinetic model)
        predicted_post = glucose - dose * self.insulin_sensitivity
        # Add IOB effect
        predicted_post -= self._iob * self.insulin_sensitivity * 0.3

        # Hoeffding bound: P(actual < predicted - margin) ≤ delta
        if n_observations > 0 and prediction_std > 0:
            margin = prediction_std * math.sqrt(
                math.log(1.0 / max(1e-15, effective_delta)) / (2.0 * n_observations)
            )
        else:
            margin = prediction_std * 3.0  # Fallback conservative

        lower_bound = predicted_post - margin
        p_harm = 1.0 - self._normal_cdf(
            (predicted_post - self.th.glucose_danger_low) / max(1e-6, prediction_std))

        if lower_bound < self.th.glucose_danger_low:
            # Binary search for maximum safe dose
            safe_dose = self._find_safe_dose(glucose, prediction_std, n_observations,
                                              effective_delta, dose)
            return TierResult(
                SafetyTier.TIER_3_SELDONIAN, False, SafetyAction.REDUCE, safe_dose,
                f"Seldonian: lower bound {lower_bound:.1f} < {self.th.glucose_danger_low}",
                p_harm=p_harm)

        return TierResult(
            SafetyTier.TIER_3_SELDONIAN, True, SafetyAction.ALLOW, dose,
            "Tier 3 PASS", p_harm=p_harm)

    def _find_safe_dose(
        self, glucose: float, std: float, n_obs: int,
        delta: float, max_dose: float,
    ) -> float:
        """Binary search for the maximum dose satisfying the Seldonian constraint."""
        lo, hi = 0.0, max_dose
        for _ in range(20):
            mid = (lo + hi) / 2.0
            post = glucose - mid * self.insulin_sensitivity
            post -= self._iob * self.insulin_sensitivity * 0.3
            if n_obs > 0 and std > 0:
                margin = std * math.sqrt(math.log(1.0 / max(1e-15, delta)) / (2.0 * n_obs))
            else:
                margin = std * 3.0
            if post - margin >= self.th.glucose_danger_low:
                lo = mid
            else:
                hi = mid
        return lo

    @staticmethod
    def _normal_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))