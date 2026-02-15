"""Layer 5: Safety Supervisor — Three-Tier Simplex Architecture.

Implements the AEGIS 3.0 safety verification system:
- Tier 1: Reflex Controller (immediate, model-free hard thresholds)
- Tier 2: STL Monitor (predictive, Signal Temporal Logic reachability)
- Tier 3: Seldonian Constraints (probabilistic high-confidence safety bounds)

Also includes cold-start schedule and hysteresis logic.
"""

import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np
from scipy.stats import norm


class SafetyTier(IntEnum):
    """Safety tier classification.

    Lower number = higher priority.
    """
    TIER_1_REFLEX = 1
    TIER_2_STL = 2
    TIER_3_SELDONIAN = 3
    TIER_4_NOMINAL = 4


class SafetyAction(IntEnum):
    """Actions the safety system can take, ordered by severity."""
    SUSPEND = 0       # Suspend all insulin delivery immediately
    BLOCK = 1         # Block the proposed dose entirely
    REDUCE = 2        # Reduce the proposed dose
    ALLOW = 3         # Allow the proposed dose


@dataclass
class SafetyThresholds:
    """Configurable safety thresholds for the supervisor."""
    glucose_danger_low: float = 54.0     # mg/dL — severe hypoglycemia
    glucose_warning_low: float = 70.0    # mg/dL — mild hypoglycemia
    glucose_target: float = 110.0        # mg/dL — target glucose
    glucose_warning_high: float = 180.0  # mg/dL — hyperglycemia
    glucose_danger_high: float = 250.0   # mg/dL — severe hyperglycemia
    max_bolus: float = 15.0              # Maximum single bolus (units)
    max_iob: float = 25.0               # Maximum insulin on board (units)


@dataclass
class TierResult:
    """Result from a single safety tier check."""
    tier: SafetyTier
    safe: bool
    action: SafetyAction
    dose: float
    reason: str
    latency_ms: float = 0.0
    p_harm: Optional[float] = None


@dataclass
class SafetyVerification:
    """Complete safety verification result from all tiers."""
    overall_safe: bool
    active_tier: SafetyTier
    tier_results: dict  # {SafetyTier: TierResult}
    original_dose: float
    safe_dose: float
    violations: list
    total_latency_ms: float = 0.0


@dataclass
class STLSpecification:
    """Signal Temporal Logic specification."""
    name: str
    formula: str  # Human-readable formula
    check_fn: object = None  # Callable for verification


class SafetySupervisor:
    """Three-tier safety supervisor with Simplex architecture.

    Ensures safety through hierarchical verification:
    1. Reflex: Hard thresholds (fastest, always active)
    2. STL: Temporal logic on predicted trajectories
    3. Seldonian: Probabilistic safety guarantees

    Tier priority is enforced: if Tier 1 triggers, its decision
    overrides Tier 2 and 3 regardless of their output.
    """

    def __init__(
        self,
        thresholds: Optional[SafetyThresholds] = None,
        seldonian_delta: float = 0.01,
        seldonian_alpha: float = 0.05,
        cold_start_days: int = 30,
        hysteresis_buffer: float = 2.0,
        hysteresis_hold_steps: int = 3,
    ):
        self.thresholds = thresholds or SafetyThresholds()
        self.seldonian_delta = seldonian_delta  # Max harm probability
        self.seldonian_alpha = seldonian_alpha  # Confidence level
        self.cold_start_days = cold_start_days
        self.hysteresis_buffer = hysteresis_buffer  # mg/dL buffer around thresholds
        self.hysteresis_hold_steps = hysteresis_hold_steps  # Min steps before tier transition

        # Hysteresis state
        self._last_tier = SafetyTier.TIER_4_NOMINAL
        self._tier_hold_counter = 0
        self._tier_transition_count = 0

        # Cold start state
        self._current_day = 0

        # STL specifications
        self.stl_specs = [
            STLSpecification(
                name="no_severe_hypo",
                formula="□[0,T](G ≥ 54)",
            ),
            STLSpecification(
                name="no_extreme_hyper",
                formula="□[0,T](G ≤ 400)",
            ),
            STLSpecification(
                name="hypo_recovery",
                formula="G<70 → ◇[0,30min](G ≥ 80)",
            ),
        ]

    def verify(
        self,
        glucose: float,
        predicted_trajectory: np.ndarray,
        recommended_dose: float,
        prediction_std: float = 20.0,
        n_observations: int = 100,
    ) -> SafetyVerification:
        """Run complete three-tier safety verification.

        Args:
            glucose: Current glucose reading (mg/dL)
            predicted_trajectory: Array of predicted future glucose values
            recommended_dose: Proposed insulin dose (units)
            prediction_std: Standard deviation of glucose prediction
            n_observations: Number of observations for Seldonian bound

        Returns:
            SafetyVerification with tier results and safe dose
        """
        start = time.perf_counter()

        # --- Tier 1: Reflex Controller ---
        t1_start = time.perf_counter()
        tier1 = self._tier1_reflex(glucose, recommended_dose)
        tier1.latency_ms = (time.perf_counter() - t1_start) * 1000

        # --- Tier 2: STL Monitor ---
        t2_start = time.perf_counter()
        tier2 = self._tier2_stl(glucose, predicted_trajectory, recommended_dose)
        tier2.latency_ms = (time.perf_counter() - t2_start) * 1000

        # --- Tier 3: Seldonian Constraints ---
        t3_start = time.perf_counter()
        tier3 = self._tier3_seldonian(
            glucose, recommended_dose, prediction_std, n_observations
        )
        tier3.latency_ms = (time.perf_counter() - t3_start) * 1000

        # --- Tier Priority: Lowest tier number wins ---
        tier_results = {
            SafetyTier.TIER_1_REFLEX: tier1,
            SafetyTier.TIER_2_STL: tier2,
            SafetyTier.TIER_3_SELDONIAN: tier3,
        }

        # Determine active tier (highest priority = lowest number that triggered)
        active_tier = SafetyTier.TIER_4_NOMINAL
        for tier_level in [SafetyTier.TIER_1_REFLEX, SafetyTier.TIER_2_STL,
                           SafetyTier.TIER_3_SELDONIAN]:
            if not tier_results[tier_level].safe:
                active_tier = tier_level
                break

        # Apply hysteresis to prevent chattering
        active_tier = self._apply_hysteresis(active_tier)

        # Calculate safe dose (minimum from all triggered tiers)
        safe_dose = self._calculate_safe_dose(
            recommended_dose, tier1, tier2, tier3, active_tier
        )

        # Collect violations
        violations = []
        for t, result in tier_results.items():
            if not result.safe:
                violations.append(result.reason)

        overall_safe = active_tier == SafetyTier.TIER_4_NOMINAL

        total_latency = (time.perf_counter() - start) * 1000

        return SafetyVerification(
            overall_safe=overall_safe,
            active_tier=active_tier,
            tier_results=tier_results,
            original_dose=recommended_dose,
            safe_dose=safe_dose,
            violations=violations,
            total_latency_ms=total_latency,
        )

    def _tier1_reflex(self, glucose: float, dose: float) -> TierResult:
        """Tier 1: Model-free reflex controller with hard thresholds.

        This is the fastest safety check — pure threshold comparisons.
        No model predictions needed.
        """
        th = self.thresholds

        # CRITICAL: Glucose below severe hypoglycemia threshold
        if glucose < th.glucose_danger_low:
            return TierResult(
                tier=SafetyTier.TIER_1_REFLEX,
                safe=False,
                action=SafetyAction.SUSPEND,
                dose=0.0,
                reason=f"CRITICAL: Glucose {glucose:.1f} < {th.glucose_danger_low} mg/dL — suspend all insulin",
            )

        # WARNING: Glucose below mild hypoglycemia and dose > 0
        if glucose < th.glucose_warning_low and dose > 0:
            return TierResult(
                tier=SafetyTier.TIER_1_REFLEX,
                safe=False,
                action=SafetyAction.BLOCK,
                dose=0.0,
                reason=f"LOW: Glucose {glucose:.1f} < {th.glucose_warning_low} mg/dL — block bolus",
            )

        # LIMIT: Dose exceeds maximum bolus
        if dose > th.max_bolus:
            return TierResult(
                tier=SafetyTier.TIER_1_REFLEX,
                safe=False,
                action=SafetyAction.REDUCE,
                dose=th.max_bolus,
                reason=f"LIMIT: Dose {dose:.1f} > max {th.max_bolus} — capped",
            )

        # All reflex checks passed
        return TierResult(
            tier=SafetyTier.TIER_1_REFLEX,
            safe=True,
            action=SafetyAction.ALLOW,
            dose=dose,
            reason="All reflex checks passed",
        )

    def _tier2_stl(
        self,
        glucose: float,
        predicted_trajectory: np.ndarray,
        dose: float,
    ) -> TierResult:
        """Tier 2: Signal Temporal Logic monitor.

        Checks predicted trajectories against temporal specifications:
        - φ₁: □[0,T](G ≥ 54) — never severe hypo
        - φ₂: □[0,T](G ≤ 400) — never extreme hyper
        - φ₃: G<70 → ◇[0,30min](G ≥ 80) — recovery from hypo
        """
        th = self.thresholds

        if predicted_trajectory is None or len(predicted_trajectory) == 0:
            # No prediction available — can't verify STL
            return TierResult(
                tier=SafetyTier.TIER_2_STL,
                safe=True,
                action=SafetyAction.ALLOW,
                dose=dose,
                reason="No trajectory available — STL skipped",
            )

        # Adjust trajectory for insulin effect (simplified: dose reduces glucose)
        adjusted_trajectory = predicted_trajectory.copy()
        insulin_effect_per_unit = 25.0  # mg/dL per unit insulin
        for i in range(len(adjusted_trajectory)):
            # Gradual insulin effect onset
            effect_fraction = min(1.0, i / max(1, len(adjusted_trajectory) * 0.3))
            adjusted_trajectory[i] -= dose * insulin_effect_per_unit * effect_fraction

        predicted_min = np.min(adjusted_trajectory)
        predicted_max = np.max(adjusted_trajectory)

        # STL spec φ₁: □[0,T](G ≥ 54) — check for severe hypo
        if predicted_min < th.glucose_danger_low:
            # Calculate dose reduction needed
            glucose_deficit = th.glucose_danger_low - predicted_min
            reduction = glucose_deficit / insulin_effect_per_unit
            safe_dose = max(0.0, dose - reduction)
            return TierResult(
                tier=SafetyTier.TIER_2_STL,
                safe=False,
                action=SafetyAction.REDUCE,
                dose=safe_dose,
                reason=f"STL: Predicted min glucose {predicted_min:.0f} < {th.glucose_danger_low} mg/dL",
            )

        # STL spec φ₂: □[0,T](G ≤ 400) — check for extreme hyper
        if predicted_max > 400:
            return TierResult(
                tier=SafetyTier.TIER_2_STL,
                safe=False,
                action=SafetyAction.ALLOW,  # Don't reduce dose if hyper
                dose=dose,
                reason=f"STL: Predicted max glucose {predicted_max:.0f} > 400 mg/dL",
            )

        # All STL specs satisfied
        # Compute robustness margin (how far from violation)
        robustness = min(
            predicted_min - th.glucose_danger_low,
            400 - predicted_max
        )

        return TierResult(
            tier=SafetyTier.TIER_2_STL,
            safe=True,
            action=SafetyAction.ALLOW,
            dose=dose,
            reason=f"STL satisfied (robustness margin: {robustness:.1f} mg/dL)",
        )

    def _tier3_seldonian(
        self,
        glucose: float,
        dose: float,
        prediction_std: float,
        n_observations: int,
    ) -> TierResult:
        """Tier 3: Seldonian safety constraints.

        High-confidence probabilistic bound on harm probability:
        P(glucose < 54) ≤ δ with confidence 1-α.

        Uses Hoeffding inequality for the UCB.
        """
        th = self.thresholds

        # Predict post-dose glucose using simple model
        insulin_effect = dose * 25.0  # mg/dL per unit
        predicted_glucose = glucose - insulin_effect

        # P(G < 54) using normal approximation
        if prediction_std > 0:
            z_score = (th.glucose_danger_low - predicted_glucose) / prediction_std
            p_hypo = norm.cdf(z_score)
        else:
            p_hypo = 1.0 if predicted_glucose < th.glucose_danger_low else 0.0

        # Upper Confidence Bound on harm probability (Hoeffding)
        n = max(1, n_observations)
        hoeffding_term = math.sqrt(math.log(1.0 / self.seldonian_alpha) / (2 * n))
        p_hypo_ucb = min(1.0, p_hypo + hoeffding_term)

        # Apply cold-start relaxation
        relaxation = self._get_cold_start_relaxation()
        effective_delta = self.seldonian_delta * relaxation

        if p_hypo_ucb > effective_delta:
            # Find safe dose via binary search
            safe_dose = self._binary_search_safe_dose(
                glucose, dose, prediction_std, n, effective_delta
            )
            return TierResult(
                tier=SafetyTier.TIER_3_SELDONIAN,
                safe=False,
                action=SafetyAction.REDUCE,
                dose=safe_dose,
                reason=f"Seldonian: P(hypo) UCB = {p_hypo_ucb:.4f} > δ = {effective_delta:.4f}",
                p_harm=p_hypo_ucb,
            )

        return TierResult(
            tier=SafetyTier.TIER_3_SELDONIAN,
            safe=True,
            action=SafetyAction.ALLOW,
            dose=dose,
            reason=f"Seldonian: P(hypo) UCB = {p_hypo_ucb:.4f} ≤ δ = {effective_delta:.4f}",
            p_harm=p_hypo,
        )

    def _binary_search_safe_dose(
        self,
        glucose: float,
        max_dose: float,
        prediction_std: float,
        n: int,
        delta: float,
    ) -> float:
        """Binary search for the maximum dose that satisfies Seldonian constraint."""
        low, high = 0.0, max_dose

        for _ in range(20):  # 20 iterations for precision
            mid = (low + high) / 2.0
            predicted = glucose - mid * 25.0
            if prediction_std > 0:
                z_score = (self.thresholds.glucose_danger_low - predicted) / prediction_std
                p_hypo = norm.cdf(z_score)
            else:
                p_hypo = 1.0 if predicted < self.thresholds.glucose_danger_low else 0.0

            hoeffding = math.sqrt(math.log(1.0 / self.seldonian_alpha) / (2 * n))
            p_ucb = p_hypo + hoeffding

            if p_ucb < delta:
                low = mid
            else:
                high = mid

        # Round down to nearest 0.5 for safety
        return math.floor(low * 2) / 2.0

    def _get_cold_start_relaxation(self) -> float:
        """Get the cold-start relaxation factor for the current day.

        During cold-start (days 0-30), we relax the Seldonian constraint
        using population-derived priors. Relaxation decreases linearly:
        Day 0: relaxation = 5.0 (5× more permissive)
        Day 30: relaxation = 1.0 (standard constraint)
        """
        if self._current_day >= self.cold_start_days:
            return 1.0

        # Linear relaxation schedule
        max_relaxation = 5.0
        progress = self._current_day / self.cold_start_days
        return max_relaxation - (max_relaxation - 1.0) * progress

    def set_day(self, day: int) -> None:
        """Set the current day for cold-start schedule."""
        self._current_day = day

    def _apply_hysteresis(self, proposed_tier: SafetyTier) -> SafetyTier:
        """Apply hysteresis to prevent rapid tier oscillation (chattering).

        Requires the proposed tier to be active for `hysteresis_hold_steps`
        consecutive calls before actually switching.
        """
        if proposed_tier == self._last_tier:
            self._tier_hold_counter = 0
            return proposed_tier

        self._tier_hold_counter += 1

        if self._tier_hold_counter >= self.hysteresis_hold_steps:
            # Tier change confirmed after hold period
            self._last_tier = proposed_tier
            self._tier_hold_counter = 0
            self._tier_transition_count += 1
            return proposed_tier

        # Not enough consecutive steps — hold previous tier
        # EXCEPTION: Always immediately escalate to higher priority (lower number)
        if proposed_tier < self._last_tier:
            self._last_tier = proposed_tier
            self._tier_hold_counter = 0
            self._tier_transition_count += 1
            return proposed_tier

        return self._last_tier

    def _calculate_safe_dose(
        self,
        original: float,
        tier1: TierResult,
        tier2: TierResult,
        tier3: TierResult,
        active_tier: SafetyTier,
    ) -> float:
        """Calculate the final safe dose respecting tier priority.

        The active tier's dose takes precedence, but we also enforce
        that the dose never exceeds any tier's safe dose.
        """
        if active_tier == SafetyTier.TIER_1_REFLEX:
            return tier1.dose

        # Return the minimum safe dose from all tiers
        safe = original
        for result in [tier1, tier2, tier3]:
            if not result.safe:
                safe = min(safe, result.dose)

        return max(0.0, safe)

    def get_tier_transition_count(self) -> int:
        """Return number of tier transitions (for hysteresis testing)."""
        return self._tier_transition_count

    def reset_hysteresis(self) -> None:
        """Reset hysteresis state."""
        self._last_tier = SafetyTier.TIER_4_NOMINAL
        self._tier_hold_counter = 0
        self._tier_transition_count = 0

    def check_stl_satisfaction(
        self,
        glucose_trace: np.ndarray,
        timestep_minutes: float = 5.0,
    ) -> dict:
        """Check STL specification satisfaction on a complete glucose trace.

        Args:
            glucose_trace: Array of glucose values over time
            timestep_minutes: Time between readings in minutes

        Returns:
            Dict with satisfaction status and robustness for each spec
        """
        results = {}

        # φ₁: □[0,T](G ≥ 54) — no severe hypoglycemia
        min_glucose = np.min(glucose_trace)
        results["no_severe_hypo"] = {
            "satisfied": min_glucose >= self.thresholds.glucose_danger_low,
            "robustness": min_glucose - self.thresholds.glucose_danger_low,
            "min_value": min_glucose,
        }

        # φ₂: □[0,T](G ≤ 400) — no extreme hyperglycemia
        max_glucose = np.max(glucose_trace)
        results["no_extreme_hyper"] = {
            "satisfied": max_glucose <= 400.0,
            "robustness": 400.0 - max_glucose,
            "max_value": max_glucose,
        }

        # φ₃: G<70 → ◇[0,30min](G ≥ 80) — hypo recovery
        recovery_window = int(30.0 / timestep_minutes)
        hypo_indices = np.where(glucose_trace < self.thresholds.glucose_warning_low)[0]
        all_recovered = True
        worst_recovery_time = 0.0

        for idx in hypo_indices:
            window_end = min(len(glucose_trace), idx + recovery_window)
            window = glucose_trace[idx:window_end]
            recovered = np.any(window >= 80.0)
            if not recovered:
                all_recovered = False
            else:
                recovery_idx = np.argmax(window >= 80.0)
                recovery_time = recovery_idx * timestep_minutes
                worst_recovery_time = max(worst_recovery_time, recovery_time)

        results["hypo_recovery"] = {
            "satisfied": all_recovered,
            "hypo_episodes": len(hypo_indices),
            "worst_recovery_minutes": worst_recovery_time,
        }

        return results
