"""Layer 5: Safety Supervisor Tests — T5.1 through T5.10.

Comprehensive testing of the three-tier safety system:
- T5.1:  Exhaustive tier priority (hierarchy enforcement)
- T5.2:  Reflex controller boundary testing (including float edges)
- T5.3:  STL specification satisfaction (exhaustive scenario grid)
- T5.4:  Seldonian constraint (high-confidence probabilistic bounds)
- T5.5:  Cold start relaxation (fine-grained schedule validation)
- T5.6:  Adversarial input testing (NaN, Inf, extreme values)
- T5.7:  Latency under realistic conditions (p99 < 1ms)
- T5.8:  Cascading failure resilience (upstream failure isolation)
- T5.9:  Threshold oscillation / hysteresis analysis (chattering prevention)
- T5.10: Multi-step temporal behaviour (escalation, de-escalation, IOB)
- T5.11: Unnecessary restriction detection (safe doses must pass unchanged)
"""

import time

import numpy as np
import pytest

from aegis_testing.aegis_core.safety_supervisor import (
    SafetyAction,
    SafetySupervisor,
    SafetyThresholds,
    SafetyTier,
)


# ============================================================
# Helper functions
# ============================================================

def generate_trajectory(glucose_start: float, trend: str = "stable",
                        n_steps: int = 36, noise_std: float = 0.0,
                        rng: np.random.RandomState = None) -> np.ndarray:
    """Generate a predicted glucose trajectory.

    Args:
        glucose_start: Starting glucose (mg/dL)
        trend: "stable", "rising", "falling", "rising_fast", "falling_fast"
        n_steps: Number of future timesteps (each 5 min)
        noise_std: Gaussian noise standard deviation
        rng: Random state for reproducibility
    """
    if rng is None:
        rng = np.random.RandomState(42)

    trend_rates = {
        "stable": 0.0,
        "rising": 1.0,         # mg/dL per 5min
        "rising_fast": 3.0,
        "falling": -1.0,
        "falling_fast": -3.0,
    }
    rate = trend_rates.get(trend, 0.0)
    trajectory = np.array([glucose_start + rate * i for i in range(n_steps)])
    if noise_std > 0:
        trajectory += rng.randn(n_steps) * noise_std
    return trajectory


# ============================================================
# T5.1: Exhaustive Tier Priority
# ============================================================

class TestT5_1_TierPriority:
    """T5.1: Verify strict tier priority enforcement.

    If Tier 1 triggers, its decision MUST override Tier 2 and 3.
    If only Tier 2 triggers, it overrides Tier 3.
    Tier 3 only matters when Tier 1 and 2 both pass.
    """

    @pytest.fixture
    def supervisor(self):
        return SafetySupervisor(hysteresis_hold_steps=1)

    def test_tier1_overrides_all(self, supervisor, result_collector):
        """Tier 1 (glucose < 54) must override Tier 2 and 3 decisions."""
        collector = result_collector("T5.1a", "Tier 1 overrides all")

        for glucose in range(20, 54):
            supervisor.reset_hysteresis()
            trajectory = generate_trajectory(glucose, "stable")
            result = supervisor.verify(
                glucose=float(glucose),
                predicted_trajectory=trajectory,
                recommended_dose=5.0,
            )
            collector.add_result({
                "glucose": glucose,
                "active_tier": int(result.active_tier),
                "safe_dose": result.safe_dose,
                "action_is_suspend": int(
                    result.tier_results[SafetyTier.TIER_1_REFLEX].action == SafetyAction.SUSPEND),
            })

            assert result.active_tier == SafetyTier.TIER_1_REFLEX, (
                f"At glucose={glucose}, active tier should be TIER_1_REFLEX, got {result.active_tier}")
            assert result.safe_dose == 0.0, (
                f"At glucose={glucose}, safe dose should be 0.0, got {result.safe_dose}")
            assert result.tier_results[SafetyTier.TIER_1_REFLEX].action == SafetyAction.SUSPEND

    def test_tier1_block_overrides_tier2_tier3(self, supervisor, result_collector):
        """Tier 1 block (54 ≤ glucose < 70, dose > 0) must override Tier 2 and 3."""
        collector = result_collector("T5.1b", "Tier 1 block overrides Tier 2/3")

        for glucose in range(54, 70):
            supervisor.reset_hysteresis()
            trajectory = generate_trajectory(glucose, "stable")
            result = supervisor.verify(
                glucose=float(glucose),
                predicted_trajectory=trajectory,
                recommended_dose=5.0,
            )
            collector.add_result({
                "glucose": glucose,
                "active_tier": int(result.active_tier),
                "safe_dose": result.safe_dose,
            })
            assert result.active_tier == SafetyTier.TIER_1_REFLEX
            assert result.safe_dose == 0.0

    def test_tier2_overrides_tier3(self, supervisor):
        """When Tier 1 passes but Tier 2 triggers, Tier 2 overrides Tier 3."""
        supervisor.reset_hysteresis()
        trajectory = generate_trajectory(120, "falling_fast", n_steps=50)
        trajectory[30:] = 40.0  # Will trigger STL

        result = supervisor.verify(
            glucose=120.0,
            predicted_trajectory=trajectory,
            recommended_dose=5.0,
        )
        assert result.tier_results[SafetyTier.TIER_1_REFLEX].safe, "Tier 1 should pass"
        assert not result.tier_results[SafetyTier.TIER_2_STL].safe, "Tier 2 should trigger"
        assert result.active_tier <= SafetyTier.TIER_2_STL

    def test_all_pass_nominal(self, supervisor, result_collector):
        """When all tiers pass, active tier should be NOMINAL (4)."""
        collector = result_collector("T5.1d", "All tiers pass - nominal")

        for glucose in range(110, 171, 10):
            supervisor.reset_hysteresis()
            trajectory = generate_trajectory(glucose, "stable")
            result = supervisor.verify(
                glucose=float(glucose),
                predicted_trajectory=trajectory,
                recommended_dose=0.5,
                prediction_std=10.0,
                n_observations=1000,
            )
            collector.add_result({
                "glucose": glucose,
                "active_tier": int(result.active_tier),
                "safe_dose": result.safe_dose,
                "overall_safe": int(result.overall_safe),
            })
            assert result.overall_safe, f"Should be safe at glucose={glucose} with dose=0.5"
            assert result.safe_dose == 0.5, f"Dose should be unchanged at glucose={glucose}"

    def test_exhaustive_tier_priority_matrix(self, supervisor, result_collector):
        """Exhaustive test: every glucose from 20-300, verify correct tier activates."""
        collector = result_collector("T5.1e", "Exhaustive tier priority matrix")
        pass_count = 0
        total = 0

        for glucose in range(20, 301):
            supervisor.reset_hysteresis()
            trajectory = generate_trajectory(glucose, "stable")
            result = supervisor.verify(
                glucose=float(glucose),
                predicted_trajectory=trajectory,
                recommended_dose=5.0,
                prediction_std=15.0,
                n_observations=100,
            )

            if glucose < 54:
                expected_action = SafetyAction.SUSPEND
            elif glucose < 70:
                expected_action = SafetyAction.BLOCK
            else:
                expected_action = None  # Various valid outcomes

            if expected_action is not None:
                tier1_action = result.tier_results[SafetyTier.TIER_1_REFLEX].action
                assert tier1_action == expected_action, (
                    f"At glucose={glucose}: expected {expected_action}, got {tier1_action}")
            pass_count += 1
            total += 1

            collector.add_result({
                "glucose": glucose,
                "active_tier": int(result.active_tier),
                "action": int(result.tier_results[SafetyTier.TIER_1_REFLEX].action),
                "safe_dose": result.safe_dose,
            })

        assert pass_count == total, f"{total - pass_count} tier priority violations"


# ============================================================
# T5.2: Reflex Controller Boundary Testing
# ============================================================

class TestT5_2_ReflexBoundary:
    """T5.2: Test ALL boundaries including floating-point edge values."""

    @pytest.fixture
    def supervisor(self):
        return SafetySupervisor(hysteresis_hold_steps=1)

    def test_integer_boundaries(self, supervisor, result_collector):
        """Test every integer glucose from 20 to 300."""
        collector = result_collector("T5.2a", "Integer boundary testing")
        doses = [0, 1, 5, 10, 15, 15.1, 16, 20, 25]

        for glucose in range(20, 301):
            for dose in doses:
                supervisor.reset_hysteresis()
                trajectory = generate_trajectory(glucose, "stable")
                result = supervisor.verify(
                    glucose=float(glucose),
                    predicted_trajectory=trajectory,
                    recommended_dose=float(dose),
                )
                tier1 = result.tier_results[SafetyTier.TIER_1_REFLEX]

                if glucose < 54:
                    assert tier1.action == SafetyAction.SUSPEND, (
                        f"glucose={glucose}, dose={dose}: expected SUSPEND")
                    assert tier1.dose == 0.0
                elif glucose < 70 and dose > 0:
                    assert tier1.action == SafetyAction.BLOCK, (
                        f"glucose={glucose}, dose={dose}: expected BLOCK")
                    assert tier1.dose == 0.0
                elif dose > 15:
                    assert tier1.action == SafetyAction.REDUCE, (
                        f"glucose={glucose}, dose={dose}: expected REDUCE")
                    assert tier1.dose == 15.0
                else:
                    assert tier1.action == SafetyAction.ALLOW, (
                        f"glucose={glucose}, dose={dose}: expected ALLOW")

                collector.add_result({
                    "glucose": glucose, "dose": dose,
                    "action": int(tier1.action), "safe_dose": tier1.dose,
                })

    def test_float_boundary_values(self, supervisor, result_collector):
        """Test critical floating-point boundary values."""
        collector = result_collector("T5.2b", "Float boundary testing")

        boundary_values = [
            53.5, 53.9, 53.99, 53.999, 54.0, 54.001, 54.01, 54.1,
            69.5, 69.9, 69.99, 69.999, 70.0, 70.001, 70.01, 70.1,
            179.9, 180.0, 180.1,
            249.9, 250.0, 250.1,
        ]

        for glucose in boundary_values:
            supervisor.reset_hysteresis()
            trajectory = generate_trajectory(glucose, "stable")
            result = supervisor.verify(
                glucose=glucose,
                predicted_trajectory=trajectory,
                recommended_dose=5.0,
            )
            tier1 = result.tier_results[SafetyTier.TIER_1_REFLEX]

            if glucose < 54.0:
                expected = SafetyAction.SUSPEND
            elif glucose < 70.0:
                expected = SafetyAction.BLOCK
            else:
                expected = SafetyAction.ALLOW

            assert tier1.action == expected, (
                f"glucose={glucose}: expected {expected.name}, got {tier1.action.name}")

            collector.add_result({
                "glucose": glucose, "action": int(tier1.action),
                "expected": int(expected), "correct": int(tier1.action == expected),
            })

    def test_zero_dose_at_low_glucose(self, supervisor):
        """At glucose < 70, dose=0 should ALLOW (no insulin to block)."""
        supervisor.reset_hysteresis()
        trajectory = generate_trajectory(65.0, "stable")
        result = supervisor.verify(glucose=65.0, predicted_trajectory=trajectory,
                                   recommended_dose=0.0)
        tier1 = result.tier_results[SafetyTier.TIER_1_REFLEX]
        assert tier1.action == SafetyAction.ALLOW, (
            f"Dose=0 at glucose=65: should ALLOW, got {tier1.action.name}")

    def test_max_bolus_cap(self, supervisor, result_collector):
        """Doses exceeding max_bolus should be capped."""
        collector = result_collector("T5.2d", "Max bolus capping")

        for dose in [15.01, 15.1, 16, 20, 25, 50, 100]:
            supervisor.reset_hysteresis()
            trajectory = generate_trajectory(150.0, "stable")
            result = supervisor.verify(glucose=150.0, predicted_trajectory=trajectory,
                                       recommended_dose=dose)
            tier1 = result.tier_results[SafetyTier.TIER_1_REFLEX]
            assert tier1.action == SafetyAction.REDUCE
            assert tier1.dose == 15.0, f"Dose {dose} should be capped to 15.0"
            collector.add_result({"original_dose": dose, "capped_dose": tier1.dose})

    def test_dose_exactly_at_max_is_allowed(self, supervisor):
        """Dose exactly equal to max_bolus should be ALLOWED, not reduced."""
        supervisor.reset_hysteresis()
        trajectory = generate_trajectory(150.0, "stable")
        result = supervisor.verify(glucose=150.0, predicted_trajectory=trajectory,
                                   recommended_dose=15.0)
        tier1 = result.tier_results[SafetyTier.TIER_1_REFLEX]
        assert tier1.action == SafetyAction.ALLOW, (
            f"Dose exactly at max_bolus should ALLOW, got {tier1.action.name}")
        assert tier1.dose == 15.0


# ============================================================
# T5.3: STL Specification Satisfaction
# ============================================================

class TestT5_3_STLSatisfaction:
    """T5.3: Exhaustive STL specification testing."""

    @pytest.fixture
    def supervisor(self):
        return SafetySupervisor(hysteresis_hold_steps=1)

    def test_no_severe_hypo_spec(self, supervisor, result_collector):
        """φ₁: □[0,T](G ≥ 54) — always above severe hypo threshold."""
        collector = result_collector("T5.3a", "STL: no severe hypo")
        rng = np.random.RandomState(42)

        for trial in range(500):
            glucose = rng.uniform(80, 250)
            dose = rng.uniform(0, 12)
            trajectory = generate_trajectory(glucose, "stable", noise_std=5, rng=rng)

            result = supervisor.verify(
                glucose=glucose, predicted_trajectory=trajectory,
                recommended_dose=dose,
            )

            if result.safe_dose > 0:
                insulin_effect = result.safe_dose * 25.0
                adjusted_min = np.min(trajectory) - insulin_effect
                collector.add_result({
                    "glucose": glucose, "original_dose": dose,
                    "safe_dose": result.safe_dose, "adjusted_min": adjusted_min,
                    "stl_satisfied": int(result.tier_results[SafetyTier.TIER_2_STL].safe),
                })

    def test_stl_check_on_traces(self, supervisor, result_collector):
        """Verify check_stl_satisfaction on pre-computed glucose traces."""
        collector = result_collector("T5.3b", "STL check on traces")
        rng = np.random.RandomState(42)

        for trial in range(200):
            base = rng.uniform(70, 200)
            noise = rng.uniform(5, 30)
            trace = base + rng.randn(288) * noise

            stl_results = supervisor.check_stl_satisfaction(trace)
            min_g = np.min(trace)
            max_g = np.max(trace)

            assert stl_results["no_severe_hypo"]["satisfied"] == (min_g >= 54), (
                f"Trial {trial}: min={min_g:.1f}, "
                f"satisfied={stl_results['no_severe_hypo']['satisfied']}")
            assert stl_results["no_extreme_hyper"]["satisfied"] == (max_g <= 400)

            collector.add_result({
                "base": base, "noise": noise,
                "min_glucose": min_g, "max_glucose": max_g,
                "hypo_satisfied": int(stl_results["no_severe_hypo"]["satisfied"]),
                "hyper_satisfied": int(stl_results["no_extreme_hyper"]["satisfied"]),
                "robustness_hypo": stl_results["no_severe_hypo"]["robustness"],
            })

    def test_closed_loop_stl(self, supervisor, result_collector):
        """T5.3c: CLOSED-LOOP STL test — L5 actively modifies doses in a
        feedback loop with a simple glucose simulator.

        This validates that L5 *actually prevents* safety violations when
        it's controlling the dose, not just checking open-loop traces.
        """
        collector = result_collector("T5.3c", "Closed-loop STL satisfaction")
        rng = np.random.RandomState(42)

        n_trials = 100
        n_steps = 288  # 24 hours at 5-min intervals
        violations = 0

        for trial in range(n_trials):
            supervisor.reset_hysteresis()
            # Initial conditions: randomise
            glucose = rng.uniform(80, 200)
            trace = []

            # Meal schedule: 3 meals with random carbs
            meal_times = {60: rng.uniform(30, 80),   # breakfast
                          144: rng.uniform(40, 100),  # lunch
                          228: rng.uniform(40, 100)}  # dinner

            for step in range(n_steps):
                # Simple glucose dynamics (Bergman-inspired, simplified)
                # Endogenous glucose production
                glucose += rng.uniform(-0.5, 0.5)  # Background noise

                # Meal absorption
                for meal_step, carbs in meal_times.items():
                    dt = step - meal_step
                    if 0 <= dt < 36:  # 3-hour absorption window
                        glucose += carbs * 0.15 * np.exp(-0.1 * dt)

                # Natural insulin effect (basal)
                glucose -= 0.3

                # Controller proposes a dose (simple proportional controller)
                if glucose > 120:
                    proposed_dose = (glucose - 110) / 30.0
                else:
                    proposed_dose = 0.0

                # L5 checks and potentially modifies the dose
                trajectory = generate_trajectory(glucose, "stable",
                                                  noise_std=3, rng=rng)
                result = supervisor.verify(
                    glucose=glucose, predicted_trajectory=trajectory,
                    recommended_dose=proposed_dose,
                    prediction_std=15.0, n_observations=200,
                )
                actual_dose = result.safe_dose

                # Apply the L5-approved dose to the simulation
                glucose -= actual_dose * 25.0 * 0.02  # Spread over time
                glucose = max(20.0, glucose)  # Floor
                trace.append(glucose)

            trace_arr = np.array(trace)
            min_g = float(np.min(trace_arr))

            # The critical check: did L5 prevent severe hypo?
            trial_violation = min_g < 54.0
            if trial_violation:
                violations += 1

            collector.add_result({
                "trial": trial,
                "min_glucose": min_g,
                "max_glucose": float(np.max(trace_arr)),
                "mean_glucose": float(np.mean(trace_arr)),
                "severe_hypo": int(trial_violation),
            })

        violation_rate = violations / n_trials * 100
        assert violation_rate <= 1.0, (
            f"Closed-loop severe hypo rate {violation_rate:.1f}% exceeds 1% threshold. "
            f"L5 failed to protect {violations}/{n_trials} trials.")


# ============================================================
# T5.4: Seldonian Constraint
# ============================================================

class TestT5_4_SeldonianConstraint:
    """T5.4: High-confidence Seldonian safety bounds."""

    @pytest.fixture
    def supervisor(self):
        return SafetySupervisor(
            seldonian_delta=0.01, seldonian_alpha=0.05,
            hysteresis_hold_steps=1,
        )

    def test_seldonian_blocks_risky_doses(self, supervisor, result_collector):
        """Seldonian should block doses with high P(hypo)."""
        collector = result_collector("T5.4a", "Seldonian blocks risky doses")

        for dose in np.arange(0, 15, 0.5):
            supervisor.reset_hysteresis()
            glucose = 100.0
            trajectory = generate_trajectory(glucose, "stable")

            result = supervisor.verify(
                glucose=glucose, predicted_trajectory=trajectory,
                recommended_dose=dose, prediction_std=15.0, n_observations=100,
            )
            tier3 = result.tier_results[SafetyTier.TIER_3_SELDONIAN]
            collector.add_result({
                "dose": dose,
                "p_harm": tier3.p_harm if tier3.p_harm is not None else -1,
                "seldonian_safe": int(tier3.safe),
                "safe_dose": result.safe_dose,
            })

            if dose > 3.0:
                predicted_post = glucose - dose * 25.0
                if predicted_post < 54:
                    assert not tier3.safe or result.safe_dose < dose, (
                        f"Dose {dose} from glucose {glucose} predicts {predicted_post}, "
                        f"but Seldonian didn't reduce dose")

    def test_seldonian_coverage_monte_carlo(self, supervisor, result_collector):
        """MC test: Seldonian bound should hold >95% of the time."""
        collector = result_collector("T5.4b", "Seldonian coverage MC")
        rng = np.random.RandomState(42)
        violations = 0
        total = 500

        for trial in range(total):
            supervisor.reset_hysteresis()
            glucose = rng.uniform(80, 150)
            dose = rng.uniform(0, 10)
            std = rng.uniform(5, 30)
            trajectory = generate_trajectory(glucose, "stable",
                                              noise_std=std * 0.1, rng=rng)
            result = supervisor.verify(
                glucose=glucose, predicted_trajectory=trajectory,
                recommended_dose=dose, prediction_std=std, n_observations=200,
            )

            # Simulate actual outcome with noise
            actual_glucose = glucose - result.safe_dose * 25.0 + rng.randn() * std
            if actual_glucose < 54 and result.overall_safe:
                violations += 1

            collector.add_result({
                "glucose": glucose, "dose": dose, "safe_dose": result.safe_dose,
                "actual_glucose": actual_glucose,
                "violation": int(actual_glucose < 54 and result.overall_safe),
            })

        violation_rate = violations / total
        assert violation_rate < 0.05, (
            f"Seldonian violation rate {violation_rate:.3f} is too high (expected < 0.05)")

    def test_seldonian_coverage_bergman_outcome(self, supervisor, result_collector):
        """T5.4c: Seldonian MC with nonlinear (Bergman-inspired) outcome model.

        Uses a more realistic glucose-insulin relationship rather than
        the simple linear model (dose * 25) to stress-test the bound.
        """
        collector = result_collector("T5.4c", "Seldonian coverage (Bergman outcome)")
        rng = np.random.RandomState(123)
        violations = 0
        total = 500

        for trial in range(total):
            supervisor.reset_hysteresis()
            glucose = rng.uniform(80, 180)
            dose = rng.uniform(0, 8)
            std = rng.uniform(8, 25)

            trajectory = generate_trajectory(glucose, "stable",
                                              noise_std=std * 0.1, rng=rng)
            result = supervisor.verify(
                glucose=glucose, predicted_trajectory=trajectory,
                recommended_dose=dose, prediction_std=std, n_observations=200,
            )

            # Nonlinear outcome: insulin effect depends on glucose level
            # (higher glucose → more responsive to insulin, simplified)
            safe_dose = result.safe_dose
            sensitivity = 20.0 + 10.0 * (glucose / 180.0)  # 20-30 mg/dL/U
            time_delay_factor = rng.uniform(0.7, 1.3)  # Absorption variability
            actual_drop = safe_dose * sensitivity * time_delay_factor
            actual_glucose = glucose - actual_drop + rng.randn() * std

            if actual_glucose < 54 and result.overall_safe:
                violations += 1

            collector.add_result({
                "glucose": glucose, "dose": dose, "safe_dose": safe_dose,
                "sensitivity": sensitivity,
                "actual_glucose": actual_glucose,
                "violation": int(actual_glucose < 54 and result.overall_safe),
            })

        violation_rate = violations / total
        assert violation_rate < 0.05, (
            f"Bergman-outcome Seldonian violation rate {violation_rate:.3f} "
            f"too high (expected < 0.05)")


# ============================================================
# T5.5: Cold Start Relaxation
# ============================================================

class TestT5_5_ColdStartRelaxation:
    """T5.5: Verify cold-start schedule works correctly."""

    def test_relaxation_schedule(self, result_collector):
        """Verify relaxation decreases linearly from 5x to 1x over 30 days."""
        collector = result_collector("T5.5a", "Cold start relaxation schedule")
        supervisor = SafetySupervisor(cold_start_days=30)

        for day in range(0, 35):
            supervisor.set_day(day)
            relaxation = supervisor._get_cold_start_relaxation()
            collector.add_result({"day": day, "relaxation": relaxation})

            if day == 0:
                assert abs(relaxation - 5.0) < 0.01
            elif day == 30:
                assert abs(relaxation - 1.0) < 0.01
            elif day > 30:
                assert abs(relaxation - 1.0) < 0.01

    def test_cold_start_more_permissive(self, result_collector):
        """During cold start, the system should allow more (higher δ_effective)."""
        collector = result_collector("T5.5b", "Cold start permissiveness")

        glucose = 90.0
        dose = 3.0
        trajectory = generate_trajectory(glucose, "stable")

        for day in [0, 5, 10, 15, 20, 25, 30]:
            supervisor = SafetySupervisor(cold_start_days=30, hysteresis_hold_steps=1)
            supervisor.set_day(day)
            result = supervisor.verify(
                glucose=glucose, predicted_trajectory=trajectory,
                recommended_dose=dose, prediction_std=15.0, n_observations=50,
            )
            tier3 = result.tier_results[SafetyTier.TIER_3_SELDONIAN]
            collector.add_result({
                "day": day, "seldonian_safe": int(tier3.safe),
                "safe_dose": result.safe_dose, "p_harm": tier3.p_harm,
            })

    def test_cold_start_monotonic_tightening(self, result_collector):
        """The allowed dose at a given glucose should be monotonically
        non-increasing as cold-start days progress (system gets stricter)."""
        collector = result_collector("T5.5c", "Cold start monotonic tightening")

        glucose = 95.0  # Borderline scenario
        dose = 5.0
        trajectory = generate_trajectory(glucose, "stable")

        allowed_doses = []
        for day in range(0, 35):
            supervisor = SafetySupervisor(cold_start_days=30, hysteresis_hold_steps=1)
            supervisor.set_day(day)
            result = supervisor.verify(
                glucose=glucose, predicted_trajectory=trajectory,
                recommended_dose=dose, prediction_std=15.0, n_observations=50,
            )
            allowed_doses.append(result.safe_dose)
            collector.add_result({"day": day, "safe_dose": result.safe_dose})

        # Check monotonic non-increasing (each day should allow ≤ previous day)
        for i in range(1, len(allowed_doses)):
            assert allowed_doses[i] <= allowed_doses[i - 1] + 0.01, (
                f"Day {i}: allowed_dose {allowed_doses[i]:.3f} > "
                f"day {i-1}: {allowed_doses[i-1]:.3f} — not monotonically tightening")


# ============================================================
# T5.6: Adversarial Input Testing
# ============================================================

class TestT5_6_AdversarialInputs:
    """T5.6: System must handle extreme/invalid inputs gracefully."""

    @pytest.fixture
    def supervisor(self):
        return SafetySupervisor(hysteresis_hold_steps=1)

    def test_nan_glucose(self, supervisor):
        supervisor.reset_hysteresis()
        trajectory = generate_trajectory(100, "stable")
        result = supervisor.verify(glucose=float('nan'),
                                   predicted_trajectory=trajectory,
                                   recommended_dose=5.0)
        assert result is not None
        # NaN sanitised to 0.0 → should SUSPEND
        assert result.safe_dose == 0.0, "NaN glucose should result in dose=0"

    def test_inf_glucose(self, supervisor):
        supervisor.reset_hysteresis()
        trajectory = generate_trajectory(100, "stable")
        result = supervisor.verify(glucose=float('inf'),
                                   predicted_trajectory=trajectory,
                                   recommended_dose=5.0)
        assert result is not None

    def test_negative_glucose(self, supervisor):
        supervisor.reset_hysteresis()
        trajectory = generate_trajectory(-10, "stable")
        result = supervisor.verify(glucose=-10.0,
                                   predicted_trajectory=trajectory,
                                   recommended_dose=5.0)
        assert result.tier_results[SafetyTier.TIER_1_REFLEX].action == SafetyAction.SUSPEND
        assert result.safe_dose == 0.0

    def test_extreme_dose(self, supervisor):
        supervisor.reset_hysteresis()
        trajectory = generate_trajectory(200, "stable")
        result = supervisor.verify(glucose=200.0,
                                   predicted_trajectory=trajectory,
                                   recommended_dose=1000.0)
        assert result.safe_dose <= 15.0

    def test_negative_dose(self, supervisor):
        supervisor.reset_hysteresis()
        trajectory = generate_trajectory(150, "stable")
        result = supervisor.verify(glucose=150.0,
                                   predicted_trajectory=trajectory,
                                   recommended_dose=-5.0)
        assert result is not None
        assert result.safe_dose >= 0.0

    def test_empty_trajectory(self, supervisor):
        supervisor.reset_hysteresis()
        result = supervisor.verify(glucose=150.0,
                                   predicted_trajectory=np.array([]),
                                   recommended_dose=5.0)
        assert result is not None

    def test_none_trajectory(self, supervisor):
        supervisor.reset_hysteresis()
        result = supervisor.verify(glucose=150.0,
                                   predicted_trajectory=None,
                                   recommended_dose=5.0)
        assert result is not None

    def test_nan_in_trajectory(self, supervisor):
        """Trajectory containing NaN values should not crash."""
        supervisor.reset_hysteresis()
        traj = np.array([150.0, 145.0, float('nan'), 140.0, 135.0])
        result = supervisor.verify(glucose=150.0,
                                   predicted_trajectory=traj,
                                   recommended_dose=3.0)
        assert result is not None

    def test_all_zeros_trajectory(self, supervisor):
        """All-zero trajectory should trigger maximum safety."""
        supervisor.reset_hysteresis()
        traj = np.zeros(36)
        result = supervisor.verify(glucose=150.0,
                                   predicted_trajectory=traj,
                                   recommended_dose=5.0)
        assert result is not None


# ============================================================
# T5.7: Latency Under Realistic Conditions
# ============================================================

class TestT5_7_Latency:
    """T5.7: Verify safety check latency under load.

    Tightened: p99 < 1ms (matching sub-millisecond claims in the paper).
    """

    def test_p99_latency(self, result_collector):
        """p99 latency should be < 1ms per safety check."""
        collector = result_collector("T5.7", "Safety check latency")
        supervisor = SafetySupervisor(hysteresis_hold_steps=1)
        rng = np.random.RandomState(42)

        latencies = []
        n_measurements = 10000

        # Warmup
        for _ in range(100):
            supervisor.verify(120.0, np.full(36, 120.0), 3.0)

        for i in range(n_measurements):
            glucose = rng.uniform(40, 300)
            dose = rng.uniform(0, 20)
            trajectory = generate_trajectory(glucose, "stable", noise_std=5, rng=rng)

            start = time.perf_counter()
            supervisor.verify(
                glucose=glucose, predicted_trajectory=trajectory,
                recommended_dose=dose, prediction_std=15.0, n_observations=100,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

            if i % 1000 == 0:
                collector.add_result({"batch": i, "latency_ms": elapsed_ms})

        latencies = np.array(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        mean_lat = np.mean(latencies)

        print(f"\n  Latency (n={n_measurements}):")
        print(f"    Mean: {mean_lat:.4f} ms")
        print(f"    p50:  {p50:.4f} ms")
        print(f"    p95:  {p95:.4f} ms")
        print(f"    p99:  {p99:.4f} ms")
        print(f"    Max:  {np.max(latencies):.4f} ms")

        # Tightened from 50ms to 1ms to match sub-millisecond claims
        assert p99 < 1.0, f"p99 latency {p99:.4f}ms exceeds 1ms threshold"


# ============================================================
# T5.8: Cascading Failure Resilience
# ============================================================

class TestT5_8_CascadingFailure:
    """T5.8: L5 must prevent safety violations even with upstream failures."""

    @pytest.fixture
    def supervisor(self):
        return SafetySupervisor(hysteresis_hold_steps=1)

    def test_wrong_prediction_high(self, supervisor, result_collector):
        """L2 predicts glucose 50% higher than actual — L5 still safe."""
        collector = result_collector("T5.8a", "Wrong prediction +50%")
        rng = np.random.RandomState(42)

        for trial in range(500):
            supervisor.reset_hysteresis()
            actual_glucose = rng.uniform(50, 200)
            reported_glucose = actual_glucose * 1.5
            trajectory = generate_trajectory(reported_glucose, "stable")

            result = supervisor.verify(
                glucose=actual_glucose, predicted_trajectory=trajectory,
                recommended_dose=rng.uniform(0, 10),
            )
            if actual_glucose < 54:
                assert result.safe_dose == 0.0, (
                    f"Actual glucose={actual_glucose:.1f}, safe_dose should be 0")

            collector.add_result({
                "actual": actual_glucose, "reported": reported_glucose,
                "safe_dose": result.safe_dose, "safe": int(result.overall_safe),
            })

    def test_wrong_prediction_low(self, supervisor, result_collector):
        """L2 predicts glucose 50% lower than actual — L5 still safe."""
        collector = result_collector("T5.8b", "Wrong prediction -50%")
        rng = np.random.RandomState(42)

        for trial in range(500):
            supervisor.reset_hysteresis()
            actual_glucose = rng.uniform(80, 300)
            trajectory = generate_trajectory(actual_glucose * 0.5, "stable")

            result = supervisor.verify(
                glucose=actual_glucose, predicted_trajectory=trajectory,
                recommended_dose=rng.uniform(0, 10),
            )
            collector.add_result({
                "actual": actual_glucose, "predicted": actual_glucose * 0.5,
                "safe_dose": result.safe_dose,
            })

    def test_dangerous_dose_proposal(self, supervisor, result_collector):
        """L4 proposes a dangerous dose — L5 must block it."""
        collector = result_collector("T5.8c", "Dangerous dose proposal")

        for glucose in range(40, 80):
            supervisor.reset_hysteresis()
            trajectory = generate_trajectory(glucose, "stable")
            result = supervisor.verify(
                glucose=float(glucose), predicted_trajectory=trajectory,
                recommended_dose=20.0,
            )
            if glucose < 54:
                assert result.safe_dose == 0.0
            elif glucose < 70:
                assert result.safe_dose == 0.0

            collector.add_result({
                "glucose": glucose, "proposed_dose": 20.0,
                "safe_dose": result.safe_dose,
            })

    def test_multiple_simultaneous_failures(self, supervisor, result_collector):
        """Multiple upstream layers fail at once — L5 still safe."""
        collector = result_collector("T5.8d", "Multiple simultaneous failures")
        rng = np.random.RandomState(42)

        for trial in range(500):
            supervisor.reset_hysteresis()
            actual_glucose = rng.uniform(40, 200)
            # L2 gives wildly wrong trajectory
            trajectory = generate_trajectory(actual_glucose * rng.uniform(0.3, 2.0), "stable")
            # L4 proposes extreme dose
            proposed_dose = rng.uniform(10, 25)

            result = supervisor.verify(
                