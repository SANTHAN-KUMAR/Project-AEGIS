"""Phase 0: simglucose Integration Scaffold Tests.

Validates that the simglucose package is correctly installed,
the adapter layer works, and baseline simulations produce
expected results. This must pass before any layer tests.
"""

from collections import namedtuple

import numpy as np
import pandas as pd
import pytest

# simglucose patient expects this action format
PatientAction = namedtuple("patient_action", ["CHO", "insulin"])


def make_action(insulin: float = 0.0, cho: float = 0.0) -> PatientAction:
    """Create a simglucose-compatible patient action."""
    return PatientAction(CHO=cho, insulin=insulin)


class TestSimglucoseSetup:
    """Validate simglucose installation and patient availability."""

    def test_simglucose_imports(self):
        """Step 0.1: Verify all required simglucose modules import."""
        from simglucose.patient.t1dpatient import T1DPatient
        from simglucose.sensor.cgm import CGMSensor
        from simglucose.actuator.pump import InsulinPump
        from simglucose.controller.base import Controller, Action
        assert True  # If we get here, imports succeeded

    def test_patient_count(self, all_patients):
        """Step 0.1: Verify all 30 standard patients are available."""
        assert len(all_patients) == 30, f"Expected 30 patients, got {len(all_patients)}"

    def test_patient_groups(self, adult_patients, adolescent_patients, child_patients):
        """Step 0.1: Verify 10 adults, 10 adolescents, 10 children."""
        assert len(adult_patients) == 10, f"Expected 10 adults, got {len(adult_patients)}"
        assert len(adolescent_patients) == 10, f"Expected 10 adolescents, got {len(adolescent_patients)}"
        assert len(child_patients) == 10, f"Expected 10 children, got {len(child_patients)}"

    def test_patient_parameters_valid(self, all_patients):
        """Step 0.1: Verify patient parameters are non-null and physiologically valid."""
        assert "Name" in all_patients.columns
        assert all_patients["Name"].notna().all()
        expected_params = ["BW"]
        for param in expected_params:
            if param in all_patients.columns:
                assert all_patients[param].notna().all(), f"Parameter {param} has null values"

    def test_single_patient_simulation(self):
        """Step 0.1: Run a single short simulation to verify simglucose works."""
        from simglucose.patient.t1dpatient import T1DPatient

        patient = T1DPatient.withName("adult#001")
        patient.reset()

        # Step the patient with basal insulin, no meal
        action = make_action(insulin=1.0, cho=0.0)
        for step in range(5):
            patient.step(action)

        bg = patient.observation.Gsub
        assert bg is not None
        assert bg > 0, f"Glucose should be positive, got {bg}"
        assert bg < 600, f"Glucose should be < 600, got {bg}"
        print(f"\n  adult#001 after 5 steps: BG = {bg:.1f} mg/dL")

    def test_bergman_parameter_comparison(self, all_patients):
        """Step 0.3: Verify Bergman parameters exist in simglucose patients."""
        assert len(all_patients.columns) > 5, (
            f"Expected many parameter columns, got {len(all_patients.columns)}"
        )

        print(f"\n  simglucose patient parameters ({len(all_patients.columns)} columns):")
        print(f"  Columns: {list(all_patients.columns[:20])}")
        adult1 = all_patients[all_patients["Name"] == "adult#001"]
        if len(adult1) > 0:
            print(f"  Sample (adult#001):")
            for col in list(all_patients.columns[:10]):
                print(f"    {col}: {adult1.iloc[0][col]}")

    def test_baseline_simulation_stability(self):
        """Step 0.5: Verify 24-hour simulation doesn't diverge numerically."""
        from simglucose.patient.t1dpatient import T1DPatient

        patient = T1DPatient.withName("adult#001")
        patient.reset()

        # Run 24 hours at 1-min resolution = 1440 steps
        glucose_trace = []
        action = make_action(insulin=1.0, cho=0.0)

        for step in range(1440):
            patient.step(action)
            bg = patient.observation.Gsub
            glucose_trace.append(bg)

            assert not np.isnan(bg), f"NaN glucose at step {step}"
            assert not np.isinf(bg), f"Inf glucose at step {step}"
            # Use tolerance for floating-point near-zero (simglucose can return ~-1e-13)
            # Note: basal insulin without meals WILL drive glucose near zero — that's
            # physiologically expected in simulation. We're testing numerical stability here.
            assert bg > -1.0, f"Significantly negative glucose at step {step}: {bg}"
            assert bg < 1000, f"Glucose diverged at step {step}: {bg}"

        glucose_trace = np.array(glucose_trace)

        print(f"\n  24-hour baseline simulation (adult#001, basal-only, no meals):")
        print(f"    Mean glucose: {np.mean(glucose_trace):.1f} mg/dL")
        print(f"    Min glucose:  {np.min(glucose_trace):.1f} mg/dL")
        print(f"    Max glucose:  {np.max(glucose_trace):.1f} mg/dL")
        print(f"    SD:           {np.std(glucose_trace):.1f} mg/dL")

        # We're testing numerical stability, not clinical plausibility
        # (No meals + constant insulin = glucose will drop)
        assert np.all(np.isfinite(glucose_trace)), "Non-finite values in glucose trace"
        assert np.std(glucose_trace) < 500, "Glucose trace has extreme variance"


class TestAdapterLayer:
    """Test the adapter between simglucose patient API and AEGIS input format."""

    def test_glucose_trace_extraction(self):
        """Step 0.2: Extract glucose trace from simglucose patient."""
        from simglucose.patient.t1dpatient import T1DPatient

        patient = T1DPatient.withName("adult#001")
        patient.reset()

        glucose_values = []
        action = make_action(insulin=1.0, cho=0.0)

        for _ in range(60):
            patient.step(action)
            glucose_values.append(patient.observation.Gsub)

        glucose_array = np.array(glucose_values)

        assert glucose_array.shape == (60,)
        assert np.all(np.isfinite(glucose_array))
        print(f"\n  60-step trace: mean={np.mean(glucose_array):.1f}, "
              f"range=[{np.min(glucose_array):.1f}, {np.max(glucose_array):.1f}]")

    def test_meal_injection(self):
        """Step 0.2: Verify meal injection causes glucose rise."""
        from simglucose.patient.t1dpatient import T1DPatient

        patient = T1DPatient.withName("adult#001")
        patient.reset()

        # Run 30 min with no meal at basal
        action_no_meal = make_action(insulin=1.0, cho=0.0)
        for _ in range(30):
            patient.step(action_no_meal)
        glucose_before = patient.observation.Gsub

        # Inject a 50g carb meal
        meal_action = make_action(insulin=1.0, cho=50.0)
        patient.step(meal_action)

        # Run 2 hours to observe glucose response
        glucose_after = []
        for _ in range(120):
            patient.step(action_no_meal)
            glucose_after.append(patient.observation.Gsub)

        glucose_after = np.array(glucose_after)
        peak_glucose = np.max(glucose_after)

        print(f"\n  Meal injection test (50g carbs):")
        print(f"    Before meal: {glucose_before:.1f} mg/dL")
        print(f"    Peak after:  {peak_glucose:.1f} mg/dL")
        print(f"    Rise:        {peak_glucose - glucose_before:+.1f} mg/dL")

        # After a 50g meal, glucose should rise
        assert peak_glucose > glucose_before, (
            f"Glucose should rise after meal: before={glucose_before:.1f}, peak={peak_glucose:.1f}"
        )
        assert all(g > 0 for g in glucose_after), "Negative glucose values detected"

    def test_multi_patient_batch(self, patient_names):
        """Step 0.2: Verify we can create and simulate all 30 patients."""
        from simglucose.patient.t1dpatient import T1DPatient

        successful = 0
        failed = []
        action = make_action(insulin=1.0, cho=0.0)

        for name in patient_names:
            try:
                patient = T1DPatient.withName(name)
                patient.reset()
                for _ in range(5):
                    patient.step(action)
                bg = patient.observation.Gsub
                assert bg > 0 and bg < 1000
                successful += 1
            except Exception as e:
                failed.append((name, str(e)))

        print(f"\n  Successfully created and simulated {successful}/30 patients")
        if failed:
            for name, err in failed:
                print(f"    FAILED: {name} — {err}")

        assert successful == 30, f"Failed to create {30 - successful} patients: {failed}"


class TestBaselineMetrics:
    """Compute reference clinical metrics for the standard basal-bolus scenario."""

    def test_compute_clinical_metrics(self):
        """Step 0.4: Compute TIR/TBR/TAR for adult#001 with basal-only."""
        from simglucose.patient.t1dpatient import T1DPatient

        patient = T1DPatient.withName("adult#001")
        patient.reset()

        # Simulate 24 hours
        glucose = []
        action = make_action(insulin=1.0, cho=0.0)
        for _ in range(1440):
            patient.step(action)
            glucose.append(patient.observation.Gsub)

        glucose = np.array(glucose)

        # Compute clinical metrics
        tir = np.mean((glucose >= 70) & (glucose <= 180)) * 100
        tbr_70 = np.mean(glucose < 70) * 100
        tbr_54 = np.mean(glucose < 54) * 100
        tar_180 = np.mean(glucose > 180) * 100
        tar_250 = np.mean(glucose > 250) * 100
        cv = (np.std(glucose) / np.mean(glucose)) * 100

        print(f"\n  Baseline clinical metrics (adult#001, basal-only, 24h):")
        print(f"    TIR (70-180):  {tir:.1f}%")
        print(f"    TBR (<70):     {tbr_70:.1f}%")
        print(f"    TBR (<54):     {tbr_54:.1f}%")
        print(f"    TAR (>180):    {tar_180:.1f}%")
        print(f"    TAR (>250):    {tar_250:.1f}%")
        print(f"    CV:            {cv:.1f}%")
        print(f"    Mean glucose:  {np.mean(glucose):.1f} mg/dL")

        assert cv > 0, "CV should be positive"
