"""Shared test fixtures for the AEGIS 3.0 testing framework.

Provides pytest fixtures for:
- simglucose patient loading (all 30 virtual patients)
- Standard simulation scenarios
- Common DGP (Data Generating Process) factories
- Result aggregation and reporting utilities
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

# ============================================================
# Constants
# ============================================================

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Standard random seeds for reproducibility
STANDARD_SEEDS = list(range(42, 542))  # 500 seeds
SAFETY_SEEDS = list(range(42, 1042))   # 1000 seeds for safety-critical tests


# ============================================================
# simglucose Patient Fixtures
# ============================================================

@pytest.fixture(scope="session")
def all_patients():
    """Load all 30 simglucose virtual patient parameter sets."""
    import pandas as pd
    import pkg_resources
    params_file = pkg_resources.resource_filename(
        "simglucose", "params/vpatient_params.csv"
    )
    patients = pd.read_csv(params_file)
    return patients


@pytest.fixture(scope="session")
def adult_patients(all_patients):
    """Get the 10 adult virtual patients."""
    return all_patients[all_patients["Name"].str.contains("adult", case=False)]


@pytest.fixture(scope="session")
def adolescent_patients(all_patients):
    """Get the 10 adolescent virtual patients."""
    return all_patients[all_patients["Name"].str.contains("adolescent", case=False)]


@pytest.fixture(scope="session")
def child_patients(all_patients):
    """Get the 10 child virtual patients."""
    return all_patients[all_patients["Name"].str.contains("child", case=False)]


@pytest.fixture(scope="session")
def patient_names(all_patients):
    """Get all patient names as a list."""
    return list(all_patients["Name"])


# ============================================================
# Simulation Scenario Fixtures
# ============================================================

@pytest.fixture
def meal_protocols():
    """Standard meal protocols for testing."""
    return {
        "standard": [
            {"time": 7 * 60, "carbs": 45},    # Breakfast 7am
            {"time": 12 * 60, "carbs": 70},   # Lunch 12pm
            {"time": 18 * 60, "carbs": 80},   # Dinner 6pm
            {"time": 22 * 60, "carbs": 15},   # Snack 10pm
        ],
        "high_carb": [
            {"time": 7 * 60, "carbs": 80},
            {"time": 12 * 60, "carbs": 120},
            {"time": 18 * 60, "carbs": 130},
            {"time": 22 * 60, "carbs": 30},
        ],
        "low_carb": [
            {"time": 7 * 60, "carbs": 20},
            {"time": 12 * 60, "carbs": 30},
            {"time": 18 * 60, "carbs": 30},
        ],
        "skipped": [
            {"time": 12 * 60, "carbs": 70},
            {"time": 18 * 60, "carbs": 80},
        ],
        "irregular": [
            {"time": 9 * 60, "carbs": 60},    # Late breakfast
            {"time": 14 * 60, "carbs": 40},   # Late lunch
            {"time": 21 * 60, "carbs": 100},  # Late heavy dinner
        ],
    }


@pytest.fixture
def noise_levels():
    """CGM sensor noise levels (coefficient of variation %)."""
    return {
        "ideal": 0.0,
        "low": 0.05,
        "typical": 0.10,
        "high": 0.20,
    }


@pytest.fixture
def activity_levels():
    """Activity level modifiers for insulin sensitivity."""
    return {
        "sedentary": 1.0,
        "mild_exercise": 1.15,
        "intense_exercise": 1.50,
    }


# ============================================================
# DGP (Data Generating Process) Factories
# ============================================================

@pytest.fixture
def causal_dgp_factory():
    """Factory for generating causal inference test data.

    Returns a function that generates data from:
    Y = β₀ + β₁·A + γ·U + ε
    where U confounds both A and Y.
    """
    def create_dgp(
        n: int = 500,
        beta0: float = 0.0,
        beta1: float = 0.5,     # True treatment effect
        gamma: float = 1.0,     # Confounding strength
        noise_std: float = 1.0,
        proxy_quality_z: float = 0.7,  # ρ(Z, U)
        proxy_quality_w: float = 0.7,  # ρ(W, U)
        seed: int = 42,
    ):
        rng = np.random.RandomState(seed)

        # Unmeasured confounder
        U = rng.randn(n)

        # Treatment (confounded by U)
        propensity = 1.0 / (1.0 + np.exp(-gamma * U))
        A = rng.binomial(1, propensity)

        # Outcome (confounded by U)
        epsilon = rng.randn(n) * noise_std
        Y = beta0 + beta1 * A + gamma * U + epsilon

        # Treatment proxy Z (correlated with U)
        Z = proxy_quality_z * U + np.sqrt(1 - proxy_quality_z**2) * rng.randn(n)

        # Outcome proxy W (correlated with U)
        W = proxy_quality_w * U + np.sqrt(1 - proxy_quality_w**2) * rng.randn(n)

        return {
            "Y": Y, "A": A, "U": U, "Z": Z, "W": W,
            "true_ate": beta1,
            "confounding_strength": gamma,
            "proxy_quality_z": proxy_quality_z,
            "proxy_quality_w": proxy_quality_w,
        }

    return create_dgp


@pytest.fixture
def harmonic_dgp_factory():
    """Factory for generating time-varying treatment effect data.

    Creates data where the true treatment effect follows a known
    harmonic function for recovery testing.
    """
    def create_dgp(
        T: int = 500,
        effect_type: str = "sinusoidal",
        noise_std: float = 1.0,
        seed: int = 42,
    ):
        rng = np.random.RandomState(seed)
        t = np.arange(T) / T  # Normalize to [0, 1]

        # True time-varying treatment effect
        if effect_type == "constant":
            tau_true = np.full(T, 0.5)
        elif effect_type == "sinusoidal":
            tau_true = 0.5 + 0.3 * np.cos(2 * np.pi * t)
        elif effect_type == "square_wave":
            tau_true = np.where(np.sin(2 * np.pi * t) > 0, 0.8, 0.2)
        elif effect_type == "linear_trend":
            tau_true = 0.2 + 0.6 * t
        elif effect_type == "two_frequency":
            tau_true = 0.5 + 0.2 * np.cos(2 * np.pi * t) + 0.15 * np.cos(4 * np.pi * t)
        else:
            raise ValueError(f"Unknown effect type: {effect_type}")

        # Generate treatment assignments (micro-randomized)
        A = rng.binomial(1, 0.5, T)

        # Generate outcomes
        epsilon = rng.randn(T) * noise_std
        Y = tau_true * A + epsilon

        return {
            "Y": Y, "A": A, "t": t, "T": T,
            "tau_true": tau_true,
            "psi0_true": np.mean(tau_true),
        }

    return create_dgp


# ============================================================
# Result Reporting Utilities
# ============================================================

class TestResultCollector:
    """Collects and reports test results in standard format.

    Every test should use this to ensure consistent reporting:
    - Mean ± SD, median, 95% CI, worst-case
    - JSON export for reproducibility
    """

    def __init__(self, test_id: str, test_name: str):
        self.test_id = test_id
        self.test_name = test_name
        self.results = []
        self.start_time = time.time()

    def add_result(self, metrics: dict, condition: dict = None):
        """Add a single simulation result."""
        self.results.append({
            "metrics": metrics,
            "condition": condition or {},
        })

    def summarize(self) -> dict:
        """Generate summary statistics."""
        summary = {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "n_runs": len(self.results),
            "elapsed_seconds": time.time() - self.start_time,
            "metrics": {},
        }

        if not self.results:
            return summary

        # Extract all metric keys
        metric_keys = set()
        for r in self.results:
            metric_keys.update(r["metrics"].keys())

        for key in metric_keys:
            values = [
                r["metrics"][key]
                for r in self.results
                if key in r["metrics"] and r["metrics"][key] is not None
                and not (isinstance(r["metrics"][key], float) and np.isnan(r["metrics"][key]))
            ]
            if values:
                arr = np.array(values, dtype=float)
                summary["metrics"][key] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "median": float(np.median(arr)),
                    "ci_lower": float(np.percentile(arr, 2.5)),
                    "ci_upper": float(np.percentile(arr, 97.5)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "n": len(values),
                }

        return summary

    def save(self, directory: Path = None):
        """Save raw results and summary to JSON."""
        directory = directory or RESULTS_DIR
        directory.mkdir(parents=True, exist_ok=True)

        summary = self.summarize()

        # Save summary
        summary_path = directory / f"{self.test_id}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Save raw results
        raw_path = directory / f"{self.test_id}_raw.json"
        with open(raw_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        return summary_path

    def print_summary(self):
        """Print formatted summary to console."""
        summary = self.summarize()
        print(f"\n{'='*60}")
        print(f"TEST: {self.test_id} — {self.test_name}")
        print(f"Runs: {summary['n_runs']} | Time: {summary['elapsed_seconds']:.1f}s")
        print(f"{'='*60}")

        for metric, stats in summary.get("metrics", {}).items():
            print(
                f"  {metric:30s}: "
                f"{stats['mean']:.4f} ± {stats['std']:.4f} "
                f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}] "
                f"worst={stats['min']:.4f}"
            )
        print()


@pytest.fixture
def result_collector():
    """Factory fixture for creating TestResultCollectors."""
    collectors = []

    def create(test_id: str, test_name: str):
        collector = TestResultCollector(test_id, test_name)
        collectors.append(collector)
        return collector

    yield create

    # Auto-save all collectors after test
    for c in collectors:
        c.save()
        c.print_summary()
