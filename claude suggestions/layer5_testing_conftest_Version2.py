"""Pytest configuration and fixtures for Layer 5 testing."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class ResultCollector:
    """Collects per-test raw results and writes them out as JSON."""

    def __init__(self, test_id: str, description: str):
        self.test_id = test_id
        self.description = description
        self.results: list = []
        self.start_time = datetime.now()

    def add_result(self, data: dict):
        self.results.append(data)

    def save(self):
        out = {
            "test_id": self.test_id,
            "description": self.description,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "n_results": len(self.results),
            "results": self.results,
        }
        path = RESULTS_DIR / f"{self.test_id}.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=str)


@pytest.fixture
def result_collector():
    """Factory fixture — returns a function that creates ResultCollectors."""
    collectors: list[ResultCollector] = []

    def _make(test_id: str, description: str) -> ResultCollector:
        rc = ResultCollector(test_id, description)
        collectors.append(rc)
        return rc

    yield _make

    # Auto-save all collectors when the test finishes
    for rc in collectors:
        rc.save()