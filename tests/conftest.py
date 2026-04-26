"""
Shared pytest fixtures for unit and integration tests.
"""

import json

import pytest

# ─────────────────────────────────────────────────────────────────
# Reusable trace payloads
# ─────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_trace_json():
    """Minimal single-step trace for fast unit tests."""
    return json.dumps(
        {
            "steps": [{"name": "research", "latency_ms": 2450}],
            "loop_count": 0,
            "retry_count": 1,
        }
    )


@pytest.fixture
def full_trace_json():
    """Multi-step trace with handoffs, tool calls, and bottlenecks."""
    return json.dumps(
        {
            "steps": [
                {"name": "research", "latency_ms": 2450, "type": "handoff"},
                {
                    "name": "tool_call",
                    "latency_ms": 800,
                    "tool_calls": [{"tool": "web_search", "latency_ms": 700}],
                },
                {"name": "finalize", "latency_ms": 3200},
            ],
            "loop_count": 1,
            "retry_count": 2,
            "tool_selection_accuracy": 0.95,
            "tool_input_correctness": 0.90,
        }
    )


@pytest.fixture
def sample_baseline():
    return {"p95_latency_ms": 3000, "safety_violation_rate": 0}


@pytest.fixture
def sample_test_case(simple_trace_json, sample_baseline):
    return {
        "test_case_id": "TC-FIXTURE-001",
        "trace": simple_trace_json,
        "expected_outcome": "Paris is the capital of France",
        "baseline": json.dumps(sample_baseline),
    }