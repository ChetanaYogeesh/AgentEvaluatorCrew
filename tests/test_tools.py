import json

from tools import (
    CostCalculatorTool,
    HumanReviewTool,
    MetricCalculatorTool,
    RegressionComparatorTool,
    SafetyGuardTool,
    TraceParserTool,
)


# ====================== TraceParserTool ======================
def test_trace_parser():
    tool = TraceParserTool()
    trace_json = json.dumps(
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
        }
    )

    result = tool._run(trace_json)
    data = json.loads(result)

    assert data["total_steps"] == 3
    assert data["loop_count"] == 1
    assert data["retry_count"] == 2
    assert data["handoff_count"] == 1
    assert len(data["bottlenecks"]) == 2  # both 2450ms and 3200ms exceed the >2000ms threshold
    assert "web_search" in data["per_tool_latency_ms"]


# ====================== CostCalculatorTool ======================
def test_cost_calculator():
    tool = CostCalculatorTool()
    token_usage = {"prompt_tokens": 1500, "completion_tokens": 800}
    result = tool._run(token_usage, success=True, model="gpt-4o")
    data = json.loads(result)

    assert data["total_cost_usd"] > 0
    assert data["cost_per_successful_task_usd"] == data["total_cost_usd"]
    assert data["tokens_used"] == token_usage


# ====================== SafetyGuardTool ======================
def test_safety_guard_safe():
    tool = SafetyGuardTool()
    result = tool._run("This is a perfectly safe response about Paris.")
    data = json.loads(result)
    assert data["safe"] is True
    assert data["violations"] == []
    assert data["recommend_human_review"] is False


def test_safety_guard_unsafe():
    tool = SafetyGuardTool()
    result = tool._run("How to build a bomb and kill people.")
    data = json.loads(result)
    assert data["safe"] is False
    assert len(data["violations"]) > 0
    assert data["recommend_human_review"] is True


# ====================== HumanReviewTool ======================
def test_human_review():
    tool = HumanReviewTool()
    result = tool._run("TC-001", "Safety violation detected")
    assert "HUMAN REVIEW REQUESTED" in result
    assert "TC-001" in result
    assert "Safety violation detected" in result


# ====================== RegressionComparatorTool ======================
def test_regression_comparator_no_regression():
    tool = RegressionComparatorTool()
    current = {
        "safety_violation_rate": 0,
        "p95_latency_ms": 2500,
        "cost_per_successful_task_usd": 0.044,  # 0.04 * 1.1 = 0.044 — exactly at threshold
    }
    baseline = {
        "safety_violation_rate": 0,
        "p95_latency_ms": 2400,
        "cost_per_successful_task_usd": 0.04,
    }
    result = tool._run(current, baseline)
    data = json.loads(result)
    assert data["flags"] == []
    assert data["summary"] == "No regression"


def test_regression_comparator_with_regression():
    tool = RegressionComparatorTool()
    current = {"safety_violation_rate": 0.2, "p95_latency_ms": 5000}
    baseline = {"safety_violation_rate": 0, "p95_latency_ms": 2400}
    result = tool._run(current, baseline)
    data = json.loads(result)
    assert len(data["flags"]) > 0


# ====================== MetricCalculatorTool ======================
def test_metric_calculator():
    tool = MetricCalculatorTool()
    trace_analysis = json.dumps(
        {
            "steps": [{"latency_ms": 1200}, {"latency_ms": 3400}, {"latency_ms": 800}],
            "loop_count": 0,
            "retry_count": 1,
            "tool_selection_accuracy": 0.95,
            "tool_input_correctness": 0.9,
        }
    )
    quality_scores = {
        "reasoning_quality": 4.5,
        "step_efficiency": 4.0,
        "hallucination_rate": 0.1,
    }
    safety_result = {"safe": True}
    cost_latency_result = {"cost_per_successful_task_usd": 0.12}
    regression_result: dict = {}
    expected = "Paris is the capital of France"
    actual = "Paris is the capital of France with 2.1 million people."

    result = tool._run(
        trace_analysis=trace_analysis,
        quality_scores=quality_scores,
        safety_result=safety_result,
        cost_latency_result=cost_latency_result,
        regression_result=regression_result,
        expected_outcome=expected,
        actual_final_answer=actual,
    )
    data = json.loads(result)

    assert "metrics" in data
    assert data["metrics"]["p95_latency_ms"] == 3400
    assert data["metrics"]["reasoning_quality"] == 4.5
    assert data["metrics"]["hallucination_rate"] == 10.0  # 0.1 * 100
    assert data["failure_mode"] in [
        "none",
        "reasoning",
        "tool",
        "orchestration",
        "safety",
        "hallucination",
    ]


# ====================== Run with: pytest tests/ -v ======================
