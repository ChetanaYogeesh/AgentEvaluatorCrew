import pytest
import json
from eval_crew import AgentEvaluatorCrew, EvaluationReport
from crewai import Crew

@pytest.mark.integration
def test_full_crew_runs_successfully():
    """Integration test: Full crew can run end-to-end with valid inputs"""
    crew_instance = AgentEvaluatorCrew().crew()
    
    test_input = {
        "test_case_id": "INT-001",
        "trace": json.dumps({
            "steps": [{"name": "research", "latency_ms": 1200}],
            "loop_count": 0,
            "retry_count": 0
        }),
        "expected_outcome": "Paris is the capital of France",
        "baseline": json.dumps({"p95_latency_ms": 3000, "safety_violation_rate": 0})
    }

    result = crew_instance.kickoff(inputs=test_input)
    
    # Assertions
    assert isinstance(result, EvaluationReport)
    assert result.test_case_id == "INT-001"
    assert result.pass_fail in ["PASS", "FAIL"]
    assert isinstance(result.metrics, dict)
    assert "reasoning_quality" in result.metrics
    assert "p95_latency_ms" in result.metrics
    assert result.release_decision in ["approve", "approve_with_caution", "block"]
    assert len(result.recommendations) >= 0


@pytest.mark.integration
def test_metric_calculator_integration():
    """Integration test: MetricCalculatorTool is called and produces valid metrics"""
    from tools import MetricCalculatorTool
    tool = MetricCalculatorTool()
    
    result = tool._run(
        trace_analysis=json.dumps({"steps": [{"latency_ms": 1500}, {"latency_ms": 4500}]}),
        quality_scores={"reasoning_quality": 4.8},
        safety_result={"safe": True},
        cost_latency_result={"cost_per_successful_task_usd": 0.08},
        regression_result={},
        expected_outcome="Expected answer",
        actual_final_answer="Correct answer"
    )
    
    data = json.loads(result)
    assert "metrics" in data
    assert data["metrics"]["p95_latency_ms"] == 4500
    assert data["failure_mode"] in ["none", "reasoning", "tool", "orchestration", "safety", "hallucination"]