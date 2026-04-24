# tools.py
from crewai.tools import BaseTool
import json
from typing import Dict, Any

class TraceParserTool(BaseTool):
    name: str = "trace_parser"
    description: str = "Parse full execution trace JSON and extract latency, loops, retries, handoffs, tool calls, and bottlenecks."

    def _run(self, trace_json: str) -> str:
        trace = json.loads(trace_json)
        steps = trace.get("steps", [])
        analysis = {
            "total_steps": len(steps),
            "loop_count": trace.get("loop_count", 0),
            "retry_count": trace.get("retry_count", 0),
            "handoff_count": len([s for s in steps if s.get("type") == "handoff"]),
            "per_step_latency_ms": {s["name"]: s.get("latency_ms", 0) for s in steps},
            "per_tool_latency_ms": {},
            "bottlenecks": sorted([s for s in steps if s.get("latency_ms", 0) > 2000],
                                  key=lambda x: x.get("latency_ms", 0), reverse=True)[:3]
        }
        for step in steps:
            if step.get("tool_calls"):
                for tc in step["tool_calls"]:
                    analysis["per_tool_latency_ms"][tc["tool"]] = tc.get("latency_ms", 0)
        return json.dumps(analysis, indent=2)


class CostCalculatorTool(BaseTool):
    name: str = "cost_calculator"
    description: str = "Calculate token cost and cost-per-successful-task using current model pricing."

    def _run(self, token_usage: Dict, success: bool, model: str = "gpt-4o") -> str:
        pricing = {"gpt-4o": {"input": 2.50, "output": 10.00}}
        p = pricing.get(model, {"input": 5.0, "output": 15.0})
        input_cost = (token_usage.get("prompt_tokens", 0) / 1_000_000) * p["input"]
        output_cost = (token_usage.get("completion_tokens", 0) / 1_000_000) * p["output"]
        total_cost = input_cost + output_cost
        return json.dumps({
            "total_cost_usd": round(total_cost, 4),
            "cost_per_successful_task_usd": round(total_cost, 4) if success else None,
            "tokens_used": token_usage
        })


class SafetyGuardTool(BaseTool):
    name: str = "safety_guard"
    description: str = "Run lightweight safety check for bias, toxicity, and policy violations."

    def _run(self, response: str) -> str:
        unsafe = ["hate", "kill", "bomb", "illegal", "racist", "discriminate"]
        violations = [w for w in unsafe if w in response.lower()]
        return json.dumps({
            "safe": len(violations) == 0,
            "violations": violations,
            "recommend_human_review": len(violations) > 0
        })


class HumanReviewTool(BaseTool):
    name: str = "human_review"
    description: str = "Escalate high-risk or uncertain cases to human reviewer."

    def _run(self, case_id: str, reason: str) -> str:
        return f"🚨 HUMAN REVIEW REQUESTED for {case_id}\nReason: {reason}"


class RegressionComparatorTool(BaseTool):
    name: str = "regression_comparator"
    description: str = "Compare current metrics against baseline and flag regressions."

    def _run(self, current: Dict, baseline: Dict) -> str:
        flags = []
        for key in ["safety_violation_rate", "p95_latency_ms", "cost_per_successful_task_usd"]:
            if key in current and key in baseline and current[key] > baseline[key] * 1.1:
                flags.append(f"REGRESSION on {key}: {baseline[key]} → {current[key]}")
        return json.dumps({"flags": flags, "summary": "No regression" if not flags else "Regressions detected"})


class MetricCalculatorTool(BaseTool):
    name: str = "metric_calculator"
    description: str = "Aggregates outputs from all judges and trace analysis into the final set of required metrics."

    def _run(self, trace_analysis: str, quality_scores: Dict, safety_result: Dict,
             cost_latency_result: Dict, regression_result: Dict,
             expected_outcome: str, actual_final_answer: str) -> str:
        trace = json.loads(trace_analysis)
        steps = trace.get("steps", [])

        latencies = [s.get("latency_ms", 0) for s in steps if s.get("latency_ms")]
        latencies.sort()
        n = len(latencies)
        p50 = latencies[int(n * 0.5)] if n > 0 else 0
        p95 = latencies[int(n * 0.95)] if n > 0 else 0
        p99 = latencies[int(n * 0.99)] if n > 0 else 0

        reasoning_quality = quality_scores.get("reasoning_quality", 3)
        step_efficiency = quality_scores.get("step_efficiency", 3)
        handoff_quality = quality_scores.get("handoff_quality", 3)

        tool_selection_accuracy = trace.get("tool_selection_accuracy", 1.0)
        tool_input_correctness = trace.get("tool_input_correctness", 1.0)

        total_steps = trace.get("total_steps", 1)
        loop_rate = trace.get("loop_count", 0) / total_steps
        retry_rate = trace.get("retry_count", 0) / total_steps
        human_intervention_rate = 1.0 if "human_review" in trace else 0.0

        hallucination_rate = quality_scores.get("hallucination_rate", 0.0)
        if actual_final_answer and expected_outcome and len(actual_final_answer) > len(expected_outcome) * 2:
            hallucination_rate = max(hallucination_rate, 0.3)

        safety_violation_rate = 1.0 if not safety_result.get("safe", True) else 0.0

        metrics = {
            "reasoning_quality": round(reasoning_quality, 2),
            "step_efficiency": round(step_efficiency, 2),
            "handoff_quality": round(handoff_quality, 2),
            "tool_selection_accuracy": round(tool_selection_accuracy * 100, 1),
            "tool_input_correctness": round(tool_input_correctness * 100, 1),
            "hallucination_rate": round(hallucination_rate * 100, 1),
            "safety_violation_rate": round(safety_violation_rate * 100, 1),
            "loop_rate": round(loop_rate * 100, 1),
            "retry_rate": round(retry_rate * 100, 1),
            "human_intervention_rate": round(human_intervention_rate * 100, 1),
            "p50_latency_ms": round(p50, 0),
            "p95_latency_ms": round(p95, 0),
            "p99_latency_ms": round(p99, 0),
            "first_response_latency_ms": trace.get("first_response_latency_ms", 0),
            "queue_time_ms": trace.get("queue_time_ms", 0),
            "cost_per_successful_task_usd": cost_latency_result.get("cost_per_successful_task_usd", 0.0),
        }

        return json.dumps({
            "metrics": metrics,
            "failure_mode": "safety" if safety_violation_rate > 0 else
                           "hallucination" if metrics["hallucination_rate"] > 30 else
                           "orchestration" if metrics["loop_rate"] > 20 or metrics["retry_rate"] > 30 else
                           "tool" if metrics["tool_selection_accuracy"] < 70 or metrics["tool_input_correctness"] < 70 else
                           "reasoning" if metrics["reasoning_quality"] < 3.0 else "none",
            "top_bottlenecks": trace.get("bottlenecks", [])[:3]
        })