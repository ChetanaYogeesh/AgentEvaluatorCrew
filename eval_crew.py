import json
import os
import traceback
from typing import Any

import yaml
from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel

from tools import (
    CostCalculatorTool,
    HumanReviewTool,
    MetricCalculatorTool,
    RegressionComparatorTool,
    SafetyGuardTool,
    TraceParserTool,
)

# ====================== OpenTelemetry (disabled) ======================
print("⚠️  OpenTelemetry disabled (set ENABLE_OTEL=true to enable)")


# ====================== Pydantic Output Model ======================
class EvaluationReport(BaseModel):
    test_case_id: str
    pass_fail: str
    metrics: dict[str, Any]
    failure_mode: str
    recommendations: list[str]
    release_decision: str
    top_bottlenecks: list[str]
    top_regressions: list[str]


# ====================== Smart Model Selection ======================
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_OPENROUTER_KEY = os.getenv("OPENAI_API_KEY")

_MODEL_MAP: dict[str, str] = {
    "evaluator_coordinator": "anthropic/claude-3.5-sonnet",
    "quality_judge": "anthropic/claude-3.5-sonnet",
    "trace_analyst": "google/gemini-2.0-flash-thinking-exp",
    "cost_latency_analyst": "google/gemini-2.0-flash-thinking-exp",
    "safety_judge": "openai/gpt-4o",
    "regression_monitor": "openai/gpt-4o",
}


def get_model_for_agent(agent_name: str) -> dict:
    """Return LLM config for the given agent role, falling back to gpt-4o."""
    return {
        "model": _MODEL_MAP.get(agent_name, "openai/gpt-4o"),
        "api_key": _OPENROUTER_KEY,
        "base_url": _OPENROUTER_BASE,
    }


# ====================== Crew Definition ======================
class AgentEvaluatorCrew:
    """Production Evaluation Crew — per-agent smart model routing via OpenRouter.

    Note: manager_agent (evaluator_coordinator) has no tools intentionally.
    Tools are assigned only to worker agents and the final coordination task.
    """

    def __init__(self) -> None:
        with open("config/agents.yaml") as f:
            self.agents_config = yaml.safe_load(f)
        with open("config/tasks.yaml") as f:
            self.tasks_config = yaml.safe_load(f)

    def evaluator_coordinator(self) -> Agent:
        # Manager agent must NOT have tools in hierarchical process
        return Agent(
            config=self.agents_config["evaluator_coordinator"],
            verbose=True,
            llm=get_model_for_agent("evaluator_coordinator"),
        )

    def trace_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["trace_analyst"],
            verbose=True,
            tools=[TraceParserTool()],
            llm=get_model_for_agent("trace_analyst"),
        )

    def quality_judge(self) -> Agent:
        return Agent(
            config=self.agents_config["quality_judge"],
            verbose=True,
            llm=get_model_for_agent("quality_judge"),
        )

    def safety_judge(self) -> Agent:
        return Agent(
            config=self.agents_config["safety_judge"],
            verbose=True,
            tools=[SafetyGuardTool(), HumanReviewTool()],
            llm=get_model_for_agent("safety_judge"),
        )

    def cost_latency_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["cost_latency_analyst"],
            verbose=True,
            tools=[CostCalculatorTool()],
            llm=get_model_for_agent("cost_latency_analyst"),
        )

    def regression_monitor(self) -> Agent:
        return Agent(
            config=self.agents_config["regression_monitor"],
            verbose=True,
            tools=[RegressionComparatorTool()],
            llm=get_model_for_agent("regression_monitor"),
        )

    def analyze_trace(self) -> Task:
        return Task(config=self.tasks_config["analyze_trace"], agent=self.trace_analyst())

    def judge_quality(self) -> Task:
        return Task(config=self.tasks_config["judge_quality"], agent=self.quality_judge())

    def judge_safety(self) -> Task:
        return Task(config=self.tasks_config["judge_safety"], agent=self.safety_judge())

    def analyze_cost_latency(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_cost_latency"],
            agent=self.cost_latency_analyst(),
        )

    def monitor_regression(self) -> Task:
        return Task(
            config=self.tasks_config["monitor_regression"],
            agent=self.regression_monitor(),
        )

    def coordinate_evaluation(self) -> Task:
        return Task(
            config=self.tasks_config["coordinate_evaluation"],
            agent=self.evaluator_coordinator(),
            tools=[MetricCalculatorTool()],  # tool scoped to final task only
            output_pydantic=EvaluationReport,
            async_execution=False,
        )

    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.evaluator_coordinator(),
                self.trace_analyst(),
                self.quality_judge(),
                self.safety_judge(),
                self.cost_latency_analyst(),
                self.regression_monitor(),
            ],
            tasks=[
                self.analyze_trace(),
                self.judge_quality(),
                self.judge_safety(),
                self.analyze_cost_latency(),
                self.monitor_regression(),
                self.coordinate_evaluation(),
            ],
            process=Process.hierarchical,
            manager_agent=self.evaluator_coordinator(),
            verbose=True,
            memory=True,
            enable_otel=False,
            output_json=True,
        )


# ====================== Batch Runner ======================
if __name__ == "__main__":
    print("🚀 Starting Agent Evaluator Crew with Smart Model Routing...")

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY is not set!")
        print("   Run: export OPENAI_API_KEY=sk-or-v1-...")
        raise SystemExit(1)

    test_cases = [
        {
            "id": "TC-001",
            "trace": {
                "steps": [{"name": "research", "latency_ms": 2450}],
                "loop_count": 0,
                "retry_count": 1,
            },
            "expected": "Paris is the capital of France",
            "baseline": {"p95_latency_ms": 3000, "safety_violation_rate": 0},
        }
    ]

    try:
        crew_obj = AgentEvaluatorCrew()
        crew = crew_obj.crew()
        print("✅ Crew created successfully.")

        results = []
        for case in test_cases:
            inputs = {
                "test_case_id": case["id"],
                "trace": json.dumps(case["trace"]),
                "expected_outcome": case["expected"],
                "baseline": json.dumps(case.get("baseline", {})),
            }
            result = crew.kickoff(inputs=inputs)
            results.append(result)
            print(f"✅ Evaluated {case['id']} → {getattr(result, 'pass_fail', 'UNKNOWN')}")

        with open("evaluation_results.json", "w") as f:
            json.dump(
                [r.model_dump() if hasattr(r, "model_dump") else dict(r) for r in results],
                f,
                indent=2,
            )

        print("\n" + "=" * 80)
        print("✅ SUCCESS! JSON file created")
        print("📁 evaluation_results.json has been saved")
        print("=" * 80)

    except Exception as e:
        print("❌ ERROR occurred:")
        print(e)
        traceback.print_exc()