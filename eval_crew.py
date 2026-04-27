import json
import os
import traceback
from typing import Any

import yaml
from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv
from pydantic import BaseModel

from tools import (
    CostCalculatorTool,
    HumanReviewTool,
    MetricCalculatorTool,
    RegressionComparatorTool,
    SafetyGuardTool,
    TraceParserTool,
)

load_dotenv()


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


# ====================== Smart Model Switch ======================
OPENROUTER_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENROUTER_API_KEY:
    print("❌ ERROR: OPENAI_API_KEY not found in .env!")
    raise SystemExit(1)

_MODEL_MAP: dict[str, str] = {
    "quality": "openrouter/openai/gpt-4o",
    "coordinator": "openrouter/openai/gpt-4o",
    "safety": "openrouter/anthropic/claude-3-haiku",
}
_DEFAULT_MODEL = "openrouter/google/gemini-flash-1.5"

_TOKEN_MAP: dict[str, int] = {
    "quality": 4000,
    "coordinator": 4000,
    "safety": 3000,
}
_DEFAULT_TOKENS = 2000


def get_llm_config(agent_name: str) -> dict:
    """Return LiteLLM config for the given agent role."""
    key = next((k for k in _MODEL_MAP if k in agent_name.lower()), None)
    return {
        "model": _MODEL_MAP.get(key, _DEFAULT_MODEL) if key else _DEFAULT_MODEL,
        "api_key": OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
        "temperature": 0.0,
        "max_tokens": _TOKEN_MAP.get(key, _DEFAULT_TOKENS) if key else _DEFAULT_TOKENS,
    }


print("🚀 Starting Final CrewAI Evaluator with Smart Model Switching...")
print("✅ Smart model switching + memory disabled configured")


# ====================== Crew Definition ======================
_TOOL_MAP: dict[str, list] = {
    "trace": [TraceParserTool()],
    "safety": [SafetyGuardTool(), HumanReviewTool()],
    "cost": [CostCalculatorTool()],
    "regression": [RegressionComparatorTool()],
}


class AgentEvaluatorCrew:
    """Production Evaluation Crew — dynamic agent factory with smart model routing."""

    def __init__(self) -> None:
        with open("config/agents.yaml") as f:
            self.agents_config = yaml.safe_load(f)
        with open("config/tasks.yaml") as f:
            self.tasks_config = yaml.safe_load(f)

    def get_agent(self, name: str) -> Agent:
        tools = next(
            (v for k, v in _TOOL_MAP.items() if k in name.lower()),
            [],
        )
        return Agent(
            config=self.agents_config.get(name, {}),
            verbose=True,
            llm=get_llm_config(name),
            tools=tools,
        )

    def coordinator(self) -> Agent:
        # Manager agent must NOT have tools in hierarchical process
        return Agent(
            config=self.agents_config["evaluator_coordinator"],
            verbose=True,
            llm=get_llm_config("coordinator"),
        )

    def coordinate_evaluation(self) -> Task:
        return Task(
            config=self.tasks_config.get("coordinate_evaluation", {}),
            agent=self.coordinator(),
            tools=[MetricCalculatorTool()],  # scoped to final task only
            output_pydantic=EvaluationReport,
            async_execution=False,
        )

    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.coordinator(),
                self.get_agent("trace_analyst"),
                self.get_agent("quality_judge"),
                self.get_agent("safety_judge"),
                self.get_agent("cost_latency_analyst"),
                self.get_agent("regression_monitor"),
            ],
            tasks=[self.coordinate_evaluation()],
            process=Process.hierarchical,
            manager_agent=self.coordinator(),
            verbose=True,
            memory=False,  # disabled to avoid embedding dependency
            output_json=True,
        )


# ====================== Run ======================
if __name__ == "__main__":
    try:
        crew_obj = AgentEvaluatorCrew()
        crew = crew_obj.crew()
        print("✅ Final Crew created with smart model switching.")

        inputs = {
            "test_case_id": "TC-001",
            "trace": json.dumps({"steps": [{"name": "research", "latency_ms": 2450}]}),
            "expected_outcome": "Paris is the capital of France",
            "baseline": json.dumps({"p95_latency_ms": 3000}),
        }

        result = crew.kickoff(inputs=inputs)
        print(f"✅ Evaluation finished: {getattr(result, 'pass_fail', 'UNKNOWN')}")

        with open("evaluation_results.json", "w") as f:
            json.dump(
                result.model_dump() if hasattr(result, "model_dump") else dict(result),
                f,
                indent=2,
            )

        print("📁 Saved to evaluation_results.json")

    except Exception as e:
        print("❌ Final error:", e)
        traceback.print_exc()
