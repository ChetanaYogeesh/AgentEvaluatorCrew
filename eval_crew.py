import json
import os
import traceback
from typing import Any

import litellm
import yaml
from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv
from pydantic import BaseModel

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


print("🚀 Starting Agent Evaluator Crew with Direct LiteLLM...")

OPENROUTER_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENROUTER_API_KEY:
    print("❌ ERROR: OPENAI_API_KEY not found in .env!")
    raise SystemExit(1)


def llm_call(messages: list[dict], model: str = "gpt-4o-mini") -> str:
    """Direct LiteLLM call — bypasses CrewAI's internal LLM wiring."""
    try:
        response = litellm.completion(
            model=f"openrouter/openai/{model}",
            messages=messages,
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LiteLLM call failed: {e}")
        raise


print("✅ Direct LiteLLM OpenRouter configured")


# ====================== Crew Definition ======================
class AgentEvaluatorCrew:
    def __init__(self) -> None:
        with open("config/agents.yaml") as f:
            self.agents_config = yaml.safe_load(f)
        with open("config/tasks.yaml") as f:
            self.tasks_config = yaml.safe_load(f)

    def evaluator_coordinator(self) -> Agent:
        return Agent(
            config=self.agents_config["evaluator_coordinator"],
            verbose=True,
            llm_call=llm_call,
        )

    def trace_analyst(self) -> Agent:
        from tools import TraceParserTool

        return Agent(
            config=self.agents_config["trace_analyst"],
            verbose=True,
            tools=[TraceParserTool()],
            llm_call=llm_call,
        )

    def quality_judge(self) -> Agent:
        return Agent(
            config=self.agents_config["quality_judge"],
            verbose=True,
            llm_call=llm_call,
        )

    def safety_judge(self) -> Agent:
        from tools import HumanReviewTool, SafetyGuardTool

        return Agent(
            config=self.agents_config["safety_judge"],
            verbose=True,
            tools=[SafetyGuardTool(), HumanReviewTool()],
            llm_call=llm_call,
        )

    def cost_latency_analyst(self) -> Agent:
        from tools import CostCalculatorTool

        return Agent(
            config=self.agents_config["cost_latency_analyst"],
            verbose=True,
            tools=[CostCalculatorTool()],
            llm_call=llm_call,
        )

    def regression_monitor(self) -> Agent:
        from tools import RegressionComparatorTool

        return Agent(
            config=self.agents_config["regression_monitor"],
            verbose=True,
            tools=[RegressionComparatorTool()],
            llm_call=llm_call,
        )

    def analyze_trace(self) -> Task:
        return Task(config=self.tasks_config["analyze_trace"], agent=self.trace_analyst())

    def judge_quality(self) -> Task:
        return Task(config=self.tasks_config["judge_quality"], agent=self.quality_judge())

    def judge_safety(self) -> Task:
        return Task(config=self.tasks_config["judge_safety"], agent=self.safety_judge())

    def analyze_cost_latency(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_cost_latency"], agent=self.cost_latency_analyst()
        )

    def monitor_regression(self) -> Task:
        return Task(config=self.tasks_config["monitor_regression"], agent=self.regression_monitor())

    def coordinate_evaluation(self) -> Task:
        from tools import MetricCalculatorTool

        return Task(
            config=self.tasks_config["coordinate_evaluation"],
            agent=self.evaluator_coordinator(),
            tools=[MetricCalculatorTool()],
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
            memory=False,
            output_json=True,
        )


# ====================== Run ======================
if __name__ == "__main__":
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
        print("✅ Crew created successfully. Running evaluation...")

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
