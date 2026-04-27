from crewai import Agent, Task, Crew, Process, Flow
from crewai.flow import start, listen
from pydantic import BaseModel
import json
from typing import Dict, Any, List
import os

# ====================== Pydantic Output Model ======================
class EvaluationReport(BaseModel):
    test_case_id: str
    pass_fail: str
    metrics: Dict[str, Any]
    failure_mode: str
    recommendations: List[str]
    release_decision: str
    top_bottlenecks: List[str]
    top_regressions: List[str]

# ====================== OpenTelemetry Setup ======================
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://api.langsmith.com"
os.environ["OTEL_SERVICE_NAME"] = "agent-evaluator-crew"

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
print("✅ OpenTelemetry enabled")

# ====================== Crew Definition ======================
class AgentEvaluatorCrew:
    """Production Evaluation Crew for Multi-Agent Systems"""

    def __init__(self):
        self.agents_config = "config/agents.yaml"
        self.tasks_config = "config/tasks.yaml"

    def evaluator_coordinator(self) -> Agent:
        return Agent(config=self.agents_config["evaluator_coordinator"], verbose=True, tools=[MetricCalculatorTool()])

    def trace_analyst(self) -> Agent:
        return Agent(config=self.agents_config["trace_analyst"], verbose=True, tools=[TraceParserTool()])

    def quality_judge(self) -> Agent:
        return Agent(config=self.agents_config["quality_judge"], verbose=True)

    def safety_judge(self) -> Agent:
        return Agent(config=self.agents_config["safety_judge"], verbose=True, tools=[SafetyGuardTool(), HumanReviewTool()])

    def cost_latency_analyst(self) -> Agent:
        return Agent(config=self.agents_config["cost_latency_analyst"], verbose=True, tools=[CostCalculatorTool()])

    def regression_monitor(self) -> Agent:
        return Agent(config=self.agents_config["regression_monitor"], verbose=True, tools=[RegressionComparatorTool()])

    def analyze_trace(self) -> Task:
        return Task(config=self.tasks_config["analyze_trace"], agent=self.trace_analyst())

    def judge_quality(self) -> Task:
        return Task(config=self.tasks_config["judge_quality"], agent=self.quality_judge())

    def judge_safety(self) -> Task:
        return Task(config=self.tasks_config["judge_safety"], agent=self.safety_judge())

    def analyze_cost_latency(self) -> Task:
        return Task(config=self.tasks_config["analyze_cost_latency"], agent=self.cost_latency_analyst())

    def monitor_regression(self) -> Task:
        return Task(config=self.tasks_config["monitor_regression"], agent=self.regression_monitor())

    def coordinate_evaluation(self) -> Task:
        return Task(
            config=self.tasks_config["coordinate_evaluation"],
            agent=self.evaluator_coordinator(),
            output_pydantic=EvaluationReport,
            async_execution=False
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
            enable_otel=True,
            output_json=True
        )


# ====================== Flow for Batch Evaluation ======================
class EvaluationFlow(Flow):
    @start()
    def load_batch(self, test_cases: List[Dict]):
        self.test_cases = test_cases
        return "Batch loaded"

    @listen("load_batch")
    def run_evaluations(self):
        crew = AgentEvaluatorCrew().crew()
        results = []
        for case in self.test_cases:
            inputs = {
                "test_case_id": case["id"],
                "trace": json.dumps(case["trace"]),
                "expected_outcome": case["expected"],
                "baseline": json.dumps(case.get("baseline", {}))
            }
            result = crew.kickoff(inputs=inputs)
            results.append(result)
        return results

    @listen("run_evaluations")
    def finalize_batch(self, results):
        with open("evaluation_results.json", "w") as f:
            json.dump([r.model_dump() if hasattr(r, "model_dump") else dict(r) for r in results], f, indent=2)
        
        print("\n" + "="*80)
        print("BATCH EVALUATION COMPLETE")
        print("="*80)
        print(f"Total cases: {len(results)}")
        passes = sum(1 for r in results if getattr(r, "pass_fail", "FAIL") == "PASS")
        print(f"Pass rate: {passes / len(results):.1%}")
        print("📁 Results saved to evaluation_results.json")


# ====================== Example Batch Runner ======================
if __name__ == "__main__":
    print("🚀 Starting Agent Evaluator Crew...")

    test_cases = [
        {
            "id": "TC-001",
            "trace": {"steps": [{"name": "research", "latency_ms": 2450}], "loop_count": 0, "retry_count": 1},
            "expected": "Paris is the capital of France",
            "baseline": {"p95_latency_ms": 3000, "safety_violation_rate": 0}
        }
    ]

    flow = EvaluationFlow()
    flow.kickoff(inputs={"test_cases": test_cases})