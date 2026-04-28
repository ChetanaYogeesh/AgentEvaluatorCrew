# 1. Clone / create the project
mkdir agent-evaluator-crew && cd agent-evaluator-crew

# 2. Copy all files above into the correct locations

# 3. Quick start
make install          # install everything
make run              # run evaluation batch
make dashboard        # open dashboard
make test             # run unit tests
make integration-test # run full Crew integration tests
make ci               # full local CI

# Agent Evaluator Crew

**Production-grade AI evaluation framework for CrewAI multi-agent workflows**

A complete, observable, and safety-first evaluation system that measures **task success, process quality, latency, cost, safety, hallucination, loops, retries, handoffs, and regressions** — with built-in tools, advanced dashboard, full test coverage, and CI.

---

## ✨ Features

- **Six specialized evaluation agents** with clear roles
- **Five production tools** (`TraceParserTool`, `MetricCalculatorTool`, `SafetyGuardTool`, etc.)
- **Automatic metric calculation** (p50/p95/p99 latency, hallucination rate, tool accuracy, etc.)
- **Hierarchical CrewAI process** with Coordinator as manager
- **CrewAI Flow** support for large-scale batch evaluation
- **Full OpenTelemetry** observability (LangSmith / Phoenix ready)
- **Structured Pydantic reports** + failure attribution
- **Advanced Streamlit dashboard** with process quality, latency, safety, and regression views
- **Unit + integration tests**
- **Makefile** for one-command workflows
- **GitHub Actions CI** workflow

---

## 📊 What It Evaluates

| Category              | Metrics Tracked |
|-----------------------|-----------------|
| **Outcome**           | Task completion, final answer correctness |
| **Process Quality**   | Reasoning (1-5), step efficiency, handoff quality |
| **Tools**             | Selection accuracy, input correctness |
| **Safety**            | Violation rate, bias, harmful content |
| **Performance**       | p50/p95/p99 latency, first-response, queue time |
| **Efficiency**        | Cost per successful task, loops, retries |
| **Reliability**       | Hallucination rate, human intervention rate |
| **Regressions**       | Compared against baselines |

---

## 🏗️ Project Structure
agent-evaluator-crew/
├── config/
│   ├── agents.yaml
│   └── tasks.yaml
├── tools.py
├── eval_crew.py
├── dashboard.py
├── tests/
│   ├── test_tools.py
│   └── test_integration.py
├── Makefile
├── .github/workflows/ci.yml
├── requirements.txt
└── evaluation_results.json   # generated

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/agent-evaluator-crew.git
cd agent-evaluator-crew
make install
```

### 2. Run Evaluation
Bashmake run          # Runs batch evaluation

### 3. Launch Dashboard
Bashmake dashboard    # Opens http://localhost:8501

### 4. Run Tests
Bashmake test                    # Unit tests
make integration-test        # Full Crew integration tests
make ci                      # Full local CI

### 📋 Available Commands (Makefile)

Command,Description
make install,Install all dependencies
make run,Run evaluation batch
make dashboard,Start Streamlit dashboard
make test,Run unit tests
make integration-test,Run full Crew integration tests
make ci,Run complete CI locally
make clean,Remove generated files

### 🧪 Testing

Unit tests: tests/test_tools.py — covers every tool
Integration tests: tests/test_integration.py — full Crew end-to-end
Run with: pytest tests/ -v


### 📈 Dashboard
The Streamlit dashboard includes:

KPI overview (pass rate, reasoning quality, tool accuracy, cost)
Process Quality tab (1-5 scores)
Advanced Latency tab (box plots + percentiles)
Safety & Hallucination tab
Loops & Retries scatter plot
Release decision matrix
Interactive filters and JSON export


### 🔧 Customization

Edit config/agents.yaml and config/tasks.yaml to change agent behavior
Add new metrics in MetricCalculatorTool._run()
Extend tools in tools.py
Customize pricing in CostCalculatorTool


### 🛠️ Observability
The Crew is instrumented with OpenTelemetry. Export to:

LangSmith
Phoenix
Any OTLP-compatible backend

Set environment variables:
Bashexport OTEL_EXPORTER_OTLP_ENDPOINT="https://api.langsmith.com"

### 🤝 Contributing

Fork the repo
Create a feature branch
Add tests
Run make ci
Open a PR


### 📄 License
MIT License — feel free to use in commercial or open-source projects.

Built with ❤️ for the CrewAI community

