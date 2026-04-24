# Changelog

All notable changes to **Agent Evaluator Crew** are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- Initial multi-agent evaluation framework with six specialized agents
- Five production tools: `TraceParserTool`, `MetricCalculatorTool`, `SafetyGuardTool`, `HumanReviewTool`, `RegressionComparatorTool`
- Streamlit dashboard with KPI overview, safety analysis, bottleneck view
- Simple (Ollama) and full CrewAI evaluation runners
- Hallucination, bias, and toxicity detectors
- Evaluation history persistence (`evaluation_history.json`)
- GitHub Actions CI/CD pipeline (lint, unit tests, integration tests, GHCR publish)
- Streamlit Community Cloud deployment (staging on `main`, prod on tag)
- Dockerfile with multi-stage build and GHCR publishing
