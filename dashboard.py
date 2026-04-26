import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Agent Evaluation Dashboard", layout="wide")
st.title("🧠 Multi-Agent Evaluation Dashboard")


# ====================== Load Results (History Support) ======================
def load_all_results():
    file_path = Path("evaluation_results.json")
    history_path = Path("evaluation_history.json")

    results = []
    if file_path.exists():
        try:
            with open(file_path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            results.extend(data)
        except Exception:
            pass

    # Load historical results
    if history_path.exists():
        try:
            with open(history_path) as f:
                history = json.load(f)
            if isinstance(history, list):
                results.extend(history)
        except Exception:
            pass

    return results


results_list = load_all_results()

if not results_list:
    st.error("No evaluation results found. Run the evaluation script first.")
    st.stop()

# Sidebar
st.sidebar.header("📋 Test Cases")
selected_id = st.sidebar.selectbox(
    "Select Evaluation",
    [f"{r.get('test_case_id', 'Unknown')} - {r.get('timestamp', '')}" for r in results_list],
)

current = next(
    (
        r
        for r in results_list
        if f"{r.get('test_case_id')} - {r.get('timestamp', '')}" == selected_id
    ),
    results_list[0],
)

# ====================== Configuration Details ======================
st.sidebar.header("⚙️ Configuration")
st.sidebar.write("**Model Used:** Ollama (llama3.2)")
st.sidebar.write("**Memory:** Disabled")
st.sidebar.write("**Process:** Sequential / Hierarchical")
st.sidebar.write(f"**Run Time:** {current.get('timestamp', 'N/A')}")

# ====================== Main KPIs ======================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Test Case", current.get("test_case_id", "N/A"))
col2.metric("Pass/Fail", current.get("pass_fail", "UNKNOWN"))
col3.metric("Release Decision", current.get("release_decision", "N/A"))
col4.metric("Failure Mode", current.get("failure_mode", "none"))

# Charts
st.subheader("📊 Performance Charts")
metrics = current.get("metrics", {})
if metrics:
    df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    fig = px.bar(df, x="Metric", y="Value", title="Key Metrics", text="Value")
    st.plotly_chart(fig, use_container_width=True)

# Safety Detectors
st.subheader("🔍 Safety & Failure Analysis")
safety_cols = st.columns(3)
safety_cols[0].metric("Hallucination", "Yes" if current.get("hallucination_detected") else "No")
safety_cols[1].metric("Bias", "Yes" if current.get("bias_detected") else "No")
safety_cols[2].metric("Toxicity", "Yes" if current.get("toxicity_detected") else "No")

# Details
st.subheader("💡 Recommendations")
for rec in current.get("recommendations", []):
    st.write(f"• {rec}")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Top Bottlenecks")
    for item in current.get("top_bottlenecks", []):
        st.write(f"• {item}")
with col_b:
    st.subheader("Top Regressions")
    for item in current.get("top_regressions", []):
        st.write(f"• {item}")

# Raw Data
with st.expander("📜 View Raw JSON"):
    st.json(current)

st.success(f"Dashboard updated at {datetime.now().strftime('%H:%M:%S')}")
