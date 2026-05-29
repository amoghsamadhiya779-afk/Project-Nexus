#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="NEXUS OS | Control Plane",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Session State for Navigation
if 'nav_selection' not in st.session_state:
    st.session_state.nav_selection = "System Overview"

def set_nav(view):
    st.session_state.nav_selection = view

# ==========================================
# CUSTOM ENTERPRISE CSS & TYPOGRAPHY
# ==========================================
# Note: No empty lines are allowed inside this multiline string to prevent Streamlit's Markdown parser from leaking the raw text!
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
    /* Global Theme */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0f172a 0%, #020617 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    /* Clean up native Streamlit elements */
    header[data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    .block-container { padding-top: 1rem !important; max-width: 95% !important; }
    /* Typography */
    h1, h2, h3, h4, h5 { font-family: 'Inter', sans-serif; font-weight: 700 !important; color: #f8fafc !important; letter-spacing: -0.02em; }
    /* Premium Metric Cards */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.4) 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(14, 165, 233, 0.4);
        transform: translateY(-3px);
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-weight: 500 !important; font-size: 13px !important; text-transform: uppercase; letter-spacing: 0.05em; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-family: 'Inter', sans-serif; font-weight: 700 !important; font-size: 32px !important; margin-top: 8px; }
    [data-testid="stMetricDelta"] { font-weight: 600 !important; font-size: 13px !important; }
    /* Button Base */
    .stButton > button {
        height: 42px;
        transition: all 0.2s ease;
        width: 100%;
    }
    /* Primary Action Buttons & Active Nav */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.02em;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.2);
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(14, 165, 233, 0.4);
    }
    /* Secondary (Inactive) Nav Buttons */
    .stButton > button[kind="secondary"] {
        background: transparent;
        color: #94a3b8;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        font-weight: 500;
    }
    .stButton > button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.05);
        color: #f8fafc;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    /* macOS-style Terminal Window */
    .mac-window {
        background: #020617;
        border: 1px solid #1e293b;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 20px 40px -10px rgba(0,0,0,0.5);
    }
    .mac-header {
        background: #0f172a;
        padding: 12px 16px;
        display: flex;
        align-items: center;
        border-bottom: 1px solid #1e293b;
    }
    .mac-dots { display: flex; gap: 8px; flex: 1; }
    .mac-dot { width: 12px; height: 12px; border-radius: 50%; }
    .mac-title { font-family: 'Fira Code', monospace; font-size: 12px; color: #64748b; flex: 2; text-align: center; }
    .mac-spacer { flex: 1; }
    .mac-body {
        padding: 20px;
        font-family: 'Fira Code', monospace;
        font-size: 13px;
        color: #e2e8f0;
        height: 380px;
        overflow-y: auto;
        line-height: 1.6;
    }
    /* Custom Scrollbar for Terminal */
    .mac-body::-webkit-scrollbar { width: 6px; }
    .mac-body::-webkit-scrollbar-track { background: transparent; }
    .mac-body::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    /* Terminal Colors */
    .t-prompt { color: #0ea5e9; font-weight: 600; margin-right: 8px; }
    .t-sys { color: #8b5cf6; font-weight: 600; }
    .t-success { color: #10b981; }
    .t-warn { color: #f59e0b; }
    .t-dim { color: #64748b; }
    .t-json-key { color: #7dd3fc; }
    .t-json-val { color: #fcd34d; }
    /* Glass Panels */
    .glass-panel {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
    }
    /* Status Badges */
    .badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;
    }
    .badge-ok { background: rgba(16,185,129,0.1); color: #34d399; border: 1px solid rgba(16,185,129,0.2); }
    .badge-sync { background: rgba(56,189,248,0.1); color: #38bdf8; border: 1px solid rgba(56,189,248,0.2); }
    /* Pulsing Dot */
    @keyframes pulse-dot { 0% { box-shadow: 0 0 0 0 rgba(16,185,129,0.4); } 70% { box-shadow: 0 0 0 6px rgba(16,185,129,0); } 100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); } }
    .status-dot { width: 8px; height: 8px; background: #10b981; border-radius: 50%; animation: pulse-dot 2s infinite; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# TOP NAVIGATION BAR
# ==========================================
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([3, 2, 2, 2, 3], gap="small")

with nav_col1:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; padding-top: 4px;">
            <div style="width: 32px; height: 32px; background: linear-gradient(135deg, #0ea5e9, #4f46e5); border-radius: 8px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 15px rgba(14,165,233,0.3);">
                <i class="fas fa-layer-group" style="color: white; font-size: 14px;"></i>
            </div>
            <span style="color: white; font-size: 20px; font-weight: 800; letter-spacing: 1px; font-family: 'Inter', sans-serif;">NEXUS<span style="color: #0ea5e9; font-weight: 300;">OS</span></span>
        </div>
    """, unsafe_allow_html=True)
    
with nav_col2:
    st.button("System Overview", type="primary" if st.session_state.nav_selection == "System Overview" else "secondary", use_container_width=True, on_click=set_nav, args=("System Overview",))
    
with nav_col3:
    st.button("Inference Gateway", type="primary" if st.session_state.nav_selection == "Inference Gateway" else "secondary", use_container_width=True, on_click=set_nav, args=("Inference Gateway",))
    
with nav_col4:
    st.button("MLOps Pipelines", type="primary" if st.session_state.nav_selection == "MLOps Pipelines" else "secondary", use_container_width=True, on_click=set_nav, args=("MLOps Pipelines",))
    
with nav_col5:
    st.markdown("""
        <div style="display: flex; justify-content: flex-end; align-items: center; padding-top: 2px;">
            <div style="background: rgba(15,23,42,0.6); border: 1px solid #1e293b; padding: 6px 16px; border-radius: 20px; display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 12px; color: #f8fafc; font-weight: 500;">Production (us-west-2)</span>
                <div class="status-dot"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown('<div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); margin-bottom: 2rem; margin-top: 1rem;"></div>', unsafe_allow_html=True)

# ==========================================
# VIEW: SYSTEM OVERVIEW
# ==========================================
if st.session_state.nav_selection == "System Overview":
    st.markdown('<h2 style="margin-bottom: 4px;"><i class="fas fa-server" style="color: #0ea5e9; margin-right: 12px;"></i>Global Telemetry</h2>', unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 14px; margin-bottom: 32px;'>Real-time aggregate performance across all Nexus microservices and models.</p>", unsafe_allow_html=True)
    
    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(label="Gateway Latency (p99)", value="12.4 ms", delta="-0.4ms")
    m2.metric(label="Inference Throughput", value="4,250 RPS", delta="124")
    m3.metric(label="Redis Cache Hit Rate", value="98.2%", delta="Stable", delta_color="off")
    m4.metric(label="GPU Utilization (Ray)", value="64%", delta="2%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Layout: Traffic Chart (Left) + Subsystems (Right)
    col_chart, col_table = st.columns([1.5, 1])
    
    with col_chart:
        st.markdown('<h4 style="margin-bottom: 16px; font-size: 16px;"><i class="fas fa-wave-square" style="color: #6366f1; margin-right: 8px;"></i>Live Traffic (RPS)</h4>', unsafe_allow_html=True)
        
        # Generate dummy data for a beautiful area chart
        times = pd.date_range(end=datetime.now(), periods=40, freq='1min')
        rps_data = np.random.normal(4250, 150, 40)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=rps_data,
            mode='lines',
            line=dict(color='#0ea5e9', width=3, shape='spline'),
            fill='tozeroy',
            fillcolor='rgba(14, 165, 233, 0.1)',
            hoverinfo='y'
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            height=260,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', gridwidth=1, tickfont=dict(color='#64748b', family='Fira Code', size=10)),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col_table:
        st.markdown('<h4 style="margin-bottom: 16px; font-size: 16px;"><i class="fas fa-network-wired" style="color: #10b981; margin-right: 8px;"></i>Active Subsystems</h4>', unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-panel" style="padding: 16px;">
            <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 12px 8px; color: #f8fafc; font-weight: 500;">Serving Gateway</td>
                    <td style="padding: 12px 8px; text-align: right;"><span class="badge badge-ok"><i class="fas fa-check"></i> Healthy</span></td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 12px 8px; color: #f8fafc; font-weight: 500;">Redis Online Store</td>
                    <td style="padding: 12px 8px; text-align: right;"><span class="badge badge-ok"><i class="fas fa-check"></i> Healthy</span></td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 12px 8px; color: #f8fafc; font-weight: 500;">Flink Stream Processor</td>
                    <td style="padding: 12px 8px; text-align: right;"><span class="badge badge-sync"><i class="fas fa-sync fa-spin"></i> Syncing</span></td>
                </tr>
                <tr>
                    <td style="padding: 12px 8px; color: #f8fafc; font-weight: 500;">MLflow Registry</td>
                    <td style="padding: 12px 8px; text-align: right;"><span class="badge badge-ok"><i class="fas fa-check"></i> Healthy</span></td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# VIEW: INFERENCE GATEWAY
# ==========================================
elif st.session_state.nav_selection == "Inference Gateway":
    st.markdown('<h2 style="margin-bottom: 4px;"><i class="fas fa-bolt" style="color: #0ea5e9; margin-right: 12px;"></i>Recommendation Inference</h2>', unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 14px; margin-bottom: 32px;'>Live execution of the PyTorch Multi-Task Ranker and FAISS Candidate Generator.</p>", unsafe_allow_html=True)
    
    col_controls, col_terminal = st.columns([1, 2])
    
    with col_controls:
        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-bottom: 24px; font-size: 15px; color: #e2e8f0;'><i class='fas fa-sliders-h' style='color: #94a3b8; margin-right: 10px;'></i>Query Parameters</h4>", unsafe_allow_html=True)
        
        user_id = st.text_input("Target User ID", value="usr_7492_alpha", help="UUID of the customer to generate recommendations for.")
        k_cands = st.slider("Retrieval Pool (K)", min_value=10, max_value=100, value=50, step=10)
        use_cache = st.checkbox("Bypass L1 Response Cache", value=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        execute_btn = st.button("EXECUTE PIPELINE", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_terminal:
        # Create an empty placeholder for the dynamic terminal
        terminal_placeholder = st.empty()
        
        # Helper to generate the macOS terminal HTML
        def build_terminal_html(content):
            return f"""
            <div class="mac-window">
                <div class="mac-header">
                    <div class="mac-dots">
                        <div class="mac-dot mac-red"></div>
                        <div class="mac-dot mac-yellow"></div>
                        <div class="mac-dot mac-green"></div>
                    </div>
                    <div class="mac-title">gateway_server.py</div>
                    <div class="mac-spacer"></div>
                </div>
                <div class="mac-body">{content}</div>
            </div>
            """

        if execute_btn:
            lines = [
                "<span class='t-dim'>// Establishing connection to Inference Gateway...</span>",
                "<div><span class='t-prompt'>~ %</span> curl -X POST /api/v1/recommend -d '{\"user_id\": \"" + user_id + "\"}'</div>",
                "<br>",
                "<div><span class='t-sys'>[SYSTEM]</span> Initializing Request Pipeline...</div>",
                "<div><span class='t-sys'>[REDIS]</span> Fetching user features for: " + user_id + "... <span class='t-success'>Done (1.2ms)</span></div>",
                "<div><span class='t-sys'>[PYTORCH]</span> Executing Two-Tower Candidate Gen... <span class='t-success'>Done (4.5ms)</span></div>",
                "<div><span class='t-sys'>[FAISS]</span> Retrieved " + str(k_cands) + " candidates from ANN index.</div>",
                "<div><span class='t-sys'>[PYTORCH]</span> Rescoring via MMoE Ranker (Tasks: CTR, CVR)... <span class='t-success'>Done (8.1ms)</span></div>",
                "<br>",
                "<div class='t-dim'>// Response Payload — 200 OK (14.2ms)</div>",
                "{"
            ]
            
            # Simulate streaming terminal
            for i in range(1, len(lines)+1):
                out = "<br>".join(lines[:i])
                terminal_placeholder.markdown(build_terminal_html(out), unsafe_allow_html=True)
                time.sleep(0.12)
                
            # Add JSON Payload
            mock_payload = f"""
  <span class='t-json-key'>"user_id"</span>: <span class='t-json-val'>"{user_id}"</span>,
  <span class='t-json-key'>"items"</span>: [
    {{
      <span class='t-json-key'>"item_id"</span>: <span class='t-json-val'>"itm_992_headphones"</span>,
      <span class='t-json-key'>"p_ctr"</span>: <span class='t-success'>0.8421</span>,
      <span class='t-json-key'>"p_cvr"</span>: <span class='t-success'>0.1204</span>,
      <span class='t-json-key'>"final_score"</span>: <span class='t-success'>0.6789</span>
    }},
    <span class='t-dim'>... ({k_cands - 1} more items)</span>
  ],
  <span class='t-json-key'>"latency_ms"</span>: <span class='t-success'>14.2</span>
}}"""
            out = "<br>".join(lines) + "<br>" + mock_payload.replace('\n', '<br>')
            terminal_placeholder.markdown(build_terminal_html(out), unsafe_allow_html=True)
            
        else:
            # Default empty terminal
            terminal_placeholder.markdown(build_terminal_html("""
                <span class='t-dim'>// Awaiting API execution trigger...</span><br>
                <span class='t-prompt'>~ %</span> _
            """), unsafe_allow_html=True)

# ==========================================
# VIEW: MLOPS DAGSTER
# ==========================================
elif st.session_state.nav_selection == "MLOps Pipelines":
    st.markdown('<h2 style="margin-bottom: 4px;"><i class="fas fa-project-diagram" style="color: #0ea5e9; margin-right: 12px;"></i>Dagster Orchestration</h2>', unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 14px; margin-bottom: 32px;'>Automated drift detection and continuous model retraining DAG.</p>", unsafe_allow_html=True)
    
    col_btn, _ = st.columns([1, 4])
    trigger_dag = col_btn.button("RUN PIPELINE", type="primary")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Premium DAG Visualization
    st.markdown("""
        <div class="glass-panel" style="padding: 40px 20px; display: flex; justify-content: space-between; position: relative;">
            <div style="position: absolute; top: 50%; left: 8%; right: 8%; height: 2px; background: rgba(255,255,255,0.05); z-index: 0; transform: translateY(-50%);"></div>
            <div style="text-align: center; z-index: 1; padding: 0 15px;">
                <div style="width: 54px; height: 54px; border-radius: 14px; background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.4); margin: 0 auto 12px; display: flex; align-items: center; justify-content: center; backdrop-filter: blur(4px); box-shadow: 0 4px 10px rgba(0,0,0,0.2);"><i class="fas fa-chart-line" style="color: #34d399; font-size: 22px;"></i></div>
                <div style="color: #f8fafc; font-size: 13px; font-weight: 600; letter-spacing: 0.02em;">Drift Monitor</div>
            </div>
            <div style="text-align: center; z-index: 1; padding: 0 15px;">
                <div style="width: 54px; height: 54px; border-radius: 14px; background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.4); margin: 0 auto 12px; display: flex; align-items: center; justify-content: center; backdrop-filter: blur(4px); box-shadow: 0 4px 10px rgba(0,0,0,0.2);"><i class="fas fa-database" style="color: #34d399; font-size: 22px;"></i></div>
                <div style="color: #f8fafc; font-size: 13px; font-weight: 600; letter-spacing: 0.02em;">Data Snapshot</div>
            </div>
            <div style="text-align: center; z-index: 1; padding: 0 15px;">
                <div style="width: 54px; height: 54px; border-radius: 14px; background: rgba(14,165,233,0.15); border: 1px solid #0ea5e9; margin: 0 auto 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 20px rgba(14,165,233,0.3); backdrop-filter: blur(4px);"><i class="fas fa-microchip" style="color: #38bdf8; font-size: 22px;"></i></div>
                <div style="color: #f8fafc; font-size: 13px; font-weight: 600; letter-spacing: 0.02em;">Ray Training</div>
            </div>
            <div style="text-align: center; z-index: 1; padding: 0 15px;">
                <div style="width: 54px; height: 54px; border-radius: 14px; background: rgba(15,23,42,0.8); border: 1px solid rgba(255,255,255,0.1); margin: 0 auto 12px; display: flex; align-items: center; justify-content: center; backdrop-filter: blur(4px);"><i class="fas fa-server" style="color: #64748b; font-size: 22px;"></i></div>
                <div style="color: #94a3b8; font-size: 13px; font-weight: 600; letter-spacing: 0.02em;">Model Registry</div>
            </div>
            <div style="text-align: center; z-index: 1; padding: 0 15px;">
                <div style="width: 54px; height: 54px; border-radius: 14px; background: rgba(15,23,42,0.8); border: 1px solid rgba(255,255,255,0.1); margin: 0 auto 12px; display: flex; align-items: center; justify-content: center; backdrop-filter: blur(4px);"><i class="fas fa-check-circle" style="color: #64748b; font-size: 22px;"></i></div>
                <div style="color: #94a3b8; font-size: 13px; font-weight: 600; letter-spacing: 0.02em;">Shadow Deploy</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    log_placeholder = st.empty()
    
    def build_log_html(content):
        return f"""
        <div style="background: #020617; border: 1px solid #1e293b; border-radius: 12px; padding: 20px; font-family: 'Fira Code', monospace; font-size: 12px; color: #94a3b8; height: 280px; overflow-y: auto;">
            {content}
        </div>
        """
    
    if trigger_dag:
        logs = [
            "<span style='color: #10b981;'>[2026-05-29 14:02:11] SUCCESS: Asset drift_detection_alert materialized.</span>",
            "<span style='color: #10b981;'>[2026-05-29 14:02:15] SUCCESS: Extracted 1,000,000 rows to offline Parquet store.</span>",
            "<span style='color: #0ea5e9;'>[2026-05-29 14:02:16] INFO: Provisioning Ray cluster for distributed training...</span>",
            "<span style='color: #e2e8f0;'>[2026-05-29 14:02:18] INFO: Epoch 1/10 | Contrastive InfoNCE Loss: 0.1245</span>",
            "<span style='color: #e2e8f0;'>[2026-05-29 14:02:22] INFO: Epoch 2/10 | Contrastive InfoNCE Loss: 0.0891</span>",
            "<span style='color: #e2e8f0;'>[2026-05-29 14:02:25] INFO: Epoch 3/10 | Contrastive InfoNCE Loss: 0.0412</span>",
            "<span style='color: #10b981;'>[2026-05-29 14:02:28] SUCCESS: Ray Training Complete.</span>",
            "<span style='color: #0ea5e9;'>[2026-05-29 14:02:29] INFO: Artifacts synced to MLflow registry.</span>"
        ]
        
        for i in range(1, len(logs)+1):
            out = "<br>".join(logs[:i])
            log_placeholder.markdown(build_log_html(out + "<br><span style='color: #0ea5e9;'>_</span>"), unsafe_allow_html=True)
            time.sleep(0.3)
    else:
        log_placeholder.markdown(build_log_html("<span style='color: #475569;'>// Execution log empty. Trigger DAG to begin.</span>"), unsafe_allow_html=True)