#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import time
import json

# ==========================================
# PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="NEXUS OS | Control Plane",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject FontAwesome and Custom Enterprise Dark Mode CSS
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
    /* Base Theme */
    .stApp {
        background: radial-gradient(circle at top right, #0d1326, #050914);
        color: #cbd5e1;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0a0f1c !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Hide top header bar */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Metric Cards (Glassmorphism) */
    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: #0ea5e9;
        transform: translateY(-2px);
    }
    [data-testid="stMetricValue"] {
        color: #f8fafc;
        font-family: 'Fira Code', monospace;
        font-size: 28px !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #0ea5e9;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #0284c7;
        box-shadow: 0 0 15px rgba(14, 165, 233, 0.4);
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        background: rgba(15, 23, 42, 0.4);
        border-radius: 12px;
        border: 1px solid #1e293b;
        padding: 10px;
    }
    
    /* Custom Terminal/Console Box */
    .console-box {
        background-color: #020617;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 15px;
        font-family: 'Fira Code', monospace;
        font-size: 12px;
        color: #94a3b8;
        height: 400px;
        overflow-y: auto;
    }
    .console-box::-webkit-scrollbar { width: 6px; }
    .console-box::-webkit-scrollbar-track { background: transparent; }
    .console-box::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    
    .log-info { color: #38bdf8; }
    .log-success { color: #34d399; }
    .log-warn { color: #fbbf24; }
    .log-err { color: #f87171; }
    
    /* Custom Headers */
    h1, h2, h3 { color: #f8fafc !important; font-weight: 700 !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("""
        <div style="display: flex; items-center; gap: 10px; margin-bottom: 30px;">
            <div style="width: 32px; height: 32px; background: linear-gradient(135deg, #0ea5e9, #4f46e5); border-radius: 8px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 15px rgba(14,165,233,0.3);">
                <i class="fas fa-microchip" style="color: white; font-size: 14px;"></i>
            </div>
            <span style="color: white; font-size: 20px; font-weight: 800; letter-spacing: 1px;">NEXUS<span style="color: #0ea5e9; font-weight: 300;">OS</span></span>
        </div>
    """, unsafe_allow_html=True)
    
    nav_selection = st.radio(
        "NAVIGATION",
        ["System Overview", "Inference Gateway", "Semantic Search", "MLOps Pipelines"],
        label_visibility="hidden"
    )
    
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="background: #0f172a; border: 1px solid #1e293b; padding: 15px; border-radius: 8px;">
            <div style="font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Cluster Status</div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 13px; color: #e2e8f0; font-weight: 500;">Production (us-west-2)</span>
                <div style="width: 8px; height: 8px; background: #10b981; border-radius: 50%; box-shadow: 0 0 8px #10b981;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# VIEW: SYSTEM OVERVIEW
# ==========================================
if nav_selection == "System Overview":
    st.markdown('<h2><i class="fas fa-server" style="color: #0ea5e9; margin-right: 10px;"></i> Global Telemetry</h2>', unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 14px; margin-top: -10px; margin-bottom: 30px;'>Real-time aggregate performance across all Nexus microservices.</p>", unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(label="Gateway Latency (p99)", value="12.4 ms", delta="-0.4ms")
    m2.metric(label="Inference Throughput", value="4,250 RPS", delta="124")
    m3.metric(label="Redis Cache Hit Rate", value="98.2%", delta="Stable", delta_color="off")
    m4.metric(label="GPU Utilization (Ray)", value="64%", delta="2%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_table, col_map = st.columns([2, 1])
    
    with col_table:
        st.markdown('<h4><i class="fas fa-network-wired" style="color: #6366f1; margin-right: 8px;"></i> Active Subsystems</h4>', unsafe_allow_html=True)
        # Custom HTML table for absolute design control
        st.markdown("""
        <div style="background: rgba(15,23,42,0.4); border: 1px solid #1e293b; border-radius: 12px; padding: 1px; overflow: hidden;">
            <table style="width: 100%; text-align: left; border-collapse: collapse; font-size: 13px;">
                <tr style="background: #0f172a; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">
                    <th style="padding: 12px 16px; font-weight: 600;">Service Component</th>
                    <th style="padding: 12px 16px; font-weight: 600;">Architecture</th>
                    <th style="padding: 12px 16px; font-weight: 600;">Status</th>
                </tr>
                <tr style="border-top: 1px solid #1e293b;">
                    <td style="padding: 12px 16px; color: #e2e8f0; font-weight: 500;">Serving Gateway</td>
                    <td style="padding: 12px 16px; color: #94a3b8; font-family: monospace;">FastAPI / Uvicorn</td>
                    <td style="padding: 12px 16px;"><span style="background: rgba(16,185,129,0.1); color: #34d399; padding: 4px 8px; border-radius: 4px; font-size: 11px; border: 1px solid rgba(16,185,129,0.2);"><i class="fas fa-check-circle"></i> Healthy</span></td>
                </tr>
                <tr style="border-top: 1px solid #1e293b;">
                    <td style="padding: 12px 16px; color: #e2e8f0; font-weight: 500;">Feature Store (Online)</td>
                    <td style="padding: 12px 16px; color: #94a3b8; font-family: monospace;">Redis Cluster</td>
                    <td style="padding: 12px 16px;"><span style="background: rgba(16,185,129,0.1); color: #34d399; padding: 4px 8px; border-radius: 4px; font-size: 11px; border: 1px solid rgba(16,185,129,0.2);"><i class="fas fa-check-circle"></i> Healthy</span></td>
                </tr>
                <tr style="border-top: 1px solid #1e293b;">
                    <td style="padding: 12px 16px; color: #e2e8f0; font-weight: 500;">Stream Processor</td>
                    <td style="padding: 12px 16px; color: #94a3b8; font-family: monospace;">Apache Flink</td>
                    <td style="padding: 12px 16px;"><span style="background: rgba(56,189,248,0.1); color: #38bdf8; padding: 4px 8px; border-radius: 4px; font-size: 11px; border: 1px solid rgba(56,189,248,0.2);"><i class="fas fa-sync fa-spin"></i> Syncing</span></td>
                </tr>
                <tr style="border-top: 1px solid #1e293b;">
                    <td style="padding: 12px 16px; color: #e2e8f0; font-weight: 500;">Model Registry</td>
                    <td style="padding: 12px 16px; color: #94a3b8; font-family: monospace;">MLflow</td>
                    <td style="padding: 12px 16px;"><span style="background: rgba(16,185,129,0.1); color: #34d399; padding: 4px 8px; border-radius: 4px; font-size: 11px; border: 1px solid rgba(16,185,129,0.2);"><i class="fas fa-check-circle"></i> Healthy</span></td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_map:
        st.markdown('<h4><i class="fas fa-project-diagram" style="color: #10b981; margin-right: 8px;"></i> Pipeline Topology</h4>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background: rgba(15,23,42,0.4); border: 1px solid #1e293b; border-radius: 12px; padding: 20px;">
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <div style="width: 30px; height: 30px; border-radius: 50%; background: rgba(56,189,248,0.1); border: 1px solid rgba(56,189,248,0.3); display: flex; align-items: center; justify-content: center;"><i class="fas fa-database" style="color: #38bdf8; font-size: 12px;"></i></div>
                <div style="margin-left: 15px;">
                    <div style="color: #e2e8f0; font-size: 13px; font-weight: 600;">Kafka Ingestion</div>
                    <div style="color: #64748b; font-size: 11px; font-family: monospace;">14,200 msg/sec</div>
                </div>
            </div>
            <div style="width: 2px; height: 20px; background: #1e293b; margin-left: 14px; margin-top: -20px; margin-bottom: 5px;"></div>
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <div style="width: 30px; height: 30px; border-radius: 50%; background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.3); display: flex; align-items: center; justify-content: center;"><i class="fas fa-microchip" style="color: #818cf8; font-size: 12px;"></i></div>
                <div style="margin-left: 15px;">
                    <div style="color: #e2e8f0; font-size: 13px; font-weight: 600;">Redis Materialization</div>
                    <div style="color: #64748b; font-size: 11px; font-family: monospace;">Sub-5ms writes</div>
                </div>
            </div>
            <div style="width: 2px; height: 20px; background: #1e293b; margin-left: 14px; margin-top: -20px; margin-bottom: 5px;"></div>
            <div style="display: flex; align-items: center;">
                <div style="width: 30px; height: 30px; border-radius: 50%; background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); display: flex; align-items: center; justify-content: center;"><i class="fas fa-bolt" style="color: #34d399; font-size: 12px;"></i></div>
                <div style="margin-left: 15px;">
                    <div style="color: #e2e8f0; font-size: 13px; font-weight: 600;">PyTorch Inference</div>
                    <div style="color: #64748b; font-size: 11px; font-family: monospace;">Two-Tower + MMoE</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# VIEW: INFERENCE GATEWAY
# ==========================================
elif nav_selection == "Inference Gateway":
    st.markdown('<h2><i class="fas fa-bolt" style="color: #0ea5e9; margin-right: 10px;"></i> Recommendation Inference</h2>', unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 14px; margin-top: -10px; margin-bottom: 30px;'>Live execution of the PyTorch Multi-Task Ranker and FAISS Candidate Generator.</p>", unsafe_allow_html=True)
    
    col_controls, col_terminal = st.columns([1, 2])
    
    with col_controls:
        st.markdown("<div style='background: rgba(15,23,42,0.4); border: 1px solid #1e293b; border-radius: 12px; padding: 20px;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-bottom: 20px; font-size: 16px;'><i class='fas fa-sliders-h' style='color: #94a3b8; margin-right: 8px;'></i> Query Parameters</h4>", unsafe_allow_html=True)
        
        user_id = st.text_input("Target User ID", value="usr_7492_alpha")
        k_cands = st.slider("Retrieval Pool (K)", min_value=10, max_value=100, value=50, step=10)
        use_cache = st.checkbox("Bypass L1 Response Cache", value=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        execute_btn = st.button("EXECUTE PIPELINE", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_terminal:
        terminal_placeholder = st.empty()
        
        if execute_btn:
            # Simulated Terminal Output for Portfolio Demo
            lines = [
                "<span class='log-info'>[SYSTEM]</span> Initializing Request Pipeline...",
                "<span class='log-info'>[REDIS]</span> Fetching user features for: " + user_id + "...",
                "<span class='log-success'>[REDIS]</span> Done (1.2ms)",
                "<span class='log-info'>[PYTORCH]</span> Executing Two-Tower Candidate Gen...",
                "<span class='log-success'>[FAISS]</span> Retrieved " + str(k_cands) + " candidates (4.5ms)",
                "<span class='log-info'>[PYTORCH]</span> Rescoring via MMoE Ranker (Tasks: CTR, CVR)...",
                "<span class='log-success'>[PYTORCH]</span> Done (8.1ms)",
                "<br><span class='log-warn'>// Response Payload</span>",
                "{"
            ]
            
            # Simulate streaming terminal
            for i in range(1, len(lines)+1):
                out = "<br>".join(lines[:i])
                terminal_placeholder.markdown(f"<div class='console-box'>{out}</div>", unsafe_allow_html=True)
                time.sleep(0.15)
                
            # Add JSON Payload
            mock_payload = f"""
  <span style="color: #818cf8;">"user_id"</span>: <span style="color: #fbbf24;">"{user_id}"</span>,
  <span style="color: #818cf8;">"items"</span>: [
    {{
      <span style="color: #818cf8;">"item_id"</span>: <span style="color: #fbbf24;">"itm_992_headphones"</span>,
      <span style="color: #818cf8;">"p_ctr"</span>: <span style="color: #34d399;">0.8421</span>,
      <span style="color: #818cf8;">"p_cvr"</span>: <span style="color: #34d399;">0.1204</span>,
      <span style="color: #818cf8;">"final_score"</span>: <span style="color: #34d399;">0.6789</span>
    }},
    <span style="color: #64748b;">... ({k_cands - 1} more items)</span>
  ],
  <span style="color: #818cf8;">"latency_ms"</span>: <span style="color: #34d399;">14.2</span>
}}"""
            out = "<br>".join(lines) + "<br>" + mock_payload.replace('\n', '<br>')
            terminal_placeholder.markdown(f"<div class='console-box'>{out}</div>", unsafe_allow_html=True)
            
        else:
            # Default empty terminal
            terminal_placeholder.markdown("""
                <div class='console-box'>
                <span style='color: #64748b;'>// Awaiting API execution trigger...</span><br>
                <span class='log-info'>$</span> ready
                </div>
            """, unsafe_allow_html=True)

# ==========================================
# VIEW: MLOPS DAGSTER
# ==========================================
elif nav_selection == "MLOps Pipelines":
    st.markdown('<h2><i class="fas fa-project-diagram" style="color: #0ea5e9; margin-right: 10px;"></i> Dagster Orchestration</h2>', unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 14px; margin-top: -10px; margin-bottom: 30px;'>Automated drift detection and model retraining DAG.</p>", unsafe_allow_html=True)
    
    col_btn, _ = st.columns([1, 4])
    trigger_dag = col_btn.button("RUN DIAGNOSTICS")
    
    # CSS based DAG visualization
    st.markdown("""
        <div style="background: rgba(15,23,42,0.4); border: 1px solid #1e293b; border-radius: 12px; padding: 40px 20px; display: flex; justify-content: space-between; position: relative;">
            <div style="position: absolute; top: 50%; left: 5%; right: 5%; height: 2px; background: #1e293b; z-index: 0; transform: translateY(-50%);"></div>
            
            <div style="text-align: center; z-index: 1; background: #0a0f1c; padding: 0 10px;">
                <div style="width: 50px; height: 50px; border-radius: 50%; background: rgba(16,185,129,0.1); border: 2px solid #10b981; margin: 0 auto 10px; display: flex; align-items: center; justify-content: center;"><i class="fas fa-chart-line" style="color: #34d399; font-size: 20px;"></i></div>
                <div style="color: #f8fafc; font-size: 14px; font-weight: 600;">Drift Monitor</div>
                <div style="color: #64748b; font-size: 11px; font-family: monospace;">PSI > 0.1</div>
            </div>
            
            <div style="text-align: center; z-index: 1; background: #0a0f1c; padding: 0 10px;">
                <div style="width: 50px; height: 50px; border-radius: 50%; background: rgba(16,185,129,0.1); border: 2px solid #10b981; margin: 0 auto 10px; display: flex; align-items: center; justify-content: center;"><i class="fas fa-database" style="color: #34d399; font-size: 20px;"></i></div>
                <div style="color: #f8fafc; font-size: 14px; font-weight: 600;">Data Snapshot</div>
                <div style="color: #64748b; font-size: 11px; font-family: monospace;">PG Extract</div>
            </div>
            
            <div style="text-align: center; z-index: 1; background: #0a0f1c; padding: 0 10px;">
                <div style="width: 50px; height: 50px; border-radius: 50%; background: rgba(14,165,233,0.1); border: 2px solid #0ea5e9; margin: 0 auto 10px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 15px rgba(14,165,233,0.4);"><i class="fas fa-microchip" style="color: #38bdf8; font-size: 20px;"></i></div>
                <div style="color: #f8fafc; font-size: 14px; font-weight: 600;">Ray Training</div>
                <div style="color: #64748b; font-size: 11px; font-family: monospace;">PyTorch DDP</div>
            </div>
            
            <div style="text-align: center; z-index: 1; background: #0a0f1c; padding: 0 10px;">
                <div style="width: 50px; height: 50px; border-radius: 50%; background: #0f172a; border: 2px solid #334155; margin: 0 auto 10px; display: flex; align-items: center; justify-content: center;"><i class="fas fa-server" style="color: #64748b; font-size: 20px;"></i></div>
                <div style="color: #94a3b8; font-size: 14px; font-weight: 600;">Model Registry</div>
                <div style="color: #475569; font-size: 11px; font-family: monospace;">MLflow Log</div>
            </div>
            
            <div style="text-align: center; z-index: 1; background: #0a0f1c; padding: 0 10px;">
                <div style="width: 50px; height: 50px; border-radius: 50%; background: #0f172a; border: 2px solid #334155; margin: 0 auto 10px; display: flex; align-items: center; justify-content: center;"><i class="fas fa-check-circle" style="color: #64748b; font-size: 20px;"></i></div>
                <div style="color: #94a3b8; font-size: 14px; font-weight: 600;">Shadow Deploy</div>
                <div style="color: #475569; font-size: 11px; font-family: monospace;">Triton Server</div>
            </div>
        </div>
        <br>
    """, unsafe_allow_html=True)
    
    log_placeholder = st.empty()
    
    if trigger_dag:
        logs = [
            "<span class='log-success'>[2026-05-29 14:02:11] SUCCESS: Asset drift_detection_alert materialized.</span>",
            "<span class='log-success'>[2026-05-29 14:02:15] SUCCESS: Extracted 1,000,000 rows to offline Parquet store.</span>",
            "<span class='log-info'>[2026-05-29 14:02:16] INFO: Provisioning Ray cluster for distributed training...</span>",
            "<span style='color: #cbd5e1;'>[2026-05-29 14:02:18] INFO: Epoch 1/10 | Contrastive InfoNCE Loss: 0.1245</span>",
            "<span style='color: #cbd5e1;'>[2026-05-29 14:02:22] INFO: Epoch 2/10 | Contrastive InfoNCE Loss: 0.0891</span>",
            "<span style='color: #cbd5e1;'>[2026-05-29 14:02:25] INFO: Epoch 3/10 | Contrastive InfoNCE Loss: 0.0412</span>",
            "<span class='log-success'>[2026-05-29 14:02:28] SUCCESS: Ray Training Complete.</span>"
        ]
        
        for i in range(1, len(logs)+1):
            out = "<br>".join(logs[:i])
            log_placeholder.markdown(f"<div class='console-box' style='height: 250px;'>{out}<br><span class='log-info'>_</span></div>", unsafe_allow_html=True)
            time.sleep(0.4)
    else:
        log_placeholder.markdown("<div class='console-box' style='height: 250px;'><span style='color: #64748b;'>// Execution log empty. Trigger DAG to begin.</span></div>", unsafe_allow_html=True)

elif nav_selection == "Semantic Search":
    st.markdown('<h2><i class="fas fa-search" style="color: #0ea5e9; margin-right: 10px;"></i> Semantic Search Gateway</h2>', unsafe_allow_html=True)
    st.info("Module linked to internal LambdaMART LTR APIs. (UI rendering pending backend sync)")