import streamlit as st
import pyvista as pv
pv.OFF_SCREEN = True
from stpyvista import stpyvista
import numpy as np
import pandas as pd
import plotly.express as px
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frontend.simulation_engine import AntigravityEngine
from antigravity_data import PublicDataEngine, RealLiteratureEngine

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ANTIGRAVITY | Researcher Console",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚛️"
)

# ... (CSS remains same) ...

# --- INITIALIZE ENGINES ---
@st.cache_resource
def load_engines():
    # Load Simulator
    sim_engine = AntigravityEngine()
    # Load Scientific Data
    data_engine = PublicDataEngine()
    real_engine = RealLiteratureEngine()
    # Pre-fetch mirror data
    c2db_data = data_engine.fetch_c2db_mirror()
    dtu_data = data_engine.fetch_dtu_defect_mirror()
    # Load Real Experimental Data
    real_data = real_engine.load_literature_data()
    return sim_engine, c2db_data, dtu_data, real_data

# Lazy load 
try:
    engine, df_c2db, df_dtu, df_hts = load_engines()
except Exception as e:
    st.error(f"Failed to load Engines: {e}")
    st.stop()

# --- SIDEBAR CONTROLS ---
st.sidebar.title("🎛️ CONTROL DECK")
st.sidebar.markdown("---")
mode = st.sidebar.radio("Research Mode", [
    "🚀 Discovery & Screening (HTS)",
    "Material Explorer (Public Data)", 
    "AI Assistant (Dopant Guide)",
    "Synthesis Cockpit (Digital Twin)"
])

# --- TAB 0: REAL LITERATURE DATABASE ---
if mode == "🚀 Discovery & Screening (HTS)":
    st.title("📚 Real Experimental Database")
    st.markdown("Verified synthesis conditions and properties from literature (MatSyn25 / ResearchGate). NO synthetic data.")
    
    st.info("💡 **Dataset Source**: Manually curated from recent high-impact 2D material papers (2019-2023). Values are experimental.")
    
    # Simple Filters
    host_filter = st.multiselect("Filter by Material", options=list(df_hts["Material"].unique()), default=[])
    dopant_filter = st.multiselect("Filter by Dopant", options=list(df_hts["Dopant"].unique()), default=[])
    
    # Filter Data
    view_df = df_hts.copy()
    if host_filter:
        view_df = view_df[view_df["Material"].isin(host_filter)]
    if dopant_filter:
        view_df = view_df[view_df["Dopant"].isin(dopant_filter)]
        
    st.markdown(f"### 🧪 {len(view_df)} Verified Experiments Found")
    st.dataframe(
        view_df, 
        use_container_width=True,
        column_config={
            "Reference": st.column_config.TextColumn("Source Paper", help="Citation"),
            "Bandgap_eV": st.column_config.NumberColumn("Bandgap (eV)", format="%.2f"),
            "Mobility_cm2Vs": st.column_config.NumberColumn("Mobility (cm²/Vs)", format="%.1f")
        }
    )
    
    st.markdown("### 📊 Experimental Trends")
    c1, c2 = st.columns(2)
    with c1:
        # Real Data Plot: Mobility vs Temp
        fig = px.scatter(
            view_df, 
            x="Temp_C", 
            y="Mobility_cm2Vs", 
            color="Dopant", 
            symbol="Material",
            hover_data=["Reference"],
            title="Temp vs Mobility (Real Experiments)",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        # Real Data Plot: Bandgap vs Conc
        fig2 = px.scatter(
            view_df[view_df["Dopant"] != "None"], 
            x="Concentration_%", 
            y="Bandgap_eV", 
            color="Dopant", 
            symbol="Material",
            title="Doping Concentration vs Bandgap",
            template="plotly_dark"
        )
        st.plotly_chart(fig2, use_container_width=True)

# --- TAB 1: MATERIAL EXPLORER (PUBLIC DATA) ---
elif mode == "Material Explorer (Public Data)":
    st.title("🌍 Scientific Data Explorer (Public Data)")
    st.markdown("Explore public 2D materials data (MoS2, WS2, etc.) loaded from Verified Sources.")
    
    if df_c2db.empty:
        st.error("⚠️ DATA ERROR: C2DB Reference Data could not be loaded. Please check antigravity_data.py")
    else:
        st.success(f"✅ Loaded {len(df_c2db)} Standard Reference Materials (MoS2, WS2, Graphene...)")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### Filters")
        min_bg, max_bg = st.slider("Bandgap Range (eV)", 0.0, 3.0, (1.0, 2.0))
        min_mob = st.slider("Min Mobility (cm²/Vs)", 0, 200, 50)
        
        # Filter Data
        filtered_df = df_c2db[
            (df_c2db["Bandgap_eV"] >= min_bg) & 
            (df_c2db["Bandgap_eV"] <= max_bg) &
            (df_c2db["Mobility_cm2Vs"] >= min_mob)
        ]
        
        st.metric("Candidates Found", len(filtered_df), delta=f"{len(filtered_df)/len(df_c2db):.1%}")
        
    with col2:
        # Plotly Visualization
        st.markdown("### Property Landscape")
        fig = px.scatter(
            filtered_df, 
            x="Bandgap_eV", 
            y="Mobility_cm2Vs", 
            color="Stability",
            size="Exfoliation_Energy",
            hover_data=["Formula"],
            template="plotly_dark",
            title="Bandgap vs Mobility (Size = Exfoliation Energy)",
            color_discrete_map={"High": "#238636", "Medium": "#d29922", "Low": "#da3633"}
        )
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("### Dataset Preview")
    st.dataframe(filtered_df.head(10), use_container_width=True)

# --- TAB 2: AI ASSISTANT (DTU DOPANTS) ---
elif mode == "AI Assistant (Dopant Guide)":
    st.title("🤖 AI Research Assistant")
    st.markdown("Expert guidance on dopant selection based on the DTU Impurities Database.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Query")
        target_prop = st.selectbox("I want to improve:", ["n-type conductivity", "p-type conductivity", "Magnetism", "Catalytic Activity"])
        host_mat = st.selectbox("Target Material:", ["MoS2", "WS2", "WSe2"])
        
        # Simple Expert System Logic based on DTU DF
        if target_prop == "n-type conductivity":
            rec = df_dtu[(df_dtu["Host"] == host_mat) & (df_dtu["Type"] == "n-type")]
        elif target_prop == "p-type conductivity":
            rec = df_dtu[(df_dtu["Host"] == host_mat) & (df_dtu["Type"] == "p-type")]
        elif target_prop == "Magnetism":
            rec = df_dtu[(df_dtu["Host"] == host_mat) & (df_dtu["Type"] == "Magnetic")]
        else:
            rec = df_dtu[(df_dtu["Host"] == host_mat) & (df_dtu["Type"] == "Catalytic")]
            
    with col2:
        st.markdown("### AI Advice")
        if not rec.empty:
            best_pick = rec.iloc[0]
            st.success(f"**Recommendation:** Doping with **{best_pick['Dopant']}**")
            st.info(f"**Mechanism:** {best_pick['Effect_Description']}")
            st.write(f"**Formation Energy:** {best_pick['Formation_Energy_eV']} eV (Lower = Easier to Dope)")
        else:
            st.warning("No suitable dopant found in current literature database.")

# --- TAB 3: SYNTHESIS COCKPIT ---
elif mode == "Synthesis Cockpit (Digital Twin)":
    st.title("🔥 Synthesis Cockpit: CVD Furnace Twin")
    
    # Dashboard Layout
    c1, c2, c3 = st.columns(3)
    temp = c1.slider("Furnace Temp (°C)", 500, 1200, 750)
    pressure = c2.slider("Chamber Pressure (Torr)", 0.1, 760.0, 10.0)
    flow = c3.slider("Gas Flow (sccm)", 0, 100, 20)
    
    st.markdown("---")
    
    # FNO Real-time Visualization
    col_main, col_metrics = st.columns([3, 1])
    
    with col_main:
        st.markdown("### 🌊 Physics-Informed Growth Field (Arrhenius Kinetics)")
        grid = engine.run_synthesis_twin(temp, pressure, flow)
        
        # Extract Data for 2D Plot (Stable on Mac)
        conc_data = grid["Gas Concentration"].reshape((64, 64))
        
        fig = px.imshow(
            conc_data,
            color_continuous_scale="inferno",
            origin="lower",
            labels=dict(x="Reactor X", y="Reactor Y", color="Concentration"),
            title="Real-time Growth Rate Distrubution"
        )
        fig.update_layout(height=500, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#c9d1d9")
        st.plotly_chart(fig, use_container_width=True)
        
    with col_metrics:
        st.markdown("### Explainable AI Metrics")
        # Scientific Explainability for Researcher
        growth_rate = np.exp(-50000 / (8.314 * (temp + 273))) * pressure * flow # Approx Arrhenius
        
        st.metric("Predicted Growth Rate", f"{growth_rate:.2e} ML/s")
        st.metric("Kinetic Regime", "Mass Transport" if temp > 800 else "Surface Reaction")
        st.metric("Reaction Efficiency", f"{(temp/1200)*100:.1f}%")
