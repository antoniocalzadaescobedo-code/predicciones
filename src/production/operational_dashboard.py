import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.production.live_observability import LiveObservabilityEngine
from src.production.live_guardrails import LiveGuardrailsEngine
from src.production.incident_registry import IncidentRegistry, IncidentType, Severity
from src.production.live_drift_analysis import LiveDriftAnalyzer
from src.production.feature_registry import FeatureRegistry

# Page configuration
st.set_page_config(
    page_title="FIFA Production Observability Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-pass {
        color: #00cc00;
        font-weight: bold;
    }
    .status-fail {
        color: #ff0000;
        font-weight: bold;
    }
    .status-warn {
        color: #ff9900;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'observability_engine' not in st.session_state:
    st.session_state.observability_engine = None
if 'guardrails_engine' not in st.session_state:
    st.session_state.guardrails_engine = None
if 'incident_registry' not in st.session_state:
    st.session_state.incident_registry = IncidentRegistry()
if 'drift_analyzer' not in st.session_state:
    st.session_state.drift_analyzer = LiveDriftAnalyzer()
if 'feature_registry' not in st.session_state:
    st.session_state.feature_registry = FeatureRegistry()

def initialize_systems():
    """Initialize all production systems."""
    try:
        registry = st.session_state.feature_registry
        observability = LiveObservabilityEngine(
            feature_registry=registry,
            psi_threshold=0.20,
            fallback_threshold=0.05
        )
        guardrails = LiveGuardrailsEngine(
            observability_engine=observability,
            incident_registry=st.session_state.incident_registry
        )
        
        st.session_state.observability_engine = observability
        st.session_state.guardrails_engine = guardrails
        
        return True
    except Exception as e:
        st.error(f"Failed to initialize systems: {e}")
        return False

def render_header():
    """Render dashboard header."""
    st.markdown('<div class="main-header">⚽ FIFA Production Observability Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f"**Last Updated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

def render_feature_status():
    """Render feature status section."""
    st.subheader("📊 Feature Status")
    
    registry = st.session_state.feature_registry
    approved = registry.get_approved_features()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Approved Features", len(approved))
    
    with col2:
        st.metric("Experimental Features", 
                 len([f for f, meta in registry._registry.items() if meta.status.value == "EXPERIMENTAL"]))
    
    with col3:
        st.metric("Rejected Features", 
                 len([f for f, meta in registry._registry.items() if meta.status.value == "REJECTED"]))
    
    # Feature details table
    feature_data = []
    for feat_name, meta in registry._registry.items():
        feature_data.append({
            "Feature": feat_name,
            "Status": meta.status.value,
            "Calibration Impact": meta.calibration_impact,
            "Drift Risk": meta.drift_risk,
            "Production Owner": meta.production_owner
        })
    
    df_features = pd.DataFrame(feature_data)
    st.dataframe(df_features, use_container_width=True)

def render_uplift_status():
    """Render uplift system status."""
    st.subheader("🚀 Uplift System Status")
    
    if st.session_state.guardrails_engine:
        guardrail_status = st.session_state.guardrails_engine.get_guardrail_status()
        
        col1, col2 = st.columns(2)
        
        with col1:
            uplift_status = "❌ DISABLED" if guardrail_status["uplift_disabled"] else "✅ ENABLED"
            st.metric("Uplift Status", uplift_status)
            
            if guardrail_status["uplift_disabled_since"]:
                st.info(f"Disabled since: {guardrail_status['uplift_disabled_since']}")
        
        with col2:
            disabled_count = len(guardrail_status["disabled_features"])
            st.metric("Disabled Features", disabled_count)
            
            if disabled_count > 0:
                st.warning(f"Disabled: {', '.join(guardrail_status['disabled_features'].keys())}")
        
        # Guardrail configurations
        st.markdown("### Guardrail Configurations")
        guardrail_configs = guardrail_status["guardrail_configs"]
        
        config_data = []
        for key, config in guardrail_configs.items():
            config_data.append({
                "Guardrail": key,
                "Enabled": "✅" if config["enabled"] else "❌",
                "Threshold": config["threshold"],
                "Action": config["action"],
                "In Cooldown": "⏸️" if config["in_cooldown"] else "▶️"
            })
        
        df_configs = pd.DataFrame(config_data)
        st.dataframe(df_configs, use_container_width=True)

def render_prediction_health():
    """Render prediction health metrics."""
    st.subheader("🎯 Prediction Health")
    
    if st.session_state.observability_engine:
        summary = st.session_state.observability_engine.get_health_summary()
        pred_health = summary["prediction_health"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", pred_health["total_predictions"])
        
        with col2:
            nan_status = "✅" if pred_health["nan_probabilities"] == 0 else "❌"
            st.metric("NaN Probabilities", f"{pred_health['nan_probabilities']} {nan_status}")
        
        with col3:
            invalid_status = "✅" if pred_health["invalid_lambdas"] == 0 else "❌"
            st.metric("Invalid Lambdas", f"{pred_health['invalid_lambdas']} {invalid_status}")
        
        with col4:
            prob_dev = pred_health["probability_sum_deviation"]
            dev_status = "✅" if prob_dev < 0.01 else "⚠️" if prob_dev < 0.05 else "❌"
            st.metric("Prob Sum Deviation", f"{prob_dev:.4f} {dev_status}")

def render_operational_metrics():
    """Render operational health metrics."""
    st.subheader("⚙️ Operational Metrics")
    
    if st.session_state.observability_engine:
        summary = st.session_state.observability_engine.get_health_summary()
        ops_health = summary["operational_health"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Requests", ops_health["total_requests"])
        
        with col2:
            fallback_rate = ops_health["fallback_rate"]
            fallback_status = "✅" if fallback_rate < 0.05 else "⚠️" if fallback_rate < 0.10 else "❌"
            st.metric("Fallback Rate", f"{fallback_rate:.2%} {fallback_status}")
        
        with col3:
            leakage_rate = ops_health["leakage_rejection_rate"]
            st.metric("Leakage Rejection", f"{leakage_rate:.2%}")
        
        with col4:
            divergence = ops_health["shadow_live_divergence"]
            div_status = "✅" if divergence < 0.10 else "⚠️"
            st.metric("Shadow/Live Divergence", f"{divergence:.2%} {div_status}")
        
        # Latency metrics
        st.markdown("### Latency Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("P50 Latency", f"{ops_health['latency_p50']:.2f} ms")
        
        with col2:
            st.metric("P95 Latency", f"{ops_health['latency_p95']:.2f} ms")
        
        with col3:
            st.metric("P99 Latency", f"{ops_health['latency_p99']:.2f} ms")

def render_calibration_health():
    """Render calibration health metrics."""
    st.subheader("📈 Calibration Health")
    
    if st.session_state.observability_engine:
        summary = st.session_state.observability_engine.get_health_summary()
        calib_health = summary["calibration_health"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sample Size", calib_health["sample_size"])
        
        with col2:
            brier = calib_health["rolling_brier"]
            st.metric("Rolling Brier", f"{brier:.4f}")
        
        with col3:
            slope = calib_health["rolling_calibration_slope"]
            slope_range = st.session_state.observability_engine.calibration_slope_range
            slope_status = "✅" if slope_range[0] <= slope <= slope_range[1] else "❌"
            st.metric("Calibration Slope", f"{slope:.4f} {slope_status}")
        
        with col4:
            ece = calib_health["rolling_ece"]
            st.metric("Rolling ECE", f"{ece:.4f}")
        
        # Calibration slope visualization
        if calib_health["sample_size"] > 0:
            st.markdown("### Calibration Slope Target")
            
            fig = go.Figure()
            
            # Target range
            fig.add_hrect(
                y0=slope_range[0], y1=slope_range[1],
                fillcolor="green", opacity=0.2,
                annotation_text="Target Range"
            )
            
            # Current value
            fig.add_trace(go.Scatter(
                x=[0], y=[slope],
                mode='markers',
                marker=dict(size=20, color='red'),
                name='Current Slope'
            ))
            
            fig.update_layout(
                title="Calibration Slope vs Target",
                yaxis_title="Slope",
                xaxis_showgrid=False,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_drift_metrics():
    """Render drift analysis metrics."""
    st.subheader("🌊 Drift Analysis")
    
    if st.session_state.drift_analyzer:
        drift_summary = st.session_state.drift_analyzer.get_drift_summary()
        
        # Regime state
        regime = drift_summary["regime_state"]
        col1, col2 = st.columns(2)
        
        with col1:
            regime_status = "⚠️ POST-TOURNAMENT" if regime["is_post_tournament"] else "✅ NORMAL"
            st.metric("Regime Status", regime_status)
            
            if regime["is_post_tournament"]:
                st.warning(f"Days since tournament: {regime['days_since_tournament']}")
        
        with col2:
            st.metric("Regime Confidence", f"{regime['regime_confidence']:.2%}")
        
        # Reference distributions
        st.markdown("### Reference Distributions")
        ref_dist = drift_summary["reference_distributions"]
        
        if ref_dist:
            ref_data = []
            for feat, info in ref_dist.items():
                ref_data.append({
                    "Feature": feat,
                    "Sample Count": info["sample_count"],
                    "Set At": info["set_at"]
                })
            
            df_ref = pd.DataFrame(ref_data)
            st.dataframe(df_ref, use_container_width=True)
        else:
            st.info("No reference distributions set")

def render_live_incidents():
    """Render live incidents section."""
    st.subheader("🚨 Live Incidents")
    
    incident_registry = st.session_state.incident_registry
    summary = incident_registry.get_incident_summary()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Incidents", summary["total_incidents"])
    
    with col2:
        st.metric("Active Incidents", summary["active_incidents"])
    
    with col3:
        st.metric("Resolved Incidents", summary["resolved_incidents"])
    
    # Incident type breakdown
    st.markdown("### Incidents by Type")
    by_type = summary["by_type"]
    
    if by_type:
        fig = px.bar(
            x=list(by_type.keys()),
            y=list(by_type.values()),
            title="Incidents by Type",
            labels={"x": "Incident Type", "y": "Count"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Severity breakdown
    st.markdown("### Incidents by Severity")
    by_severity = summary["by_severity"]
    
    if by_severity:
        severity_colors = {
            "INFO": "blue",
            "WARNING": "orange",
            "CRITICAL": "red",
            "EMERGENCY": "purple"
        }
        
        fig = px.pie(
            values=list(by_severity.values()),
            names=list(by_severity.keys()),
            title="Incidents by Severity",
            color=list(by_severity.keys()),
            color_discrete_map=severity_colors
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent incidents
    st.markdown("### Recent Incidents")
    recent = summary["recent_incidents"]
    
    if recent:
        incident_data = []
        for inc in recent:
            incident_data.append({
                "ID": inc["incident_id"][:8],
                "Type": inc["incident_type"],
                "Severity": inc["severity"],
                "Message": inc["message"],
                "Timestamp": inc["timestamp_utc"],
                "Resolved": "✅" if inc["resolved"] else "❌"
            })
        
        df_incidents = pd.DataFrame(incident_data)
        st.dataframe(df_incidents, use_container_width=True)
    else:
        st.info("No recent incidents")

def render_system_actions():
    """Render system action buttons."""
    st.subheader("🔧 System Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Initialize Systems", key="init"):
            if initialize_systems():
                st.success("Systems initialized successfully!")
                st.rerun()
    
    with col2:
        if st.button("Reset Metrics", key="reset"):
            if st.session_state.observability_engine:
                st.session_state.observability_engine.reset_metrics()
                st.success("Metrics reset!")
                st.rerun()
    
    with col3:
        if st.button("Clear Old Incidents", key="clear"):
            st.session_state.incident_registry.clear_old_incidents(days_to_keep=30)
            st.success("Old incidents cleared!")
            st.rerun()
    
    # Uplift control
    st.markdown("### Uplift Control")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Enable Uplift", key="enable_uplift"):
            if st.session_state.guardrails_engine:
                st.session_state.guardrails_engine.enable_uplift()
                st.success("Uplift enabled!")
                st.rerun()
    
    with col2:
        if st.button("Disable Uplift", key="disable_uplift"):
            if st.session_state.guardrails_engine:
                st.session_state.guardrails_engine._disable_uplift()
                st.warning("Uplift disabled!")
                st.rerun()

def main():
    """Main dashboard function."""
    render_header()
    
    # Auto-initialize if needed
    if st.session_state.observability_engine is None:
        with st.spinner("Initializing production systems..."):
            initialize_systems()
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Overview", "Feature Status", "Prediction Health", "Operational Metrics", 
         "Calibration", "Drift Analysis", "Incidents", "System Actions"]
    )
    
    if page == "Overview":
        render_feature_status()
        st.markdown("---")
        render_uplift_status()
        st.markdown("---")
        render_prediction_health()
        st.markdown("---")
        render_operational_metrics()
    
    elif page == "Feature Status":
        render_feature_status()
    
    elif page == "Prediction Health":
        render_prediction_health()
    
    elif page == "Operational Metrics":
        render_operational_metrics()
    
    elif page == "Calibration":
        render_calibration_health()
    
    elif page == "Drift Analysis":
        render_drift_metrics()
    
    elif page == "Incidents":
        render_live_incidents()
    
    elif page == "System Actions":
        render_system_actions()
    
    # Auto-refresh option
    if st.sidebar.checkbox("Auto-refresh (30s)", value=False):
        import time
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
