import streamlit as st
import requests
import json
import subprocess
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_digital_twin import EnhancedDigitalTwin

st.set_page_config(
    page_title="CsPbBr₃ Digital Twin",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_digital_twin():
    """Load the digital twin system"""
    return EnhancedDigitalTwin()

def main():
    st.title("🧬 CsPbBr₃ Digital Twin")
    st.markdown("AI-Powered Perovskite Synthesis Prediction & Optimization")
    
    # Load system
    try:
        twin = load_digital_twin()
        st.success("✅ Digital Twin System Loaded")
    except Exception as e:
        st.error(f"❌ Error loading system: {e}")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🔮 Prediction", 
        "🤖 AI Suggestions", 
        "🎯 Optimization",
        "📊 System Status"
    ])
    
    if page == "🔮 Prediction":
        prediction_page(twin)
    elif page == "🤖 AI Suggestions":
        suggestions_page(twin)
    elif page == "🎯 Optimization":
        optimization_page(twin)
    elif page == "📊 System Status":
        status_page(twin)

def prediction_page(twin):
    st.header("🔮 Synthesis Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Synthesis Conditions")
        
        cs_conc = st.slider("Cs-Br Concentration (mol/L)", 0.5, 3.0, 1.1, 0.1)
        pb_conc = st.slider("Pb-Br₂ Concentration (mol/L)", 0.5, 3.0, 1.1, 0.1)
        temperature = st.slider("Temperature (°C)", 80, 300, 190, 10)
        oa_conc = st.slider("Oleic Acid (mol/L)", 0.0, 2.0, 0.4, 0.1)
        oam_conc = st.slider("Oleylamine (mol/L)", 0.0, 1.0, 0.2, 0.1)
        reaction_time = st.slider("Reaction Time (min)", 5, 200, 75, 5)
        solvent = st.selectbox("Solvent", ["DMSO", "DMF", "Toluene", "Octadecene", "Mixed"])
        
        solvent_map = {"DMSO": 0, "DMF": 1, "Toluene": 2, "Octadecene": 3, "Mixed": 4}
        
        if st.button("🔮 Predict Outcome"):
            conditions = {
                'cs_br_concentration': cs_conc,
                'pb_br2_concentration': pb_conc,
                'temperature': temperature,
                'oa_concentration': oa_conc,
                'oam_concentration': oam_conc,
                'reaction_time': reaction_time,
                'solvent_type': solvent_map[solvent]
            }
            
            with st.spinner("Making prediction..."):
                prediction = twin.predict_with_uncertainty(conditions)
            
            display_prediction(prediction)
    
    with col2:
        st.subheader("Prediction Results")
        st.info("Enter synthesis conditions and click 'Predict Outcome' to see results.")

def display_prediction(prediction):
    st.subheader("📊 Prediction Results")
    
    # Main prediction
    phase = prediction['predicted_phase']
    confidence = prediction['confidence']
    
    if prediction['has_uncertainty']:
        uncertainty = prediction['uncertainty_score']
        st.metric("Predicted Phase", phase, f"Confidence: {confidence:.1%}")
        st.metric("Uncertainty Score", f"{uncertainty:.3f}", "Lower is better")
    else:
        st.metric("Predicted Phase", phase, f"Confidence: {confidence:.1%}")
    
    # Phase probabilities
    st.subheader("Phase Probabilities")
    phase_names = ['CsPbBr₃ 3D', 'Cs₄PbBr₆ 0D', 'CsPb₂Br₅ 2D', 'Mixed', 'Failed']
    probs = prediction['phase_probabilities']
    
    for name, prob in zip(phase_names, probs):
        st.progress(prob, text=f"{name}: {prob:.1%}")
    
    # Properties
    if prediction['properties']:
        st.subheader("Predicted Properties")
        for prop_name, prop_data in prediction['properties'].items():
            if prediction['has_uncertainty']:
                mean_val = prop_data['mean']
                std_val = prop_data['std']
                st.metric(
                    prop_name.replace('_', ' ').title(),
                    f"{mean_val:.3f} ± {std_val:.3f}",
                    f"95% CI: [{prop_data['confidence_interval'][0]:.3f}, {prop_data['confidence_interval'][1]:.3f}]"
                )
            else:
                st.metric(prop_name.replace('_', ' ').title(), f"{prop_data['mean']:.3f}")

def suggestions_page(twin):
    st.header("🤖 AI Experiment Suggestions")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        strategy = st.selectbox("Strategy", ["mixed", "uncertainty", "bayesian", "diversity"])
        num_suggestions = st.slider("Number of Suggestions", 1, 10, 3)
        
        if st.button("🎯 Generate Suggestions"):
            with st.spinner("Generating AI suggestions..."):
                suggestions = twin.suggest_experiments(
                    strategy=strategy,
                    num_suggestions=num_suggestions
                )
            
            st.session_state['suggestions'] = suggestions
    
    with col2:
        if 'suggestions' in st.session_state:
            st.subheader("Experiment Suggestions")
            
            for i, suggestion in enumerate(st.session_state['suggestions']):
                with st.expander(f"Suggestion {i+1} - Priority: {suggestion['priority'].title()}"):
                    st.write(f"**Rationale**: {suggestion['rationale']}")
                    
                    # Conditions
                    conditions = suggestion['conditions']
                    st.write("**Conditions**:")
                    st.write(f"- Temperature: {conditions['temperature']:.0f}°C")
                    st.write(f"- Cs-Br: {conditions['cs_br_concentration']:.2f} mol/L")
                    st.write(f"- Pb-Br₂: {conditions['pb_br2_concentration']:.2f} mol/L")
                    st.write(f"- Oleic Acid: {conditions['oa_concentration']:.2f} mol/L")
                    st.write(f"- Oleylamine: {conditions['oam_concentration']:.2f} mol/L")
                    
                    # Prediction
                    predicted = suggestion['predicted_outcome']
                    st.write(f"**Predicted**: {predicted['predicted_phase']} (confidence: {predicted['confidence']:.1%})")

def optimization_page(twin):
    st.header("🎯 Synthesis Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        objective = st.selectbox("Optimization Objective", [
            "cspbbr3_probability", "confidence", "uncertainty"
        ])
        method = st.selectbox("Method", ["bayesian", "grid"])
        
        if st.button("⚡ Optimize"):
            with st.spinner("Running optimization..."):
                result = twin.optimize_synthesis(objective=objective, method=method)
            
            if result['success']:
                st.success("✅ Optimization Complete")
                
                # Display optimal conditions
                optimal = result['optimal_conditions']
                st.subheader("Optimal Conditions")
                st.write(f"Temperature: {optimal['temperature']:.0f}°C")
                st.write(f"Cs-Br: {optimal['cs_br_concentration']:.2f} mol/L")
                st.write(f"Pb-Br₂: {optimal['pb_br2_concentration']:.2f} mol/L")
                
                # Display prediction
                pred = result['prediction']
                st.subheader("Expected Outcome")
                st.write(f"Phase: {pred['predicted_phase']}")
                st.write(f"Confidence: {pred['confidence']:.1%}")
                
            else:
                st.error(f"❌ Optimization failed: {result.get('error', 'Unknown error')}")
    
    with col2:
        st.info("Select optimization parameters and click 'Optimize' to find optimal synthesis conditions.")

def status_page(twin):
    st.header("📊 System Status")
    
    info = twin.get_system_info()
    
    # System overview
    st.subheader("System Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("System Status", info['system_status'].title())
    
    with col2:
        st.metric("Model Type", info['model_info']['uncertainty_model_type'] or 'Baseline')
    
    with col3:
        st.metric("Features", info['model_info']['feature_count'])
    
    # Capabilities
    st.subheader("Capabilities")
    capabilities = info['capabilities']
    
    for capability, available in capabilities.items():
        status = "✅ Available" if available else "❌ Not Available"
        st.write(f"**{capability.replace('_', ' ').title()}**: {status}")
    
    # Validation status
    validation_status = twin.get_validation_status()
    if 'validation_metrics' in validation_status:
        metrics = validation_status['validation_metrics']
        
        st.subheader("Validation Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Experiments", metrics['total_experiments'])
        with col2:
            st.metric("Phase Accuracy", f"{metrics['phase_accuracy']:.1f}%")
        with col3:
            st.metric("Mean Confidence", f"{metrics['mean_confidence']:.3f}")

if __name__ == "__main__":
    main()