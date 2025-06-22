#!/usr/bin/env python3
"""
Heroku Deployment Setup for CsPbBr₃ Digital Twin
Creates necessary files for cloud deployment
"""

import os
from pathlib import Path

def create_heroku_files():
    """Create Heroku deployment files"""
    
    # Procfile for Heroku
    procfile_content = """web: python web_interface.py"""
    
    with open("Procfile", "w") as f:
        f.write(procfile_content)
    
    # Runtime specification
    runtime_content = """python-3.8.18"""
    
    with open("runtime.txt", "w") as f:
        f.write(runtime_content)
    
    # Requirements for Heroku
    requirements_content = """
Flask==2.3.3
torch==2.0.1
torchvision==0.15.2
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
Pillow==9.5.0
requests==2.31.0
gunicorn==21.2.0
"""
    
    with open("requirements_heroku.txt", "w") as f:
        f.write(requirements_content.strip())
    
    # Heroku-compatible web interface
    heroku_web_content = '''#!/usr/bin/env python3
"""
Heroku-compatible Web Interface for CsPbBr₃ Digital Twin
"""

import os
from web_interface import app, initialize_system

if __name__ == "__main__":
    # Initialize the system
    initialize_system()
    
    # Get port from environment (Heroku requirement)
    port = int(os.environ.get("PORT", 5000))
    
    # Run with production settings
    app.run(host="0.0.0.0", port=port, debug=False)
'''
    
    with open("heroku_app.py", "w") as f:
        f.write(heroku_web_content)
    
    # Update Procfile to use gunicorn
    procfile_prod = """web: gunicorn heroku_app:app --bind 0.0.0.0:$PORT"""
    
    with open("Procfile", "w") as f:
        f.write(procfile_prod)
    
    print("✅ Heroku deployment files created:")
    print("   - Procfile")
    print("   - runtime.txt") 
    print("   - requirements_heroku.txt")
    print("   - heroku_app.py")
    
    print("\n🚀 Deploy to Heroku:")
    print("1. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli")
    print("2. heroku create your-app-name")
    print("3. git init && git add . && git commit -m 'Initial deployment'")
    print("4. heroku git:remote -a your-app-name")
    print("5. git push heroku main")
    print("6. heroku open")

def create_docker_files():
    """Create Docker deployment files"""
    
    dockerfile_content = """
FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for models and data
RUN mkdir -p uncertainty_training_output experimental_data

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=web_interface.py
ENV FLASK_ENV=production

# Run the application
CMD ["python", "web_interface.py"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content.strip())
    
    # Docker Compose for easy local deployment
    docker_compose_content = """
version: '3.8'

services:
  digital-twin:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./experimental_data:/app/experimental_data
      - ./uncertainty_training_output:/app/uncertainty_training_output
    environment:
      - FLASK_ENV=production
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose_content.strip())
    
    print("\n✅ Docker deployment files created:")
    print("   - Dockerfile")
    print("   - docker-compose.yml")
    
    print("\n🐳 Deploy with Docker:")
    print("1. docker build -t cspbbr3-digital-twin .")
    print("2. docker run -p 5000:5000 cspbbr3-digital-twin")
    print("3. Or use: docker-compose up")
    print("4. Access at: http://localhost:5000")

def create_railway_files():
    """Create Railway deployment files"""
    
    railway_toml = """
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "python web_interface.py"
healthcheckPath = "/"
healthcheckTimeout = 100
restartPolicyType = "ON_FAILURE"

[env]
PYTHON_VERSION = "3.8"
"""
    
    with open("railway.toml", "w") as f:
        f.write(railway_toml.strip())
    
    # Railway start script
    start_script = """#!/bin/bash
echo "Starting CsPbBr₃ Digital Twin..."
python web_interface.py
"""
    
    with open("start.sh", "w") as f:
        f.write(start_script.strip())
    
    os.chmod("start.sh", 0o755)
    
    print("\n✅ Railway deployment files created:")
    print("   - railway.toml")
    print("   - start.sh")
    
    print("\n🚂 Deploy to Railway:")
    print("1. Visit: https://railway.app")
    print("2. Connect your GitHub repository")
    print("3. Deploy automatically from main branch")
    print("4. Get public URL instantly")

def create_streamlit_version():
    """Create Streamlit version for easy deployment"""
    
    streamlit_app = '''
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
'''
    
    with open("streamlit_app.py", "w") as f:
        f.write(streamlit_app.strip())
    
    print("\n✅ Streamlit app created: streamlit_app.py")
    print("\n🎈 Deploy to Streamlit Cloud:")
    print("1. pip install streamlit")
    print("2. streamlit run streamlit_app.py  # Local testing")
    print("3. Push to GitHub")
    print("4. Visit: https://share.streamlit.io")
    print("5. Deploy from GitHub repository")
    print("6. Get public URL instantly")

if __name__ == "__main__":
    print("🚀 Creating Deployment Files for CsPbBr₃ Digital Twin")
    print("=" * 60)
    
    create_heroku_files()
    create_docker_files()
    create_railway_files()
    create_streamlit_version()
    
    print("\n✅ All deployment files created!")
    print("\nChoose your preferred deployment method above.")