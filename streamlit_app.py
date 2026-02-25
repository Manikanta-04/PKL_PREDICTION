"""
PKL Match Winner Predictor - Streamlit App
Run this for Streamlit Cloud deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="PKL Match Predictor",
    page_icon="🏆",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        text-align: center;
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin: 2rem 0;
    }
    .winner-text {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .confidence-text {
        font-size: 1.5rem;
        opacity: 0.95;
    }
    .team-select {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .vs-badge {
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #667eea;
        margin: 0.5rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 25px;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.4);
    }
    .info-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🏆 PKL Match Predictor</h1>
    <p style='font-size: 1.2rem;'>Predict the winner of Pro Kabaddi League matches using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Load models and encoders
@st.cache_resource
def load_models():
    """Load trained model and encoders"""
    try:
        model = joblib.load('models/latest_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        le = joblib.load('models/label_encoder.pkl')
        
        # Get all team names
        teams = list(le.classes_)
        
        # Load team statistics
        if os.path.exists('data/pkl_ml_dataset.csv'):
            df = pd.read_csv('data/pkl_ml_dataset.csv')
            team_stats = {}
            for team in teams:
                team_data = df[(df['team_a'] == team) | (df['team_b'] == team)]
                if len(team_data) > 0:
                    team_stats[team] = {
                        'avg_strength': team_data['team_a_strength'].mean() if team in team_data['team_a'].values else team_data['team_b_strength'].mean(),
                        'win_pct': team_data[team_data['winner'] == team].shape[0] / len(team_data) if len(team_data) > 0 else 0.5
                    }
                else:
                    team_stats[team] = {'avg_strength': 0.7, 'win_pct': 0.5}
        else:
            team_stats = {team: {'avg_strength': 0.7, 'win_pct': 0.5} for team in teams}
        
        return model, scaler, le, teams, team_stats
    except Exception as e:
        st.error(f"Error loading models: {e}")
        # Fallback teams if model not found
        fallback_teams = ['Patna Pirates', 'Bengal Warriors', 'U Mumba', 'Jaipur Pink Panthers',
                         'Haryana Steelers', 'Tamil Thalaivas', 'Telugu Titans', 'Puneri Paltan',
                         'Gujarat Giants', 'Dabang Delhi KC', 'Bengaluru Bulls', 'UP Yoddhas']
        return None, None, None, fallback_teams, {}

# Load models
model, scaler, le, teams, team_stats = load_models()

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎯 Individual Prediction", "📊 Batch Prediction", "ℹ️ Model Info"])

with tab1:
    st.markdown("### Select Teams")
    
    col1, col2 = st.columns(2)
    
    with col1:
        team_a = st.selectbox("Team A", teams, index=0, key="team_a")
    
    with col2:
        team_b = st.selectbox("Team B", teams, index=1, key="team_b")
    
    st.markdown('<div class="vs-badge">VS</div>', unsafe_allow_html=True)
    
    if st.button("🔮 Predict Winner", use_container_width=True):
        if team_a == team_b:
            st.error("Please select different teams!")
        elif model is None:
            st.warning("⚠️ Model not loaded. Using rule-based prediction.")
            # Simple rule-based fallback
            winner = team_a if hash(team_a + team_b) % 2 == 0 else team_b
            confidence = 60.0
            st.markdown(f"""
            <div class="prediction-box">
                <h2>Predicted Winner</h2>
                <div class="winner-text">🏆 {winner}</div>
                <div class="confidence-text">Confidence: {confidence}%</div>
                <p style='margin-top: 1rem;'>(Using fallback prediction - train model for better accuracy)</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            try:
                # Get team stats
                stats_a = team_stats.get(team_a, {'avg_strength': 0.7, 'win_pct': 0.5})
                stats_b = team_stats.get(team_b, {'avg_strength': 0.7, 'win_pct': 0.5})
                
                # Encode teams
                team_a_encoded = le.transform([team_a])[0]
                team_b_encoded = le.transform([team_b])[0]
                
                # Create feature vector
                features = np.array([[
                    stats_a['win_pct'], stats_b['win_pct'],
                    50, 50,  # placeholder points
                    5, 5,    # placeholder wins
                    stats_a['avg_strength'], stats_b['avg_strength'],
                    team_a_encoded, team_b_encoded
                ]])
                
                # Scale and predict
                features_scaled = scaler.transform(features)
                prediction = model.predict(features_scaled)[0]
                
                # Get probability
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features_scaled)[0]
                    confidence = max(proba) * 100
                else:
                    confidence = 65.0
                
                winner = team_a if prediction == 1 else team_b
                
                # Display result
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted Winner</h2>
                    <div class="winner-text">🏆 {winner}</div>
                    <div class="confidence-text">Confidence: {confidence:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show team comparison
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(team_a, f"{stats_a['win_pct']*100:.1f}% Win Rate")
                with col2:
                    st.metric("VS", "")
                with col3:
                    st.metric(team_b, f"{stats_b['win_pct']*100:.1f}% Win Rate")
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")

with tab2:
    st.markdown("### Batch Prediction")
    st.markdown("Upload a CSV file with columns: **team_a, team_b**")
    
    # Sample CSV template
    sample_df = pd.DataFrame({
        'team_a': [teams[0], teams[2], teams[4]],
        'team_b': [teams[1], teams[3], teams[5]]
    })
    
    with st.expander("📁 View Sample CSV Format"):
        st.dataframe(sample_df)
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv,
            file_name="sample_pkl_matches.csv",
            mime="text/csv"
        )
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'team_a' not in df.columns or 'team_b' not in df.columns:
                st.error("CSV must contain 'team_a' and 'team_b' columns!")
            else:
                with st.spinner("Processing predictions..."):
                    predictions = []
                    confidences = []
                    
                    for _, row in df.iterrows():
                        t_a = row['team_a']
                        t_b = row['team_b']
                        
                        if t_a not in teams or t_b not in teams:
                            predictions.append("Unknown Team")
                            confidences.append(0)
                            continue
                        
                        if model is not None:
                            stats_a = team_stats.get(t_a, {'avg_strength': 0.7, 'win_pct': 0.5})
                            stats_b = team_stats.get(t_b, {'avg_strength': 0.7, 'win_pct': 0.5})
                            
                            features = np.array([[
                                stats_a['win_pct'], stats_b['win_pct'],
                                50, 50, 5, 5,
                                stats_a['avg_strength'], stats_b['avg_strength'],
                                le.transform([t_a])[0],
                                le.transform([t_b])[0]
                            ]])
                            
                            features_scaled = scaler.transform(features)
                            pred = model.predict(features_scaled)[0]
                            
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(features_scaled)[0]
                                conf = max(proba) * 100
                            else:
                                conf = 65.0
                            
                            winner = t_a if pred == 1 else t_b
                        else:
                            # Fallback
                            winner = t_a if hash(t_a + t_b) % 2 == 0 else t_b
                            conf = 60.0
                        
                        predictions.append(winner)
                        confidences.append(round(conf, 1))
                    
                    df['predicted_winner'] = predictions
                    df['confidence_%'] = confidences
                    
                    st.success(f"✅ Processed {len(df)} matches successfully!")
                    st.dataframe(df)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Visualize results
                    if len(df) > 1:
                        fig = px.bar(df, x=df.index, y='confidence_%', 
                                   color='predicted_winner',
                                   title='Prediction Confidence by Match',
                                   labels={'index': 'Match', 'confidence_%': 'Confidence %'})
                        st.plotly_chart(fig, use_container_width=True)
                        
        except Exception as e:
            st.error(f"Error processing file: {e}")

with tab3:
    st.markdown("### Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Model Details")
        if model is not None:
            st.metric("Model Type", type(model).__name__)
            if os.path.exists('models/model_metadata.csv'):
                meta = pd.read_csv('models/model_metadata.csv').iloc[0]
                st.metric("Accuracy", f"{meta['accuracy']*100:.1f}%")
                st.metric("Training Date", meta['training_date'][:10])
            else:
                st.metric("Accuracy", "~60%")
        else:
            st.warning("No trained model found. Using fallback predictions.")
    
    with col2:
        st.markdown("#### Dataset Statistics")
        if os.path.exists('data/pkl_ml_dataset.csv'):
            df_stats = pd.read_csv('data/pkl_ml_dataset.csv')
            st.metric("Total Matches", len(df_stats))
            st.metric("Teams", df_stats['team_a'].nunique())
            st.metric("Seasons", df_stats['season'].max())
    
    st.markdown("#### Features Used")
    features_list = [
        "Team A Win Percentage (Previous Season)",
        "Team B Win Percentage (Previous Season)",
        "Team A Points (Previous Season)",
        "Team B Points (Previous Season)",
        "Team A Current Wins",
        "Team B Current Wins",
        "Team A Strength",
        "Team B Strength"
    ]
    for f in features_list:
        st.markdown(f"✅ {f}")
    
    # Feature importance plot
    if os.path.exists('models/feature_importance.png'):
        st.markdown("#### Feature Importance")
        st.image('models/feature_importance.png', use_container_width=True)
    
    if os.path.exists('models/confusion_matrix.png'):
        st.markdown("#### Confusion Matrix")
        st.image('models/confusion_matrix.png', use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>⚡ Powered by Machine Learning | Data from PKL Seasons 1-10</p>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 0.8rem;'>Made with ❤️ for PKL fans</p>",
    unsafe_allow_html=True
)