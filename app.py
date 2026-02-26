"""
PKL Match Winner Predictor - Flask Web Application
Run this to start the web interface
"""

from flask import Flask, request, jsonify, render_template, send_file
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load models and encoders
try:
    model = joblib.load('models/latest_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    le = joblib.load('models/label_encoder.pkl')
    
    # Get all team names from the encoder
    teams = list(le.classes_)
    
    print(f"✅ Model and encoders loaded successfully")
    print(f"✅ Found {len(teams)} teams: {teams}")
    
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    model = None
    scaler = None
    le = None
    # Fallback teams if model not loaded
    teams = ['Patna Pirates', 'Bengal Warriors', 'U Mumba', 'Jaipur Pink Panthers',
             'Haryana Steelers', 'Tamil Thalaivas', 'Telugu Titans', 'Puneri Paltan',
             'Gujarat Giants', 'Dabang Delhi KC', 'Bengaluru Bulls', 'UP Yoddhas']

# Load team statistics for better predictions
team_stats = {}
try:
    df = pd.read_csv('data/pkl_ml_dataset.csv')
    for team in teams:
        team_data = df[(df['team_a'] == team) | (df['team_b'] == team)]
        if len(team_data) > 0:
            team_stats[team] = {
                'avg_strength': team_data['team_a_strength'].mean() if team in team_data['team_a'].values else 
                                team_data['team_b_strength'].mean(),
                'win_pct': team_data[team_data['winner'] == team].shape[0] / len(team_data) if len(team_data) > 0 else 0.5
            }
        else:
            team_stats[team] = {'avg_strength': 0.7, 'win_pct': 0.5}
    print("✅ Team statistics loaded")
except Exception as e:
    print(f"⚠️ Could not load team stats: {e}")

@app.route('/')
def home():
    """Render the home page with team dropdowns"""
    return render_template('index.html', teams=teams)

@app.route('/predict', methods=['POST'])
def predict():
    """Make a single prediction"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        team_a = data['team_a']
        team_b = data['team_b']
        
        # Validate different teams
        if team_a == team_b:
            return jsonify({'error': 'Please select different teams'}), 400
        
        # Get team stats
        stats_a = team_stats.get(team_a, {'avg_strength': 0.7, 'win_pct': 0.5})
        stats_b = team_stats.get(team_b, {'avg_strength': 0.7, 'win_pct': 0.5})
        
        # Encode teams
        team_a_encoded = le.transform([team_a])[0]
        team_b_encoded = le.transform([team_b])[0]
        
        # Create feature vector (must match training features)
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
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_scaled)[0]
            confidence = max(proba) * 100
        else:
            confidence = 65.0  # default confidence
        
        winner = team_a if prediction == 1 else team_b
        
        return jsonify({
            'success': True,
            'winner': winner,
            'confidence': round(confidence, 1),
            'team_a': team_a,
            'team_b': team_b
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Process batch predictions from CSV"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        file = request.files['file']
        df = pd.read_csv(file)
        
        if 'team_a' not in df.columns or 'team_b' not in df.columns:
            return jsonify({'error': 'CSV must have team_a and team_b columns'}), 400
        
        predictions = []
        confidences = []
        
        for _, row in df.iterrows():
            team_a = row['team_a']
            team_b = row['team_b']
            
            if team_a not in le.classes_ or team_b not in le.classes_:
                predictions.append('Unknown Team')
                confidences.append(0)
                continue
            
            stats_a = team_stats.get(team_a, {'avg_strength': 0.7, 'win_pct': 0.5})
            stats_b = team_stats.get(team_b, {'avg_strength': 0.7, 'win_pct': 0.5})
            
            features = np.array([[
                stats_a['win_pct'], stats_b['win_pct'],
                50, 50, 5, 5,
                stats_a['avg_strength'], stats_b['avg_strength'],
                le.transform([team_a])[0],
                le.transform([team_b])[0]
            ]])
            
            features_scaled = scaler.transform(features)
            pred = model.predict(features_scaled)[0]
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_scaled)[0]
                conf = max(proba) * 100
            else:
                conf = 65.0
            
            winner = team_a if pred == 1 else team_b
            predictions.append(winner)
            confidences.append(round(conf, 1))
        
        df['predicted_winner'] = predictions
        df['confidence'] = confidences
        
        # Save to CSV
        output_file = f'batch_predictions_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(output_file, index=False)
        
        return jsonify({
            'success': True,
            'message': f'✅ Processed {len(df)} matches',
            'download_url': f'/download/{output_file}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download a file"""
    return send_file(filename, as_attachment=True)

@app.route('/teams')
def get_teams():
    """API endpoint to get list of teams"""
    return jsonify({'teams': teams})

# FIXED: Single main block for Render deployment
if __name__ == '__main__':
    # Get port from environment variable (for Render) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "=" * 60)
    print("🚀 STARTING PKL MATCH PREDICTOR WEB APP")
    print("=" * 60)
    print(f"\n📱 Server starting on port: {port}")
    print(f"🏆 Available teams: {len(teams)}")
    # Use debug=False for production
    app.run(host='0.0.0.0', port=port, debug=False)