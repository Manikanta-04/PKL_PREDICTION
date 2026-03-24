<div align="center">

<img src="https://img.shields.io/badge/PKL%20Predictor-Match%20Winner%20AI-ff6b00?style=for-the-badge&logo=python&logoColor=white" alt="PKL Predictor Banner"/>

# 🏉 PKL Match Winner Predictor

### *ML-Powered Pro Kabaddi League Match Outcome Prediction*

<br/>

[![Live Demo](https://img.shields.io/badge/🌐%20Live%20Demo-Visit%20Now-ff6b00?style=for-the-badge)](https://pkl-prediction.vercel.app)

<br/>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-black?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-red?style=flat-square)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen?style=flat-square)](CONTRIBUTING.md)

</div>

---

## 🚀 Live Demo

| Service | URL |
|---|---|
| 🌐 **Web Application** | [pkl-prediction.vercel.app](https://pkl-prediction.vercel.app) *(update with your actual URL)* |

> 💡 Select any two PKL teams and get an instant winner prediction with a confidence score.

---

## 🎥 Demo Video

> 📽️ *(Add a Loom / YouTube demo walkthrough here)*
>
> [![Watch Demo](https://img.shields.io/badge/▶%20Watch%20Demo-YouTube-red?style=for-the-badge&logo=youtube)](https://youtube.com)

---

## 🧠 Problem Statement

Pro Kabaddi League matches are unpredictable — fans and analysts rely on gut feeling, recent form, or incomplete statistics to guess outcomes. Current challenges include:

- 📉 No data-driven tool for match outcome prediction specific to PKL
- 🔢 Historical season data across 10 seasons is scattered and unstructured
- 🧩 No single platform combines team stats, form, and ML predictions in one UI
- 📁 Manual analysis is time-consuming — no bulk/batch prediction support

**Kabaddi deserves the same analytical depth that cricket and football already have.**

---

## 💡 Solution

**PKL Match Winner Predictor** is a Flask-powered ML web application that analyzes 10 seasons of team performance data — win rates, point totals, strength ratings, and current form — to predict match outcomes with confidence scores. Supports both individual match predictions and bulk CSV batch processing.

> *"Don't guess the winner. Let the data decide."*

---

## 🖼️ Screenshots

| Match Prediction UI | Confidence Score Output |
|---|---|
| ![Prediction](screenshots/prediction.png) | ![Confidence](screenshots/confidence.png) |

| Team Stats Dashboard | Batch CSV Upload |
|---|---|
| ![Stats](screenshots/stats.png) | ![Batch](screenshots/batch.png) |

> 📌 *(Replace with actual screenshots from your deployed app)*

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      CLIENT LAYER                         │
│           HTML5 + CSS3 + Vanilla JavaScript               │
│   Match Selector | Batch Upload | Team Stats Dashboard    │
└───────────────────────────┬──────────────────────────────┘
                            │  HTTP POST (form / CSV upload)
                            ▼
┌──────────────────────────────────────────────────────────┐
│                  APPLICATION LAYER                        │
│                 Flask (Python 2.x)                        │
│    Prediction API | Batch Processor | Stats Endpoint      │
└───────────────────────────┬──────────────────────────────┘
                            │
              ┌─────────────┴──────────────┐
              │                            │
              ▼                            ▼
┌─────────────────────┐      ┌────────────────────────────┐
│    ML MODEL LAYER   │      │       DATA LAYER            │
│  latest_model.pkl   │      │  pkl_ml_dataset.csv         │
│  scaler.pkl         │      │  raw_matches.csv            │
│  label_encoder.pkl  │      │  raw_standings.csv          │
│  (Logistic Reg.)    │      │  10 seasons, 12 teams       │
└─────────────────────┘      └────────────────────────────┘
```

---

## ⚙️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | HTML5, CSS3, JavaScript | Web interface |
| **Backend** | Flask (Python) | REST API + routing |
| **ML Models** | Scikit-learn, XGBoost | Match outcome classification |
| **Data Processing** | Pandas, NumPy | Feature engineering & preprocessing |
| **Serialization** | Joblib | Save/load trained models |
| **Visualization** | Matplotlib, Seaborn | Confusion matrix, feature importance |

---

## ✨ Features

### 🏆 Match Prediction
- Select any two PKL teams from dropdowns
- Get **instant winner prediction** with confidence percentage
- Powered by Logistic Regression ensemble model

### 📁 Batch Processing
- Upload a CSV file with multiple match pairs
- Get predictions for **all matches at once**
- Download results as CSV

### 📊 Team Statistics Dashboard
- Browse historical performance metrics per team
- View strength ratings, win percentages, and point totals across 10 seasons

### 🤖 Ensemble ML Model
- 4 classifiers trained and compared automatically
- Best model saved and deployed
- Outperforms random 50% baseline with **59.85% test accuracy**

---

## 🤖 ML Model Details

### Training Pipeline

```
Raw Data (raw_matches.csv + raw_standings.csv)
         │
         ▼
[create_pkl_dataset.py]
  Feature Engineering:
  - Win percentage per team per season
  - Historical points accumulated
  - Team strength ratings
  - Current season form
  - Encoded team identifiers
         │
         ▼
[train_model.py]
  4 Classifiers evaluated:
  1. Logistic Regression  ← Best (saved as latest_model.pkl)
  2. Random Forest
  3. Gradient Boosting
  4. XGBoost
         │
         ▼
[Flask API]  →  Prediction served with confidence score
```

### Model Performance

| Metric | Value |
|---|---|
| **Test Accuracy** | **59.85%** |
| Cross-Validation Accuracy | 55.49% |
| Baseline (random) | 50.00% |
| Training Samples | 1,056 |
| Test Samples | 264 |
| Best Model | Logistic Regression (tuned) |
| Seasons Covered | 10 |
| Teams | 12 |
| Total Matches | ~1,320 |

> ✅ The model consistently outperforms the random 50% baseline with balanced predictions across both teams.

---

## 📊 System Design

```
Individual Prediction Flow:

[User selects Team A + Team B]
         │
         ▼
POST /predict
         │
         ▼
[Flask loads latest_model.pkl + scaler.pkl + label_encoder.pkl]
         │
         ▼
[Feature vector constructed from team stats]
         │
         ▼
[model.predict_proba() → confidence scores]
         │
         ▼
[Winner + confidence % returned as JSON]
         │
         ▼
[Frontend renders result card]
```

```
Batch Prediction Flow:

[User uploads CSV: team_a, team_b columns]
         │
         ▼
POST /batch-predict
         │
         ▼
[Pandas reads CSV row by row]
[Each row → same prediction pipeline]
         │
         ▼
[Results appended: predicted_winner, confidence]
         │
         ▼
[Output CSV returned for download]
```

---

## 🔄 Workflow

```
1. Run create_pkl_dataset.py    →  Generates pkl_ml_dataset.csv from raw data
2. Run train_model.py           →  Trains 4 classifiers, saves best as latest_model.pkl
3. Run app.py                   →  Flask server starts at http://127.0.0.1:5000
4. User opens web UI            →  Selects Team A + Team B from dropdown
5. POST /predict called         →  Features extracted → model inference
6. Confidence score returned    →  Winner displayed with percentage
7. (Optional) CSV uploaded      →  Batch predictions processed + downloaded
```

---

## 📈 Performance & Metrics

| Metric | Value |
|---|---|
| Test Accuracy | 59.85% |
| Cross-Validation Accuracy | 55.49% |
| Improvement over baseline | +9.85% over random 50% |
| Seasons in dataset | 10 |
| Teams supported | 12 |
| Total training matches | ~1,320 |
| Prediction response time | < 200ms |
| Batch processing | Unlimited rows via CSV |

---

## 🧪 Testing

```bash
# Run the Flask app
python app.py

# Test individual prediction (curl)
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"team_a": "Patna Pirates", "team_b": "Bengal Warriors"}'
# Expected: { "winner": "Patna Pirates", "confidence": 67.3 }

# Test batch prediction
curl -X POST http://127.0.0.1:5000/batch-predict \
  -F "file=@matches.csv"
# Expected: CSV file download with predicted_winner + confidence columns

# Test team stats endpoint
curl http://127.0.0.1:5000/api/teams
# Expected: JSON array of all 12 teams with stats

# Verify model files exist
ls models/
# Expected: latest_model.pkl, scaler.pkl, label_encoder.pkl
```

---

## 📁 Project Structure

```
PKL_PREDICTION/
│
├── data/
│   ├── pkl_ml_dataset.csv          # ML-ready feature dataset
│   ├── raw_matches.csv             # Raw match results (10 seasons)
│   └── raw_standings.csv           # Season standings per team
│
├── models/
│   ├── latest_model.pkl            # Best trained model (Logistic Regression)
│   ├── scaler.pkl                  # Feature scaler (StandardScaler)
│   ├── label_encoder.pkl           # Team name → numeric encoder
│   ├── confusion_matrix.png        # Model performance visualization
│   ├── feature_importance.png      # Top features driving predictions
│   └── model_metadata.csv          # Model info, metrics, timestamp
│
├── templates/
│   └── index.html                  # Web interface (prediction UI)
│
├── app.py                          # Flask application entry point
├── create_pkl_dataset.py           # Dataset generation from raw CSVs
├── train_model.py                  # Model training + evaluation script
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 🔐 Security

- **No user data stored** — predictions are stateless, no input logged
- **File validation** — CSV uploads validated for correct columns before processing
- **Error handling** — invalid team names or malformed CSVs return clean error responses

---

## 🏉 Dataset Details

**Synthetic dataset** covering 10 PKL seasons across 12 teams (~1,320 matches):

| Team | |
|---|---|
| Patna Pirates | Bengal Warriors |
| U Mumba | Jaipur Pink Panthers |
| Haryana Steelers | Tamil Thalaivas |
| Telugu Titans | Puneri Paltan |
| Gujarat Giants | Dabang Delhi KC |
| Bengaluru Bulls | UP Yoddhas |

**Features used for prediction:**

| Feature | Description |
|---|---|
| `win_pct` | Team win percentage from prior seasons |
| `points_total` | Historical points accumulated |
| `team_strength` | Composite strength rating |
| `current_form` | Recent season performance score |
| `team_encoded` | Numeric team identifier |

---

## ⚙️ Local Development Setup

### Prerequisites

- Python `3.8+`
- pip

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Manikanta-04/PKL_PREDICTION.git
cd PKL_PREDICTION
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Generate Dataset

```bash
python create_pkl_dataset.py
```

### 4️⃣ Train the Model

```bash
python train_model.py
```

### 5️⃣ Run the Application

```bash
python app.py
```

Open: `http://127.0.0.1:5000`

---

## 🔑 Environment Variables

This project requires no environment variables for local development — all config is embedded in `app.py`.

For production deployment, optionally set:

```env
FLASK_ENV=production
PORT=5000
```

---

## 🚀 Deployment

### Deploy on Render (Flask Backend)

| Setting | Value |
|---|---|
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt && python create_pkl_dataset.py && python train_model.py` |
| Start Command | `python app.py` |

### Deploy on Railway / Heroku

```bash
# Add Procfile
echo "web: python app.py" > Procfile
git push heroku main
```

---

## 🔮 Future Improvements

- [ ] 🔗 Integrate real PKL data via official / scraped APIs
- [ ] 👤 Add player-level statistics (raiders, defenders, all-rounders)
- [ ] 🆚 Head-to-head historical matchup records
- [ ] 🧠 Neural network models (LSTM for time-series form)
- [ ] 🏟️ Venue and travel impact analysis
- [ ] 📱 Mobile app version (React Native)
- [ ] 📡 Live match data integration for in-game predictions
- [ ] 🏆 Season-long tournament simulator

---

## 🤝 Contributing

Contributions are welcome and appreciated!

```bash
# 1. Fork this repository
# 2. Create your feature branch
git checkout -b feature/your-feature-name

# 3. Commit with conventional commits
git commit -m "feat: describe your change"

# 4. Push and open a Pull Request
git push origin feature/your-feature-name
```

Please follow [Conventional Commits](https://www.conventionalcommits.org/) and test model training end-to-end before submitting ML-related PRs.

---

## 👨‍💻 Author

**Manikanta Naripeddi** — ML & Full Stack Developer

[![GitHub](https://img.shields.io/badge/GitHub-Manikanta--04-181717?style=flat-square&logo=github)](https://github.com/Manikanta-04)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Manikanta%20Naripeddi-0077b5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/manikanta-naripeddi-4326232a5/)

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

- [scikit-learn](https://scikit-learn.org/) — ML model framework
- [XGBoost](https://xgboost.readthedocs.io/) — Gradient boosting classifier
- [Flask](https://flask.palletsprojects.com/) — Python web framework
- [Pandas](https://pandas.pydata.org/) — Data processing
- [Joblib](https://joblib.readthedocs.io/) — Model serialization
- [Pro Kabaddi League](https://www.prokabaddi.com/) — Sport and inspiration

---

<div align="center">

**Built with ❤️ for Kabaddi analytics**

⭐ **Star this repo** if PKL Predictor impressed you!

[![GitHub Stars](https://img.shields.io/github/stars/Manikanta-04/PKL_PREDICTION?style=social)](https://github.com/Manikanta-04/PKL_PREDICTION)

---

*🏉 Don't guess the winner. Let the data decide.*

</div>
