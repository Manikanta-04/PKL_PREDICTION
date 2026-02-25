# PKL Match Winner Predictor

> A machine learning application that predicts Pro Kabaddi League match outcomes using historical team performance data across 10 seasons.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

PKL Match Winner Predictor analyzes historical team statistics — win rates, point totals, team strengths, and current form — to generate match outcome predictions with confidence scores. The system supports both individual match predictions and bulk batch processing via CSV upload.

---

## Features

- **Match Prediction** — Select any two teams and get an instant winner prediction with a confidence percentage
- **Batch Processing** — Upload a CSV file to predict outcomes for multiple matches at once
- **Team Statistics Dashboard** — Browse historical performance metrics and team strength ratings
- **Ensemble ML Model** — Trained on 10 seasons of PKL data using multiple classifiers
- **Interactive Web Interface** — Clean, responsive UI built with HTML5/CSS3/JavaScript

---

## Model Performance

The model was trained on 1,320 synthetic matches across 12 teams and 10 seasons.

| Metric | Value |
|---|---|
| Test Accuracy | 59.85% |
| Cross-Validation Accuracy | 55.49% |
| Training Samples | 1,056 |
| Test Samples | 264 |
| Best Model | Logistic Regression (tuned) |

> The model performs better than the random 50% baseline with balanced predictions across both teams.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3, JavaScript |
| Backend | Flask (Python) |
| ML Models | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Serialization | Joblib |
| Visualization | Matplotlib, Seaborn |

---

## Project Structure

```
PKL_PREDICTION/
├── data/
│   ├── pkl_ml_dataset.csv      # ML-ready feature dataset
│   ├── raw_matches.csv         # Raw match results
│   └── raw_standings.csv       # Season standings
├── models/
│   ├── latest_model.pkl        # Best trained model
│   ├── scaler.pkl              # Feature scaler
│   ├── label_encoder.pkl       # Team name encoder
│   ├── confusion_matrix.png    # Performance visualization
│   ├── feature_importance.png  # Feature importance plot
│   └── model_metadata.csv      # Model info and metrics
├── templates/
│   └── index.html              # Web interface
├── app.py                      # Flask application
├── create_pkl_dataset.py       # Dataset generation
├── train_model.py              # Model training
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/Manikanta-04/PKL_PREDICTION.git
cd PKL_PREDICTION
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Generate the dataset**

```bash
python create_pkl_dataset.py
```

**4. Train the model**

```bash
python train_model.py
```

**5. Run the application**

```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000`.

---

## Usage

### Individual Match Prediction

1. Select **Team A** from the first dropdown
2. Select **Team B** from the second dropdown
3. Click **Predict Winner**
4. View the predicted winner and confidence score

### Batch Prediction via CSV

Prepare a CSV file with two columns:

```csv
team_a,team_b
Patna Pirates,Bengal Warriors
U Mumba,Jaipur Pink Panthers
```

Upload the file, click **Process Batch**, and download the results.

---

## Dataset

The synthetic dataset covers **10 seasons**, **12 teams**, and approximately **1,320 matches**.

**Teams included:**

Patna Pirates · Bengal Warriors · U Mumba · Jaipur Pink Panthers · Haryana Steelers · Tamil Thalaivas · Telugu Titans · Puneri Paltan · Gujarat Giants · Dabang Delhi KC · Bengaluru Bulls · UP Yoddhas

**Features used for prediction:**
- Team win percentages from prior seasons
- Historical points accumulated
- Team strength ratings
- Current season form
- Encoded team identifiers

---

## Model Training

Four classifiers are trained and evaluated; the best-performing model is saved automatically:

- Logistic Regression *(best)*
- Random Forest
- Gradient Boosting
- XGBoost

To adjust training behavior, edit the following in `train_model.py`:
- `test_size` — Train/test split ratio
- `cv` — Number of cross-validation folds
- Hyperparameter grid in `param_grid`

---

## Roadmap

- [ ] Integrate real PKL data via official APIs
- [ ] Add player-level statistics
- [ ] Head-to-head historical matchup records
- [ ] Neural network models
- [ ] Venue and travel impact analysis
- [ ] Mobile app version

---

## Contributing

Contributions are welcome!

```bash
# Fork the repo, then:
git checkout -b feature/your-feature
git commit -m "Add your feature"
git push origin feature/your-feature
# Open a Pull Request
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Manikanta** — [@Manikanta-04](https://github.com/Manikanta-04)
