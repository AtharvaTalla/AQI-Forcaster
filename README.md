# AQI Prediction using Machine Learning

A machine learning project that predicts the next-day Air Quality Index (AQI) for major Indian cities using historical pollution data.  
The model is trained on time-series features and deployed through a Streamlit web application for real-time forecasting.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Setup and Usage](#setup-and-usage)
- [Model and Evaluation](#model-and-evaluation)
- [Future Scope](#future-scope)
- [Contact](#contact)

---

## Project Overview
This project uses the *Air Quality in India* dataset (`city_day.csv`) to train a regression model that predicts the AQI for the following day.  
The workflow includes preprocessing, feature engineering, model training with hyperparameter tuning, and deployment via Streamlit.

---

## Features
- Data cleaning with time-based interpolation for missing values  
- Feature engineering:
  - Lag features and rolling averages  
  - Pollutant lag features  
  - Date-based features (month, weekday, day of year)
- Models:
  - Baseline: Linear Regression  
  - Final: Random Forest Regressor with RandomizedSearchCV
- TimeSeriesSplit for cross-validation  
- Model and scaler persistence using joblib  
- Streamlit app for interactive inference

---

## Repository Structure
AQI_Prediction/
```
├── aqi_prediction_notebook.py    # Model training pipeline
├── streamlit_app.py              # Streamlit inference app
├── requirements.txt              # Dependencies
├── aqi_rf_model.joblib           # Trained RandomForest model (output)
├── aqi_scaler.joblib             # Scaler used for features (output)
├── assets/
│   └── prediction_vs_actual.jpg  # Plot generated during training (recommended)
└── README.md
```

---

## Setup and Usage

### 1. Create Environment
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python aqi_prediction_notebook.py
```
This script:
- Loads and preprocesses `city_day.csv`
- Creates lag and rolling features
- Trains and tunes the Random Forest model
- Saves the model, scaler, and plots to the project (e.g., `aqi_rf_model.joblib`, `aqi_scaler.joblib`, `assets/`)

### 4. Run the Streamlit App
```bash
streamlit run streamlit_app.py
```
- Upload a dataset (schema matching `city_day.csv`)
- Select a city
- View predicted next-day AQI values and visualizations

---

## Model and Evaluation

| Model              | R²    | MAE    | RMSE   |
|--------------------|-------|--------|--------|
| Linear Regression  | 0.902 | 5.88   | 5.91   |
| Random Forest      | 0.892 | 27.84  | 37.72  |

Notes:
- The Random Forest model captures temporal dependencies and benefits from lag/rolling features.
- City-specific tuning and inclusion of meteorological variables can further improve performance.

---

## Future Scope
- Integrate weather variables (temperature, humidity, wind speed)  
- Experiment with sequence models (LSTM, GRU, Transformer)  
- Try advanced forecasting frameworks (Prophet, Darts)  
- Deploy using Streamlit Cloud or Hugging Face Spaces  
- Add prediction confidence intervals / uncertainty quantification

---

## Contact
Author: Atharva Talla

This project is open-source and intended for research and educational use.  
For collaboration or support, please connect via GitHub.

---
