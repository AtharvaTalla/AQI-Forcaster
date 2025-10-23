import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Configuration - MUST MATCH YOUR NOTEBOOK
# -------------------------------------------------------------------
# These are the files your notebook *created*.
MODEL_FILE = 'aqi_rf_model.joblib'
SCALER_FILE = 'aqi_scaler.joblib'

# These are the column names from your notebook.
DATE_COL = 'Date'
TARGET_COL = 'AQI'
POLLUTANT_COLS = [
    'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
    'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'
]
WEATHER_COLS = [] # Keep this in sync with your notebook

# -------------------------------------------------------------------
# Feature Engineering Function
# This function MUST be identical to Cell 5 in your notebook
# -------------------------------------------------------------------
def create_features(df_in, target_col, pollutant_cols, weather_cols):
    """Creates time-series features from a raw dataframe."""
    df = df_in.copy()
    
    # Check if target_col exists
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in the uploaded file.")
        return None

    # 1. Target (Not needed for prediction, but good to have)
    df['target_AQI_next_day'] = df[target_col].shift(-1)

    # 2. Lag Features
    df['AQI_lag_1'] = df[target_col].shift(1)
    df['AQI_lag_2'] = df[target_col].shift(2)
    df['AQI_lag_3'] = df[target_col].shift(3)

    # 3. Rolling Average Features
    df['AQI_rolling_7d_mean'] = df[target_col].shift(1).rolling(window=7).mean()
    df['AQI_rolling_30d_mean'] = df[target_col].shift(1).rolling(window=30).mean()

    # 4. Lags for other features
    features_to_lag = [col for col in pollutant_cols + weather_cols if col in df.columns]
    for col in features_to_lag:
        df[f'{col}_lag_1'] = df[col].shift(1)

    # 5. Date Features
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear
    
    return df

# -------------------------------------------------------------------
# Load Model and Data (with caching)
# -------------------------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    """Loads the saved model and scaler."""
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: Model ('{MODEL_FILE}') or Scaler ('{SCALER_FILE}') not found.")
        st.error("Please run the `aqi_prediction_notebook.py` (Cell 11) to save them first.")
        return None, None

@st.cache_data
def load_data(uploaded_file):
    """Loads and preprocesses the raw data from an uploaded file."""
    try:
        df_raw = pd.read_csv(uploaded_file)
        if DATE_COL not in df_raw.columns:
            st.error(f"Date column '{DATE_COL}' not found in the uploaded file.")
            return None
        df_raw[DATE_COL] = pd.to_datetime(df_raw[DATE_COL])
        return df_raw
    except Exception as e:
        st.error(f"Error loading or parsing the CSV file: {e}")
        return None

# -------------------------------------------------------------------
# Main App UI
# -------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("Air Quality Index (AQI) Forecaster 🌬️")

st.info("This app loads your pre-trained model to forecast AQI. "
        "It requires the `aqi_rf_model.joblib` and `aqi_scaler.joblib` files "
        "to be in the same folder.")

# --- 1. Load Model ---
model, scaler = load_model_and_scaler()

if model is not None and scaler is not None:
    st.success("Pre-trained model and scaler loaded successfully!")

    # --- 2. Upload Data ---
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

    if uploaded_file is not None:
        df_raw = load_data(uploaded_file)
        
        if df_raw is not None:
            
            # --- 3. User Input (City) ---
            if 'City' not in df_raw.columns:
                st.error("The uploaded file must contain a 'City' column.")
            else:
                all_cities = df_raw['City'].unique()
                city = st.selectbox("Select a City to Forecast:", all_cities)
                
                st.info(f"Forecasting for **{city}**. Loading and processing data...")

                # --- 4. Data Processing ---
                # Filter for the city
                df_city_raw = df_raw[df_raw['City'] == city].copy()
                
                # Set index and select columns
                ALL_COLS_TO_USE = [TARGET_COL] + POLLUTANT_COLS + WEATHER_COLS
                # Only use columns that *actually exist* in the uploaded file
                cols_that_exist = [col for col in ALL_COLS_TO_USE if col in df_city_raw.columns]
                
                if TARGET_COL not in cols_that_exist:
                    st.error(f"Target column '{TARGET_COL}' not found for {city}.")
                else:
                    df_city_base = df_city_raw.set_index(DATE_COL)[cols_that_exist].sort_index()

                    # Preprocess: Interpolate (same as notebook)
                    df_city_clean = df_city_base.interpolate(method='time')
                    df_city_clean.ffill(inplace=True)
                    df_city_clean.bfill(inplace=True)
                    
                    # Engineer features (same as notebook)
                    df_city_featured = create_features(df_city_clean, TARGET_COL, POLLUTANT_COLS, WEATHER_COLS)
                    
                    if df_city_featured is None:
                        st.stop()
                        
                    # --- 5. Get Features for Prediction ---
                    # We get the *last available day* in the dataset to predict the *next* day
                    # .dropna() is crucial to ensure we don't get a partially-calculated row
                    df_predictable = df_city_featured.dropna()
                    
                    if df_predictable.empty:
                        st.error(f"Not enough data for {city} to create features (need at least 30 days). Please try another city or file.")
                    else:
                        # Get the feature columns *exactly* as the model was trained on
                        try:
                            model_feature_cols = model.feature_names_in_
                        except AttributeError:
                            st.error("The loaded model doesn't have 'feature_names_in_'. It might be an older model. Please re-run the notebook.")
                            st.stop()

                        # Get the last row of features
                        features_today = df_predictable[model_feature_cols].iloc[[-1]] # Use [[-1]] to keep it a DataFrame
                        last_date = features_today.index[0]

                        # --- 6. Make Prediction ---
                        # Scale the features using the *saved* scaler
                        features_today_scaled = scaler.transform(features_today)
                        
                        # Predict!
                        prediction = model.predict(features_today_scaled)
                        predicted_aqi = int(prediction[0])

                        # --- 7. Display Results ---
                        st.subheader(f"Forecast for {city} on {last_date.date() + pd.Timedelta(days=1)}")
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Predicted AQI (for Tomorrow)", predicted_aqi)
                        
                        last_actual_aqi = int(df_predictable[TARGET_COL].iloc[-1])
                        col2.metric(f"Last Actual AQI (for {last_date.date()})", last_actual_aqi)

                        # --- 8. Show Data ---
                        st.subheader(f"Recent Data for {city}")
                        # Show the most recent *raw* data
                        st.dataframe(df_city_raw.set_index(DATE_COL).sort_index(ascending=False).head(10))
                        
                        st.subheader("Model Feature Importance")
                        st.write("This shows what features the model learned are most important.")
                        if hasattr(model, 'feature_importances_'):
                            importances = pd.Series(model.feature_importances_, index=model_feature_cols)
                            top_15 = importances.nlargest(15)
                            
                            fig, ax = plt.subplots(figsize=(10, 8))
                            top_15.sort_values().plot(kind='barh', ax=ax)
                            ax.set_title('Top 15 Most Important Features')
                            st.pyplot(fig)
