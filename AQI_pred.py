# -------------------------------------------------------------------
# AQI PREDICTION PROJECT (Real Dataset Version)
# -------------------------------------------------------------------
# This script is formatted as a "notebook" using '#%%' cell markers.
# If you use an IDE like VS Code, you can run each cell individually.
# -------------------------------------------------------------------

#%% 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import joblib

# Set plot style
plt.style.use('fivethirtyeight')
sns.set_context("talk")
print("Libraries imported.")

#%% 2. Project Configuration (!!! USER TO EDIT !!!)
# --- This is the ONLY cell you need to edit ---

# --- File and City ---
# !!! Path updated to the file you uploaded
FILE_PATH = 'city_day.csv' 
# !!! You can change this to any city in the file, e.g., 'Mumbai', 'Ahmedabad'
CITY_TO_MODEL = 'Delhi' 

# --- Column Names ---
# !!! These have been updated to match 'city_day.csv'

DATE_COL = 'Date'       # The column with the date
TARGET_COL = 'AQI'      # The column we are trying to predict

# Features we will use (all available pollutants from your file)
POLLUTANT_COLS = [
    'PM2.5', 
    'PM10', 
    'NO',
    'NO2', 
    'NOx',
    'NH3',
    'CO', 
    'SO2', 
    'O3',
    'Benzene',
    'Toluene',
    'Xylene'
]
# 'city_day.csv' does not have weather data, so we set this to an empty list.
# The script is designed to handle this and will just use the pollutant columns.
WEATHER_COLS = [] 

# --- Model Settings ---
TIME_SPLIT_PERCENTAGE = 0.8 # 80% train, 20% test
RANDOM_SEED = 42

print("Configuration set.")
print(f"Project initialized to model AQI for: {CITY_TO_MODEL}")
print(f"Loading data from: {FILE_PATH}")


#%% 3. Load Data
print("Step 3: Loading data...")
try:
    df_raw = pd.read_csv(FILE_PATH)
    print("File loaded successfully.")
except FileNotFoundError:
    print(f"--- ERROR: File not found at {FILE_PATH} ---")
    print("Please update FILE_PATH in Cell 2 and re-run.")
    # Stop execution if file isn't found
    raise

# Convert date column to datetime objects
df_raw[DATE_COL] = pd.to_datetime(df_raw[DATE_COL])

# Filter for the specific city
df_city = df_raw[df_raw['City'] == CITY_TO_MODEL].copy()

if df_city.empty:
    print(f"--- ERROR: No data found for city '{CITY_TO_MODEL}' ---")
    print(f"Available cities: {df_raw['City'].unique()}")
    print("Please check CITY_TO_MODEL in Cell 2.")
    raise
else:
    print(f"Found {len(df_city)} records for {CITY_TO_MODEL}.")

# Select only the columns we need (and set date as index)
ALL_COLS_TO_USE = [TARGET_COL] + POLLUTANT_COLS + WEATHER_COLS

# Check for missing columns
missing_cols = [col for col in ALL_COLS_TO_USE if col not in df_city.columns]
if missing_cols:
    print(f"--- WARNING: The following columns are missing from your file: {missing_cols} ---")
    print("Please check the COLUMN_NAMES in Cell 2.")
    # Continue with only the columns that *do* exist
    ALL_COLS_TO_USE = [col for col in ALL_COLS_TO_USE if col in df_city.columns]
    
df_base = df_city.set_index(DATE_COL)[ALL_COLS_TO_USE].sort_index()
print("Data loading and filtering complete.")
df_base.info()


#%% 4. Preprocessing & Exploratory Data Analysis (EDA)
print("\nStep 4: Preprocessing and EDA...")

# 4.1. Handle Missing Data
# Real sensor data is full of gaps. We use time-based interpolation.
print(f"Missing values before interpolation:\n{df_base.isnull().sum()}")
df_clean = df_base.interpolate(method='time')
# Fill any remaining NaNs (e.g., at the very start or end)
df_clean.ffill(inplace=True)
df_clean.bfill(inplace=True)
print("Missing values handled.")

# 4.2. EDA: Plot AQI over time
print("Plotting AQI trend...")
df_clean[TARGET_COL].plot(figsize=(15, 6), title=f'AQI Trend for {CITY_TO_MODEL}')
plt.ylabel('AQI')
plt.show()

# 4.3. EDA: Correlation Heatmap
print("Plotting feature correlation heatmap...")
plt.figure(figsize=(12, 8))
sns.heatmap(df_clean.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()


#%% 5. Feature Engineering
print("\nStep 5: Engineering time-series features...")

# We will create features based on "past" data to predict "future" data.
df_featured = df_clean.copy()

# 5.1. Create the Target Variable (y)
# Our goal is to predict the *next day's* AQI.
# We shift the AQI column *backward* by 1 day.
df_featured['target_AQI_next_day'] = df_featured[TARGET_COL].shift(-1)

# 5.2. Create Lag Features (Model's "Memory")
# What was the AQI *today*, *yesterday*, *3 days ago*?
df_featured['AQI_lag_1'] = df_featured[TARGET_COL].shift(1)
df_featured['AQI_lag_2'] = df_featured[TARGET_COL].shift(2)
df_featured['AQI_lag_3'] = df_featured[TARGET_COL].shift(3)

# 5.3. Create Rolling Average Features (Model's "Trend")
# What was the average AQI over the last week?
# We .shift(1) to prevent data leakage from the current day.
df_featured['AQI_rolling_7d_mean'] = df_featured[TARGET_COL].shift(1).rolling(window=7).mean()
df_featured['AQI_rolling_30d_mean'] = df_featured[TARGET_COL].shift(1).rolling(window=30).mean()

# 5.4. Create Lags for other important features
for col in POLLUTANT_COLS + WEATHER_COLS:
    if col in df_featured.columns:
        df_featured[f'{col}_lag_1'] = df_featured[col].shift(1)

# 5.5. Create Date Features (Model's "Seasonality")
df_featured['month'] = df_featured.index.month
df_featured['day_of_week'] = df_featured.index.dayofweek
df_featured['day_of_year'] = df_featured.index.dayofyear

# 5.6. Clean up NaNs
# All this shifting and rolling creates NaNs at the start and end.
# We must drop them before modeling.
rows_before = len(df_featured)
df_featured.dropna(inplace=True)
rows_after = len(df_featured)
print(f"Dropped {rows_before - rows_after} rows containing NaNs created by feature engineering.")
print("Feature engineering complete.")


#%% 6. Helper Functions (for plotting)

def plot_actual_vs_predicted(y_test, y_pred, model_name):
    """Generates a time-series plot of actual vs. predicted values."""
    plt.figure(figsize=(15, 6))
    plt.plot(y_test.index, y_test, label='Actual AQI', alpha=0.9, color='blue')
    plt.plot(y_test.index, y_pred, label=f'Predicted AQI ({model_name})', linestyle='--', color='red')
    plt.legend()
    plt.title(f'Actual vs. Predicted AQI - {model_name}')
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names):
    """Generates a bar chart of the most important features."""
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_names)
        top_15 = importances.nlargest(15)
        
        plt.figure(figsize=(10, 8))
        top_15.sort_values().plot(kind='barh')
        plt.title('Top 15 Most Important Features (Random Forest)')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Model {type(model)} does not have feature_importances_ attribute.")

print("Helper functions defined.")


#%% 7. Model Preparation (Split & Scale)
print("\nStep 7: Preparing data for modeling...")

# 7.1. Define X (features) and y (target)
y = df_featured['target_AQI_next_day']
X = df_featured.drop('target_AQI_next_day', axis=1)

# 7.2. Time-Series Train-Test Split
# --- CRITICAL: We MUST NOT shuffle time-series data! ---
# We split by date: train on the past, test on the "future".
split_point = int(len(X) * TIME_SPLIT_PERCENTAGE)

X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

print(f"Data split into {len(X_train)} train samples and {len(X_test)} test samples.")
print(f"Train date range: {X_train.index.min()} to {X_train.index.max()}")
print(f"Test date range: {X_test.index.min()} to {X_test.index.max()}")

# 7.3. Feature Scaling
# We fit the scaler ONLY on the training data to prevent data leakage.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# We transform the test data using the *same* scaler.
X_test_scaled = scaler.transform(X_test)

print("Train-test split and scaling complete.")


#%% 8. Model 1: Linear Regression (Baseline)
print("\nStep 8: Training Model 1 - Linear Regression (Baseline)...")

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

# Evaluate
print("--- Linear Regression Results ---")
print(f"MAE (Mean Absolute Error): {mean_absolute_error(y_test, y_pred_lr):.2f}")
print(f"RMSE (Root Mean Squared Error): {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}")
print(f"R-squared (R²): {r2_score(y_test, y_pred_lr):.2f}")


#%% 9. Model 2: Random Forest Regressor (Improved)
print("\nStep 9: Training Model 2 - Random Forest Regressor...")

# n_jobs=-1 uses all available CPU cores
rf_model = RandomForestRegressor(n_estimators=100, 
                                 random_state=RANDOM_SEED, 
                                 n_jobs=-1,
                                 oob_score=True) # Out-of-bag score is a good cross-validation estimate

rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate
print("--- Random Forest Results ---")
print(f"OOB Score (like R² on training data): {rf_model.oob_score_:.2f}")
print(f"MAE (Mean Absolute Error): {mean_absolute_error(y_test, y_pred_rf):.2f}")
print(f"RMSE (Root Mean Squared Error): {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
print(f"R-squared (R²): {r2_score(y_test, y_pred_rf):.2f}")


#%% 9.5. Model Tuning with RandomizedSearchCV (Advanced)
print("\nStep 9.5: Tuning the Random Forest model...")

# --- This is the "must-have" next step for a top project ---
# We will find the *best* combination of parameters instead of guessing.

# 1. Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2'] # Use 'sqrt' or 'log2' which are good defaults
}

# 2. Set up the Time-Series Cross-Validator
# We MUST use this to respect the order of the data during cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# 3. Set up RandomizedSearchCV
# We use RandomizedSearchCV because it's much faster than searching *every* combination.
# n_iter=20 means it will try 20 different random combinations.
rf_tuner = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
    param_distributions=param_grid,
    n_iter=20,  # Increase this (e.g., to 50) for better results, but it will be slower
    cv=tscv,
    verbose=2,  # Shows you the progress
    random_state=RANDOM_SEED,
    n_jobs=-1  # Use all cores
)

# 4. Fit the tuner
print("Starting model tuning. This may take a few minutes...")
rf_tuner.fit(X_train_scaled, y_train)

print("Tuning complete.")
print(f"Best parameters found: {rf_tuner.best_params_}")

# 5. Get the best model
best_rf_model = rf_tuner.best_estimator_

# 6. Evaluate the *tuned* model
print("\n--- Tuned Random Forest Results ---")
y_pred_rf_tuned = best_rf_model.predict(X_test_scaled)
print(f"MAE (Mean Absolute Error): {mean_absolute_error(y_test, y_pred_rf_tuned):.2f}")
print(f"RMSE (Root Mean Squared Error): {np.sqrt(mean_squared_error(y_test, y_pred_rf_tuned)):.2f}")
print(f"R-squared (R²): {r2_score(y_test, y_pred_rf_tuned):.2f}")


#%% 10. Final Evaluation & Visualization
print("\nStep 10: Final Evaluation & Visualization...")

# 10.1. Plot Actual vs. Predicted
# We'll use the *tuned* Random Forest model
plot_actual_vs_predicted(y_test, y_pred_rf_tuned, "Tuned Random Forest")

# 10.2. Plot Feature Importance
# This is key for interviews - what did the model "learn"?
plot_feature_importance(best_rf_model, X.columns)

# 10.3. Final Insights
print("\n--- Project Complete ---")
print("You have successfully built and evaluated an AQI prediction pipeline.")
print("Key things to talk about:")
print("1. How you handled missing data (interpolation).")
print("2. The importance of Feature Engineering (lags, rolling averages).")
print("3. Why you did a time-series split (no shuffling).")
print("4. The Feature Importance chart - which features *really* matter?")
print("5. Your R-squared and MAE scores, and what they mean (e.g., 'My model is, on average, X points off the actual AQI.')")
print("6. How you improved the model with Hyperparameter Tuning.")


#%% 11. Save Final Model & Scaler
print("\nStep 11: Saving model and scaler...")

# --- This is the "production" step ---
# We save the tuned model and the scaler so our web app can use them.
try:
    joblib.dump(best_rf_model, 'aqi_rf_model.joblib')
    joblib.dump(scaler, 'aqi_scaler.joblib')
    print("Model and scaler saved successfully as 'aqi_rf_model.joblib' and 'aqi_scaler.joblib'")
except Exception as e:
    print(f"Error saving model: {e}")

