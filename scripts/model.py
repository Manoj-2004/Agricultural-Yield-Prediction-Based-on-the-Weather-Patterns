import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
import ipywidgets as widgets
from IPython.display import display

# Load dataset
df = pd.read_csv("/content/drive/MyDrive/dataset/crop_noswvl4.csv")
df["Log_Yield"] = np.log1p(df["Yield"])

# Feature identification
categorical_cols = df.select_dtypes(include='object').columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove("Yield")

# Treat Year as categorical if needed
if 'Year' in numerical_cols:
    numerical_cols.remove('Year')
    categorical_cols.append('Year')

feature_cols = categorical_cols + numerical_cols
X = df[feature_cols].copy()
y = df["Log_Yield"]

# Label encoding for RF and XGBoost
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Train-test split
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)

# --- Random Forest ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# --- XGBoost ---
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest = xgb.DMatrix(X_test)

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtrain, 'train'), (dvalid, 'valid')],
    early_stopping_rounds=50,
    verbose_eval=False
)

# --- CatBoost ---
cat_features = [X.columns.get_loc(col) for col in categorical_cols]
train_pool = Pool(X_train, y_train, cat_features=cat_features)
valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

cat_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    loss_function='RMSE',
    eval_metric='RMSE',
    early_stopping_rounds=100,
    random_seed=42,
    verbose=0
)
cat_model.fit(train_pool, eval_set=valid_pool)

# --- Evaluation ---
rf_log_preds = rf_model.predict(X_test)
xgb_log_preds = xgb_model.predict(xgb.DMatrix(X_test))
cat_log_preds = cat_model.predict(test_pool)

rf_preds = np.expm1(rf_log_preds)
xgb_preds = np.expm1(xgb_log_preds)
cat_preds = np.expm1(cat_log_preds)
true_y = np.expm1(y_test)

metrics = {
    "Random Forest": {
        "mae": mean_absolute_error(true_y, rf_preds),
        "mse": mean_squared_error(true_y, rf_preds),
        "rmse": np.sqrt(mean_squared_error(true_y, rf_preds)),
        "r2": r2_score(true_y, rf_preds)
    },
    "XGBoost": {
        "mae": mean_absolute_error(true_y, xgb_preds),
        "mse": mean_squared_error(true_y, xgb_preds),
        "rmse": np.sqrt(mean_squared_error(true_y, xgb_preds)),
        "r2": r2_score(true_y, xgb_preds)
    },
    "CatBoost": {
        "mae": mean_absolute_error(true_y, cat_preds),
        "mse": mean_squared_error(true_y, cat_preds),
        "rmse": np.sqrt(mean_squared_error(true_y, cat_preds)),
        "r2": r2_score(true_y, cat_preds)
    }
}

# --- Utility Functions ---
def calculate_avg_weather(state, crop, season):
    filtered = df[(df["State"] == state) & (df["Crop"] == crop) & (df["cropSeason"] == season)]
    sorted_data = filtered.sort_values("Year", ascending=False).head(6)
    if sorted_data.empty:
        return None
    avg_weather = sorted_data[[col for col in numerical_cols if col != 'Year']].mean().to_dict()
    return avg_weather

def get_avg_yield(state, crop, season):
    filtered = df[(df["State"] == state) & (df["Crop"] == crop) & (df["cropSeason"] == season)]
    sorted_data = filtered.sort_values("Year", ascending=False).head(6)
    return sorted_data["Yield"].mean() if not sorted_data.empty else None

def predict_yield_model(state, crop, season, model_type="Random Forest"):
    avg_weather = calculate_avg_weather(state, crop, season)
    if avg_weather is None or any(pd.isnull(list(avg_weather.values()))):
        return "❌ Prediction failed: Insufficient historical data for the selected combination."

    input_data = {}
    for col in categorical_cols:
        le = label_encoders[col]
        if col == "State":
            input_data[col] = le.transform([state])[0]
        elif col == "Crop":
            input_data[col] = le.transform([crop])[0]
        elif col == "cropSeason":
            input_data[col] = le.transform([season])[0]
        elif col == "Year":
            filtered = df[(df["State"] == state) & (df["Crop"] == crop) & (df["cropSeason"] == season)]
            if filtered.empty:
                return "❌ Prediction failed: Cannot determine Year."
            latest_year = filtered.sort_values("Year", ascending=False).iloc[0]["Year"]
            input_data[col] = le.transform([latest_year])[0]

    for col in numerical_cols:
        input_data[col] = avg_weather.get(col, 0)

    input_df = pd.DataFrame([{col: input_data.get(col, 0) for col in X_train.columns}])

    if model_type == "Random Forest":
        log_pred = rf_model.predict(input_df)[0]
    elif model_type == "XGBoost":
        log_pred = xgb_model.predict(xgb.DMatrix(input_df))[0]
    elif model_type == "CatBoost":
        input_pool = Pool(input_df, cat_features=cat_features)
        log_pred = cat_model.predict(input_pool)[0]
    else:
        return "❌ Invalid model selected."

    pred_yield = np.expm1(log_pred)
    avg_hist_yield = get_avg_yield(state, crop, season)

    m = metrics[model_type]
    result = f"✅ Predicted Yield using {model_type}: {pred_yield:.2f} kg/hectare\n"
    if avg_hist_yield:
        result += f"📊 Historical Average (last 6 years): {avg_hist_yield:.2f} kg/hectare\n\n"
    result += "📌 Model Evaluation on Test Set:\n"
    result += f" - MAE: {m['mae']:.2f}\n - MSE: {m['mse']:.2f}\n - RMSE: {m['rmse']:.2f}\n - R²: {m['r2']:.4f}"
    return result

# --- UI with ipywidgets ---
state_widget = widgets.Dropdown(options=sorted(df['State'].unique()), description='State:')
crop_widget = widgets.Dropdown(options=sorted(df['Crop'].unique()), description='Crop:')
season_widget = widgets.Dropdown(options=sorted(df['cropSeason'].unique()), description='Season:')
model_widget = widgets.Dropdown(options=["Random Forest", "XGBoost", "CatBoost"], description='Model:')
predict_button = widgets.Button(description='Predict Yield', button_style='success')
output_box = widgets.Output()

def on_click(b):
    output_box.clear_output()
    with output_box:
        result = predict_yield_model(state_widget.value, crop_widget.value, season_widget.value, model_widget.value)
        print(result)

predict_button.on_click(on_click)

# --- Display UI ---
display(widgets.VBox([
    widgets.HTML("<h3>🌾 Crop Yield Prediction Interface</h3>"),
    state_widget,
    crop_widget,
    season_widget,
    model_widget,
    predict_button,
    output_box
]))