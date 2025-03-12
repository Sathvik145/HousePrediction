import numpy as np
import pandas as pd
import joblib
from scipy.stats import randint
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Handle missing values
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_imputed.drop(columns=['target']))
df_scaled = pd.DataFrame(scaled_features, columns=data.feature_names)
df_scaled['target'] = df_imputed['target']

# Train-Test Split
X = df_scaled.drop(columns=['target'])
y = df_scaled['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Random Forest
rf_params = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5)
}

rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), rf_params, n_iter=10, scoring='r2', cv=3, verbose=1, random_state=42, n_jobs=-1)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

# Hyperparameter tuning for Gradient Boosting
gb_params = {
    'n_estimators': randint(50, 200),
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 10]
}

gb_search = RandomizedSearchCV(GradientBoostingRegressor(random_state=42), gb_params, n_iter=10, scoring='r2', cv=3, verbose=1, random_state=42, n_jobs=-1)
gb_search.fit(X_train, y_train)
best_gb = gb_search.best_estimator_

# Evaluate both models
models = {
    "Random Forest (Tuned)": best_rf,
    "Gradient Boosting (Tuned)": best_gb
}

best_model = None
best_score = -np.inf

for name, model in models.items():
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    print(f"\n{name}")
    print("R² Score:", score)
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    
    if score > best_score:
        best_score = score
        best_model = model

# Save the best model and scaler
joblib.dump(best_model, 'best_house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(f"\nBest model selected: {best_model.__class__.__name__} with R² Score: {best_score}")
print("Model and Scaler saved successfully.")
