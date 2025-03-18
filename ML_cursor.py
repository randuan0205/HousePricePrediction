import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('data/train.csv')

# Separate features and target
X = df.drop('SalePrice', axis=1)
y = np.log(df['SalePrice'])

# Split into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=200, random_state=42)

# Handle categorical variables
categorical_cols = X_train.select_dtypes(include=['object']).columns
X_train = pd.get_dummies(X_train, columns=categorical_cols)
X_valid = pd.get_dummies(X_valid, columns=categorical_cols)

# Align train and validation features
common_cols = set(X_train.columns) & set(X_valid.columns)
X_train = X_train[list(common_cols)]
X_valid = X_valid[list(common_cols)]

# Handle missing values
X_train = X_train.fillna(X_train.mean())
X_valid = X_valid.fillna(X_valid.mean())

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# 1. Train a simple XGBregressor
### simple XGBregressor
simplexgb=XGBRegressor(eval_metric='rmse')
simplexgb.fit(X_train_scaled, y_train)
simplexgb_pred = simplexgb.predict(X_valid_scaled)
simplexgb_rmse = np.sqrt(mean_squared_error(y_valid, simplexgb_pred))
print(f"Simple XGBregressor Validation RMSE: ${simplexgb_rmse:,.4f}")

# 2. XGBoost with RandomizedSearchCV
xgb_param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

xgb_random = RandomizedSearchCV(
    XGBRegressor(random_state=42),
    param_distributions=xgb_param_dist,
    n_iter=20,
    cv=5,
    random_state=42,
    n_jobs=-1
)

xgb_random.fit(X_train_scaled, y_train)
xgb_predictions = xgb_random.predict(X_valid_scaled)
xgb_rmse = np.sqrt(mean_squared_error(y_valid, xgb_predictions))
print(f"\nXGBoost Best Parameters: {xgb_random.best_params_}")
print(f"XGBoost Validation RMSE: ${xgb_rmse:,.4f}")


