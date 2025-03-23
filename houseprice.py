## training a XGBregressor
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import uniform, randint, loguniform

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from xgboost import XGBClassifier, DMatrix, XGBRegressor
from category_encoders import TargetEncoder
import pickle

import sys
print(sys.executable)

def data_preprocessing(traindataloc = "data/train.csv", testdataloc = "data/test.csv"):
  ## read in the data
  traindata = pd.read_csv(traindataloc)
  print("Full train dataset shape is {}".format(traindata.shape))
  testdata = pd.read_csv(testdataloc)
  print("Full test dataset shape is {}".format(testdata.shape))

  ##separate X and y: create X_train_combo and ylog (X_train_combo will be split into X_train and X_val later)
  X_mlb = traindata.drop('SalePrice', axis=1)
  X_mlb.set_index('Id', inplace=True)
  ylog = np.log(traindata['SalePrice']) ## apply log to target based on competition host's request
  ## set up the dataset X_test for final submission
  X_test = testdata.copy(deep=True) ##create a data copy
  X_test.set_index('Id', inplace=True)

  ylog.index=X_mlb.index

  ## feature cleaning
  ## drop 2 categorical features with super high NA%
  X_mlb.drop(['PoolQC','MiscFeature'], axis=1, inplace=True)
  ### fill NA for a bunch of categorical features - fill with 'None'
  catna_none = ['Alley','Fence','MasVnrType', 'FireplaceQu', 'GarageType', 'GarageQual', 'GarageCond', 'GarageFinish', 'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond']
  X_mlb[catna_none] = X_mlb[catna_none].fillna('None')
  ### fill na for cat features 'Electrical' - fill with most frequent value
  most_frequent_value = X_mlb['Electrical'].mode()[0]
  X_mlb['Electrical'] = X_mlb['Electrical'].fillna(most_frequent_value)
  ### fill NA for 2 numerical features
  X_mlb['GarageYrBlt'] = X_mlb['GarageYrBlt'].fillna(1700) ##fill na of garageyrblt with 1700
  X_mlb['MasVnrArea'] = X_mlb['MasVnrArea'].fillna(0) ##fill na of MasVnrArea with 0
  ##fill na of LotFrontage with mean
  X_mlb['LotFrontage'] = X_mlb['LotFrontage'].fillna(X_mlb['LotFrontage'].mean())
  print("Full train dataset shape after cleaning is {}".format(X_mlb.shape))
  print("Full test dataset shape after cleaning is {}".format(X_test.shape))
  return X_mlb, ylog, X_test

def mean_encoding(X_mlb, ylog):
  # Split the data into training and validation sets with fixed validation size of 200 samples
  X_train, X_valid, y_train, y_valid = train_test_split(X_mlb, ylog, test_size=200, random_state=42)

  # Get list of categorical columns from Xcombo
  catcollist = X_mlb.select_dtypes(include=['object']).columns.tolist()

  # Apply mean encoding on the training set
  mean_encoder = TargetEncoder(cols=catcollist)
  X_train_encoded = mean_encoder.fit_transform(X_train, y_train)
  # Transform the test set using the fitted mean encoder
  X_valid_encoded = mean_encoder.transform(X_valid)

  print("Encoded Training Set shape: ", X_train_encoded.shape)
  print("Encoded Valid Set shape: ", X_valid_encoded.shape)
  return mean_encoder,X_train_encoded, X_valid_encoded, y_train, y_valid

def feature_selection(X_train_encoded, X_valid_encoded, y_train):
  # Initialize the model
  xgbmodel = XGBRegressor(objective='reg:squarederror',
                          n_estimators=100,
                          learning_rate=0.1,
                          eval_metric='rmse',
                          n_jobs=-1)

  # Perform RFE
  selector = RFE(estimator=xgbmodel, n_features_to_select=60, step=1)
  selector = selector.fit(X_train_encoded, y_train)

  # Get the selected features
  selected_features = X_train_encoded.columns[selector.support_]
  X_train_RFE = selector.transform(X_train_encoded)
  X_valid_RFE = selector.transform(X_valid_encoded)

  print("RFE Training Set shape: ", X_train_RFE.shape)
  print("RFE Valid Set shape: ", X_valid_RFE.shape)
  return selector, selected_features,X_train_RFE, X_valid_RFE

def model_training(model, param_dist, n_iter, cv, scoring, X_train_RFE, y_train, X_valid_RFE, y_valid):
  # Initialize RandomizedSearchCV
  random_search_rfe = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=n_iter, cv=cv, scoring=scoring, random_state=42, verbose=2, n_jobs=3)
  # Fit the model
  random_search_rfe.fit(X_train_RFE, y_train, eval_set=[(X_valid_RFE, y_valid)],verbose=False)
  # Print the best score
  print("Best score: ", np.sqrt((-random_search_rfe.best_score_)))
  # Print the best parameters
  print("Best parameters found: ", random_search_rfe.best_params_)
  # Use the best model to make predictions
  tunedmodel_rfe = random_search_rfe.best_estimator_
  # Evaluate the model using RMSE
  y_pred = tunedmodel_rfe.predict(X_valid_RFE)
  rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
  print(f'Root Mean Squared Error for validset: {rmse:.4f}')
  ytrain_pred = tunedmodel_rfe.predict(X_train_RFE)
  rmse_train = np.sqrt(mean_squared_error(y_train, ytrain_pred))
  print(f'Root Mean Squared Error for trainset: {rmse_train:.4f}')
  return random_search_rfe.best_params_

def test_data_cleaning(X_test, mean_encoder, selector):
  ## drop 2 categorical features with super high NA%
  ## drop 2 categorical features with super high NA%
  X_test2 = X_test.copy(deep=True)
  X_test2.drop(['PoolQC','MiscFeature'], axis=1, inplace=True)
  ### fill NA for a bunch of categorical features - fill with 'None'
  catna_none = ['Alley','Fence','MasVnrType', 'FireplaceQu', 'GarageType', 'GarageQual', 'GarageCond', 'GarageFinish', 'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond']
  X_test2[catna_none] = X_test2[catna_none].fillna('None')
  ### fill na for cat features 'Electrical' - fill with most frequent value
  most_frequent_value = X_test2['Electrical'].mode()[0]
  X_test2['Electrical'] = X_test2['Electrical'].fillna(most_frequent_value)
  ### fill NA for 2 numerical features
  X_test2['GarageYrBlt'] = X_test2['GarageYrBlt'].fillna(1700) ##fill na of garageyrblt with 1700
  X_test2['MasVnrArea'] = X_test2['MasVnrArea'].fillna(0) ##fill na of MasVnrArea with 0
  ##fill na of LotFrontage with mean
  X_test2['LotFrontage'] = X_test2['LotFrontage'].fillna(X_test2['LotFrontage'].mean())

  ## implement mean-encoding
  X_test_encoded = mean_encoder.transform(X_test2)
  print("Test Set shape: ", X_test_encoded.shape)
  ##implement RFE selection
  X_test_RFE = selector.transform(X_test_encoded)
  print(X_test_RFE.shape)
  return X_test_RFE

def model_inference(X_test, X_test_RFE, selected_features, X_train_RFE, X_valid_RFE, y_train, y_valid, best_params):
  ## combine the train data and valid data
  X_train_RFEdf = pd.DataFrame(X_train_RFE, columns=selected_features)
  X_valid_RFEdf = pd.DataFrame(X_valid_RFE, columns=selected_features)
  X_trainrfe_combo = pd.concat([X_train_RFEdf, X_valid_RFEdf], axis=0)
  ## combine the train data and valid data
  y_train_RFE = pd.DataFrame(y_train)
  y_valid_RFE = pd.DataFrame(y_valid)
  y_trainrfe_combo = pd.concat([y_train_RFE, y_valid_RFE], axis=0, ignore_index=True)

  # Retrieve the best parameters from RandomizedSearchCV
  additional_params = {
      'objective':'reg:squarederror',
      'eval_metric':'rmse'
  }
  # Update the best parameters with the new ones
  best_params.update(additional_params)
  # Create a new model with the updated parameters
  model_inference = XGBRegressor(**best_params)
  # Train the final model on the combined dataset (or any dataset you choose)
  model_inference.fit(X_trainrfe_combo, y_trainrfe_combo)
  ytrain_pred_rfe = model_inference.predict(X_trainrfe_combo)
  rmse_train = np.sqrt(mean_squared_error(y_trainrfe_combo, ytrain_pred_rfe))
  print(f'Root Mean Squared Error for trainset: {rmse_train:.4f}')

  y_pred_test = model_inference.predict(X_test_RFE)
  testpred = np.exp(y_pred_test)
  print(f'average sales prediction: {np.mean(testpred)}')

  test_id=X_test.index.tolist()
  sub = pd.DataFrame()
  sub['Id'] = test_id
  sub['SalePrice'] = testpred
  print(f'snapshot of inference results: {sub.head()}')
  return sub

## training a  MLP model
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import optuna
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32         # Batch size

# ====== Data Pipeline - normalize and pack train/valid data to Loaders ======
def dnn_loader (batch_size, X_train_encoded, X_valid_encoded, y_train, y_valid):
  ## give the pre-RFE dataset to DNN to let it implement feature engineering
  train_features_np = X_train_encoded.values
  valid_features_np = X_valid_encoded.values

  # Normalize features using StandardScaler
  scaler = StandardScaler()
  train_features_np = scaler.fit_transform(train_features_np)
  valid_features_np = scaler.transform(valid_features_np)

  ## Normalize prediction target
  target_scaler = StandardScaler()
  y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
  y_valid_scaled = target_scaler.transform(y_valid.values.reshape(-1, 1))

  # Convert NumPy arrays to PyTorch tensors after normalization
  train_features = torch.tensor(train_features_np, dtype=torch.float32).to(device)
  valid_features = torch.tensor(valid_features_np, dtype=torch.float32).to(device)
  train_labels = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1).to(device)
  valid_labels = torch.tensor(y_valid_scaled, dtype=torch.float32).view(-1, 1).to(device)

  # Create dataloaders
  train_dataset = TensorDataset(train_features, train_labels)
  valid_dataset = TensorDataset(valid_features, valid_labels)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
  return train_loader, val_loader, valid_features, valid_labels, scaler, target_scaler


# Fixed hyperparameters (for aspects not tuned by Optuna)
dropout_rate = 0.2      # Dropout rate in the MLP
max_epochs = 100        # Maximum number of epochs per trial
patience = 10           # Patience for early stopping (used in scheduler)
criterion = nn.MSELoss()  # Mean Squared Error for regression

# ====== Model Definition Functions ======
def define_model(trial, X_train_encoded):
    """Define the MLP architecture for a given Optuna trial.
       Tuning both number of hidden layers and neurons per layer."""
    n_layers = trial.suggest_int("n_layers", 2, 4)
    layers = []
    input_dim = X_train_encoded.shape[1]  # number of features
    for i in range(n_layers):
        units = trial.suggest_int(f"n_units_l{i}", 64, 256)
        layers.append(nn.Linear(input_dim, units))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        input_dim = units
    layers.append(nn.Linear(input_dim, 1))  # Output layer for regression
    model = nn.Sequential(*layers)
    return model

def build_model_from_config(config, X_train_encoded):
    """Rebuild a model from a configuration dictionary (e.g., best found params)."""
    n_layers = config["n_layers"]
    layers = []
    input_dim = X_train_encoded.shape[1]
    for i in range(n_layers):
        units = config[f"n_units_l{i}"]
        layers.append(nn.Linear(input_dim, units))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        input_dim = units
    layers.append(nn.Linear(input_dim, 1))
    model = nn.Sequential(*layers)
    return model

## generate the data loaders for DNN model training
# train_loader, val_loader, valid_features, valid_labels, target_scaler = dnn_loader(batch_size, X_train_encoded, X_valid_encoded, y_train, y_valid)

# ====== Training Function with ReduceLROnPlateau ======
def train_model(model, lr, train_loader, valid_features, valid_labels):
    """
    Train the model for up to max_epochs with early stopping using
    ReduceLROnPlateau for dynamic learning rate adjustment.
    Returns the best validation loss and the best model state dict.
    """
    # Create optimizer with the tuned learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Setup ReduceLROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=patience, factor=0.5, min_lr=1e-6, verbose=True
    )

    best_val_loss = float('inf')
    best_state_dict = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        total_samples = 0
        with torch.no_grad():
            preds = model(valid_features)
            val_loss = F.mse_loss(preds, valid_labels).item()

        # Update best validation loss and state dict if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

        # Step the scheduler using the validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if epoch % 100 == 0:
          print(f"Epoch {epoch}: val_loss={val_loss:.6f}, lr={current_lr:.1e}")

        # Early stopping: if lr has reached the minimum threshold, stop training
        if current_lr <= scheduler.min_lrs[0]:
            print(f"Learning rate reached minimum threshold ({scheduler.min_lrs[0]}). Early stopping.")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return best_val_loss, best_state_dict

# ====== Global Tracking for Best Model ======
best_global_model_state = None
best_global_val_loss = float('inf')
best_global_params = None

# ====== Optuna Objective Function ======
def objective(trial, X_train_encode, train_loader, valid_features, valid_labels):
    global best_global_model_state, best_global_val_loss, best_global_params

    # Tune the initial learning rate as well as the network architecture
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    model = define_model(trial, X_train_encode).to(device)
    val_loss, best_state = train_model(model, lr, train_loader, valid_features, valid_labels)

    # Update global best if current trial is better
    if val_loss < best_global_val_loss:
        best_global_val_loss = val_loss
        best_global_params = trial.params  # includes architecture & learning rate
        best_global_model_state = best_state
        torch.save(best_global_model_state, "best_dnnmodel_statedict_prodrun.pth")
        print(f"*** New best model found! Val MSE={val_loss:.6f}, Params={best_global_params}")
    return val_loss  # Optuna minimizes this

def create_objective(X_train_encoded, train_loader, valid_features, valid_labels):
    def objective_wrapper(trial):
        return objective(trial, X_train_encoded, train_loader, valid_features, valid_labels)
    return objective_wrapper

def main():
## data preprocessing, feature enginnering, feature selection
    X_mlb, ylog, X_test = data_preprocessing()
    mean_encoder, X_train_encoded, X_valid_encoded, y_train, y_valid = mean_encoding(X_mlb, ylog)
    selector, selected_features, X_train_RFE, X_valid_RFE = feature_selection(X_train_encoded, X_valid_encoded, y_train)
    ## Model training with parameter tuning
    model = XGBRegressor(objective='reg:squarederror',eval_metric='rmse')
    n_iter=50
    cv=3
    scoring='neg_mean_squared_error'
    # Define the parameter distribution
    param_dist = { 'n_estimators': randint(150, 500)
                    ,'max_depth': randint(3, 10)  # Randomly pick between 3 and 10
                    ,'learning_rate': loguniform(0.005, 0.3)  # Log-uniform distribution between 0.01 and 0.3
                    ,'subsample': uniform(0.5, 0.5)  # Randomly pick between 0.6 and 1.0
                    ,'colsample_bytree': uniform(0.4, 0.6)  # Randomly pick between 0.6 and 1.0
                    ,'gamma': uniform(0, 0.5)  # Randomly pick between 0 and 0.5
                    ,'min_child_weight': randint(1, 6)  # Randomly pick between 1 and 5
                    #,'reg_alpha': uniform(0, 10)  # Randomly pick between 0 and 10
                    #,'reg_lambda': uniform(0, 10)  # Randomly pick between 0 and 10
                }

    best_params = model_training(model, param_dist, n_iter, cv, scoring, X_train_RFE, y_train, X_valid_RFE, y_valid)

    ## Model Inference
    X_test_RFE = test_data_cleaning(X_test, mean_encoder, selector)
    pred_sub=model_inference(X_test, X_test_RFE, selected_features, X_train_RFE, X_valid_RFE, y_train, y_valid, best_params)

    ### DNN model training
    train_loader, val_loader, valid_features, valid_labels, scaler, target_scaler = dnn_loader(batch_size, X_train_encoded, X_valid_encoded, y_train, y_valid)
    # ====== Run Optuna Study ======
    study = optuna.create_study(direction="minimize")
    objective_wrapper = create_objective(X_train_encoded, train_loader, valid_features, valid_labels)
    print("Starting hyperparameter search...")
    study.optimize(objective_wrapper, n_trials=100, gc_after_trial=True)  # 100 trials

    # ====== Hyperparameter Tuning Results ======
    print("\nOptuna search completed.")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial index: {study.best_trial.number}")
    print(f"Best validation MSE: {study.best_value:.6f}")
    best_params = study.best_params
    print("Best hyperparameter configuration:", best_params)
    best_val_rmse = study.best_value ** 0.5
    print(f"Best validation RMSE: {best_val_rmse:.6f}")

    # Save the best parameters to a file
    with open('dnn_best_params_prodrun.pkl', 'wb') as file:
        pickle.dump(best_params, file)


    # rebuild the DNN
    with open('dnn_best_params_prodrun.pkl', 'rb') as file:
        best_params = pickle.load(file)
    best_model = build_model_from_config(best_params, X_train_encoded).to(device)
    best_model.load_state_dict(torch.load("best_dnnmodel_statedict_prodrun.pth"))

    # Evaluation on validation set
    best_model.eval()
    with torch.no_grad():
        valid_predictions = best_model(valid_features)
        valid_predictions = valid_predictions.cpu().numpy()  # if using GPU
        valid_predictions_inv = target_scaler.inverse_transform(valid_predictions)
        rmse = np.sqrt(mean_squared_error(y_valid, valid_predictions_inv))
        print(f"RMSE on validation set: {rmse:.4f}")

if __name__ == "__main__":
    main()