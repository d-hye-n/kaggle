import pandas as pd
import os


base_path = os.path.dirname(os.path.abspath(__file__))
train_file_path = os.path.join('..', 'data', 'playground-series-s6e2', 'train.csv')
test_file_path = os.path.join('..', 'data', 'playground-series-s6e2', 'test.csv')

train = pd.read_csv(train_file_path)
test = pd.read_csv(test_file_path)

import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from optuna.integration import XGBoostPruningCallback
from optuna.integration import LightGBMPruningCallback

db_url = "sqlite:///heart_disease_optuna.db"

train['id'] = train['Heart Disease'].map({'Presence': 1, 'Absence': 0})

cat_cols = train.select_dtypes(include=['object']).columns
le = LabelEncoder()

for col in cat_cols:
    if col != 'Heart Disease':
        train[col] = le.fit_transform(train[col])
        test[col] = le.transform(test[col])

X = train.drop(columns=['Heart Disease', 'id'])
y = train['id']
X_test = test.drop(columns=['id'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'device': 'cuda',
        'tree_method': 'hist',
        'random_state': 42,
        'early_stopping_rounds': 50,
        'eval_metric': 'auc',
        'n_jobs': -1,
    }

    pruning_callback = XGBoostPruningCallback(trial, "validation_0-auc")

    model = XGBClassifier(**params, callbacks=[pruning_callback])
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    return auc

def objective_lgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'device': 'gpu',
        'random_state': 42,
        'early_stopping_rounds': 50,
        'metric': 'auc',
        'n_jobs': -1
    }

    pruning_callback = LightGBMPruningCallback(trial, "auc")

    model = LGBMClassifier(**params, callbacks=[pruning_callback])
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    return auc

if __name__ == "__main__":
    storage_url = "sqlite:///D:/Dev/python/kaggle/Predicting_Heart_Disease/heart_disease_optuna.db"

    study_xgb = optuna.create_study(study_name="xgb_v1", storage=storage_url, direction="maximize", load_if_exists=True)
    study_xgb.optimize(objective_xgb, n_trials=100)

    study_lgbm = optuna.create_study(study_name="lgbm_v1", storage=storage_url, direction="maximize", load_if_exists=True)
    study_lgbm.optimize(objective_lgbm, n_trials=100, n_jobs=1)

study_xgb = optuna.load_study(study_name="xgb_v1", storage=db_url)
study_lgbm = optuna.load_study(study_name="lgbm_v1", storage=db_url)

print('Best XGBoost parameters:', study_xgb.best_params)
best_xgb_params = study_xgb.best_params

print('Best LightGBM parameters:', study_lgbm.best_params)
best_lgbm_params = study_lgbm.best_params

'''
xgb_model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    tree_method="hist",
    device = "cuda",
    random_state=42,
    early_stopping_rounds=50
)

print('Training the XGBoost model...')
xgb_model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          verbose=100)

xgb_preds = xgb_model.predict_proba(X_test)[:, 1]

lgbm_model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    device = "cuda",
    early_stopping_rounds=50,
    random_state=42,
    verbose=100
)

print('Training the LightGBM model...')
lgbm_model.fit(X_train, y_train,
               eval_set=[(X_val, y_val)],
               )

lgbm_preds = lgbm_model.predict_proba(X_test)[:, 1]
final_preds = xgb_preds*0.5 + lgbm_preds*0.5
'''

final_xgb = XGBClassifier(**best_xgb_params, device='cuda', tree_method='hist')
final_lgbm = LGBMClassifier(**best_lgbm_params, device='gpu')

final_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
final_lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])


w_xgb = study_xgb.best_value
w_lgbm = study_lgbm.best_value

xgb_preds = final_xgb.predict_proba(X_test)[:, 1]
lgbm_preds = final_lgbm.predict_proba(X_test)[:, 1]

final_preds = (xgb_preds * w_xgb + lgbm_preds * w_lgbm) / (w_xgb + w_lgbm)

submission = pd.DataFrame({
    'id': test['id'],
    'Heart Disease': final_preds
})
submission_file_path = os.path.join(base_path, 'submission5.csv')
submission.to_csv(submission_file_path, index=False)
print(f'Submission file saved to {submission_file_path}')