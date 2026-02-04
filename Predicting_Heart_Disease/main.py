import pandas as pd
import os


base_path = os.path.dirname(os.path.abspath(__file__))
train_file_path = os.path.join('..', 'data', 'playground-series-s6e2', 'train.csv')
test_file_path = os.path.join('..', 'data', 'playground-series-s6e2', 'test.csv')

train = pd.read_csv(train_file_path)
test = pd.read_csv(test_file_path)


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

from lightgbm import LGBMClassifier

lgbm_model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
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


submission = pd.DataFrame({
    'id': test['id'],
    'Heart Disease': final_preds
})
submission_file_path = os.path.join(base_path, 'submission2.csv')
submission.to_csv(submission_file_path, index=False)
print(f'Submission file saved to {submission_file_path}')