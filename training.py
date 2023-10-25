import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler
import os
import pickle

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "keibaapp.settings")

import django
django.setup()

from keibaapp.models import Race, Horse, RaceDetail

# 1. データベースからデータを取得
details = RaceDetail.objects.all()

# 2. データを前処理
df_details = pd.DataFrame.from_records(details.values())

# One-Hot Encoding
encoder = OneHotEncoder(drop='first')
encoded_features = encoder.fit_transform(df_details[['horse_name', 'jockey_name']])
encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names_out(['horse_name', 'jockey_name']))

# 元のDataFrameと結合
df_details = pd.concat([df_details, encoded_df], axis=1)

# 不要なカラムを削除
X = df_details.drop(columns=['id', 'horse_name', 'jockey_name', 'result'])
y = df_details['result']

# データのスケーリング前にカラム名を保存
trained_columns = X.columns.tolist()

# データのスケーリング
scaler = StandardScaler()
X = scaler.fit_transform(X)

# scalerの状態を保存
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# オーバーサンプリング
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# データをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# クラスの最小のメンバー数を取得
min_class_size = y_resampled.value_counts().min()

# StratifiedKFoldのn_splitsの値をクラスの最小のメンバー数に合わせる
cv = StratifiedKFold(n_splits=min(min_class_size, 5)) 

# 3. ランダムフォレストモデルのハイパーパラメータのチューニング
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}

clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=cv, scoring='accuracy')
grid_search.fit(X_resampled, y_resampled)

# 最適なパラメータでモデルをトレーニング
best_clf = grid_search.best_estimator_

# データをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 4. モデルの性能を評価
y_pred = best_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# F1スコアを計算
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1 Score: {f1:.2f}")

# トレーニング時の特徴名を保存
with open('trained_columns.pkl', 'wb') as f:
    pickle.dump(trained_columns, f)

# トレーニングされたモデルを保存
with open('model.pkl', 'wb') as f:
    pickle.dump(best_clf, f)

print("Model saved as model.pkl")