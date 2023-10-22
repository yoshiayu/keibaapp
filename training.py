import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "keibaapp.settings")

import django
django.setup()

from keibaapp.models import Race, Horse, RaceDetail
# 1. データベースからデータを取得
races = Race.objects.all()
horses = Horse.objects.all()
details = RaceDetail.objects.all()

# 2. データを前処理
# ここでは、簡単のためにpandas DataFrameを使用してデータを前処理します。
# 実際には、より詳細な前処理が必要になるかもしれません。
df_races = pd.DataFrame.from_records(races.values())
df_horses = pd.DataFrame.from_records(horses.values())
df_details = pd.DataFrame.from_records(details.values())

print(df_details.columns)
# データの結合と前処理の詳細を追加...

# 3. データをトレーニングセットとテストセットに分割
X = df_horses  # ターゲット変数を除外
y = df_details['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. ランダムフォレスト分類器をトレーニング
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# 5. モデルの性能を評価
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 6. モデルを保存
joblib.dump(clf, 'model.pkl')