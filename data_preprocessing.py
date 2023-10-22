from scraper.models import Race, Horse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# データベースからデータを取得
races = Race.objects.all().values()
horses = Horse.objects.all().values()

# QuerySetをDataFrameに変換
df_races = pd.DataFrame(races)
df_horses = pd.DataFrame(horses)

# ここでカラムを出力
print("df_races columns:", df_races.columns)
print("df_horses columns:", df_horses.columns)

# カラムが存在する場合のみ結合を実行
if 'race_id' in df_horses.columns and 'id' in df_races.columns:
    # Race IDをキーとしてデータを結合
    merged_data = pd.merge(df_horses, df_races, left_on='race_id', right_on='id', how='inner')
    
    # 欠損値の処理
    for column in merged_data.columns:
        if merged_data[column].dtype == 'object':  # カテゴリ変数の場合
            merged_data[column].fillna(merged_data[column].mode()[0], inplace=True)
        else:  # 数値変数の場合
            merged_data[column].fillna(merged_data[column].median(), inplace=True)
    
    # 不要な列の削除
    columns_to_drop = ['race_id']  # 例として'race_id'を削除
    merged_data.drop(columns=columns_to_drop, inplace=True)
else:
    print("Required columns for merging are missing.")

# 予測対象のカラムを指定 (例: 'winner')
y = merged_data['winner']
X = merged_data.drop('winner', axis=1)

# トレーニングデータとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ランダムフォレストモデルのインスタンスを作成
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# モデルのトレーニング
clf.fit(X_train, y_train)

# テストデータでの予測
y_pred = clf.predict(X_test)

# 精度の評価
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# トレーニングされたモデルを保存
joblib.dump(clf, 'trained_model.pkl')