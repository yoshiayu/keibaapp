from django.shortcuts import render, redirect
from .models import Race, Horse
from .forms import RacePredictionForm
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# グローバル変数としてスケーラを定義
scaler = StandardScaler()

trained_columns = ['race_location_東京', 'race_location_京都', 'race_weather_sunny', 'race_weather_rainy', 'race_condition_good', 'race_condition_bad']

def preprocess_data(data):
    # データをDataFrameに変換
    df = pd.DataFrame([data])

    # カテゴリ変数をワンホットエンコーディング
    df = pd.get_dummies(df, columns=['race_location', 'race_weather', 'race_condition'])

    # トレーニング時に存在していたが、予測時のデータには存在しないカテゴリの列を追加し、0で埋める
    for col in trained_columns:
        if col not in df.columns:
            df[col] = 0

    # 欠損値の処理
    for column in df.columns:
        if df[column].dtype == 'object':  # カテゴリ変数の場合
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:  # 数値変数の場合
            df[column].fillna(df[column].median(), inplace=True)

    # その他のカテゴリ変数のワンホットエンコーディング
    cat_columns = df.select_dtypes(include=['object']).columns
    if len(cat_columns) > 0:
        encoder = OneHotEncoder(drop='first', sparse=False)
        encoded_data = encoder.fit_transform(df[cat_columns])
        df_encoded = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_columns))
        df.drop(columns=cat_columns, inplace=True)
        df = pd.concat([df, df_encoded], axis=1)

    # 数値変数のスケーリング
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    print("Numeric columns:", numeric_columns)  # ここで数値のカラムを出力
    if len(numeric_columns) > 0:  # 数値のカラムが存在する場合のみスケーリングを実行
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    else:
        print("No numeric columns found for scaling.")
    
    print(df.values) 
    return df[trained_columns].values

def index(request):
    # 最新のレース予測結果を取得
    latest_predictions = Race.objects.all().order_by('-date')[:5]

    # 過去のレース結果を取得
    past_results = Race.objects.filter(is_finished=True).order_by('-date')[:5]

    # 馬の統計情報を取得
    horse_stats = Horse.objects.all().order_by('-win_rate')[:5]

    context = {
        'predictions': latest_predictions,
        'past_results': past_results,
        'horse_stats': horse_stats,
    }

    return render(request, 'index.html', context)

def predict_race(request):
    if request.method == 'POST':
        form = RacePredictionForm(request.POST)
        if form.is_valid():
            # ユーザーからの入力データの取得
            data = {
                'race_name': form.cleaned_data['race_name'],
                'race_date': form.cleaned_data['race_date'],
                'race_location': form.cleaned_data['race_location'],
                'race_weather': form.cleaned_data['race_weather'],
                'race_condition': form.cleaned_data['race_condition']
            }

            # データの前処理
            processed_data = preprocess_data(data)

            # 予測モデルをロード
            model = joblib.load('model.pkl')
            print(processed_data)
            # 予測の実行
            prediction = model.predict(processed_data)[0]
            print("Prediction:", prediction)

            # 予測結果の表示
            return render(request, 'prediction_result.html', {'prediction': prediction})
    else:
        form = RacePredictionForm()

    return render(request, 'predict.html', {'form': form})

# def prediction_result(request):
#     # 予測モデルをロード
#     with open('trained_model.pkl', 'rb') as f:
#         model = pickle.load(f)

#     # リクエストから入力データを取得
#     race_name = request.POST.get('race_name')
#     weather = request.POST.get('weather')
#     course_condition = request.POST.get('course_condition')
#     race_distance = request.POST.get('race_distance')
#     horse_age = request.POST.get('horse_age')

#     # 入力データをモデルが受け入れる形式に変換
#     input_data = {
#         'race_name': race_name,
#         'weather': weather,
#         'course_condition': course_condition,
#         'race_distance': race_distance,
#         'horse_age': horse_age,
#     }

#     # 予測を実行
#     predicted_horses = model.predict([input_data])

#     # 予測結果をコンテキストに格納
#     context = {
#         'first_place': predicted_horses[0],
#         'second_place': predicted_horses[1],
#         'third_place': predicted_horses[2],
#     }

#     return render(request, 'prediction_result.html', context)
