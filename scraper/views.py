import pickle, os
from django.shortcuts import render, redirect
from django.conf import settings
from .models import Race, Horse
from .forms import RacePredictionForm
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from django.http import JsonResponse

# 予測を行う前に、trained_columnsをロード
with open('trained_columns.pkl', 'rb') as f:
    trained_columns = pickle.load(f)

# グローバル変数としてスケーラを定義
scaler = StandardScaler()
# トレーニング時に保存したscalerの状態をロード
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def preprocess_data(data):
    # データをDataFrameに変換
    df = pd.DataFrame([data])

    # カテゴリ変数をワンホットエンコーディング
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    df_encoded = encoder.fit_transform(df[['race_location', 'race_weather', 'race_condition']])
    df_encoded = pd.DataFrame(df_encoded, columns=encoder.get_feature_names_out(['race_location', 'race_weather', 'race_condition']))
    df = pd.concat([df, df_encoded], axis=1)
    df.drop(columns=['race_location', 'race_weather', 'race_condition'], inplace=True)

    # トレーニング時に存在していたが、予測時のデータには存在しないカテゴリの列を追加し、0で埋める
    for col in trained_columns:
        if col not in df.columns:
            df[col] = 0

    # トレーニング時の特徴の順序に従ってデータを並べ替え
    df = df[trained_columns]

    # 数値変数のスケーリング
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) > 0:
        df[numeric_columns] = scaler.transform(df[numeric_columns])  # fit_transformからtransformに変更

    return df

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
    print("predict_race function called") 
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

            # データベースに指定されたレース名が存在するか確認
            try:
                race_instance = Race.objects.get(name=data['race_name'])
            except Race.DoesNotExist:
                # 指定されたレース名が存在しない場合、エラーメッセージを表示
                context = {
                    'form': form,
                    'error_message': '指定されたレース名は存在しません。'
                }
                return render(request, 'predict.html', context)

            # データの前処理
            try:
                processed_data = preprocess_data(data)
            except Exception as e:
                print(f"Error during preprocessing: {e}")
                context = {
                    'form': form,
                    'error_message': f"データの前処理中にエラーが発生しました: {e}"
                }
                return render(request, 'predict.html', context)
            
            try:
                # 予測モデルをロード
                model_path = os.path.join(settings.BASE_DIR, 'model.pkl')
                model = joblib.load(model_path)
                # 予測の実行
                prediction_values = model.predict(processed_data)
            except Exception as e:
                print(f"Error during prediction: {e}")
                context = {
                    'form': form,
                    'error_message': f"予測中にエラーが発生しました: {e}"
                }
                return render(request, 'predict.html', context)

            # 参加馬のリストと予測値を組み合わせて、予測値の降順にソート
            participating_horses = Horse.objects.filter(race=race_instance).values_list('name', flat=True)
            sorted_horses = sorted(zip(participating_horses, prediction_values), key=lambda x: x[1], reverse=True)

            # ソートされたリストから1位、2位、3位の馬名を取得
            first_place = sorted_horses[0][0]
            second_place = sorted_horses[1][0]
            third_place = sorted_horses[2][0]

            context = {
                'prediction': {
                    'first_place': first_place,
                    'second_place': second_place,
                    'third_place': third_place,
                },
                'race_name': data['race_name'],
                'weather': data['race_weather'],
                'course_condition': data['race_condition'],
                'participating_horses': [horse[0] for horse in sorted_horses],
            }
            print("Rendering prediction_result.html")
            return render(request, 'prediction_result.html', context)

        else:
            # フォームが無効な場合、エラーメッセージを表示
            context = {
                'form': form,
                'error_message': 'フォームの入力に問題があります。'
            }
            return render(request, 'predict.html', context)

    else:
        form = RacePredictionForm()
    context = {'form': form}
    print("Context content:", context)
    return render(request, 'predict.html', context)
