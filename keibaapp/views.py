from django.shortcuts import render
from .models import Race, Horse

def analyze_race_data(races):
    # ここでレースデータの分析を行う
    # 例: 最も勝率の高い馬を取得
    best_horse = max(races, key=lambda race: race.horse.win_rate)
    return best_horse

def index(request):
    # データベースからレースデータを取得
    races = Race.objects.all().order_by('-date')

    # レースデータの分析
    best_horse = analyze_race_data(races)

    context = {
        'races': races,
        'best_horse': best_horse,
    }

    return render(request, 'index.html', context)
