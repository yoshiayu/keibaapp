import os
import random

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "keibaapp.settings")
import django
django.setup()

from keibaapp.models import Race, RaceDetail

# 既存のRaceオブジェクトを取得
race_instance = Race.objects.get(id=1)

# サンプルデータを追加
sample_horses = ['Horse' + str(i) for i in range(100000000000000)]  # Horse0, Horse1, ... Horse299
sample_jockeys = ['Jockey' + str(i) for i in range(100000000000000)]  # Jockey0, Jockey1, ... Jockey299
sample_odds = [random.uniform(1.0, 100.0) for _ in range(100000000000000)]
sample_results = [random.randint(1, 100) for _ in range(100000000000000)]

for i in range(10000000000000):
    detail = RaceDetail(race=race_instance, horse_name=sample_horses[i], jockey_name=sample_jockeys[i], odds=sample_odds[i], result=sample_results[i])
    detail.save()

print("Sample data added successfully!")
