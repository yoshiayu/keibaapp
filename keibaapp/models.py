from django.db import models

class Race(models.Model):
    name = models.CharField(max_length=255)  # レース名
    date = models.DateField()  # レース日
    result = models.TextField()  # レース結果

    def __str__(self):
        return self.name

class Horse(models.Model):
    name = models.CharField(max_length=255)  # 馬の名前
    win_rate = models.FloatField()  # 勝率
    avg_odds = models.FloatField()  # 平均オッズ

    def __str__(self):
        return self.name
class RaceDetail(models.Model):
    race = models.ForeignKey(Race, on_delete=models.CASCADE)
    horse_name = models.CharField(max_length=100, default="Unknown Horse")  # デフォルト値を"Unknown Horse"として設定
    jockey_name = models.CharField(max_length=100, default="Unknown Jockey")  # 必要に応じてデフォルト値を設定
    odds = models.FloatField(default=0.0)  # 必要に応じてデフォルト値を設定
    result = models.IntegerField(default=0)

    # result = models.IntegerField()
    # result = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return self.horse_name