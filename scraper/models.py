from django.db import models

class Race(models.Model):
    name = models.CharField(max_length=200)
    date = models.DateField()
    place = models.CharField(max_length=50)
    distance = models.IntegerField()
    weather = models.CharField(max_length=50)
    ground_status = models.CharField(max_length=50)
    prize_distribution = models.TextField()
    race_category = models.CharField(max_length=50)
    is_finished = models.BooleanField(default=False)
    predicted_winner = models.CharField(max_length=200, null=True, blank=True)
    
class Horse(models.Model):
    race = models.ForeignKey(Race, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    jockey_name = models.CharField(max_length=200)
    frame_number = models.IntegerField()
    horse_number = models.IntegerField()
    odds = models.FloatField()
    weight = models.FloatField()
    training_center = models.CharField(max_length=50)
    previous_race_result = models.CharField(max_length=50)
    age = models.IntegerField()
    weight_carried = models.FloatField()
    win_rate = models.FloatField()
    top3_rate = models.FloatField()
    trainer = models.CharField(max_length=200)
    owner = models.CharField(max_length=200)
    breeder = models.CharField(max_length=200)
    pedigree = models.TextField()
    weight_change = models.FloatField()
    popularity_rank = models.IntegerField()
    days_since_last_race = models.IntegerField()
    sire = models.CharField(max_length=200)
    dam = models.CharField(max_length=200)
    dams_sire = models.CharField(max_length=200)

class RaceDetail(models.Model):
    race = models.ForeignKey(Race, on_delete=models.CASCADE)
    horse_name = models.CharField(max_length=100)
    jockey_name = models.CharField(max_length=100)
    odds = models.FloatField()
    result = models.IntegerField(default=0)


    def __str__(self):
        return self.horse_name