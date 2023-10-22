# Generated by Django 4.2.6 on 2023-10-22 09:19

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('scraper', '0002_race_is_finished'),
    ]

    operations = [
        migrations.AddField(
            model_name='race',
            name='predicted_winner',
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
        migrations.CreateModel(
            name='RaceDetail',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('horse_name', models.CharField(max_length=100)),
                ('jockey_name', models.CharField(max_length=100)),
                ('odds', models.FloatField()),
                ('result', models.IntegerField()),
                ('race', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='scraper.race')),
            ],
        ),
    ]