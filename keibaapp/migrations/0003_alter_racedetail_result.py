# Generated by Django 4.2.6 on 2023-10-22 12:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('keibaapp', '0002_remove_racedetail_horse_remove_racedetail_position_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='racedetail',
            name='result',
            field=models.IntegerField(blank=True, null=True),
        ),
    ]
