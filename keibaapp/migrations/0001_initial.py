# Generated by Django 4.2.6 on 2023-10-22 12:38

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Horse',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('win_rate', models.FloatField()),
                ('avg_odds', models.FloatField()),
            ],
        ),
        migrations.CreateModel(
            name='Race',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('date', models.DateField()),
                ('result', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='RaceDetail',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('position', models.IntegerField()),
                ('odds', models.FloatField()),
                ('horse', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='keibaapp.horse')),
                ('race', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='keibaapp.race')),
            ],
        ),
    ]
