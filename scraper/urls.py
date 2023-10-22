from django.urls import path
from . import views

urlpatterns = [
    # path('race_list/', views.race_list, name='race_list'),  # URLパターンを変更
    # path('race/<int:race_id>/', views.race_detail, name='race_detail'),
    path('', views.index, name='index'),
    path('predict/', views.predict_race, name='predict'),
    # path('prediction_result/', views.prediction_result, name='prediction_result'),
]
