from django import forms

class RacePredictionForm(forms.Form):
    race_name = forms.CharField(label='レース名', max_length=100)
    race_date = forms.DateField(label='レースの日付', widget=forms.SelectDateWidget())
    race_location = forms.CharField(label='レースの場所', max_length=100) 
    participating_horses = forms.CharField(label='参加する馬の情報', widget=forms.Textarea)
    race_weather = forms.ChoiceField(label='天気', choices=[
        ('sunny', '晴れ'),
        ('cloudy', '曇り'),
        ('rainy', '雨'),
        ('stormy', '嵐'),
    ])
    race_condition = forms.ChoiceField(label='コースの状態', choices=[
        ('good', '良'),
        ('muddy', '稍重'),
        ('heavy', '重'),
        ('bad', '不良'),
    ])
