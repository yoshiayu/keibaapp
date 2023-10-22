import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

BASE_URL = "https://www.netkeiba.com/"
RACE_LIST_URL = BASE_URL + "?pid=race_list&word=&track=&start_year=2018&end_year=2023&page={}"

def analyze_data(data):
    # データをpandasのDataFrameに変換
    df = pd.DataFrame(data)

    # 1. 各騎手の勝利数を集計
    jockey_wins = df['horses'].apply(lambda x: x[0]['jockey_name']).value_counts()

    # 2. 馬場の状態ごとの勝利数を集計
    ground_status_wins = df['race_ground_status'].value_counts()

    # 3. 血統ごとの勝利数を集計
    sire_wins = df['horses'].apply(lambda x: x[0]['sire']).value_counts()
    dam_wins = df['horses'].apply(lambda x: x[0]['dam']).value_counts()
    dams_sire_wins = df['horses'].apply(lambda x: x[0]['dams_sire']).value_counts()

    # 4. 左回りと右回りのコースでの勝利数を集計
    # この部分はサンプルデータに左回りと右回りの情報が含まれていないため、仮のコードとなります。
    # course_wins = df[df['course_type'] == '左回り']['horses'].apply(lambda x: x[0]['horse_name']).value_counts()
    # course_wins = df[df['course_type'] == '右回り']['horses'].apply(lambda x: x[0]['horse_name']).value_counts()

    print("各騎手の勝利数:\n", jockey_wins)
    print("\n馬場の状態ごとの勝利数:\n", ground_status_wins)
    print("\n血統ごとの勝利数 (Sire):\n", sire_wins)
    print("\n血統ごとの勝利数 (Dam):\n", dam_wins)
    print("\n血統ごとの勝利数 (Dam's Sire):\n", dams_sire_wins)
    # print("\n左回りと右回りのコースでの勝利数:\n", course_wins)

def extended_analysis(data):
    df = pd.DataFrame(data)

    # 1. 各年ごとのレース数を集計
    race_counts_by_year = df['race_year'].value_counts()

    # 2. 各騎手の平均オッズを計算
    df['average_odds'] = df['horses'].apply(lambda x: x[0]['odds'])
    avg_odds_by_jockey = df.groupby(df['horses'].apply(lambda x: x[0]['jockey_name']))['average_odds'].mean()

    # 3. 各馬場の状態ごとの平均オッズを計算
    avg_odds_by_ground_status = df.groupby('race_ground_status')['average_odds'].mean()

    # 4. 血統ごとの平均オッズを計算
    avg_odds_by_sire = df.groupby(df['horses'].apply(lambda x: x[0]['sire']))['average_odds'].mean()
    avg_odds_by_dam = df.groupby(df['horses'].apply(lambda x: x[0]['dam']))['average_odds'].mean()

    print("各年ごとのレース数:\n", race_counts_by_year)
    print("\n各騎手の平均オッズ:\n", avg_odds_by_jockey)
    print("\n各馬場の状態ごとの平均オッズ:\n", avg_odds_by_ground_status)
    print("\n血統(Sire)ごとの平均オッズ:\n", avg_odds_by_sire)
    print("\n血統(Dam)ごとの平均オッズ:\n", avg_odds_by_dam)

def preprocess_data(details):
    # 1. 不要な文字や空白の除去
    details["race_name"] = details["race_name"].replace("\n", "").strip()
    details["race_date"] = details["race_date"].replace("\n", "").strip()
    details["race_place"] = details["race_place"].replace("\n", "").strip()
    details["race_distance"] = details["race_distance"].replace("m", "").strip()

    # 2. データ型の変換
    details["race_distance"] = int(details["race_distance"])

    # レースの日付を年、月、日に分割
    date_parts = details["race_date"].split("/")
    details["race_year"] = int(date_parts[0])
    details["race_month"] = int(date_parts[1])
    details["race_day"] = int(date_parts[2])
    del details["race_date"]  # 元の日付の情報は削除

    # 賞金分布を整数のリストに変換
    details["prize_distribution"] = [int(prize.replace(",", "").replace("万円", "")) for prize in details["prize_distribution"]]

    for horse in details["horses"]:
        horse["frame_number"] = int(horse["frame_number"])
        horse["horse_number"] = int(horse["horse_number"])
        horse["odds"] = float(horse["odds"].replace(",", ""))
        horse["weight"] = int(horse["weight"].replace("kg", "").strip())

        # 馬の勝率とトップ3率を0-1の範囲の浮動小数点数に変換
        horse["win_rate"] = float(horse["win_rate"].replace("%", "")) / 100
        horse["top3_rate"] = float(horse["top3_rate"].replace("%", "")) / 100

        # 体重変動を整数に変換
        if horse["weight_change"].startswith("+"):
            horse["weight_change"] = int(horse["weight_change"].replace("+", ""))
        elif horse["weight_change"].startswith("-"):
            horse["weight_change"] = -int(horse["weight_change"].replace("-", ""))
        else:
            horse["weight_change"] = 0

    return details

def get_race_details(race_url):
    response = requests.get(race_url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # レース名の取得
    race_name = soup.select_one(".RaceName").text.strip()
    
    # レースの日付の取得
    race_date = soup.select_one(".RaceData01 .RaceData01_Data").text.strip()
    
    # レースの場所の取得
    race_place = soup.select_one(".RaceData01 .RaceData01_Data span").text.strip()
    
    # レースの距離の取得
    race_distance = soup.select_one(".RaceData01 span.Icon_Small").next_sibling.strip()
    
    # レースの天気の取得
    race_weather = soup.select_one(".RaceData01 img.Weather").get("alt")
    
    # レースの馬場の状態の取得
    race_ground_status = soup.select_one(".RaceData01 img.Baba").get("alt")

    # レースの賞金分布の取得
    prize_distribution = [td.text.strip() for td in soup.select(".RaceData02 td")]

    # レースのグレード・カテゴリーの取得
    race_category = soup.select_one(".RaceData02 span").text.strip()

    # 出走馬の情報の取得
    horse_details = []
    horse_rows = soup.select(".Shutuba_Table tbody tr")
    for row in horse_rows:
        horse_name = row.select_one(".HorseInfo a").text.strip()
        jockey_name = row.select_one(".Jockey a").text.strip()
        frame_number = row.select_one(".Waku").text.strip()
        horse_number = row.select_one(".Num").text.strip()
        odds = row.select_one(".Odds").text.strip()
        weight = row.select_one(".Weight").text.strip()
        training_center = row.select_one(".TrainingCenter").text.strip()
        previous_race_result = row.select_one(".PastPerformance a").text.strip()
        weight_change = row.select_one(".WeightDiff").text.strip()
        popularity_rank = row.select_one(".Popularity").text.strip()
        days_since_last_race = row.select_one(".PastPerformance span").text.strip()
        sire = row.select_one(".Pedigree01 a").text.strip()
        dam = row.select_one(".Pedigree02 a").text.strip()
        dams_sire = row.select_one(".Pedigree03 a").text.strip()

        horse_detail = {
            "horse_name": horse_name,
            "jockey_name": jockey_name,
            "frame_number": frame_number,
            "horse_number": horse_number,
            "odds": odds,
            "weight": weight,
            "training_center": training_center,
            "previous_race_result": previous_race_result,
            "weight_change": weight_change,
            "popularity_rank": popularity_rank,
            "days_since_last_race": days_since_last_race,
            "sire": sire,
            "dam": dam,
            "dams_sire": dams_sire
        }
        horse_details.append(horse_detail)

    return {
        "race_name": race_name,
        "race_date": race_date,
        "race_place": race_place,
        "race_distance": race_distance,
        "race_weather": race_weather,
        "race_ground_status": race_ground_status,
        "horses": horse_details,  # こちらを修正しました
        "prize_distribution": prize_distribution,
        "race_category": race_category
    }

def scrape_netkeiba():
    all_data = []
    for page in range(1, 5):  # 例として最初の4ページを対象にします
        url = RACE_LIST_URL.format(page)
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        
        race_links = [a["href"] for a in soup.select(".RaceList_Item02 a")]
        
        for race_link in race_links:
            race_url = BASE_URL + race_link
            details = get_race_details(race_url)
            
            # データの前処理
            processed_data = preprocess_data(details)
            all_data.append(processed_data)
            
            # サーバに負荷をかけないように、リクエストの間に適切な間隔を空ける
            time.sleep(2)

    return all_data

def build_prediction_model(data):
    # データをDataFrameに変換
    df = pd.DataFrame(data)

    # 特徴量として使用するカラムを選択
    features = ['horse_age', 'horse_weight', 'jockey_win_rate', 'horse_top3_rate']
    X = df[features]

    # ターゲット変数（予測したい変数）としてオッズを選択
    y = df['odds']

    # データを訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 線形回帰モデルをインスタンス化
    model = LinearRegression()

    # モデルを訓練データで訓練
    model.fit(X_train, y_train)

    # テストデータでの予測を実行
    y_pred = model.predict(X_test)

    # 予測の精度を評価（平均二乗誤差）
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    return model

def advanced_prediction_model(data):
    df = pd.DataFrame(data)

    # 特徴量エンジニアリング
    df['jockey_win_rate'] = df['jockey_win_rate'].str.rstrip('%').astype('float') / 100.0
    df['horse_top3_rate'] = df['horse_top3_rate'].str.rstrip('%').astype('float') / 100.0

    # 特徴量として使用するカラムを選択
    features = ['horse_age', 'horse_weight', 'jockey_win_rate', 'horse_top3_rate']
    X = df[features]
    y = df['odds']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ランダムフォレスト回帰モデルを使用
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    return model

if __name__ == "__main__":
    data = scrape_netkeiba()
    analyze_data(data)
    extended_analysis(data)
    model = advanced_prediction_model(data)