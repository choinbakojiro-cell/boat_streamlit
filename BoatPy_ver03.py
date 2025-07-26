import streamlit as st
import random
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup
import base64
##__________________________________________________________________##

# 演出部分
# ✅ スプラッシュ一度だけ表示するためのフラグ
if "splash_done" not in st.session_state:
    st.session_state.splash_done = False

# ✅ スプラッシュがまだ表示されていなければ表示
# ランダムに画像ファイル名を選ぶ
splash_images = ["top_anime01.png", "top_anime02.png", "top_anime03.png"]
selected_image = random.choice(splash_images)

# 選ばれた画像をbase64エンコード
with open(selected_image, "rb") as f:
    data_url = base64.b64encode(f.read()).decode()

    # splash_html 変数に HTML+CSS を格納
    splash_html = f"""
    <style>
    #splash {{
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-color: white;
        z-index: 9999;
        display: flex;
        justify-content: center;
        align-items: center;
        animation: fadeout 6s ease-in-out forwards;
    }}
    @keyframes fadeout {{
        0% {{ opacity: 0; }}
        10% {{ opacity: 1; }}
        90% {{ opacity: 1; }}
        100% {{ opacity: 0; visibility: hidden; }}
    }}
    #splash img {{
        width: 60%;
        max-width: 600px;
        animation: float 10s ease-in-out infinite;
        border-radius: 10px; /* ← 角を丸くする */
    }}
    @keyframes float {{
        0%, 100% {{ transform: translateY(0); }}
        80% {{ transform: translateY(-40px); }}
    }}

    </style>

    <div id="splash">
        <img src="data:image/png;base64,{data_url}" alt="Splash">
    </div>
    """

    # 🔽 実際に表示
    st.markdown(splash_html, unsafe_allow_html=True)

    # スプラッシュが消える時間分スリープ
    time.sleep(2)

    # 2回目以降非表示にする
    st.session_state.splash_done = True

    # チラ見防止のためにアプリ描画を一時停止
    #st.stop()
##____________________________________________________________#
# --- セッション状態の初期化 ---
if "weather_info" not in st.session_state:
    st.session_state.weather_info = None

if "dice_results" not in st.session_state:
    st.session_state.dice_results = None

if "race_data" not in st.session_state:
    st.session_state.race_data = None

# --- タイトル部分 ---
col1, col2, col3 = st.columns([2, 4, 2])

with col1:
    st.image("dog_boat01.png", width=250)
with col2:
    st.image("BoatRaceAI.png", width=800)
with col3:
    st.image("dog_boat02.png", width=250)

# --- 🎲 本日のラッキーナンバー ---
if st.button("🎲本日のラッキーナンバー ！"):
    with st.spinner("🥁 ドコドコドコドコドコドコドコドコ..."):
        time.sleep(2)
    triplets = []
    used = set()
    while len(triplets) < 6:
        comb = sorted(random.sample(range(1, 7), 3))
        if tuple(comb) not in used:
            triplets.append(comb)
            used.add(tuple(comb))
    df = pd.DataFrame({"組み合わせ": [f"{x[0]}-{x[1]}-{x[2]}" for x in triplets]})
    st.session_state.dice_results = df  # 結果を保存

# --- 🎲 結果表示（セッションから、横並びに、お言葉表示） ---
if st.session_state.dice_results is not None:
    st.markdown(
        """
        <h3 style='font-size: 20px; font-weight: bold; color: #333; border-bottom:2px solid #0288d1; padding-bottom: 2px; margin-bottom: 16px; margin-top: 0;' >
            🎲 気に入ったお言葉の組み合わせがラッキーを呼ぶかも？
        </h3>
        """,
        unsafe_allow_html=True
    )

    # お言葉.csv 読み込み（A列:組み合わせ文字列、B列:言葉）
    words_df = pd.read_csv("お言葉.csv", dtype=str)
    words_dict = dict(zip(words_df.iloc[:, 0], words_df.iloc[:, 1]))  # 辞書化

    cols = st.columns(6)  # 6列に分割
    for i, col in enumerate(cols):
        combination = st.session_state.dice_results.iloc[i, 0]  # 組み合わせ文字列
        message = words_dict.get(combination, "メッセージなし")  # 完全一致で取得

        # 組み合わせとお言葉を表示
        col.markdown(
            f"""
            <div style='background-color: #b3e5fc; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold;'>
                {combination}
            </div>
            <div style='margin-top: 10px; text-align: center; font-size: 14px;'>
                {message}
            </div>
            """,
            unsafe_allow_html=True
        )


##ここから
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import datetime

# --- 競艇場リスト ---
boat_places = {
    "桐生": "01", "戸田": "02", "江戸川": "03", "平和島": "04", "多摩川": "05",
    "浜名湖": "06", "蒲郡": "07", "常滑": "08", "津": "09", "三国": "10",
    "びわこ": "11", "住之江": "12", "尼崎": "13", "鳴門": "14", "丸亀": "15",
    "児島": "16", "宮島": "17", "徳山": "18", "下関": "19", "若松": "20",
    "芦屋": "21", "福岡": "22", "唐津": "23", "大村": "24"
}

# --- ユーザー選択 ---
# --- セクションタイトル ---
st.markdown(
    """
    <div style='font-size: 20px; font-weight: bold; color: #333;
                border-bottom:2px solid #0288d1; padding-bottom: 2px;
                margin-bottom: 16px; margin-top: 0;'>
        🏁 予想したいレースを選択
    </div>
    """,
    unsafe_allow_html=True
)

# --- 横並びの入力欄 streamlitは縦になりがちなので---
col1, col2, col3 = st.columns([1.5, 1, 1.5])  # 幅調整

with col1:
    selected_place = st.selectbox("競艇場", list(boat_places.keys()))
with col2:
    selected_race = st.selectbox("レース", [f"{i}R" for i in range(1, 13)])
with col3:
    selected_date = st.date_input("日付", value=datetime.date.today())

# --- 出走表URL生成 rno=何レース目か、jcd=レース場ナンバー、hd=日付---
# --- 天気取得 ---
lat_lon_map = {
    "01": (36.38898, 139.29339), "02": (35.80385, 139.66036), "03": (35.79592, 139.82672),
    "04": (35.72445, 139.75333), "05": (35.68074, 139.46954), "06": (34.69304, 137.59065),
    "07": (34.82570, 137.20900), "08": (34.97417, 136.82735), "09": (34.66085, 136.52399),
    "10": (36.22338, 136.18149), "11": (35.12239, 135.84877), "12": (34.60851, 135.46222),
    "13": (34.71315, 135.35947), "14": (34.16435, 134.63042), "15": (34.39480, 133.77905),
    "16": (34.48734, 133.80762), "17": (34.36146, 132.27628), "18": (34.01161, 131.83329),
    "19": (34.01322, 130.99754), "20": (33.94340, 130.76600), "21": (33.87134, 130.64759),
    "22": (33.57809, 130.38448), "23": (33.42890, 129.98980), "24": (32.90029, 129.93981),
}
jcd = boat_places[selected_place]
rno = selected_race.replace("R", "")
hd = selected_date.strftime("%Y%m%d")
detail_url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd}&hd={hd}"

## --- 出走表URLを表示 ---
if st.button("出走表URLを表示"):
    st.markdown(f"🔗 [出走表URLはこちら]({detail_url})", unsafe_allow_html=True)

#-----------------------------------------------------------------------------------------------------------------------
# --- 天気取得用：緯度経度マップ ---


    lat, lon = lat_lon_map[jcd]
    api_key = "4911c2965cd7773a50320aa3ddee3fe0"
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&lang=ja&units=metric"

    try:
        response = requests.get(weather_url)
        data = response.json()

        if response.status_code == 200:
            weather = data["weather"][0]["description"]
            wind_speed = data["wind"]["speed"]
            wind_deg = data["wind"].get("deg", "不明")

            st.session_state.weather_info = {
                "天候": data["weather"][0]["description"],
                "風速": data["wind"]["speed"],
                "風向": data["wind"].get("deg", "不明"),
                "気温": data["main"]["temp"]
            }
        else:
            st.session_state.weather_info = {"error": f"ステータスコード: {response.status_code}"}

    except Exception as e:
        st.session_state.weather_info = {"error": str(e)}
# --- 天気表示 ---
st.markdown(
    """
    <div style='font-size: 20px; font-weight: bold; color: #333;
                border-bottom:2px solid #0288d1; padding-bottom: 2px;
                margin-bottom: 16px; margin-top: 0;'>
        🌤 現地の情報（風向は0が北、90が東、180が南、となります）
    </div>
    """,
    unsafe_allow_html=True
)
if st.session_state.weather_info is not None and "error" not in st.session_state.weather_info:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"☀ **天候**: {st.session_state.weather_info['天候']}")
    with col2:
        st.markdown(f"💨 **風速**: {st.session_state.weather_info['風速']} m/s")
    with col3:
        st.markdown(f"🧭 **風向**: {st.session_state.weather_info['風向']}°")
    with col4:
        st.markdown(f"🌡 **気温**: {st.session_state.weather_info['気温']} ℃")

elif st.session_state.weather_info is not None and "error" in st.session_state.weather_info:
    st.error(f"天気情報の取得に失敗しました: {st.session_state.weather_info['error']}")

#=======================================================================================================================
# --- レース場のレースの特徴。水面特性を読み込む） ---
if st.session_state.dice_results is not None:
    st.markdown(
        """
        <h3 style='font-size: 20px; font-weight: bold; color: #333; border-bottom:2px solid #0288d1; padding-bottom: 2px; margin-bottom: 16px; margin-top: 0;' >
            🗺 レースの特徴。水面特性
        </h3>
        """,
        unsafe_allow_html=True
    )

    # お言葉.csv 読み込み（A列:組み合わせ文字列、B列:言葉）
    words_df = pd.read_csv("レース場.csv", dtype=str)
    words_dict = dict(zip(words_df.iloc[:, 0], words_df.iloc[:, 1]))  # 辞書化

# --- rcd（2桁コード）を取得 ---
rcd = boat_places[selected_place]  # 例: "18"

# --- レース場データを読み込み ---
place_df = pd.read_csv("レース場.csv", dtype=str)

# --- 対象行を抽出（A列と一致） ---
matched_row = place_df[place_df.iloc[:, 0] == rcd]

# --- 表示 ---
if not matched_row.empty:
    col_B = matched_row.iloc[0, 1]  # B列（2列目）
    col_C = matched_row.iloc[0, 2]  # C列（3列目）

    st.markdown(
        f"""
        <div style='font-size: 14px; line-height: 1.5;'>
            {col_B}
        </div>
        <div style='font-size: 14px; line-height: 1.5;'>
            {col_C}
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("⚠ レース場の情報が見つかりませんでした。")

#=======================================================================================================================
# --- 緯度経度に基づく Google マップ表示 ---
if (
    "weather_info" in st.session_state
    and st.session_state.weather_info is not None
    and "error" not in st.session_state.weather_info
    and jcd in lat_lon_map
):
    lat, lon = lat_lon_map[jcd]

    map_df = pd.DataFrame({
        'lat': [lat],
        'lon': [lon]
    })

    st.markdown(
        """
        <div style='font-size: 20px; font-weight: bold; color: #333;
                    border-bottom:2px solid #0288d1; padding-bottom: 2px;
                    margin-bottom: 16px; margin-top: 0;' >
            🗺 現地マップ（気象地点）
        </div>
        """,
        unsafe_allow_html=True
    )

    st.map(map_df)


#-----------------------------------------------------------------------------------------------------------------------

# --- 選手情報取得関数 ---
def get_race_data_from_url(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    soup = BeautifulSoup(res.content, "html.parser")

    tbodies = soup.find_all("tbody", class_="is-fs12")
    racers = []

    for tbody in tbodies:
        tr = tbody.find("tr")
        if not tr:
            continue
        tds = tr.find_all("td")
        if len(tds) < 7:
            continue

        try:
            reg_info = tds[2].find("div", class_="is-fs11")
            reg_num = reg_info.text.split("/")[0].strip()
            winrate = tds[4].text.strip().split("\n")[0]
            motor_info = tds[6].text.strip().split("\n")
            motor_rate = motor_info[1] if len(motor_info) > 1 else ""
            boat_info = tds[7].text.strip().split("\n")
            boat_power = boat_info[1] if len(boat_info) > 1 else ""

            racers.append({
                "登録番号": reg_num,
                "全国勝率": winrate,
                "モーター2連率": motor_rate,
                "ボート2連率": boat_power,
            })
        except:
            continue

    return pd.DataFrame(racers) if racers else None

# --- URLから選手情報取得 ---
st.markdown(
    """
    <div style='font-size: 20px; font-weight: bold; color: #333; padding-bottom: 10px; border-bottom:2px solid #0288d1; padding-bottom: 2px; margin-bottom: 16px; margin-top: 0;'>
        🔗 出走表URLから選手情報を取得
    </div>
    """,
    unsafe_allow_html=True
)
#st.caption("ランクの変換: A1=1, A2=0.7, B1=0.4, B2=0.1")  # ← 表示追加

input_url = st.text_input("出走表のURLを入力してください", placeholder=detail_url)

if st.button("👀 情報を取得"):
    with st.spinner("🥁 ドコドコドコドコドコドコドコドコ..."):
        time.sleep(2)
    if not input_url:
        st.warning("⚠ URLを入力してください。")
    else:
        try:
            df = get_race_data_from_url(input_url)

            if df is not None:
                st.session_state.df = df  # 🔶 ここを追加：セッションに保存
                st.success("✅ 出走選手データ取得成功")
                st.dataframe(df)
            else:
                st.warning("⚠ 選手データが見つかりませんでした。URLやページ構造を確認してください。")
        except Exception as e:
            st.error(f"❌ エラー発生: {e}")

#ここから予測########################################################################################
import itertools
import torch
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st
import torch.nn as nn

# 🔐 安全なオブジェクトを明示的に許可
torch.serialization.add_safe_globals([
    StandardScaler,
    np.dtype,
    np.core.multiarray._reconstruct,
    np.core.multiarray
])

# --- モデルとスケーラーの読み込み関数 ---
@st.cache_resource
def load_model_and_scaler(model_path):
    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    model_state = checkpoint['model_state_dict']
    scaler = checkpoint['scaler']

    class ClassificationModel(torch.nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 48)
            self.fc4 = nn.Linear(48, 32)
            self.fc5 = nn.Linear(32, 24)
            self.fc6 = nn.Linear(24, 16)
            self.fc7 = nn.Linear(16, 8)
            self.output = nn.Linear(8, num_classes)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = torch.relu(self.fc4(x))
            x = torch.relu(self.fc5(x))
            x = torch.relu(self.fc6(x))
            x = torch.relu(self.fc7(x))
            x = self.output(x)
            return x

    model = ClassificationModel(input_dim=4, num_classes=6)
    model.load_state_dict(model_state)
    model.eval()

    return model, scaler

# --- パス指定とモデル読み込み ---
MODEL_PATH = r"E:\04_学習環境\classification_model.pth"
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ モデルファイルが存在しません: {MODEL_PATH}")
    st.stop()

model, scaler = load_model_and_scaler(MODEL_PATH)

# --- 特徴量整形・予測処理 ---
if 'df' in st.session_state:
    df = st.session_state.df.copy()

    # コース割り当て（1〜6）
    df["course"] = [i + 1 for i in range(len(df))]

    # 特徴量整備
    df["motor_win_rate"] = df["モーター2連率"].str.replace('%', '').astype(float)
    df["course"] = df["course"]   # 小さい方が強いので反転（やめる）
    df["Zenkoku_power"] = pd.to_numeric(df["全国勝率"], errors='coerce').fillna(0)
    df["Boat_power"] = df["ボート2連率"].str.replace('%', '').astype(float)

    # 特徴量と変換
    feature_cols = ["motor_win_rate", "course", "Zenkoku_power", "Boat_power"]
    X = scaler.transform(df[feature_cols])
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X_tensor)
        predicted_ranks = outputs.argmax(dim=1).numpy()  # 出力が6クラス分類（0=1位, ..., 5=6位）

    df["予測着順スコア"] = predicted_ranks
    df_sorted = df.sort_values(by="予測着順スコア").reset_index(drop=True)

    st.markdown(
    """
    <div style='font-size: 20px; font-weight: bold; color: #333; font-family: "Meiryo", sans-serif; border-bottom:2px solid #0288d1; padding-bottom: 2px; margin-bottom: 16px; margin-top: 0;'>
        🎯 モデルによる予測順位（上位3名）
    </div>
    """,
    unsafe_allow_html=True
    )
    st.dataframe(df_sorted[["登録番号", "モーター2連率","全国勝率", "ボート2連率", "予測着順スコア"]].head(3))

    top3 = df_sorted.iloc[:3]["course"].astype(int).tolist()

    # 3連単の全組み合わせを作成（順列）
    combinations = list(itertools.permutations(top3, 3))

    # DataFrameに変換（列名を指定）
    df_combinations = pd.DataFrame(combinations, columns=["1着", "2着", "3着"])

    # 表形式で表示
    st.markdown(
    """
    <div style='font-size: 20px; font-weight: bold; color: #333; font-family: "Meiryo", sans-serif; border-bottom:2px solid #0288d1; padding-bottom: 2px; margin-bottom: 16px; margin-top: 0;'>
        🎰 3連単予測（コース番号ベース）
    </div>
    """,
    unsafe_allow_html=True
    )
    st.dataframe(df_combinations)

else:
    st.warning("⚠ 先に出走選手データを取得してください。")