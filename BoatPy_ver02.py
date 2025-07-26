import streamlit as st
import random
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup

# --- セッション状態の初期化 ---
if "dice_results" not in st.session_state:
    st.session_state.dice_results = None

if "race_data" not in st.session_state:
    st.session_state.race_data = None

# --- タイトル部分 ---
col1, col2, col3 = st.columns([2, 4, 2])

with col1:
    st.image("dog_boat01.png", width=250)

with col2:
    st.markdown("""
        <div style='text-align: center;'>
            <h1 style='font-family: "Yu Mincho", "Hiragino Mincho Pro", "MS PMincho", serif;
                       font-style: italic;
                       color: #C62828;
                       font-size: 32px;
                       margin: 0;'>
            BOATRACE_AI_ver02
            </h1>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.image("dog_boat02.png", width=250)

# --- 🎲 本日のラッキーナンバー ---
if st.button("🎲本日のラッキーナンバー ！"):
    with st.spinner("🥁 ドコドコドコドコ..."):
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
    st.subheader("🎲 サイコロ占い")

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
st.subheader("🏁 予想したいレースを選択")
selected_place = st.selectbox("競艇場を選択", list(boat_places.keys()))
selected_race = st.selectbox("レース番号を選択", [f"{i}R" for i in range(1, 13)])
selected_date = st.date_input("日付を選択", value=datetime.date.today())

# --- 出走表URL生成 ---
jcd = boat_places[selected_place]
rno = selected_race.replace("R", "")
hd = selected_date.strftime("%Y%m%d")
detail_url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd}&hd={hd}"

# --- 出走表URLを表示 ---
if st.button("出走表URLを表示"):
    st.markdown(f"🔗 [出走表URLはこちら]({detail_url})", unsafe_allow_html=True)

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
            grade = reg_info.find("span").text.strip() if reg_info.find("span") else ""

            winrate = tds[4].text.strip().split("\n")[0]
            motor_info = tds[6].text.strip().split("\n")
            motor_rate = motor_info[1] if len(motor_info) > 1 else ""

            racers.append({
                "登録番号": reg_num,
                "級別": grade,
                "全国勝率": winrate,
                "モーター2連率": motor_rate
            })
        except:
            continue

    return pd.DataFrame(racers) if racers else None

# --- URLから選手情報取得 ---
st.subheader("🔗 出走表URLから選手情報を取得")
st.caption("級別ランクの変換: A1=1, A2=0.7, B1=0.4, B2=0.1")  # ← 表示追加

input_url = st.text_input("出走表のURLを入力してください", placeholder=detail_url)

if st.button("👀 情報を取得"):
    if not input_url:
        st.warning("⚠ URLを入力してください。")
    else:
        try:
            df = get_race_data_from_url(input_url)

            if df is not None:
                grade_mapping = {"A1": 1.0, "A2": 0.7, "B1": 0.4, "B2": 0.1}
                df["級別ランク"] = df["級別"].map(grade_mapping).fillna(0)
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
            self.fc3 = nn.Linear(64, 32)
            self.output = nn.Linear(32, num_classes)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
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
    df["Rank"] = df["級別ランク"]
    df["Zenkoku_power"] = pd.to_numeric(df["全国勝率"], errors='coerce').fillna(0)
    df["motor_win_rate"] = df["モーター2連率"].str.replace('%', '').astype(float)
    df["course"] = -df["course"] * 2  # 小さい方が強いので反転

    # 特徴量と変換
    feature_cols = ["motor_win_rate", "course", "Rank", "Zenkoku_power"]
    X = scaler.transform(df[feature_cols])
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X_tensor)
        predicted_ranks = outputs.argmax(dim=1).numpy()  # 出力が6クラス分類（0=1位, ..., 5=6位）

    df["予測着順スコア"] = predicted_ranks
    df_sorted = df.sort_values(by="予測着順スコア").reset_index(drop=True)

    st.subheader("🎯 モデルによる予測順位（上位3名）")
    st.dataframe(df_sorted[["登録番号", "級別", "全国勝率", "モーター2連率", "予測着順スコア"]].head(3))

#    top3 = df_sorted.iloc[:3]["登録番号"].tolist()
#    combinations = list(itertools.permutations(top3, 3))

#    st.subheader("🎰 3連単予測（6通り）")
#    for comb in combinations:
#        st.write(" → ".join(comb))

#    top3 = (df_sorted.iloc[:3]["course"] * -0.5).astype(int).tolist()
#    combinations = list(itertools.permutations(top3, 3))

#    st.subheader("🎰 3連単予測（コース番号ベース）")
#    for comb in combinations:
#        st.write(" → ".join(map(str, comb)))
# コース番号（1〜6）に戻す
    top3 = (df_sorted.iloc[:3]["course"] * -0.5).astype(int).tolist()

    # 3連単の全組み合わせを作成（順列）
    combinations = list(itertools.permutations(top3, 3))

    # DataFrameに変換（列名を指定）
    df_combinations = pd.DataFrame(combinations, columns=["1着", "2着", "3着"])

    # 表形式で表示
    st.subheader("🎰 3連単予測（コース番号ベース）")
    st.dataframe(df_combinations)

else:
    st.warning("⚠ 先に出走選手データを取得してください。")