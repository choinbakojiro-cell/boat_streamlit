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
col1, col2= st.columns([1, 5])
with col1:
    st.image("boat_icon.png", width=120)
with col2:
    st.markdown("""
        <h1 style='font-family: "Yu Mincho", "Hiragino Mincho Pro", "MS PMincho", serif;
                   font-style: italic;
                   color: #C62828;
                   margin-top: 10px;'>
        BOATRACE_AI
        </h1>
    """, unsafe_allow_html=True)

# --- 🎲 サイコロシステム ---
if st.button("🎲 サイコロを振る！"):
    with st.spinner("🥁 ドラムロール中..."):
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

# --- 結果表示（セッションから） ---
if st.session_state.dice_results is not None:
    st.subheader("🎲 サイコロシミュレータ")
    styled_df = st.session_state.dice_results.style.set_table_styles([
        {'selector': 'thead th', 'props': [('background-color', '#eeeeee'), ('width', '100px')]},
        {'selector': 'tbody td', 'props': [('width', '100px')]},
        {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', '#ffffff')]},
        {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#b3e5fc')]}
    ])
    st.write(styled_df.hide(axis='index'), unsafe_allow_html=True)

# --- 📋 ボートレース出走データ取得 ---
# --- 📋 ボートレース出走データ取得（完全版） ---
st.subheader("📋 ボートレース出走データ取得")
race_url = st.text_input("レースURLを入力してください")

if st.button("取得する") and race_url:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "ja-JP",
            "Referer": "https://www.boatrace.jp/"
        }
        res = requests.get(race_url, headers=headers)
        if res.status_code != 200:
            st.error(f"❌ ステータスコード {res.status_code} によりアクセス失敗")
        else:
            soup = BeautifulSoup(res.content, "html.parser")
            tbodies = soup.find_all("tbody", class_="is-fs12")
            data = []
            for tbody in tbodies:
                tds = tbody.find_all("td")
                if len(tds) >= 10:
                    try:
                        # コース
                        course = tds[0].get_text(strip=True)

                        # 登録番号と級別
                        reg_div = tds[2].find("div", class_="is-fs11")
                        if reg_div:
                            reg_num = reg_div.get_text(strip=True).split("/")[0]
                            grade_span = reg_div.find("span", class_="is-fColor1 ")
                            grade = grade_span.get_text(strip=True) if grade_span else ""
                        else:
                            reg_num = ""
                            grade = ""

                        # 全国勝率
                        nation_td = tds[4]
                        nation_text = nation_td.get_text(separator="\n", strip=True).split("\n")
                        nation_win_rate = nation_text[0] if nation_text else ""

                        # モーター2連率
                        motor_td = tds[6]
                        motor_text = motor_td.get_text(separator="\n", strip=True).split("\n")
                        motor_rate = motor_text[1] if len(motor_text) > 1 else ""

                        # 性別（性別の位置が不明な場合、ここで追加対応可）
                        sex = "?"  # 必要なら後で追加

                        data.append([reg_num, course, motor_rate, grade, nation_win_rate, sex])
                    except Exception as e:
                        continue

            if data:
                df_race = pd.DataFrame(data, columns=[
                    "登録番号", "コース", "モーター2連率", "級別", "全国勝率", "性別"
                ])
                st.session_state.race_data = df_race
                st.success("✅ データ取得成功")
                st.subheader("🚤 取得した出走データ")
                st.dataframe(df_race)
            else:
                st.warning("⚠ データが見つかりませんでした。URLやページ構造を確認してください。")
    except Exception as e:
        st.error(f"❌ エラー発生: {e}")

# --- レース結果表示（セッションから） ---
if st.session_state.race_data is not None:
    st.subheader("🚤 出走選手情報")

    try:
        # CSVの読み込み（3列：登録番号・今季能力指数・性別）
        ability_df = pd.read_csv("今季能力指数.csv", dtype=str)
        ability_df = ability_df.rename(columns={
            ability_df.columns[0]: "登録番号",
            ability_df.columns[1]: "今季能力指数",
            ability_df.columns[2]: "性別"
        })

        # 整形（登録番号：全角→半角、空白除去、ゼロ埋め）
        ability_df["登録番号"] = ability_df["登録番号"].str.strip().str.zfill(6)\
            .str.replace("　", "").str.replace(" ", "")\
            .str.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
        df_race = st.session_state.race_data.copy()
        df_race["登録番号"] = df_race["登録番号"].astype(str).str.strip().str.zfill(6)\
            .str.replace("　", "").str.replace(" ", "")\
            .str.translate(str.maketrans("０１２３４５６７８９", "0123456789"))

        # マージ（性別も含める）
        merged_df = pd.merge(df_race, ability_df, on="登録番号", how="left")

        # 数値型へ変換
        merged_df["モーター2連率"] = pd.to_numeric(merged_df["モーター2連率"], errors="coerce")
        merged_df["今季能力指数"] = pd.to_numeric(merged_df["今季能力指数"], errors="coerce")

        # 相対値計算
        motor_avg = merged_df["モーター2連率"].mean()
        ability_avg = merged_df["今季能力指数"].mean()
        merged_df["相対モーター2連率"] = merged_df["モーター2連率"] - motor_avg
        merged_df["相対今季能力指数"] = merged_df["今季能力指数"] - ability_avg

        # ✅ 性別指数を追加（1なら1.0、2なら0.8）
        merged_df["性別指数"] = merged_df["性別"].map({"1": 1.0, "2": 0.9})

        # 更新
        st.session_state.race_data = merged_df

        # 表示
        st.dataframe(merged_df)

        # 情報表示
        st.caption(f"📊 モーター2連率の平均: {motor_avg:.2f}")
        st.caption(f"📊 今季能力指数の平均: {ability_avg:.2f}")

    except Exception as e:
        st.warning(f"📄 能力指数の読み込みまたは結合でエラーが発生しました: {e}")

import torch
import torch.nn as nn
import pandas as pd
import itertools
import streamlit as st

# --- MLPモデル（ListMLE用） ---
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 32)  # 特徴量数を5に
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)  # shape: (batch,)

# --- 🔮 モデルで予測して組み合わせ表示ボタン ---
if st.button("🔮 モデルで予測して組み合わせ表示"):
    try:
        df = st.session_state.race_data.copy()

        # 列名変換・追加
        df["course"] = pd.to_numeric(df["コース"], errors="coerce")
        df["motor_win_rate"] = df["相対モーター2連率"]
        df["sex_power"] = df["性別指数"]
        df["Rank"] = df["全国勝率ランク"]
        df["Zenkoku_power"] = df["全国勝率"]

        # 特徴量
        features = ["motor_win_rate", "course",  "sex_power", "Rank", "Zenkoku_power"]

        X = df[features].astype(float).values

        # モデル読み込み
        model = MLP()
        model.load_state_dict(torch.load("classification_model.pth", map_location=torch.device("cpu")))
        model.eval()

        # スコア予測
        with torch.no_grad():
            input_tensor = torch.tensor(X, dtype=torch.float32)
            scores = model(input_tensor).numpy()
            df["スコア"] = scores

        # 上位3人を抽出
        top3 = df.sort_values("スコア", ascending=False).head(3)

        # 順列6通り
        combos = list(itertools.permutations(top3["コース"], 3))[:6]
        result_df = pd.DataFrame({
            "パターン": [f"パターン{i+1}" for i in range(6)],
            "組み合わせ": [f"{a}-{b}-{c}" for a, b, c in combos]
        })

        st.subheader("🏁 予測上位3選手の組み合わせ（6通り）")
        st.dataframe(result_df)

    except Exception as e:
        st.error(f"❌ モデル予測でエラーが発生しました: {e}")

