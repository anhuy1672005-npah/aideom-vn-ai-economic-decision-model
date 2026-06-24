import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO

# ============================================================
# BÀI 3 - TÍNH CHỈ SỐ ƯU TIÊN NGÀNH PRIORITY_i
# Webapp Streamlit
# Yêu cầu chính: pandas, numpy, matplotlib, openpyxl
# ============================================================

try:
    from utils.style import load_css, hero, card
except Exception:
    def load_css():
        st.markdown(
            """
            <style>
            .stApp {background: linear-gradient(180deg, #0b1020 0%, #0f172a 100%);}
            .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px;}
            h1, h2, h3 {color: #ffffff !important;}
            .card {background: rgba(18, 26, 47, 0.95); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 20px 22px; box-shadow: 0 8px 24px rgba(0,0,0,0.25); margin-bottom: 18px;}
            .card-title {font-size: 1.1rem; font-weight: 750; color: #ffffff; margin-bottom: 8px;}
            .card-text {color: #cbd5e1; font-size: 0.95rem; line-height: 1.55;}
            .badge {display: inline-block; padding: 6px 12px; margin-right: 8px; margin-bottom: 8px; border-radius: 999px; font-size: 0.78rem; font-weight: 700; color: white; background: linear-gradient(90deg, #ff3b7f, #7c3aed);}
            </style>
            """,
            unsafe_allow_html=True
        )

    def hero(title, subtitle="", badges=None):
        if badges is None:
            badges = []
        badge_html = "".join([f'<span class="badge">{b}</span>' for b in badges])
        st.markdown(
            f"""
            <div class="card">
                <div>{badge_html}</div>
                <h1>{title}</h1>
                <div class="card-text">{subtitle}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    def card(title, text):
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">{title}</div>
                <div class="card-text">{text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


st.set_page_config(
    page_title="Bài 3 - Priority ngành",
    layout="wide"
)
load_css()

hero(
    title="🏭 Bài 3 — Tính chỉ số ưu tiên ngành Priorityᵢ",
    subtitle="Chuẩn hóa min-max dữ liệu 10 ngành Việt Nam, tính chỉ số ưu tiên, phân tích độ nhạy theo trọng số AI Readiness và so sánh hai định hướng chính sách.",
    badges=["Cấp độ dễ", "MCDM cơ bản", "pandas", "numpy", "matplotlib"]
)

st.markdown("""
Bài này xây dựng chỉ số định lượng để xếp hạng 10 ngành kinh tế Việt Nam theo mức độ nên ưu tiên chuyển đổi số và ứng dụng AI.
Các bước chính gồm: đọc dữ liệu, chuẩn hóa min-max, tính điểm Priority, phân tích độ nhạy trọng số và diễn giải chính sách.
""")

# ============================================================
# 1. HÀM TIỆN ÍCH
# ============================================================

DATA_PATH = Path("data") / "vietnam_sectors_2024.csv"
OUTPUT_DIR = Path("outputs")

DISPLAY_NAMES = {
    "growth_norm": "Growth_norm",
    "productivity_norm": "Productivity_norm",
    "spillover_norm": "Spillover_norm",
    "export_norm": "Export_norm",
    "employment_norm": "Employment_norm",
    "ai_readiness_norm": "AIReadiness_norm",
    "risk_safe_norm": "Risk_inverted_norm"
}

WEIGHT_LABELS = [
    "Growth",
    "Productivity",
    "Spillover",
    "Export",
    "Employment",
    "AIReadiness",
    "Risk_inverted"
]


def make_download_excel(sheets: dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df_sheet in sheets.items():
            safe_name = sheet_name[:31]
            df_sheet.to_excel(writer, index=False, sheet_name=safe_name)
    return output.getvalue()


def save_outputs(output_dir: Path, files: dict):
    output_dir.mkdir(exist_ok=True)
    for filename, obj in files.items():
        path = output_dir / filename
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(path, index=False, encoding="utf-8-sig")
        elif isinstance(obj, str):
            with open(path, "w", encoding="utf-8") as f:
                f.write(obj)


def norm_good(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    denom = series.max() - series.min()
    if abs(denom) < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / denom


def norm_bad_as_safe(series: pd.Series) -> pd.Series:
    """
    Dùng cho chỉ số xấu như automation risk.
    Giá trị càng thấp càng tốt, nên sau chuẩn hóa đảo chiều:
    Risk_inverted = (max - x) / (max - min)
    Kết quả càng cao nghĩa là rủi ro càng thấp/an toàn hơn.
    """
    series = series.astype(float)
    denom = series.max() - series.min()
    if abs(denom) < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series.max() - series) / denom


def normalize_weights(raw_weights):
    w = np.array(raw_weights, dtype=float)
    total = w.sum()
    if total <= 0:
        raise ValueError("Tổng trọng số phải lớn hơn 0.")
    return w / total


def detect_productivity_column(df: pd.DataFrame):
    candidates = [
        "productivity_million_VND_per_worker",
        "productivity_million_VND_worker",
        "labor_productivity_million_VND",
        "productivity_million_VND_per_labor",
        "productivity"
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def add_productivity_from_assignment(df: pd.DataFrame) -> pd.DataFrame:
    """
    File vietnam_sectors_2024.csv đi kèm thường không có cột năng suất đúng như bảng 3.3.
    Vì vậy hàm này bổ sung cột năng suất theo số liệu trong đề bài nếu chưa có.
    """
    df = df.copy()
    prod_col = detect_productivity_column(df)
    if prod_col is not None:
        df["productivity_million_VND_per_worker"] = df[prod_col].astype(float)
        source = f"Dùng cột có sẵn trong CSV: {prod_col}"
        return df, source

    productivity_by_id = {
        1: 103.4,
        2: 241.2,
        3: 168.8,
        4: 1290.5,
        5: 145.3,
        6: 1072.4,
        7: 321.4,
        8: 713.8,
        9: 205.7,
        10: 437.1,
    }
    if "sector_id" in df.columns and set(df["sector_id"].astype(int)).issubset(set(productivity_by_id.keys())):
        df["productivity_million_VND_per_worker"] = df["sector_id"].astype(int).map(productivity_by_id)
        source = "CSV chưa có cột năng suất; đã bổ sung theo bảng 3.3 trong đề bài bằng sector_id."
    elif "sector_name_vi" in df.columns:
        productivity_by_name = {
            "Nông-Lâm-Thủy sản": 103.4,
            "Công nghiệp chế biến chế tạo": 241.2,
            "Xây dựng": 168.8,
            "Khai khoáng": 1290.5,
            "Bán buôn-bán lẻ": 145.3,
            "Tài chính-Ngân hàng-Bảo hiểm": 1072.4,
            "Logistics-Vận tải-Kho bãi": 321.4,
            "Thông tin-Truyền thông-CNTT": 713.8,
            "Giáo dục-Đào tạo": 205.7,
            "Y tế-Chăm sóc sức khỏe": 437.1,
        }
        def lookup_productivity(name: str) -> float:
            name = str(name)
            for key, value in productivity_by_name.items():
                if key.lower() in name.lower() or name.lower() in key.lower():
                    return value
            raise KeyError(name)
        try:
            df["productivity_million_VND_per_worker"] = df["sector_name_vi"].apply(lookup_productivity)
            source = "CSV chưa có cột năng suất; đã bổ sung theo bảng 3.3 trong đề bài bằng tên ngành."
        except Exception as exc:
            raise ValueError(
                "CSV thiếu cột năng suất và không khớp sector_id/tên ngành trong bảng 3.3; "
                "không dùng proxy để tránh sai lệch số liệu nộp bài."
            ) from exc
    else:
        raise ValueError(
            "CSV thiếu cột năng suất, sector_id và sector_name_vi; không thể bổ sung dữ liệu theo đề bài."
        )
    return df, source


def build_normalized_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame()
    X["sector_name_vi"] = df["sector_name_vi"]
    X["growth_norm"] = norm_good(df["growth_rate_2024_pct"])
    X["productivity_norm"] = norm_good(df["productivity_million_VND_per_worker"])
    X["spillover_norm"] = norm_good(df["spillover_coef_0_1"])
    X["export_norm"] = norm_good(df["export_billion_USD"])
    X["employment_norm"] = norm_good(df["labor_million"])
    X["ai_readiness_norm"] = norm_good(df["ai_readiness_0_100"])
    X["risk_safe_norm"] = norm_bad_as_safe(df["automation_risk_pct"])
    return X


def compute_priority(norm_df: pd.DataFrame, weights) -> np.ndarray:
    cols = [
        "growth_norm",
        "productivity_norm",
        "spillover_norm",
        "export_norm",
        "employment_norm",
        "ai_readiness_norm",
        "risk_safe_norm"
    ]
    w = normalize_weights(weights)
    return norm_df[cols].to_numpy(dtype=float) @ w


def build_rank_table(df: pd.DataFrame, norm_df: pd.DataFrame, weights, score_col="Priority") -> pd.DataFrame:
    out = df[[
        "sector_id",
        "sector_name_vi",
        "growth_rate_2024_pct",
        "productivity_million_VND_per_worker",
        "spillover_coef_0_1",
        "export_billion_USD",
        "labor_million",
        "ai_readiness_0_100",
        "automation_risk_pct"
    ]].copy()
    out[score_col] = compute_priority(norm_df, weights)
    out = out.sort_values(score_col, ascending=False).reset_index(drop=True)
    out["Rank"] = np.arange(1, len(out) + 1)
    return out[["Rank"] + [c for c in out.columns if c != "Rank"]]


def create_policy_text(default_rank_df, growth_rank_df, inclusive_rank_df, sensitivity_top3_df):
    top3_default = default_rank_df.head(3)["sector_name_vi"].tolist()
    top3_growth = growth_rank_df.head(3)["sector_name_vi"].tolist()
    top3_inclusive = inclusive_rank_df.head(3)["sector_name_vi"].tolist()
    mining_rank = int(default_rank_df.loc[default_rank_df["sector_name_vi"].str.contains("Khai khoáng", case=False, regex=False), "Rank"].iloc[0])

    all_top3 = sensitivity_top3_df["Top_3_text"].unique().tolist()
    stable_text = "ổn định" if len(all_top3) == 1 else "có thay đổi"

    html = f"""
    <h3>a) Theo kết quả, ba ngành nào nên được ưu tiên đẩy mạnh chuyển đổi số và AI trước?</h3>
    <p>
    Theo bộ trọng số mặc định, ba ngành đứng đầu là: <b>{top3_default[0]}</b>, <b>{top3_default[1]}</b> và <b>{top3_default[2]}</b>.
    Đây là các ngành có sự kết hợp tương đối tốt giữa tăng trưởng, năng lực AI, khả năng lan tỏa, xuất khẩu và năng suất.
    Kết quả này phù hợp với tinh thần ưu tiên khoa học - công nghệ, đổi mới sáng tạo và chuyển đổi số, vì các ngành đứng đầu đều có khả năng tạo tác động lan tỏa cho nền kinh tế.
    </p>

    <p>
    Trong phân tích độ nhạy, khi thay đổi trọng số AI Readiness từ 0,05 đến 0,40, nhóm top-3 <b>{stable_text}</b>.
    Điều này cho thấy kết quả xếp hạng không chỉ phụ thuộc vào một tiêu chí AI đơn lẻ, mà còn chịu tác động đồng thời của xuất khẩu, lan tỏa, năng suất và rủi ro tự động hóa.
    </p>

    <h3>b) Tại sao ngành Khai khoáng có năng suất rất cao nhưng vẫn không nằm trong nhóm ưu tiên?</h3>
    <p>
    Trong dữ liệu, Khai khoáng có năng suất rất cao nhưng theo kết quả mặc định chỉ xếp hạng <b>{mining_rank}</b>.
    Lý do là chỉ số ưu tiên không chỉ xét năng suất, mà còn xét tăng trưởng, lan tỏa, xuất khẩu, việc làm, AI Readiness và rủi ro tự động hóa.
    Khai khoáng có tăng trưởng âm, mức lan tỏa thấp, quy mô việc làm nhỏ và rủi ro tự động hóa cao, nên điểm tổng hợp bị kéo xuống.
    Điều này phản ánh đúng bản chất MCDM: một ngành mạnh ở một tiêu chí chưa chắc là lựa chọn ưu tiên nếu yếu ở nhiều tiêu chí còn lại.
    </p>

    <h3>c) Bộ trọng số nên do ai quyết định?</h3>
    <p>
    Bộ trọng số không nên chỉ do chuyên gia kỹ thuật quyết định, vì trọng số thể hiện ưu tiên phát triển và có tác động phân bổ nguồn lực xã hội.
    Cách hợp lý hơn là kết hợp ba lớp: chuyên gia kỹ thuật đề xuất phương pháp và kiểm định dữ liệu; hội đồng chính sách lựa chọn định hướng phù hợp mục tiêu quốc gia; và quy trình tham vấn công khai giúp tăng tính minh bạch, trách nhiệm giải trình và tính chính danh.
    </p>

    <p>
    Khi so sánh hai định hướng, bộ <b>Định hướng tăng trưởng</b> cho top-3 là: <b>{top3_growth[0]}</b>, <b>{top3_growth[1]}</b>, <b>{top3_growth[2]}</b>.
    Bộ <b>Định hướng bao trùm</b> cho top-3 là: <b>{top3_inclusive[0]}</b>, <b>{top3_inclusive[1]}</b>, <b>{top3_inclusive[2]}</b>.
    Sự khác biệt này cho thấy lựa chọn trọng số là một quyết định chính sách, không phải thao tác kỹ thuật trung lập.
    </p>
    """
    return html


def make_html_report(input_df, norm_df, default_weight_df, default_rank_df, sensitivity_top3_df, sensitivity_matrix_df, compare_df, policy_text):
    css = """
    <style>
        body {font-family: Arial, sans-serif; line-height: 1.5; margin: 30px;}
        h1, h2, h3 {color: #1f3b66;}
        table {border-collapse: collapse; width: 100%; margin-bottom: 25px; font-size: 13px;}
        th {background-color: #1f3b66; color: white; padding: 6px; border: 1px solid #ccc;}
        td {padding: 6px; border: 1px solid #ccc; text-align: center;}
        .box {background: #f2f6ff; padding: 15px; border-left: 5px solid #1f3b66; margin-bottom: 20px;}
    </style>
    """
    top3 = default_rank_df.head(3)["sector_name_vi"].tolist()
    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"><title>Bài 3 - Priority ngành</title>{css}</head>
    <body>
        <h1>BÀI 3 - TÍNH CHỈ SỐ ƯU TIÊN NGÀNH PRIORITYᵢ</h1>
        <div class="box">
            <p><b>Top-3 mặc định:</b> {top3[0]} | {top3[1]} | {top3[2]}</p>
            <p><b>Phương pháp:</b> Chuẩn hóa min-max, Risk đảo chiều thành Risk_inverted, trọng số được chuẩn hóa tổng = 1.</p>
        </div>
        <h2>3.5. Câu hỏi thảo luận chính sách</h2>
        {policy_text}
        <h2>Bảng 1. Dữ liệu đầu vào</h2>
        {input_df.to_html(index=False)}
        <h2>Bảng 2. Ma trận chuẩn hóa</h2>
        {norm_df.to_html(index=False)}
        <h2>Bảng 3. Trọng số mặc định</h2>
        {default_weight_df.to_html(index=False)}
        <h2>Bảng 4. Xếp hạng Priority mặc định</h2>
        {default_rank_df.to_html(index=False)}
        <h2>Bảng 5. Top-3 khi thay đổi trọng số AI Readiness</h2>
        {sensitivity_top3_df.to_html(index=False)}
        <h2>Bảng 6. Ma trận điểm độ nhạy</h2>
        {sensitivity_matrix_df.to_html(index=False)}
        <h2>Bảng 7. So sánh hai định hướng trọng số</h2>
        {compare_df.to_html(index=False)}
    </body>
    </html>
    """
    return html


# ============================================================
# 2. ĐỌC DỮ LIỆU
# ============================================================

st.header("1. Đọc dữ liệu đầu vào")

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    st.success("Đã đọc file vietnam_sectors_2024.csv từ thư mục data.")
else:
    st.warning("Chưa tìm thấy file trong thư mục data. Bạn có thể upload file vietnam_sectors_2024.csv tại đây.")
    uploaded_file = st.file_uploader("Upload file vietnam_sectors_2024.csv", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Đã upload và đọc được file CSV.")
    else:
        st.stop()

required_columns = [
    "sector_id",
    "sector_name_vi",
    "growth_rate_2024_pct",
    "labor_million",
    "export_billion_USD",
    "ai_readiness_0_100",
    "spillover_coef_0_1",
    "automation_risk_pct"
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"File dữ liệu đang thiếu cột bắt buộc: {missing_columns}")
    st.stop()

# Bổ sung năng suất nếu CSV không có cột productivity.
df, productivity_source = add_productivity_from_assignment(df)

# Sắp xếp để kết quả ổn định.
df = df.sort_values("sector_id").reset_index(drop=True)

input_cols = [
    "sector_id",
    "sector_name_vi",
    "growth_rate_2024_pct",
    "productivity_million_VND_per_worker",
    "spillover_coef_0_1",
    "export_billion_USD",
    "labor_million",
    "ai_readiness_0_100",
    "automation_risk_pct"
]
input_df = df[input_cols].copy()

st.subheader("Dữ liệu gốc sử dụng cho mô hình")
st.dataframe(input_df, width="stretch")

st.info(productivity_source)

# ============================================================
# 3. MÔ HÌNH VÀ LƯU Ý KIỂM TRA ĐỀ
# ============================================================

st.header("2. Mô hình, công thức và lưu ý kiểm tra đề")

st.markdown(r"""
Công thức chỉ số ưu tiên ngành:

$$
Priority_i = a_1 Growth_i + a_2 Productivity_i + a_3 Spillover_i + a_4 Export_i + a_5 Employment_i + a_6 AIReadiness_i - a_7 Risk_i
$$

Trong code này, rủi ro tự động hóa được xử lý theo hướng **đảo chiều Risk**:

$$
Risk\_inverted_i = \frac{max(Risk)-Risk_i}{max(Risk)-min(Risk)}
$$

Vì vậy điểm tổng hợp được viết tương đương:

$$
Priority_i = \sum a_k X_{ik} + a_7 Risk\_inverted_i
$$

Cách này làm cho ngành có rủi ro tự động hóa thấp được cộng điểm cao hơn, đúng với logic chính sách.
""")

col_note1, col_note2 = st.columns(2)
with col_note1:
    card(
        "Lưu ý 1 — Cột năng suất",
        "File CSV đi kèm thường không có cột năng suất đúng như bảng 3.3. Code đã kiểm tra tự động: nếu CSV không có, chương trình bổ sung số liệu năng suất theo bảng trong đề bài."
    )
with col_note2:
    card(
        "Lưu ý 2 — Trọng số mặc định",
        "Bộ trọng số đề cho có tổng 1,10. Code vẫn nhập đúng các trọng số gốc, sau đó chuẩn hóa về tổng = 1 để đúng tinh thần MCDM. Thứ hạng không đổi nếu chỉ nhân/chia toàn bộ trọng số cùng một tỷ lệ."
    )

st.warning(
    "Trong gợi ý mã của đề có đoạn norm_bad(Risk) rồi lại trừ w_risk * Xb. Nếu Xb đã là Risk đảo chiều thì dấu trừ sẽ làm ngành rủi ro thấp bị phạt. Code này dùng cách nhất quán: Risk_inverted được cộng vào điểm Priority."
)

# ============================================================
# 4. CÂU 3.4.1 - CHUẨN HÓA MIN-MAX
# ============================================================

st.header("3. Câu 3.4.1 - Chuẩn hóa min-max 7 tiêu chí")

norm_df = build_normalized_matrix(df)
norm_display = norm_df.rename(columns=DISPLAY_NAMES).copy()
for col in norm_display.columns:
    if col != "sector_name_vi":
        norm_display[col] = norm_display[col].round(6)

st.subheader("Ma trận đã chuẩn hóa")
st.dataframe(norm_display, width="stretch")

st.markdown("""
Trong ma trận này:
- Các tiêu chí tốt như tăng trưởng, năng suất, lan tỏa, xuất khẩu, việc làm và AI Readiness được chuẩn hóa theo hướng giá trị càng cao càng tốt.
- Rủi ro tự động hóa được đảo chiều thành `Risk_inverted_norm`, nghĩa là giá trị càng cao thì rủi ro càng thấp.
""")

# ============================================================
# 5. CÂU 3.4.2 - TÍNH PRIORITY MẶC ĐỊNH
# ============================================================

st.header("4. Câu 3.4.2 - Tính Priorityᵢ và xếp hạng 10 ngành")

raw_default_weights = np.array([0.15, 0.15, 0.20, 0.15, 0.10, 0.20, 0.15], dtype=float)
default_weights = normalize_weights(raw_default_weights)

default_weight_df = pd.DataFrame({
    "Tiêu chí": WEIGHT_LABELS,
    "Trọng số gốc theo đề": raw_default_weights,
    "Trọng số sau chuẩn hóa": np.round(default_weights, 6)
})

col_w1, col_w2 = st.columns(2)
with col_w1:
    st.metric("Tổng trọng số gốc", f"{raw_default_weights.sum():.2f}")
with col_w2:
    st.metric("Tổng sau chuẩn hóa", f"{default_weights.sum():.2f}")

st.subheader("Bộ trọng số mặc định")
st.dataframe(default_weight_df, width="stretch")

default_rank_df = build_rank_table(df, norm_df, raw_default_weights, score_col="Priority_default")
default_rank_df["Priority_default"] = default_rank_df["Priority_default"].round(6)

st.subheader("Xếp hạng 10 ngành theo Priority giảm dần")
st.dataframe(default_rank_df, width="stretch")

top3_default = default_rank_df.head(3)["sector_name_vi"].tolist()

st.success(
    f"Top-3 ngành ưu tiên theo bộ trọng số mặc định: {top3_default[0]} | {top3_default[1]} | {top3_default[2]}"
)

st.subheader("Biểu đồ điểm Priority mặc định")
fig1, ax1 = plt.subplots(figsize=(10, 5))
plot_df = default_rank_df.sort_values("Priority_default", ascending=True)
ax1.barh(plot_df["sector_name_vi"], plot_df["Priority_default"])
ax1.set_title("Xếp hạng Priority ngành theo bộ trọng số mặc định")
ax1.set_xlabel("Priority score")
ax1.set_ylabel("Ngành")
ax1.grid(axis="x", alpha=0.3)
st.pyplot(fig1)

# ============================================================
# 6. CÂU 3.4.3 - PHÂN TÍCH ĐỘ NHẠY TRỌNG SỐ AI READINESS
# ============================================================

st.header("5. Câu 3.4.3 - Phân tích độ nhạy trọng số AI Readiness")

st.markdown("""
Thay đổi trọng số gốc của tiêu chí AI Readiness từ **0,05 đến 0,40**, bước **0,05**.
Ở mỗi lần thay đổi, toàn bộ bộ trọng số được chuẩn hóa lại để tổng bằng 1.
""")

ai_weight_values = np.round(np.arange(0.05, 0.401, 0.05), 2)
sensitivity_records = []
sensitivity_score_records = []

for a6 in ai_weight_values:
    w_raw = raw_default_weights.copy()
    w_raw[5] = a6
    w_norm = normalize_weights(w_raw)
    score = compute_priority(norm_df, w_raw)

    temp = pd.DataFrame({
        "sector_name_vi": df["sector_name_vi"],
        "Priority": score
    }).sort_values("Priority", ascending=False).reset_index(drop=True)
    top3 = temp.head(3)["sector_name_vi"].tolist()
    sensitivity_records.append({
        "AI_weight_raw": a6,
        "AI_weight_after_normalization": round(w_norm[5], 6),
        "Top_1": top3[0],
        "Top_2": top3[1],
        "Top_3": top3[2],
        "Top_3_text": " | ".join(top3)
    })

    for _, row in temp.iterrows():
        sensitivity_score_records.append({
            "AI_weight_raw": a6,
            "sector_name_vi": row["sector_name_vi"],
            "Priority": row["Priority"]
        })

sensitivity_top3_df = pd.DataFrame(sensitivity_records)
sensitivity_scores_long = pd.DataFrame(sensitivity_score_records)
sensitivity_matrix_df = sensitivity_scores_long.pivot(index="sector_name_vi", columns="AI_weight_raw", values="Priority").reset_index()

for col in sensitivity_matrix_df.columns:
    if col != "sector_name_vi":
        sensitivity_matrix_df[col] = sensitivity_matrix_df[col].round(6)

st.subheader("Bảng top-3 khi thay đổi trọng số AI Readiness")
st.dataframe(sensitivity_top3_df, width="stretch")

unique_top3 = sensitivity_top3_df["Top_3_text"].unique()
if len(unique_top3) == 1:
    st.success("Kết luận độ nhạy: Top-3 không thay đổi trong toàn bộ khoảng trọng số AI Readiness từ 0,05 đến 0,40.")
else:
    st.warning("Kết luận độ nhạy: Top-3 có thay đổi khi điều chỉnh trọng số AI Readiness.")

st.subheader("Heatmap điểm Priority theo trọng số AI Readiness")
heatmap_data = sensitivity_matrix_df.set_index("sector_name_vi").to_numpy(dtype=float)
fig2, ax2 = plt.subplots(figsize=(11, 6))
im = ax2.imshow(heatmap_data, aspect="auto")
ax2.set_title("Heatmap Priority khi thay đổi trọng số AI Readiness")
ax2.set_xlabel("Trọng số AI Readiness gốc")
ax2.set_ylabel("Ngành")
ax2.set_xticks(np.arange(len(ai_weight_values)))
ax2.set_xticklabels([f"{x:.2f}" for x in ai_weight_values])
ax2.set_yticks(np.arange(len(sensitivity_matrix_df["sector_name_vi"])))
ax2.set_yticklabels(sensitivity_matrix_df["sector_name_vi"])
fig2.colorbar(im, ax=ax2, label="Priority score")
st.pyplot(fig2)

# ============================================================
# 7. CÂU 3.4.4 - SO SÁNH HAI BỘ TRỌNG SỐ
# ============================================================

st.header("6. Câu 3.4.4 - So sánh hai định hướng trọng số")

st.markdown("""
Đề bài yêu cầu so sánh hai bộ trọng số nhưng không cung cấp con số cụ thể.
Vì vậy code đặt hai bộ trọng số minh bạch như sau và chuẩn hóa tổng = 1:
- **Định hướng tăng trưởng**: ưu tiên Growth, Productivity, Export.
- **Định hướng bao trùm**: ưu tiên Employment, Spillover và Risk_inverted.
""")

growth_oriented_weights = np.array([0.25, 0.20, 0.10, 0.25, 0.05, 0.10, 0.05], dtype=float)
inclusive_weights = np.array([0.10, 0.05, 0.20, 0.05, 0.25, 0.10, 0.25], dtype=float)

growth_weight_df = pd.DataFrame({
    "Tiêu chí": WEIGHT_LABELS,
    "Định hướng tăng trưởng": normalize_weights(growth_oriented_weights),
    "Định hướng bao trùm": normalize_weights(inclusive_weights)
})
st.subheader("Hai bộ trọng số sử dụng để so sánh")
st.dataframe(growth_weight_df, width="stretch")

growth_rank_df = build_rank_table(df, norm_df, growth_oriented_weights, score_col="Priority_growth")
inclusive_rank_df = build_rank_table(df, norm_df, inclusive_weights, score_col="Priority_inclusive")

growth_rank_df["Priority_growth"] = growth_rank_df["Priority_growth"].round(6)
inclusive_rank_df["Priority_inclusive"] = inclusive_rank_df["Priority_inclusive"].round(6)

compare_df = df[["sector_id", "sector_name_vi"]].copy()
compare_df["Priority_default"] = compute_priority(norm_df, raw_default_weights)
compare_df["Priority_growth"] = compute_priority(norm_df, growth_oriented_weights)
compare_df["Priority_inclusive"] = compute_priority(norm_df, inclusive_weights)
compare_df["Rank_default"] = compare_df["Priority_default"].rank(ascending=False, method="min").astype(int)
compare_df["Rank_growth"] = compare_df["Priority_growth"].rank(ascending=False, method="min").astype(int)
compare_df["Rank_inclusive"] = compare_df["Priority_inclusive"].rank(ascending=False, method="min").astype(int)
for c in ["Priority_default", "Priority_growth", "Priority_inclusive"]:
    compare_df[c] = compare_df[c].round(6)
compare_df = compare_df.sort_values("Rank_default").reset_index(drop=True)

st.subheader("Bảng so sánh điểm và thứ hạng theo 3 bộ trọng số")
st.dataframe(compare_df, width="stretch")

col_cmp1, col_cmp2 = st.columns(2)
with col_cmp1:
    st.subheader("Top-3 định hướng tăng trưởng")
    st.dataframe(growth_rank_df.head(3), width="stretch")
with col_cmp2:
    st.subheader("Top-3 định hướng bao trùm")
    st.dataframe(inclusive_rank_df.head(3), width="stretch")

st.subheader("Biểu đồ so sánh điểm theo ba bộ trọng số")
fig3, ax3 = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(compare_df))
width = 0.25
ax3.bar(x - width, compare_df["Priority_default"], width, label="Mặc định")
ax3.bar(x, compare_df["Priority_growth"], width, label="Tăng trưởng")
ax3.bar(x + width, compare_df["Priority_inclusive"], width, label="Bao trùm")
ax3.set_title("So sánh điểm Priority theo các bộ trọng số")
ax3.set_xlabel("Ngành")
ax3.set_ylabel("Priority score")
ax3.set_xticks(x)
ax3.set_xticklabels(compare_df["sector_name_vi"], rotation=60, ha="right")
ax3.legend()
ax3.grid(axis="y", alpha=0.3)
st.pyplot(fig3)

# ============================================================
# 8. CÂU 3.5 - CÂU HỎI THẢO LUẬN CHÍNH SÁCH
# ============================================================

st.header("7. Câu 3.5 - Câu hỏi thảo luận chính sách")

policy_text = create_policy_text(
    default_rank_df=default_rank_df,
    growth_rank_df=growth_rank_df,
    inclusive_rank_df=inclusive_rank_df,
    sensitivity_top3_df=sensitivity_top3_df
)
st.markdown(policy_text, unsafe_allow_html=True)

# ============================================================
# 9. CHECKLIST HOÀN THÀNH YÊU CẦU
# ============================================================

st.header("8. Checklist hoàn thành yêu cầu Bài 3")

checklist_df = pd.DataFrame([
    ["3.4.1 Đọc dữ liệu vietnam_sectors_2024.csv", "Đã làm", "Đọc từ thư mục data hoặc upload CSV"],
    ["3.4.1 Chuẩn hóa min-max 7 cột", "Đã làm", "Có ma trận norm_display"],
    ["3.4.1 Đảo dấu Risk", "Đã làm", "Risk_inverted_norm càng cao nghĩa là rủi ro càng thấp"],
    ["3.4.2 Tính Priority với trọng số mặc định", "Đã làm", "Trọng số gốc theo đề được chuẩn hóa tổng = 1"],
    ["3.4.2 Xếp hạng 10 ngành giảm dần", "Đã làm", "Có bảng default_rank_df và biểu đồ"],
    ["3.4.3 Thay đổi a6 từ 0,05 đến 0,40", "Đã làm", "Có bảng top-3 theo từng trọng số"],
    ["3.4.3 Chuẩn hóa lại tổng trọng số = 1", "Đã làm", "Mỗi kịch bản AI weight đều normalize_weights"],
    ["3.4.3 Vẽ heatmap", "Đã làm", "Dùng matplotlib imshow"],
    ["3.4.4 So sánh định hướng tăng trưởng và bao trùm", "Đã làm", "Có bảng và biểu đồ so sánh"],
    ["3.5 Trả lời câu hỏi chính sách a, b, c", "Đã làm", "Có phần phân tích chính sách"],
], columns=["Yêu cầu", "Trạng thái", "Ghi chú"])

st.dataframe(checklist_df, width="stretch")

# ============================================================
# 10. TẢI KẾT QUẢ
# ============================================================

st.header("9. Tải kết quả")

html_report = make_html_report(
    input_df=input_df,
    norm_df=norm_display,
    default_weight_df=default_weight_df,
    default_rank_df=default_rank_df,
    sensitivity_top3_df=sensitivity_top3_df,
    sensitivity_matrix_df=sensitivity_matrix_df,
    compare_df=compare_df,
    policy_text=policy_text
)

save_outputs(
    OUTPUT_DIR,
    {
        "bai03_input_data.csv": input_df,
        "bai03_normalized_matrix.csv": norm_display,
        "bai03_default_priority_ranking.csv": default_rank_df,
        "bai03_sensitivity_top3.csv": sensitivity_top3_df,
        "bai03_sensitivity_matrix.csv": sensitivity_matrix_df,
        "bai03_compare_weight_scenarios.csv": compare_df,
        "bai03_report.html": html_report,
    }
)

excel_file = make_download_excel({
    "Input": input_df,
    "Normalized": norm_display,
    "Default_Ranking": default_rank_df,
    "Sensitivity_Top3": sensitivity_top3_df,
    "Sensitivity_Matrix": sensitivity_matrix_df,
    "Compare_Weights": compare_df,
    "Checklist": checklist_df,
})

col_dl1, col_dl2, col_dl3 = st.columns(3)

with col_dl1:
    st.download_button(
        label="Tải HTML report",
        data=html_report.encode("utf-8"),
        file_name="bai03_report.html",
        mime="text/html"
    )

with col_dl2:
    st.download_button(
        label="Tải Excel tổng hợp",
        data=excel_file,
        file_name="bai03_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with col_dl3:
    st.download_button(
        label="Tải bảng xếp hạng CSV",
        data=default_rank_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="bai03_default_priority_ranking.csv",
        mime="text/csv"
    )

st.success("Bài 3 đã hoàn thành đầy đủ các yêu cầu: chuẩn hóa dữ liệu, tính Priority, xếp hạng ngành, phân tích độ nhạy, heatmap, so sánh trọng số và phân tích chính sách.")
