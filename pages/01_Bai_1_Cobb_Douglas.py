import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
from utils.style import load_css, hero, card

# ============================================================
# BÀI 1 - COBB-DOUGLAS MỞ RỘNG VỚI AI VÀ SỐ HÓA
# Webapp Streamlit
# Yêu cầu: numpy, pandas, matplotlib
# ============================================================

st.set_page_config(
    page_title="Bài 1 - Cobb-Douglas",
    layout="wide"
)
load_css()

hero(
    title="📊 Bài 1 — Cobb-Douglas mở rộng với AI và số hóa",
    subtitle="Tính TFP, dự báo GDP, phân rã tăng trưởng và mô phỏng GDP Việt Nam năm 2030 dựa trên mô hình sản xuất mở rộng.",
    badges=["Cấp độ dễ", "numpy", "pandas", "matplotlib"]
)


st.markdown("""
Bài này sử dụng mô hình Cobb-Douglas mở rộng để phân tích tăng trưởng GDP Việt Nam giai đoạn 2020-2025,
có xét thêm các yếu tố số hóa, năng lực AI và vốn nhân lực số.
""")

# ============================================================
# 1. HÀM TIỆN ÍCH
# ============================================================

def make_download_excel(sheets: dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df_sheet in sheets.items():
            df_sheet.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()


def make_html_report(
    tfp_df,
    forecast_df,
    growth_df,
    contrib_df,
    scenario_df,
    Abar,
    MAPE,
    Y2030,
    policy_text
):
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Bài 1 - Cobb-Douglas mở rộng</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.5;
                margin: 30px;
            }}
            h1, h2, h3 {{
                color: #1f3b66;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 25px;
                font-size: 13px;
            }}
            th {{
                background-color: #1f3b66;
                color: white;
                padding: 6px;
                border: 1px solid #ccc;
            }}
            td {{
                padding: 6px;
                border: 1px solid #ccc;
                text-align: center;
            }}
            .box {{
                background: #f2f6ff;
                padding: 15px;
                border-left: 5px solid #1f3b66;
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>BÀI 1 - HÀM SẢN XUẤT COBB-DOUGLAS MỞ RỘNG</h1>

        <div class="box">
            <p><b>A trung bình 2020-2025:</b> {Abar:.6f}</p>
            <p><b>MAPE:</b> {MAPE:.4f}%</p>
            <p><b>GDP dự báo năm 2030:</b> {Y2030:.4f} nghìn tỷ VND</p>
        </div>

        <h2>1.5. Câu hỏi thảo luận chính sách</h2>
        {policy_text}

        <h2>Bảng 1. TFP từng năm</h2>
        {tfp_df.to_html(index=False)}

        <h2>Bảng 2. GDP thực tế, GDP dự báo và sai số</h2>
        {forecast_df.to_html(index=False)}

        <h2>Bảng 3. Phân rã tăng trưởng theo từng giai đoạn</h2>
        {growth_df.to_html(index=False)}

        <h2>Bảng 4. Đóng góp tăng trưởng bình quân</h2>
        {contrib_df.to_html(index=False)}

        <h2>Bảng 5. Kịch bản dự báo GDP 2030</h2>
        {scenario_df.to_html(index=False)}
    </body>
    </html>
    """
    return html


def save_outputs(output_dir, tfp_df, forecast_df, growth_df, contrib_df, scenario_df, html_report):
    output_dir.mkdir(exist_ok=True)

    tfp_df.to_csv(output_dir / "bai01_tfp_table.csv", index=False, encoding="utf-8-sig")
    forecast_df.to_csv(output_dir / "bai01_forecast_table.csv", index=False, encoding="utf-8-sig")
    growth_df.to_csv(output_dir / "bai01_growth_decomposition.csv", index=False, encoding="utf-8-sig")
    contrib_df.to_csv(output_dir / "bai01_average_contribution.csv", index=False, encoding="utf-8-sig")
    scenario_df.to_csv(output_dir / "bai01_scenario_2030.csv", index=False, encoding="utf-8-sig")

    with open(output_dir / "bai01_report.html", "w", encoding="utf-8") as f:
        f.write(html_report)


# ============================================================
# 2. ĐỌC DỮ LIỆU
# ============================================================

st.header("1. Đọc dữ liệu đầu vào")

DATA_PATH = Path("data") / "vietnam_macro_2020_2025.csv"
OUTPUT_DIR = Path("outputs")

def default_macro_from_assignment() -> pd.DataFrame:
    """Dữ liệu GDP 2020-2025 theo bảng đề bài, dùng khi chưa có CSV.

    Đây không phải dữ liệu tự tạo. Các biến K, L, D, AI, H bên dưới cũng lấy theo đề.
    Việc fallback giúp app chạy ổn định khi nộp code mà quên kèm thư mục data/.
    """
    return pd.DataFrame({
        "year": [2020, 2021, 2022, 2023, 2024, 2025],
        "GDP_trillion_VND": [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
    })

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    st.success("Đã đọc file vietnam_macro_2020_2025.csv từ thư mục data.")
else:
    st.warning("Chưa tìm thấy file trong thư mục data. Có thể upload CSV; nếu không upload, app dùng dữ liệu GDP 2020-2025 đúng theo đề bài để tránh dừng chương trình.")
    uploaded_file = st.file_uploader("Upload file vietnam_macro_2020_2025.csv", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Đã upload và đọc được file CSV.")
    else:
        df = default_macro_from_assignment()
        st.info("Đang dùng dữ liệu mặc định từ đề bài: GDP 2020-2025.")

required_columns = ["year", "GDP_trillion_VND"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"File dữ liệu đang thiếu cột: {missing_columns}")
    st.stop()

df = df.sort_values("year").reset_index(drop=True)

st.subheader("Dữ liệu gốc")
st.dataframe(df, width="stretch")

years = df["year"].to_numpy()
Y = df["GDP_trillion_VND"].to_numpy(dtype=float)

# ============================================================
# 3. DỮ LIỆU THEO ĐỀ BÀI
# ============================================================

st.header("2. Mô hình và tham số")

st.markdown("""
Mô hình Cobb-Douglas mở rộng:

$$
Y_t = A_t \\times K_t^\\alpha \\times L_t^\\beta \\times D_t^\\gamma \\times AI_t^\\delta \\times H_t^\\theta
$$

Trong đó:
- \(Y_t\): GDP của nền kinh tế.
- \(A_t\): năng suất nhân tố tổng hợp.
- \(K_t\): vốn vật chất.
- \(L_t\): lao động.
- \(D_t\): tỷ trọng kinh tế số/GDP.
- \(AI_t\): năng lực AI.
- \(H_t\): vốn nhân lực số.
""")

K = np.array([16500, 17800, 19600, 21300, 23500, 25900], dtype=float)
L = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4], dtype=float)
D = np.array([12.0, 12.7, 14.3, 16.5, 18.3, 19.5], dtype=float)
AI = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1], dtype=float)
H = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2], dtype=float)

alpha = 0.33
beta = 0.42
gamma = 0.10
delta = 0.08
theta = 0.07

param_df = pd.DataFrame({
    "Tham số": ["alpha", "beta", "gamma", "delta", "theta"],
    "Giá trị": [alpha, beta, gamma, delta, theta],
    "Ý nghĩa": [
        "Độ co giãn theo vốn vật chất K",
        "Độ co giãn theo lao động L",
        "Độ co giãn theo số hóa D",
        "Độ co giãn theo năng lực AI",
        "Độ co giãn theo vốn nhân lực số H"
    ]
})

st.subheader("Tham số mô hình")
st.dataframe(param_df, width="stretch")

input_df = pd.DataFrame({
    "Năm": years,
    "Y_GDP": Y,
    "K": K,
    "L": L,
    "D": D,
    "AI": AI,
    "H": H
})

st.subheader("Dữ liệu sử dụng trong mô hình")
st.dataframe(input_df, width="stretch")

# ============================================================
# 4. CÂU 1.4.1 - TÍNH TFP A_t
# ============================================================

st.header("3. Câu 1.4.1 - Tính TFP A_t từng năm")

A = Y / (
    K**alpha *
    L**beta *
    D**gamma *
    AI**delta *
    H**theta
)

tfp_df = pd.DataFrame({
    "year": years,
    "GDP_actual": np.round(Y, 2),
    "K": K,
    "L": L,
    "D": D,
    "AI": AI,
    "H": H,
    "TFP_A": np.round(A, 6)
})

st.dataframe(tfp_df, width="stretch")

st.subheader("Biểu đồ xu hướng TFP A_t")

fig1, ax1 = plt.subplots(figsize=(9, 4.5))
ax1.plot(years, A, marker="o")
ax1.set_title("Xu hướng TFP A_t giai đoạn 2020-2025")
ax1.set_xlabel("Năm")
ax1.set_ylabel("TFP A_t")
ax1.grid(True)
st.pyplot(fig1)

tfp_start = A[0]
tfp_end = A[-1]
tfp_change_pct = (tfp_end - tfp_start) / tfp_start * 100
tfp_cagr = ((tfp_end / tfp_start) ** (1 / (len(A) - 1)) - 1) * 100

if tfp_end > tfp_start:
    tfp_trend = "tăng"
elif tfp_end < tfp_start:
    tfp_trend = "giảm"
else:
    tfp_trend = "gần như không đổi"

st.markdown(f"""
**Nhận xét:** TFP có xu hướng **{tfp_trend}** trong giai đoạn 2020-2025.
Cụ thể, TFP tăng từ **{tfp_start:.4f}** năm {years[0]} lên **{tfp_end:.4f}** năm {years[-1]},
tương đương mức tăng khoảng **{tfp_change_pct:.2f}%** trong toàn giai đoạn,
hay khoảng **{tfp_cagr:.2f}%/năm**.
""")

# ============================================================
# 5. CÂU 1.4.2 - GDP DỰ BÁO VÀ MAPE
# ============================================================

st.header("4. Câu 1.4.2 - GDP dự báo và MAPE")

Abar = A.mean()

Yhat = Abar * (
    K**alpha *
    L**beta *
    D**gamma *
    AI**delta *
    H**theta
)

error_pct = np.abs((Y - Yhat) / Y) * 100
MAPE = error_pct.mean()

forecast_df = pd.DataFrame({
    "year": years,
    "GDP_actual": np.round(Y, 2),
    "GDP_predicted": np.round(Yhat, 2),
    "error_pct": np.round(error_pct, 4)
})

col1, col2 = st.columns(2)

with col1:
    st.metric("A trung bình 2020-2025", f"{Abar:.6f}")

with col2:
    st.metric("MAPE", f"{MAPE:.4f}%")

st.subheader("Bảng GDP thực tế, GDP dự báo và sai số")
st.dataframe(forecast_df, width="stretch")

st.subheader("Biểu đồ GDP thực tế và GDP dự báo")

fig2, ax2 = plt.subplots(figsize=(9, 4.5))
ax2.plot(years, Y, marker="o", label="GDP thực tế")
ax2.plot(years, Yhat, marker="o", label="GDP dự báo")
ax2.set_title("So sánh GDP thực tế và GDP dự báo")
ax2.set_xlabel("Năm")
ax2.set_ylabel("GDP nghìn tỷ VND")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

st.markdown(f"""
Khi sử dụng giá trị **A trung bình = {Abar:.6f}**, mô hình cho sai số MAPE là **{MAPE:.4f}%**.
Mức sai số này cho thấy mô hình Cobb-Douglas mở rộng có khả năng mô phỏng tương đối phù hợp xu hướng GDP
trong giai đoạn 2020-2025.
""")

# ============================================================
# 6. CÂU 1.4.3 - PHÂN RÃ TĂNG TRƯỞNG
# ============================================================

st.header("5. Câu 1.4.3 - Phân rã tăng trưởng 2020-2025")

growth_Y = np.diff(np.log(Y))
growth_A = np.diff(np.log(A))
growth_K = alpha * np.diff(np.log(K))
growth_L = beta * np.diff(np.log(L))
growth_D = gamma * np.diff(np.log(D))
growth_AI = delta * np.diff(np.log(AI))
growth_H = theta * np.diff(np.log(H))

growth_df = pd.DataFrame({
    "period": [f"{years[i]}-{years[i+1]}" for i in range(len(years) - 1)],
    "GDP_growth_log": np.round(growth_Y, 6),
    "TFP_contribution": np.round(growth_A, 6),
    "K_contribution": np.round(growth_K, 6),
    "L_contribution": np.round(growth_L, 6),
    "D_contribution": np.round(growth_D, 6),
    "AI_contribution": np.round(growth_AI, 6),
    "H_contribution": np.round(growth_H, 6)
})

st.subheader("Bảng phân rã tăng trưởng theo từng giai đoạn")
st.dataframe(growth_df, width="stretch")

avg_growth_Y = growth_Y.mean()

contrib_df = pd.DataFrame({
    "Factor": ["TFP", "K", "L", "D", "AI", "H"],
    "Average_log_contribution": [
        growth_A.mean(),
        growth_K.mean(),
        growth_L.mean(),
        growth_D.mean(),
        growth_AI.mean(),
        growth_H.mean()
    ]
})

contrib_df["Share_of_GDP_growth_pct"] = contrib_df["Average_log_contribution"] / avg_growth_Y * 100
contrib_df["Average_log_contribution"] = contrib_df["Average_log_contribution"].round(6)
contrib_df["Share_of_GDP_growth_pct"] = contrib_df["Share_of_GDP_growth_pct"].round(4)

st.subheader("Bảng đóng góp tăng trưởng bình quân")
st.dataframe(contrib_df, width="stretch")

st.subheader("Biểu đồ đóng góp vào tăng trưởng GDP bình quân")

fig3, ax3 = plt.subplots(figsize=(9, 4.5))
ax3.bar(contrib_df["Factor"], contrib_df["Share_of_GDP_growth_pct"])
ax3.axhline(0)
ax3.set_title("Đóng góp vào tăng trưởng GDP bình quân 2020-2025")
ax3.set_xlabel("Yếu tố")
ax3.set_ylabel("Tỷ trọng đóng góp (%)")
ax3.grid(axis="y")
st.pyplot(fig3)

tfp_share = float(contrib_df.loc[contrib_df["Factor"] == "TFP", "Share_of_GDP_growth_pct"].iloc[0])
d_share = float(contrib_df.loc[contrib_df["Factor"] == "D", "Share_of_GDP_growth_pct"].iloc[0])
ai_share = float(contrib_df.loc[contrib_df["Factor"] == "AI", "Share_of_GDP_growth_pct"].iloc[0])
h_share = float(contrib_df.loc[contrib_df["Factor"] == "H", "Share_of_GDP_growth_pct"].iloc[0])

new_factors = contrib_df[contrib_df["Factor"].isin(["D", "AI", "H"])].sort_values(
    "Share_of_GDP_growth_pct",
    ascending=False
)
top_new_factor = new_factors.iloc[0]["Factor"]
top_new_factor_share = float(new_factors.iloc[0]["Share_of_GDP_growth_pct"])

factor_name = {
    "D": "số hóa D",
    "AI": "năng lực AI",
    "H": "vốn nhân lực số H"
}

st.markdown(f"""
Trong phân rã tăng trưởng, TFP đóng góp khoảng **{tfp_share:.4f}%** vào tăng trưởng GDP bình quân.
Trong ba yếu tố mới gồm D, AI và H, yếu tố đóng góp lớn nhất là **{factor_name[top_new_factor]}**,
với tỷ trọng khoảng **{top_new_factor_share:.4f}%**.
""")

# ============================================================
# 7. CÂU 1.4.4 - DỰ BÁO GDP 2030
# ============================================================

st.header("6. Câu 1.4.4 - Mô phỏng GDP Việt Nam năm 2030")

K2030 = K[-1] * (1.06 ** 5)
L2030 = L[-1] * (1.06 ** 5)
D2030 = 30
AI2030 = 100
H2030 = 35
A2030 = A[-1] * (1.012 ** 5)

Y2030 = A2030 * (
    K2030**alpha *
    L2030**beta *
    D2030**gamma *
    AI2030**delta *
    H2030**theta
)

scenario_df = pd.DataFrame({
    "Variable": ["A_2030", "K_2030", "L_2030", "D_2030", "AI_2030", "H_2030", "GDP_2030"],
    "Value": [
        round(A2030, 6),
        round(K2030, 4),
        round(L2030, 4),
        D2030,
        AI2030,
        H2030,
        round(Y2030, 4)
    ]
})

st.dataframe(scenario_df, width="stretch")

st.metric("GDP dự báo năm 2030", f"{Y2030:.4f} nghìn tỷ VND")

st.markdown(f"""
Theo kịch bản đề bài, nếu đến năm 2030:
- D đạt **30%**,
- AI đạt **100 nghìn doanh nghiệp số**,
- H đạt **35%**,
- K và L tăng **6%/năm**,
- TFP tăng **1,2%/năm**,

thì GDP Việt Nam năm 2030 được dự báo đạt khoảng **{Y2030:.2f} nghìn tỷ VND**.
""")

# ============================================================
# 8. CÂU 1.5 - PHÂN TÍCH CHÍNH SÁCH
# ============================================================

st.header("7. Câu 1.5 - Câu hỏi thảo luận chính sách")

D2025 = D[-1]
required_dcagr = ((D2030 / D2025) ** (1 / 5) - 1) * 100
historical_dcagr = ((D[-1] / D[0]) ** (1 / (len(D) - 1)) - 1) * 100

if required_dcagr <= historical_dcagr:
    feasibility_text = "Về mặt mô hình, mục tiêu 30% kinh tế số/GDP vào năm 2030 tương đối khả thi vì tốc độ tăng cần thiết của D không vượt quá tốc độ tăng bình quân lịch sử giai đoạn 2020-2025."
else:
    feasibility_text = "Về mặt mô hình, mục tiêu 30% kinh tế số/GDP vào năm 2030 có thể đạt được nhưng khá thách thức vì tốc độ tăng cần thiết của D cao hơn tốc độ tăng bình quân lịch sử giai đoạn 2020-2025."

factor_explain = {
    "D": "D đóng góp cao nhất vì tỷ trọng kinh tế số/GDP tăng khá nhanh trong giai đoạn 2020-2025, làm cho thành phần γΔlnD trong phân rã tăng trưởng lớn hơn hai yếu tố AI và H.",
    "AI": "AI đóng góp cao nhất vì năng lực AI, đại diện bằng số doanh nghiệp công nghệ số, tăng nhanh và tạo tác động lan tỏa đến năng suất.",
    "H": "H đóng góp cao nhất vì tỷ lệ lao động qua đào tạo tăng ổn định, giúp nền kinh tế hấp thụ công nghệ tốt hơn."
}

policy_text = f"""
<h3>a) TFP của Việt Nam có xu hướng tăng hay giảm trong giai đoạn 2020-2025?</h3>

<p>
TFP của Việt Nam trong mô hình có xu hướng <b>{tfp_trend}</b>.
Cụ thể, TFP tăng từ <b>{tfp_start:.4f}</b> năm {years[0]} lên <b>{tfp_end:.4f}</b> năm {years[-1]},
tức tăng khoảng <b>{tfp_change_pct:.2f}%</b> trong toàn giai đoạn, tương đương <b>{tfp_cagr:.2f}%/năm</b>.
</p>

<p>
Điều này cho thấy chất lượng tăng trưởng có xu hướng cải thiện. GDP không chỉ tăng nhờ mở rộng vốn và lao động,
mà còn nhờ hiệu quả tổng hợp như công nghệ, tổ chức sản xuất, chuyển đổi số, năng lực quản trị và khả năng hấp thụ đổi mới.
Trong kết quả phân rã tăng trưởng, TFP đóng góp khoảng <b>{tfp_share:.4f}%</b> vào tăng trưởng GDP bình quân.
</p>

<h3>b) Trong các yếu tố mới D, AI, H, yếu tố nào đóng góp nhiều nhất?</h3>

<p>
Trong ba yếu tố mới gồm số hóa D, năng lực AI và vốn nhân lực số H,
yếu tố đóng góp nhiều nhất là <b>{factor_name[top_new_factor]}</b>,
với tỷ trọng khoảng <b>{top_new_factor_share:.4f}%</b> trong tăng trưởng GDP bình quân.
</p>

<p>
{factor_explain[top_new_factor]}
Cụ thể, đóng góp của D là <b>{d_share:.4f}%</b>, AI là <b>{ai_share:.4f}%</b>, và H là <b>{h_share:.4f}%</b>.
</p>

<h3>c) Mục tiêu Việt Nam đạt 30% kinh tế số/GDP vào năm 2030 có khả thi không?</h3>

<p>
Theo mô hình, GDP dự báo năm 2030 đạt khoảng <b>{Y2030:.2f} nghìn tỷ VND</b>.
Từ mức D năm 2025 là <b>{D2025:.2f}%</b>, để đạt <b>30%</b> vào năm 2030,
tỷ trọng kinh tế số/GDP cần tăng bình quân khoảng <b>{required_dcagr:.2f}%/năm</b>.
Trong khi đó, tốc độ tăng bình quân của D giai đoạn 2020-2025 là khoảng <b>{historical_dcagr:.2f}%/năm</b>.
{feasibility_text}
</p>

<p>
Tuy nhiên, mục tiêu này chỉ khả thi nếu đi kèm các ràng buộc chính sách:
</p>

<ul>
    <li><b>Hạ tầng số:</b> cần đầu tư mạng số, trung tâm dữ liệu, điện toán đám mây và kết nối vùng.</li>
    <li><b>Nhân lực số:</b> cần đào tạo lao động số, kỹ sư dữ liệu, kỹ sư AI và năng lực quản trị công nghệ.</li>
    <li><b>Vốn và ngân sách:</b> đầu tư phải phù hợp với năng lực hấp thụ, tránh dàn trải.</li>
    <li><b>Thể chế và dữ liệu:</b> cần khung pháp lý cho dữ liệu mở, an toàn thông tin, giao dịch điện tử và bảo vệ quyền riêng tư.</li>
    <li><b>Bao trùm xã hội:</b> chuyển đổi số phải giảm khoảng cách vùng miền và bất bình đẳng số.</li>
</ul>

<p>
Như vậy, mục tiêu 30% kinh tế số/GDP vào năm 2030 có cơ sở để hướng tới,
nhưng cần đồng bộ giữa số hóa, AI, nhân lực số, hạ tầng, thể chế và năng lực hấp thụ của nền kinh tế.
</p>
"""

st.markdown(policy_text, unsafe_allow_html=True)

# ============================================================
# 9. TẢI KẾT QUẢ
# ============================================================

st.header("8. Tải kết quả")

html_report = make_html_report(
    tfp_df=tfp_df,
    forecast_df=forecast_df,
    growth_df=growth_df,
    contrib_df=contrib_df,
    scenario_df=scenario_df,
    Abar=Abar,
    MAPE=MAPE,
    Y2030=Y2030,
    policy_text=policy_text
)

save_outputs(
    output_dir=OUTPUT_DIR,
    tfp_df=tfp_df,
    forecast_df=forecast_df,
    growth_df=growth_df,
    contrib_df=contrib_df,
    scenario_df=scenario_df,
    html_report=html_report
)

excel_file = make_download_excel({
    "TFP": tfp_df,
    "Forecast": forecast_df,
    "Growth": growth_df,
    "Contribution": contrib_df,
    "Scenario_2030": scenario_df
})

col_dl1, col_dl2, col_dl3 = st.columns(3)

with col_dl1:
    st.download_button(
        label="Tải HTML report",
        data=html_report.encode("utf-8"),
        file_name="bai01_report.html",
        mime="text/html"
    )

with col_dl2:
    st.download_button(
        label="Tải Excel tổng hợp",
        data=excel_file,
        file_name="bai01_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with col_dl3:
    st.download_button(
        label="Tải bảng TFP CSV",
        data=tfp_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="bai01_tfp_table.csv",
        mime="text/csv"
    )

st.success("Bài 1 đã hoàn thành đầy đủ các yêu cầu: TFP, MAPE, phân rã tăng trưởng, dự báo 2030 và phân tích chính sách.")