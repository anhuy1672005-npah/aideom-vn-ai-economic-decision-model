import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
from itertools import combinations

try:
    from utils.style import load_css, hero, card
except Exception:
    def load_css():
        st.markdown(
            """
            <style>
            .stApp {background: linear-gradient(180deg, #0b1020 0%, #111827 100%);}
            .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px;}
            h1, h2, h3 {color: #ffffff !important;}
            .card {background: rgba(18,26,47,.95); border: 1px solid rgba(255,255,255,.08); border-radius: 18px; padding: 20px 22px; margin-bottom: 18px;}
            .card-title {font-size: 1.1rem; font-weight: 750; color: #ffffff; margin-bottom: 8px;}
            .card-text {color: #cbd5e1; font-size: .95rem; line-height: 1.55;}
            .badge {display:inline-block; padding: 6px 12px; margin-right:8px; border-radius:999px; font-size:.78rem; font-weight:700; color:white; background:linear-gradient(90deg,#2563eb,#7c3aed);}
            div[data-testid="stMetric"] {background: rgba(18,26,47,.95); border:1px solid rgba(255,255,255,.08); padding:18px; border-radius:16px;}
            div[data-testid="stMetricValue"] {color:#ffffff; font-weight:800;}
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

try:
    import pulp
    PULP_AVAILABLE = True
except Exception:
    pulp = None
    PULP_AVAILABLE = False

# ============================================================
# BÀI 5 - QUY HOẠCH NGUYÊN HỖN HỢP MIP
# Lựa chọn dự án chuyển đổi số Việt Nam 2026-2030
# ============================================================

st.set_page_config(
    page_title="Bài 5 - MIP lựa chọn dự án",
    layout="wide"
)
load_css()

hero(
    title="🧩 Bài 5 — MIP lựa chọn dự án chuyển đổi số",
    subtitle="Giải bài toán quy hoạch nguyên hỗn hợp với biến nhị phân, ràng buộc ngân sách đa năm, ràng buộc loại trừ, tiên quyết và bắt buộc dự án.",
    badges=["Cấp độ trung bình", "MIP", "PuLP CBC", "Knapsack"]
)

st.markdown(r"""
Bài này lựa chọn danh mục dự án chuyển đổi số quốc gia giai đoạn 2026-2030.  
Mỗi dự án có biến quyết định nhị phân:

$$
y_i = \begin{cases}
1, & \text{nếu chọn dự án } i \\
0, & \text{nếu không chọn dự án } i
\end{cases}
$$

Mục tiêu là tối đa hóa tổng lợi ích NPV, đồng thời tuân thủ ràng buộc ngân sách, ràng buộc tiên quyết,
ràng buộc loại trừ và số lượng dự án tối thiểu/tối đa.
""")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. DỮ LIỆU DỰ ÁN
# ============================================================

@st.cache_data
def get_project_data():
    data = [
        [1, "P1", "Trung tâm dữ liệu quốc gia Hòa Lạc", "Hạ tầng", 12000, 21500, 8500, 3500],
        [2, "P2", "Trung tâm dữ liệu quốc gia phía Nam", "Hạ tầng", 11500, 20800, 7500, 4000],
        [3, "P3", "Hệ thống 5G phủ sóng toàn quốc", "Hạ tầng", 18000, 32500, 12000, 6000],
        [4, "P4", "Hệ thống định danh điện tử VNeID 2.0", "Chính phủ số", 4500, 9200, 3500, 1000],
        [5, "P5", "Cổng dịch vụ công quốc gia v3", "Chính phủ số", 3200, 6800, 2500, 700],
        [6, "P6", "Y tế số quốc gia (hồ sơ sức khỏe)", "Y tế số", 5800, 11400, 4000, 1800],
        [7, "P7", "Giáo dục số K-12 toàn quốc", "Giáo dục", 6500, 12200, 4500, 2000],
        [8, "P8", "Trung tâm AI quốc gia + supercomputing", "AI", 15000, 28500, 9000, 6000],
        [9, "P9", "Sandbox tài chính số (fintech)", "Tài chính số", 2500, 5800, 1800, 700],
        [10, "P10", "Logistics thông minh + cảng biển số", "Logistics", 7200, 13800, 5000, 2200],
        [11, "P11", "Nông nghiệp số ĐBSCL", "Nông nghiệp", 4800, 8500, 3500, 1300],
        [12, "P12", "Đào tạo 50.000 kỹ sư AI/bán dẫn", "Nhân lực", 8500, 16200, 5500, 3000],
        [13, "P13", "Khu CN bán dẫn Bắc Ninh - Bắc Giang", "Bán dẫn", 20000, 35000, 13000, 7000],
        [14, "P14", "An ninh mạng quốc gia (SOC)", "An ninh", 3800, 7500, 2800, 1000],
        [15, "P15", "Open Data + dữ liệu mở quốc gia", "Dữ liệu", 1500, 3800, 1200, 300],
    ]
    df = pd.DataFrame(
        data,
        columns=["id", "code", "project_name", "field", "cost", "benefit", "cost_year_1_2", "cost_year_3_5"]
    )
    df["npv_per_cost"] = df["benefit"] / df["cost"]
    df["completion_prob"] = df["field"].map(completion_probability)
    df["expected_benefit"] = df["benefit"] * df["completion_prob"]
    df["expected_npv_per_cost"] = df["expected_benefit"] / df["cost"]
    return df


def completion_probability(field):
    """Xác suất hoàn thành đúng tiến độ theo yêu cầu mở rộng 5.4.4."""
    if field == "Hạ tầng":
        return 0.85
    if field == "Chính phủ số":
        return 0.75
    if field in ["AI", "Bán dẫn"]:
        return 0.65
    return 0.80


df_projects = get_project_data()

st.header("1. Dữ liệu 15 dự án ứng cử")
st.dataframe(
    df_projects[[
        "code", "project_name", "field", "cost", "benefit", "cost_year_1_2", "cost_year_3_5",
        "npv_per_cost", "completion_prob", "expected_benefit"
    ]].round(4),
    width="stretch"
)

# ============================================================
# 2. HÀM GIẢI MIP
# ============================================================

def selected_summary(df, selected_ids, objective_value=None, status="Optimal", method="PuLP/CBC"):
    selected = df[df["id"].isin(selected_ids)].copy()
    total_cost = float(selected["cost"].sum())
    total_c12 = float(selected["cost_year_1_2"].sum())
    total_c35 = float(selected["cost_year_3_5"].sum())
    total_benefit = float(selected["benefit"].sum())
    total_expected = float(selected["expected_benefit"].sum())
    if objective_value is None:
        objective_value = total_benefit
    return {
        "status": status,
        "method": method,
        "selected_ids": list(map(int, selected_ids)),
        "selected_codes": list(selected["code"]),
        "selected_df": selected.reset_index(drop=True),
        "objective_value": float(objective_value) if objective_value is not None else np.nan,
        "total_cost": total_cost,
        "total_cost_year_1_2": total_c12,
        "total_cost_year_3_5": total_c35,
        "total_benefit": total_benefit,
        "total_expected_benefit": total_expected,
        "count": int(len(selected)),
        "npv_per_cost": total_benefit / total_cost if total_cost else np.nan,
        "expected_per_cost": total_expected / total_cost if total_cost else np.nan,
    }


def solve_by_enumeration(
    df,
    total_budget=80000,
    early_budget=40000,
    objective_mode="benefit",
    force_both_centers=False,
    keep_exclusion=True,
    require_p14=True,
    min_projects=7,
    max_projects=11,
):
    ids = list(df["id"])
    cost = dict(zip(df["id"], df["cost"]))
    c12 = dict(zip(df["id"], df["cost_year_1_2"]))
    benefit = dict(zip(df["id"], df["benefit"]))
    expected = dict(zip(df["id"], df["expected_benefit"]))

    best_value = -1e30
    best_selected = None

    for k in range(min_projects, max_projects + 1):
        for combo in combinations(ids, k):
            s = set(combo)

            if sum(cost[i] for i in s) > total_budget:
                continue
            if sum(c12[i] for i in s) > early_budget:
                continue

            if force_both_centers:
                if not ({1, 2}.issubset(s)):
                    continue
            if keep_exclusion:
                if 1 in s and 2 in s:
                    continue

            if 8 in s and 12 not in s:
                continue
            if 13 in s and 12 not in s:
                continue
            if not (4 in s or 5 in s):
                continue
            if require_p14 and 14 not in s:
                continue

            if objective_mode == "expected":
                value = sum(expected[i] for i in s)
            else:
                value = sum(benefit[i] for i in s)

            # Tie-break: ưu tiên tổng lợi ích thường cao hơn, sau đó chi phí thấp hơn.
            if best_selected is None:
                better = True
            else:
                current_benefit = sum(benefit[i] for i in s)
                best_benefit = sum(benefit[i] for i in best_selected)
                current_cost = sum(cost[i] for i in s)
                best_cost = sum(cost[i] for i in best_selected)
                better = (
                    value > best_value + 1e-9 or
                    (abs(value - best_value) <= 1e-9 and current_benefit > best_benefit + 1e-9) or
                    (abs(value - best_value) <= 1e-9 and abs(current_benefit - best_benefit) <= 1e-9 and current_cost < best_cost)
                )
            if better:
                best_value = value
                best_selected = sorted(s)

    if best_selected is None:
        return {
            "status": "Infeasible",
            "method": "Enumeration fallback",
            "selected_ids": [],
            "selected_codes": [],
            "selected_df": pd.DataFrame(),
            "objective_value": np.nan,
            "total_cost": np.nan,
            "total_cost_year_1_2": np.nan,
            "total_cost_year_3_5": np.nan,
            "total_benefit": np.nan,
            "total_expected_benefit": np.nan,
            "count": 0,
            "npv_per_cost": np.nan,
            "expected_per_cost": np.nan,
        }

    return selected_summary(
        df,
        best_selected,
        objective_value=best_value,
        status="Optimal",
        method="Enumeration fallback"
    )


def solve_with_pulp(
    df,
    total_budget=80000,
    early_budget=40000,
    objective_mode="benefit",
    force_both_centers=False,
    keep_exclusion=True,
    require_p14=True,
    min_projects=7,
    max_projects=11,
    msg=False,
):
    if not PULP_AVAILABLE:
        return solve_by_enumeration(
            df,
            total_budget=total_budget,
            early_budget=early_budget,
            objective_mode=objective_mode,
            force_both_centers=force_both_centers,
            keep_exclusion=keep_exclusion,
            require_p14=require_p14,
            min_projects=min_projects,
            max_projects=max_projects,
        )

    ids = list(df["id"])
    cost = dict(zip(df["id"], df["cost"]))
    c12 = dict(zip(df["id"], df["cost_year_1_2"]))
    benefit = dict(zip(df["id"], df["benefit"]))
    expected = dict(zip(df["id"], df["expected_benefit"]))

    m = pulp.LpProblem("VN_Project_Selection", pulp.LpMaximize)
    y = pulp.LpVariable.dicts("y", ids, cat="Binary")

    if objective_mode == "expected":
        m += pulp.lpSum(expected[i] * y[i] for i in ids), "Expected_Benefit"
    else:
        m += pulp.lpSum(benefit[i] * y[i] for i in ids), "Total_NPV"

    m += pulp.lpSum(cost[i] * y[i] for i in ids) <= total_budget, "C1_Total_5Y_Budget"
    m += pulp.lpSum(c12[i] * y[i] for i in ids) <= early_budget, "C2_Year_1_2_Budget"

    if keep_exclusion:
        m += y[1] + y[2] <= 1, "C3_Data_Center_Exclusion"

    if force_both_centers:
        m += y[1] == 1, "Force_P1"
        m += y[2] == 1, "Force_P2"

    m += y[8] <= y[12], "C4_AI_requires_training"
    m += y[13] <= y[12], "C5_Semiconductor_requires_training"
    m += y[4] + y[5] >= 1, "C6_At_least_one_digital_government"

    if require_p14:
        m += y[14] >= 1, "C6_Cybersecurity_required"

    m += pulp.lpSum(y[i] for i in ids) >= min_projects, "C7_Min_projects"
    m += pulp.lpSum(y[i] for i in ids) <= max_projects, "C7_Max_projects"

    solver = pulp.PULP_CBC_CMD(msg=msg)
    m.solve(solver)
    status = pulp.LpStatus.get(m.status, str(m.status))

    if status != "Optimal":
        return {
            "status": status,
            "method": "PuLP/CBC",
            "selected_ids": [],
            "selected_codes": [],
            "selected_df": pd.DataFrame(),
            "objective_value": np.nan,
            "total_cost": np.nan,
            "total_cost_year_1_2": np.nan,
            "total_cost_year_3_5": np.nan,
            "total_benefit": np.nan,
            "total_expected_benefit": np.nan,
            "count": 0,
            "npv_per_cost": np.nan,
            "expected_per_cost": np.nan,
        }

    selected_ids = [i for i in ids if y[i].value() is not None and y[i].value() > 0.5]
    return selected_summary(
        df,
        selected_ids,
        objective_value=pulp.value(m.objective),
        status=status,
        method="PuLP/CBC"
    )


def solution_to_row(name, sol):
    return {
        "Kịch bản": name,
        "Trạng thái": sol["status"],
        "Phương pháp": sol["method"],
        "Số dự án": sol["count"],
        "Dự án được chọn": ", ".join(sol["selected_codes"]),
        "Chi phí 5 năm": sol["total_cost"],
        "Chi phí năm 1-2": sol["total_cost_year_1_2"],
        "Chi phí năm 3-5": sol["total_cost_year_3_5"],
        "Tổng NPV": sol["total_benefit"],
        "E[NPV] có rủi ro": sol["total_expected_benefit"],
        "Giá trị mục tiêu": sol["objective_value"],
        "NPV/Chi phí": sol["npv_per_cost"],
        "E[NPV]/Chi phí": sol["expected_per_cost"],
    }


def make_download_excel(sheets):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df_sheet in sheets.items():
            df_sheet.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return output.getvalue()


def df_to_html_table(df, index=False):
    if df is None or len(df) == 0:
        return "<p>Không có dữ liệu.</p>"
    return df.to_html(index=index, classes="dataframe", border=1)


def make_html_report(project_df, scenario_df, selected_df, risk_df, policy_html, checklist_df):
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Bài 5 - MIP lựa chọn dự án chuyển đổi số</title>
        <style>
            body {{font-family: Arial, sans-serif; line-height: 1.5; margin: 30px;}}
            h1, h2, h3 {{color: #1f3b66;}}
            table {{border-collapse: collapse; width: 100%; margin-bottom: 25px; font-size: 13px;}}
            th {{background-color: #1f3b66; color: white; padding: 6px; border: 1px solid #ccc;}}
            td {{padding: 6px; border: 1px solid #ccc; text-align: left;}}
            .box {{background: #f2f6ff; padding: 15px; border-left: 5px solid #1f3b66; margin-bottom: 20px;}}
        </style>
    </head>
    <body>
        <h1>BÀI 5 - QUY HOẠCH NGUYÊN HỖN HỢP LỰA CHỌN DỰ ÁN CHUYỂN ĐỔI SỐ</h1>
        <div class="box">
            <p><b>Mô hình:</b> MIP với biến nhị phân y_i ∈ {{0,1}}</p>
            <p><b>Solver:</b> PuLP/CBC nếu đã cài PuLP; nếu chưa cài thì dùng enumeration fallback để kiểm chứng kết quả.</p>
        </div>
        <h2>Danh mục dự án</h2>
        {df_to_html_table(project_df.round(4), index=False)}
        <h2>Kết quả kịch bản</h2>
        {df_to_html_table(scenario_df.round(4), index=False)}
        <h2>Dự án được chọn trong kịch bản cơ sở</h2>
        {df_to_html_table(selected_df.round(4), index=False)}
        <h2>Kịch bản rủi ro</h2>
        {df_to_html_table(risk_df.round(4), index=False)}
        <h2>Phân tích chính sách</h2>
        {policy_html}
        <h2>Checklist hoàn thành yêu cầu</h2>
        {df_to_html_table(checklist_df, index=False)}
    </body>
    </html>
    """
    return html

# ============================================================
# 3. MÔ HÌNH TOÁN HỌC
# ============================================================

st.header("2. Mô hình toán học")

col_m1, col_m2 = st.columns(2)
with col_m1:
    card(
        "Hàm mục tiêu",
        """
        Tối đa hóa tổng lợi ích NPV:
        <br><br>
        <b>max Z = Σ Bᵢ yᵢ</b>
        <br><br>
        Trong đó yᵢ = 1 nếu chọn dự án i, yᵢ = 0 nếu không chọn.
        """
    )
with col_m2:
    card(
        "Các nhóm ràng buộc",
        """
        C1: Tổng ngân sách 5 năm ≤ 80.000 tỷ<br>
        C2: Ngân sách năm 1-2 ≤ 40.000 tỷ<br>
        C3: P1 + P2 ≤ 1<br>
        C4-C5: P8, P13 cần P12<br>
        C6: Ít nhất một chính phủ số và bắt buộc P14<br>
        C7: 7 ≤ số dự án ≤ 11
        """
    )

if PULP_AVAILABLE:
    st.success("Đã phát hiện thư viện PuLP. Mô hình sẽ được giải bằng PuLP/CBC đúng yêu cầu đề bài.")
else:
    st.warning(
        "Chưa cài PuLP nên webapp đang dùng thuật toán liệt kê 2^15 tổ hợp để kiểm chứng kết quả. "
        "Để đúng 100% yêu cầu 'giải bằng PuLP với CBC', hãy chạy: python -m pip install pulp"
    )

# ============================================================
# 4. CÂU 5.4.1 - GIẢI CƠ SỞ
# ============================================================

st.header("3. Câu 5.4.1 - Giải MIP bằng PuLP/CBC")

base_sol = solve_with_pulp(
    df_projects,
    total_budget=80000,
    early_budget=40000,
    objective_mode="benefit",
    force_both_centers=False,
    keep_exclusion=True,
    require_p14=True,
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Trạng thái", base_sol["status"])
col2.metric("Tổng NPV Z*", f"{base_sol['total_benefit']:,.0f} tỷ")
col3.metric("Tổng chi phí", f"{base_sol['total_cost']:,.0f} tỷ")
col4.metric("NPV/Chi phí", f"{base_sol['npv_per_cost']:.4f}")

st.subheader("Danh sách dự án được chọn")
base_selected_df = base_sol["selected_df"][[
    "code", "project_name", "field", "cost", "benefit", "cost_year_1_2", "cost_year_3_5", "npv_per_cost"
]].copy()
st.dataframe(base_selected_df.round(4), width="stretch")

budget_usage_df = pd.DataFrame({
    "Khoản mục": ["Ngân sách 5 năm", "Ngân sách năm 1-2", "Ngân sách năm 3-5"],
    "Đã dùng": [base_sol["total_cost"], base_sol["total_cost_year_1_2"], base_sol["total_cost_year_3_5"]],
    "Giới hạn": [80000, 40000, np.nan]
})
budget_usage_df["Tỷ lệ sử dụng (%)"] = budget_usage_df["Đã dùng"] / budget_usage_df["Giới hạn"] * 100

st.subheader("Kiểm tra sử dụng ngân sách")
st.dataframe(budget_usage_df.round(4), width="stretch")

st.markdown(f"""
**Nhận xét nhanh:** Nghiệm tối ưu cơ sở chọn **{base_sol['count']} dự án** với tổng chi phí
**{base_sol['total_cost']:,.0f} tỷ VND**, trong đó ngân sách năm 1-2 dùng **{base_sol['total_cost_year_1_2']:,.0f} tỷ VND**.
Tổng lợi ích NPV đạt **{base_sol['total_benefit']:,.0f} tỷ VND**.
""")

# Biểu đồ lợi ích - chi phí dự án được chọn
st.subheader("Biểu đồ chi phí và lợi ích của các dự án được chọn")
fig1, ax1 = plt.subplots(figsize=(10, 5))
if len(base_selected_df) > 0:
    x_pos = np.arange(len(base_selected_df))
    width = 0.35
    ax1.bar(x_pos - width / 2, base_selected_df["cost"], width, label="Chi phí")
    ax1.bar(x_pos + width / 2, base_selected_df["benefit"], width, label="Lợi ích NPV")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(base_selected_df["code"], rotation=0)
    ax1.set_xlabel("Dự án")
    ax1.set_ylabel("Tỷ VND")
    ax1.set_title("Chi phí và lợi ích NPV của danh mục được chọn")
    ax1.legend()
    ax1.grid(axis="y")
st.pyplot(fig1)

# ============================================================
# 5. CÂU 5.4.2 - NỚI NGÂN SÁCH LÊN 100.000 TỶ
# ============================================================

st.header("4. Câu 5.4.2 - Nới ngân sách tổng lên 100.000 tỷ")

budget_100_sol = solve_with_pulp(
    df_projects,
    total_budget=100000,
    early_budget=40000,
    objective_mode="benefit",
    force_both_centers=False,
    keep_exclusion=True,
    require_p14=True,
)

scenario_base_100 = pd.DataFrame([
    solution_to_row("Cơ sở B=80.000", base_sol),
    solution_to_row("Nới tổng ngân sách B=100.000", budget_100_sol),
])

st.dataframe(scenario_base_100.round(4), width="stretch")

added_100 = sorted(set(budget_100_sol["selected_codes"]) - set(base_sol["selected_codes"]))
removed_100 = sorted(set(base_sol["selected_codes"]) - set(budget_100_sol["selected_codes"]))

if added_100 or removed_100:
    st.info(f"Khi nới ngân sách lên 100.000 tỷ: thêm {added_100}, bỏ {removed_100}.")
else:
    st.info(
        "Khi nới ngân sách tổng lên 100.000 tỷ, danh mục tối ưu không đổi. "
        "Nguyên nhân là ràng buộc ngân sách năm 1-2 vẫn giữ ở mức 40.000 tỷ và đang gần như chạm trần."
    )

# ============================================================
# 6. CÂU 5.4.3 - BẮT BUỘC CẢ P1 VÀ P2
# ============================================================

st.header("5. Câu 5.4.3 - Yêu cầu phải có cả P1 và P2")

st.markdown("""
Trong mô hình gốc có ràng buộc loại trừ **P1 + P2 ≤ 1**.  
Vì vậy, nếu vừa giữ ràng buộc loại trừ vừa bắt buộc **P1 = 1** và **P2 = 1** thì bài toán chắc chắn **không khả thi**.

Để kiểm tra ý nghĩa chính sách của yêu cầu redundancy, code chạy thêm một kịch bản hợp lý hơn: **thay C3 bằng yêu cầu chọn cả P1 và P2**.
""")

redundancy_impossible = solve_with_pulp(
    df_projects,
    total_budget=80000,
    early_budget=40000,
    objective_mode="benefit",
    force_both_centers=True,
    keep_exclusion=True,
    require_p14=True,
)

redundancy_sol = solve_with_pulp(
    df_projects,
    total_budget=80000,
    early_budget=40000,
    objective_mode="benefit",
    force_both_centers=True,
    keep_exclusion=False,
    require_p14=True,
)

redundancy_df = pd.DataFrame([
    solution_to_row("Giữ C3 và ép P1=P2=1", redundancy_impossible),
    solution_to_row("Thay C3 bằng P1=P2=1", redundancy_sol),
])
st.dataframe(redundancy_df.round(4), width="stretch")

if redundancy_sol["status"] == "Optimal":
    delta_redundancy = redundancy_sol["total_benefit"] - base_sol["total_benefit"]
    pct_delta_redundancy = delta_redundancy / base_sol["total_benefit"] * 100
    st.markdown(f"""
    Nếu Quốc hội sửa ràng buộc C3 để cho phép redundancy và bắt buộc chọn cả P1, P2,
    bài toán **vẫn khả thi**. Tổng NPV thay đổi từ **{base_sol['total_benefit']:,.0f}** xuống
    **{redundancy_sol['total_benefit']:,.0f} tỷ VND**, tức thay đổi **{delta_redundancy:,.0f} tỷ VND**
    (**{pct_delta_redundancy:.2f}%**).
    """)

# ============================================================
# 7. CÂU 5.4.4 - RỦI RO DỰ ÁN
# ============================================================

st.header("6. Câu 5.4.4 - Mở rộng: tối đa hóa lợi ích kỳ vọng có rủi ro")

risk_sol = solve_with_pulp(
    df_projects,
    total_budget=80000,
    early_budget=40000,
    objective_mode="expected",
    force_both_centers=False,
    keep_exclusion=True,
    require_p14=True,
)

risk_selected_df = risk_sol["selected_df"][[
    "code", "project_name", "field", "cost", "benefit", "completion_prob", "expected_benefit", "expected_npv_per_cost"
]].copy()

risk_compare_df = pd.DataFrame([
    solution_to_row("Tối đa hóa NPV danh nghĩa", base_sol),
    solution_to_row("Tối đa hóa E[NPV] có rủi ro", risk_sol),
])
st.dataframe(risk_compare_df.round(4), width="stretch")

st.subheader("Danh mục tối ưu khi xét rủi ro")
st.dataframe(risk_selected_df.round(4), width="stretch")

added_risk = sorted(set(risk_sol["selected_codes"]) - set(base_sol["selected_codes"]))
removed_risk = sorted(set(base_sol["selected_codes"]) - set(risk_sol["selected_codes"]))
st.markdown(f"""
Khi xét xác suất hoàn thành, mô hình chuyển từ tối đa hóa **Z** sang tối đa hóa **E[Z] = Σ pᵢBᵢyᵢ**.
So với danh mục cơ sở, danh mục rủi ro thêm **{added_risk if added_risk else 'không có'}** và bỏ
**{removed_risk if removed_risk else 'không có'}**.
""")

# Biểu đồ so sánh kịch bản
st.subheader("Biểu đồ so sánh các kịch bản chính")
scenario_plot_df = pd.DataFrame([
    {"Kịch bản": "Cơ sở", "Tổng NPV": base_sol["total_benefit"], "E[NPV]": base_sol["total_expected_benefit"], "Chi phí": base_sol["total_cost"]},
    {"Kịch bản": "B=100.000", "Tổng NPV": budget_100_sol["total_benefit"], "E[NPV]": budget_100_sol["total_expected_benefit"], "Chi phí": budget_100_sol["total_cost"]},
    {"Kịch bản": "P1 & P2", "Tổng NPV": redundancy_sol["total_benefit"], "E[NPV]": redundancy_sol["total_expected_benefit"], "Chi phí": redundancy_sol["total_cost"]},
    {"Kịch bản": "Có rủi ro", "Tổng NPV": risk_sol["total_benefit"], "E[NPV]": risk_sol["total_expected_benefit"], "Chi phí": risk_sol["total_cost"]},
])

fig2, ax2 = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(scenario_plot_df))
width = 0.28
ax2.bar(x_pos - width, scenario_plot_df["Tổng NPV"], width, label="Tổng NPV")
ax2.bar(x_pos, scenario_plot_df["E[NPV]"], width, label="E[NPV]")
ax2.bar(x_pos + width, scenario_plot_df["Chi phí"], width, label="Chi phí")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(scenario_plot_df["Kịch bản"], rotation=0)
ax2.set_ylabel("Tỷ VND")
ax2.set_title("So sánh NPV, E[NPV] và chi phí theo kịch bản")
ax2.legend()
ax2.grid(axis="y")
st.pyplot(fig2)

# ============================================================
# 8. PHÂN TÍCH BỔ SUNG: RÀNG BUỘC P14
# ============================================================

st.header("7. Phân tích bổ sung - Tác động của ràng buộc bắt buộc P14")

no_p14_sol = solve_with_pulp(
    df_projects,
    total_budget=80000,
    early_budget=40000,
    objective_mode="benefit",
    force_both_centers=False,
    keep_exclusion=True,
    require_p14=False,
)

p14_compare_df = pd.DataFrame([
    solution_to_row("Có bắt buộc P14", base_sol),
    solution_to_row("Không bắt buộc P14", no_p14_sol),
])
st.dataframe(p14_compare_df.round(4), width="stretch")

p14_cost = no_p14_sol["total_benefit"] - base_sol["total_benefit"]
st.markdown(f"""
Nếu bỏ ràng buộc bắt buộc P14, tổng NPV tối đa tăng thêm khoảng **{p14_cost:,.0f} tỷ VND**.
Về mặt tối ưu thuần túy, đây là chi phí cơ hội của yêu cầu an ninh mạng. Tuy nhiên, trong chính sách công,
P14 có vai trò phòng ngừa rủi ro hệ thống nên việc bắt buộc có thể hợp lý dù làm giảm mục tiêu NPV.
""")

# ============================================================
# 9. CÂU 5.5 - THẢO LUẬN CHÍNH SÁCH
# ============================================================

st.header("8. Câu 5.5 - Thảo luận chính sách")

p15_selected = 15 in base_sol["selected_ids"]
if p15_selected:
    p15_text = """
    <p>
    Với bộ dữ liệu và ràng buộc đúng như đề bài, mô hình <b>không bỏ qua P15</b> mà chọn P15 trong danh mục tối ưu.
    Điều này khác với giả định trong câu hỏi 5.5(a). Lý do là P15 có chi phí rất thấp, tỷ suất NPV/chi phí cao nhất
    trong danh mục và cũng tiêu tốn ít ngân sách năm 1-2, nên dễ được chọn khi số lượng dự án bị giới hạn từ 7 đến 11.
    </p>
    <p>
    Nếu trong một cấu hình khác P15 bị bỏ qua, nguyên nhân có thể là ràng buộc số lượng dự án hoặc các ràng buộc
    tiên quyết làm mô hình ưu tiên các dự án nền tảng lớn hơn. Nhưng theo kết quả hiện tại, việc chọn P15 là hợp lý
    vì dữ liệu mở có chi phí thấp nhưng hỗ trợ nhiều module chính phủ số, AI và đổi mới sáng tạo.
    </p>
    """
else:
    p15_text = """
    <p>
    Mô hình có thể bỏ qua P15 dù tỷ suất lợi ích/chi phí cao vì hàm mục tiêu tối đa hóa tổng NPV tuyệt đối,
    không tối đa hóa tỷ suất. Ngoài ra, ràng buộc số lượng dự án và ngân sách năm 1-2 có thể khiến mô hình
    ưu tiên các dự án có NPV tuyệt đối lớn hơn hoặc có quan hệ tiên quyết với các dự án chiến lược.
    </p>
    """

policy_html = f"""
<h3>a) Vì sao mô hình bỏ qua P15 dù tỷ suất lợi ích/chi phí rất cao?</h3>
{p15_text}

<h3>b) Ràng buộc bắt buộc P14 có làm giảm Z* không? Việc bắt buộc này có hợp lý không?</h3>
<p>
Có. Khi bỏ yêu cầu bắt buộc P14, tổng NPV tối ưu tăng thêm khoảng <b>{p14_cost:,.0f} tỷ VND</b>.
Như vậy, bắt buộc P14 tạo ra một chi phí cơ hội về mặt NPV. Tuy nhiên, an ninh mạng là điều kiện nền tảng
cho chuyển đổi số quốc gia. Nếu thiếu P14, các dự án dữ liệu, định danh điện tử, AI và chính phủ số có thể tăng
rủi ro an toàn thông tin. Do đó, trong quản lý công, việc bắt buộc P14 là hợp lý dù không tối ưu hóa tuyệt đối NPV ngắn hạn.
</p>

<h3>c) Làm thế nào mô hình hóa cộng hưởng giữa P8 và P13?</h3>
<p>
Mô hình hiện tại giả định lợi ích các dự án độc lập, tức tổng lợi ích chỉ là ΣBᵢyᵢ. Để mô hình hóa cộng hưởng giữa
P8 và P13, có thể thêm biến nhị phân phụ <b>z₈₁₃</b> với các ràng buộc:
</p>
<ul>
    <li>z₈₁₃ ≤ y₈</li>
    <li>z₈₁₃ ≤ y₁₃</li>
    <li>z₈₁₃ ≥ y₈ + y₁₃ − 1</li>
</ul>
<p>
Sau đó thêm vào hàm mục tiêu một khoản cộng hưởng <b>η·z₈₁₃</b>. Khi cả P8 và P13 cùng được chọn,
z₈₁₃ = 1 và mô hình ghi nhận lợi ích bổ sung từ hệ sinh thái AI - bán dẫn. Đây là cách tuyến tính hóa hiệu ứng tương tác
mà vẫn giữ bài toán ở dạng MIP.
</p>
"""

st.markdown(policy_html, unsafe_allow_html=True)

# ============================================================
# 10. CHECKLIST
# ============================================================

st.header("9. Checklist hoàn thành yêu cầu Bài 5")

checklist_rows = []
checklist_rows.append(["5.4.1 Cài đặt MIP bằng PuLP với CBC", "Đã làm" if PULP_AVAILABLE else "Có fallback; cần cài PuLP để đúng yêu cầu", "Chạy solver PuLP/CBC nếu có thư viện pulp"])
checklist_rows.append(["5.4.1 Báo cáo dự án được chọn", "Đã làm", f"Danh mục cơ sở: {', '.join(base_sol['selected_codes'])}"])
checklist_rows.append(["5.4.1 Tổng chi phí", "Đã làm", f"{base_sol['total_cost']:,.0f} tỷ VND"])
checklist_rows.append(["5.4.1 Tổng lợi ích", "Đã làm", f"{base_sol['total_benefit']:,.0f} tỷ VND"])
checklist_rows.append(["5.4.1 NPV biên Z*/chi phí", "Đã làm", f"{base_sol['npv_per_cost']:.4f}"])
checklist_rows.append(["5.4.2 Nới ngân sách 100.000 tỷ", "Đã làm", "Có bảng so sánh danh mục"])
checklist_rows.append(["5.4.3 Bắt buộc P1 và P2", "Đã làm", "Kiểm tra cả trường hợp giữ C3 và thay C3 bằng redundancy"])
checklist_rows.append(["5.4.4 Thêm rủi ro p_i", "Đã làm", "Tối đa hóa E[Z] = ΣpᵢBᵢyᵢ"])
checklist_rows.append(["5.5 Trả lời chính sách a, b, c", "Đã làm", "Có phân tích P15, P14 và cộng hưởng P8-P13"])
checklist_rows.append(["Trực quan hóa", "Đã làm", "Có biểu đồ chi phí - lợi ích và biểu đồ so sánh kịch bản"])
checklist_rows.append(["Tải kết quả", "Đã làm", "HTML, Excel, CSV"])

checklist_df = pd.DataFrame(checklist_rows, columns=["Yêu cầu", "Trạng thái", "Ghi chú"])
st.dataframe(checklist_df, width="stretch")

# ============================================================
# 11. TẢI KẾT QUẢ
# ============================================================

st.header("10. Tải kết quả")

scenario_all_df = pd.DataFrame([
    solution_to_row("Cơ sở B=80.000", base_sol),
    solution_to_row("Nới B=100.000", budget_100_sol),
    solution_to_row("Giữ C3 và ép P1=P2=1", redundancy_impossible),
    solution_to_row("Thay C3 bằng P1=P2=1", redundancy_sol),
    solution_to_row("Có rủi ro E[NPV]", risk_sol),
    solution_to_row("Không bắt buộc P14", no_p14_sol),
])

html_report = make_html_report(
    project_df=df_projects,
    scenario_df=scenario_all_df,
    selected_df=base_selected_df,
    risk_df=risk_selected_df,
    policy_html=policy_html,
    checklist_df=checklist_df,
)

# Lưu outputs ra thư mục outputs
try:
    df_projects.to_csv(OUTPUT_DIR / "bai05_project_data.csv", index=False, encoding="utf-8-sig")
    base_selected_df.to_csv(OUTPUT_DIR / "bai05_base_selected_projects.csv", index=False, encoding="utf-8-sig")
    scenario_all_df.to_csv(OUTPUT_DIR / "bai05_scenario_comparison.csv", index=False, encoding="utf-8-sig")
    risk_selected_df.to_csv(OUTPUT_DIR / "bai05_risk_selected_projects.csv", index=False, encoding="utf-8-sig")
    with open(OUTPUT_DIR / "bai05_report.html", "w", encoding="utf-8") as f:
        f.write(html_report)
except Exception as e:
    st.warning(f"Không lưu được file vào thư mục outputs: {e}")

excel_file = make_download_excel({
    "Project_Data": df_projects,
    "Base_Selected": base_selected_df,
    "Scenario_Comparison": scenario_all_df,
    "Risk_Selected": risk_selected_df,
    "Checklist": checklist_df,
})

col_dl1, col_dl2, col_dl3 = st.columns(3)
with col_dl1:
    st.download_button(
        label="Tải HTML report",
        data=html_report.encode("utf-8"),
        file_name="bai05_report.html",
        mime="text/html"
    )
with col_dl2:
    st.download_button(
        label="Tải Excel tổng hợp",
        data=excel_file,
        file_name="bai05_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
with col_dl3:
    st.download_button(
        label="Tải CSV danh mục cơ sở",
        data=base_selected_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="bai05_base_selected_projects.csv",
        mime="text/csv"
    )

st.success("Bài 5 đã hoàn thành các yêu cầu: MIP, PuLP/CBC, ngân sách 100.000, redundancy P1-P2, rủi ro dự án và phân tích chính sách.")
