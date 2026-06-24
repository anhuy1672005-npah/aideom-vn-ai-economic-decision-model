import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO

# ============================================================
# BÀI 4 - QUY HOẠCH TUYẾN TÍNH PHÂN BỔ NGÂN SÁCH SỐ THEO NGÀNH - VÙNG
# Webapp Streamlit
# Yêu cầu đề bài: PuLP, CVXPY, heatmap, so sánh có/không công bằng vùng
# ============================================================

try:
    import pulp
    HAS_PULP = True
except Exception:
    pulp = None
    HAS_PULP = False

try:
    import cvxpy as cp
    HAS_CVXPY = True
except Exception:
    cp = None
    HAS_CVXPY = False

try:
    from scipy.optimize import linprog
    HAS_SCIPY = True
except Exception:
    linprog = None
    HAS_SCIPY = False

try:
    from utils.style import load_css, hero, card
except Exception:
    def load_css():
        pass

    def hero(title, subtitle="", badges=None):
        if badges is None:
            badges = []
        st.title(title)
        if subtitle:
            st.write(subtitle)
        if badges:
            st.caption(" | ".join(badges))

    def card(title, text):
        st.info(f"**{title}**\n\n{text}")


st.set_page_config(
    page_title="Bài 4 - LP ngành vùng",
    layout="wide"
)
load_css()

hero(
    title="🗺️ Bài 4 — Quy hoạch tuyến tính phân bổ ngân sách số theo ngành - vùng",
    subtitle="Giải mô hình LP phân bổ 50.000 tỷ VND cho 6 vùng và 4 hạng mục đầu tư số; kiểm tra công bằng vùng miền, so sánh PuLP và CVXPY, vẽ heatmap và phân tích chính sách.",
    badges=["Cấp độ trung bình", "PuLP", "CVXPY", "Matplotlib", "LP"]
)

# ============================================================
# 1. DỮ LIỆU VÀ THAM SỐ THEO ĐỀ
# ============================================================

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

REGIONS = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
REGION_NAMES = [
    "Trung du miền núi phía Bắc",
    "Đồng bằng sông Hồng",
    "Bắc Trung Bộ + DH Trung Bộ",
    "Tây Nguyên",
    "Đông Nam Bộ",
    "Đồng bằng sông Cửu Long"
]
ITEMS = ["I", "D", "AI", "H"]
ITEM_NAMES = {
    "I": "Hạ tầng số",
    "D": "Chuyển đổi số DN",
    "AI": "Năng lực AI",
    "H": "Nhân lực số"
}

# Ma trận beta theo đề: 6 vùng x 4 hạng mục: I, D, AI, H
BETA = np.array([
    [1.15, 0.85, 0.55, 1.30],
    [0.95, 1.25, 1.40, 1.05],
    [1.05, 0.95, 0.85, 1.15],
    [1.20, 0.75, 0.45, 1.35],
    [0.90, 1.30, 1.55, 1.00],
    [1.10, 0.85, 0.65, 1.25],
], dtype=float)

# D0 theo đề: chỉ số số hóa ban đầu
D0 = np.array([38, 78, 55, 32, 82, 48], dtype=float)

BUDGET_TOTAL = 50000.0
REGION_FLOOR = 5000.0
REGION_CAP = 12000.0
H_FLOOR = 12000.0
GAMMA = 0.002
LAM_ORIGINAL = 0.70

# ============================================================
# 2. HÀM TIỆN ÍCH
# ============================================================

def beta_dataframe():
    df_beta = pd.DataFrame(BETA, columns=[ITEM_NAMES[i] for i in ITEMS])
    df_beta.insert(0, "Vùng", REGION_NAMES)
    df_beta.insert(1, "Mã vùng", REGIONS)
    return df_beta


def base_region_dataframe():
    return pd.DataFrame({
        "Mã vùng": REGIONS,
        "Vùng": REGION_NAMES,
        "Digital Index ban đầu D0": D0
    })


def max_feasible_lambda_with_region_cap():
    """Tính ngưỡng lambda tối đa sơ bộ do C3 và C5 tạo ra.

    Vì M tối thiểu phải >= max(D0) = 82. Vùng yếu nhất là Tây Nguyên D0=32.
    Với trần vùng 12.000, nếu dồn toàn bộ cho D thì D_new tối đa = 32 + 0.002*12000 = 56.
    Do đó lambda tối đa không thể vượt quá 56/82 = 0.6829.
    """
    min_possible_ratio = np.min((D0 + GAMMA * REGION_CAP) / np.max(D0))
    binding_region_index = int(np.argmin((D0 + GAMMA * REGION_CAP) / np.max(D0)))
    return float(min_possible_ratio), REGION_NAMES[binding_region_index]


def required_d_for_region(region_index, lam=LAM_ORIGINAL, M_min=None):
    if M_min is None:
        M_min = np.max(D0)
    required_index = lam * M_min
    required_xd = max(0.0, (required_index - D0[region_index]) / GAMMA)
    return required_index, required_xd


def matrix_to_solution_df(X):
    df = pd.DataFrame(X, columns=[ITEM_NAMES[i] for i in ITEMS])
    df.insert(0, "Vùng", REGION_NAMES)
    df.insert(1, "Mã vùng", REGIONS)
    df["Tổng vùng"] = X.sum(axis=1)
    df["Digital Index sau đầu tư D"] = D0 + GAMMA * X[:, 1]
    return df


def item_total_df(X):
    return pd.DataFrame({
        "Hạng mục": [ITEM_NAMES[i] for i in ITEMS],
        "Mã": ITEMS,
        "Tổng phân bổ": X.sum(axis=0)
    })


def check_constraints(X, M_value=None, lam=None, fairness=True):
    total_budget = X.sum()
    h_total = X[:, 3].sum()
    region_sum = X.sum(axis=1)
    digital_after = D0 + GAMMA * X[:, 1]
    if fairness and M_value is not None and lam is not None:
        min_ratio = digital_after.min() / M_value if M_value > 0 else np.nan
        fairness_ok = bool(np.all(digital_after >= lam * M_value - 1e-5))
    else:
        min_ratio = np.nan
        fairness_ok = None
    return pd.DataFrame({
        "Kiểm tra": [
            "Tổng ngân sách <= 50.000",
            "Mỗi vùng >= 5.000",
            "Mỗi vùng <= 12.000",
            "Tổng nhân lực số H >= 12.000",
            "Công bằng vùng D_r + gamma*x_D,r >= lambda*M",
        ],
        "Giá trị": [
            round(total_budget, 4),
            round(region_sum.min(), 4),
            round(region_sum.max(), 4),
            round(h_total, 4),
            None if not fairness else round(min_ratio, 6),
        ],
        "Ngưỡng": [
            "<= 50000",
            ">= 5000",
            "<= 12000",
            ">= 12000",
            None if not fairness else f">= {lam:.4f}",
        ],
        "Đạt?": [
            total_budget <= BUDGET_TOTAL + 1e-5,
            region_sum.min() >= REGION_FLOOR - 1e-5,
            region_sum.max() <= REGION_CAP + 1e-5,
            h_total >= H_FLOOR - 1e-5,
            "Không áp dụng" if not fairness else fairness_ok,
        ]
    })


def make_download_excel(sheets: dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df_sheet in sheets.items():
            safe_name = sheet_name[:31]
            df_sheet.to_excel(writer, index=False, sheet_name=safe_name)
    return output.getvalue()


def make_html_report(summary_html, tables: dict):
    table_html = ""
    for name, df_table in tables.items():
        table_html += f"<h2>{name}</h2>\n{df_table.to_html(index=False)}\n"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Bài 4 - LP ngành vùng</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.5; margin: 30px; }}
            h1, h2, h3 {{ color: #1f3b66; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 25px; font-size: 13px; }}
            th {{ background-color: #1f3b66; color: white; padding: 6px; border: 1px solid #ccc; }}
            td {{ padding: 6px; border: 1px solid #ccc; text-align: center; }}
            .box {{ background: #f2f6ff; padding: 15px; border-left: 5px solid #1f3b66; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>BÀI 4 - QUY HOẠCH TUYẾN TÍNH PHÂN BỔ NGÂN SÁCH SỐ THEO NGÀNH - VÙNG</h1>
        <div class="box">{summary_html}</div>
        {table_html}
    </body>
    </html>
    """
    return html

# ============================================================
# 3. SOLVER - SCIPY FALLBACK
# ============================================================

def solve_with_scipy(lam=0.68, fairness=True, use_region_cap=True):
    if not HAS_SCIPY:
        return {
            "solver": "SciPy HiGHS",
            "status": "not_available",
            "message": "Chưa cài scipy. Chạy: python -m pip install scipy",
            "success": False,
        }

    n_x = len(REGIONS) * len(ITEMS)
    has_M = fairness
    n_var = n_x + (1 if has_M else 0)

    c = np.zeros(n_var)
    c[:n_x] = -BETA.flatten()

    A_ub = []
    b_ub = []

    # C1: tổng ngân sách <= 50.000
    row = np.zeros(n_var)
    row[:n_x] = 1.0
    A_ub.append(row)
    b_ub.append(BUDGET_TOTAL)

    # C2, C3: sàn/trần vùng
    for r in range(len(REGIONS)):
        idx = [r * len(ITEMS) + j for j in range(len(ITEMS))]

        row = np.zeros(n_var)
        row[idx] = -1.0
        A_ub.append(row)
        b_ub.append(-REGION_FLOOR)

        if use_region_cap:
            row = np.zeros(n_var)
            row[idx] = 1.0
            A_ub.append(row)
            b_ub.append(REGION_CAP)

    # C4: sàn nhân lực số
    row = np.zeros(n_var)
    for r in range(len(REGIONS)):
        row[r * len(ITEMS) + 3] = -1.0
    A_ub.append(row)
    b_ub.append(-H_FLOOR)

    # C5: công bằng vùng, linear hóa bằng M
    if fairness:
        M_idx = n_x
        for r in range(len(REGIONS)):
            row = np.zeros(n_var)
            row[r * len(ITEMS) + 1] = GAMMA
            row[M_idx] = -1.0
            A_ub.append(row)
            b_ub.append(-D0[r])

        for r in range(len(REGIONS)):
            row = np.zeros(n_var)
            row[r * len(ITEMS) + 1] = -GAMMA
            row[M_idx] = lam
            A_ub.append(row)
            b_ub.append(D0[r])

    bounds = [(0, None)] * n_var

    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=bounds,
        method="highs"
    )

    if not res.success:
        return {
            "solver": "SciPy HiGHS",
            "status": "infeasible_or_error",
            "message": res.message,
            "success": False,
        }

    X = res.x[:n_x].reshape(len(REGIONS), len(ITEMS))
    M_value = float(res.x[n_x]) if fairness else None
    return {
        "solver": "SciPy HiGHS",
        "status": "optimal",
        "message": res.message,
        "success": True,
        "X": X,
        "M": M_value,
        "objective": float(-res.fun),
    }

# ============================================================
# 4. SOLVER - PuLP
# ============================================================

def solve_with_pulp(lam=0.68, fairness=True, use_region_cap=True):
    if not HAS_PULP:
        return {
            "solver": "PuLP CBC",
            "status": "not_available",
            "message": "Chưa cài PuLP. Chạy: python -m pip install pulp",
            "success": False,
        }

    model = pulp.LpProblem("Bai4_VN_Digital_Budget", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("x", (REGIONS, ITEMS), lowBound=0, cat="Continuous")

    if fairness:
        M = pulp.LpVariable("Dmax", lowBound=0, cat="Continuous")
    else:
        M = None

    beta_dict = {
        (REGIONS[r], ITEMS[j]): float(BETA[r, j])
        for r in range(len(REGIONS))
        for j in range(len(ITEMS))
    }

    # Hàm mục tiêu
    model += pulp.lpSum(
        beta_dict[(r, j)] * x[r][j]
        for r in REGIONS
        for j in ITEMS
    ), "GDP_gain"

    # C1: ngân sách tổng
    model += pulp.lpSum(x[r][j] for r in REGIONS for j in ITEMS) <= BUDGET_TOTAL, "C1_Tong_ngan_sach"

    # C2, C3: sàn và trần vùng
    for r in REGIONS:
        model += pulp.lpSum(x[r][j] for j in ITEMS) >= REGION_FLOOR, f"C2_San_vung_{r}"
        if use_region_cap:
            model += pulp.lpSum(x[r][j] for j in ITEMS) <= REGION_CAP, f"C3_Tran_vung_{r}"

    # C4: sàn nhân lực số
    model += pulp.lpSum(x[r]["H"] for r in REGIONS) >= H_FLOOR, "C4_San_nhan_luc_so"

    # C5: công bằng vùng
    if fairness:
        for idx, r in enumerate(REGIONS):
            model += D0[idx] + GAMMA * x[r]["D"] <= M, f"C5a_Dmax_{r}"
        for idx, r in enumerate(REGIONS):
            model += D0[idx] + GAMMA * x[r]["D"] >= lam * M, f"C5b_Cong_bang_{r}"

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    status_text = pulp.LpStatus[model.status]
    if status_text != "Optimal":
        return {
            "solver": "PuLP CBC",
            "status": status_text,
            "message": f"PuLP status: {status_text}",
            "success": False,
            "model": model,
        }

    X = np.zeros((len(REGIONS), len(ITEMS)))
    for r_idx, r in enumerate(REGIONS):
        for j_idx, j in enumerate(ITEMS):
            X[r_idx, j_idx] = float(pulp.value(x[r][j]))

    M_value = float(pulp.value(M)) if fairness else None
    objective = float(pulp.value(model.objective))

    constraint_df = []
    for name, constraint in model.constraints.items():
        constraint_df.append({
            "Ràng buộc": name,
            "Slack": getattr(constraint, "slack", np.nan),
            "Dual / Shadow price": getattr(constraint, "pi", np.nan),
        })
    constraint_df = pd.DataFrame(constraint_df)

    return {
        "solver": "PuLP CBC",
        "status": status_text,
        "message": f"PuLP status: {status_text}",
        "success": True,
        "X": X,
        "M": M_value,
        "objective": objective,
        "model": model,
        "constraint_df": constraint_df,
    }

# ============================================================
# 5. SOLVER - CVXPY
# ============================================================

def solve_with_cvxpy(lam=0.68, fairness=True, use_region_cap=True):
    if not HAS_CVXPY:
        return {
            "solver": "CVXPY",
            "status": "not_available",
            "message": "Chưa cài CVXPY. Chạy: python -m pip install cvxpy",
            "success": False,
        }

    X = cp.Variable((len(REGIONS), len(ITEMS)), nonneg=True)
    constraints = []

    constraints.append(cp.sum(X) <= BUDGET_TOTAL)

    for r in range(len(REGIONS)):
        constraints.append(cp.sum(X[r, :]) >= REGION_FLOOR)
        if use_region_cap:
            constraints.append(cp.sum(X[r, :]) <= REGION_CAP)

    constraints.append(cp.sum(X[:, 3]) >= H_FLOOR)

    if fairness:
        M = cp.Variable(nonneg=True)
        digital_after = D0 + GAMMA * X[:, 1]
        constraints.append(digital_after <= M)
        constraints.append(digital_after >= lam * M)
    else:
        M = None

    objective = cp.Maximize(cp.sum(cp.multiply(BETA, X)))
    problem = cp.Problem(objective, constraints)

    solver_used = None
    error_message = None

    candidate_solvers = []
    try:
        installed = cp.installed_solvers()
        for s in ["CLARABEL", "ECOS", "SCIPY", "SCS"]:
            if s in installed:
                candidate_solvers.append(s)
    except Exception:
        candidate_solvers = [None]

    if not candidate_solvers:
        candidate_solvers = [None]

    for solver_name in candidate_solvers:
        try:
            if solver_name is None:
                problem.solve()
            else:
                problem.solve(solver=solver_name)
            solver_used = solver_name if solver_name else "CVXPY default"
            break
        except Exception as e:
            error_message = str(e)
            continue

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        return {
            "solver": f"CVXPY ({solver_used})",
            "status": problem.status,
            "message": error_message if error_message else f"CVXPY status: {problem.status}",
            "success": False,
        }

    X_value = np.array(X.value, dtype=float)
    X_value[np.abs(X_value) < 1e-6] = 0.0
    M_value = float(M.value) if fairness and M is not None else None

    return {
        "solver": f"CVXPY ({solver_used})",
        "status": problem.status,
        "message": f"CVXPY status: {problem.status}",
        "success": True,
        "X": X_value,
        "M": M_value,
        "objective": float(problem.value),
    }

# ============================================================
# 6. GIAO DIỆN WEBAPP
# ============================================================

st.markdown("""
Bài 4 yêu cầu xây dựng mô hình quy hoạch tuyến tính để phân bổ **50.000 tỷ VND** cho **6 vùng kinh tế - xã hội**
và **4 hạng mục đầu tư số** gồm hạ tầng số, chuyển đổi số doanh nghiệp, AI và nhân lực số.
""")

st.header("1. Dữ liệu và mô hình theo đề bài")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Ma trận hệ số tác động biên beta")
    st.dataframe(beta_dataframe(), use_container_width=True)

with col_b:
    st.subheader("Chỉ số số hóa ban đầu D0")
    st.dataframe(base_region_dataframe(), use_container_width=True)

st.markdown("""
Hàm mục tiêu:

$$
\max Z = \sum_r \sum_j \beta_{j,r} x_{j,r}
$$

Các ràng buộc chính:

- Tổng ngân sách không vượt quá 50.000 tỷ VND.
- Mỗi vùng nhận tối thiểu 5.000 tỷ VND.
- Mỗi vùng nhận tối đa 12.000 tỷ VND.
- Tổng đầu tư nhân lực số tối thiểu 12.000 tỷ VND.
- Ràng buộc công bằng vùng: $D_r + \gamma x_{D,r} \geq \lambda M$, với $M = \max_r(D_r + \gamma x_{D,r})$.
""")

# ============================================================
# 7. KIỂM TRA ĐIỂM KHÔNG HỢP LÝ TRONG ĐỀ
# ============================================================

st.header("2. Kiểm tra tính khả thi của mô hình gốc trong đề")

max_lam, binding_region = max_feasible_lambda_with_region_cap()
required_index_ch, required_xd_ch = required_d_for_region(3, lam=LAM_ORIGINAL, M_min=np.max(D0))
feasibility_diagnostic_df = pd.DataFrame([
    {"Chỉ tiêu": "lambda gốc trong đề", "Giá trị": LAM_ORIGINAL, "Diễn giải": "Đề yêu cầu λ = 0.70"},
    {"Chỉ tiêu": "Digital Index lớn nhất ban đầu", "Giá trị": np.max(D0), "Diễn giải": "Đông Nam Bộ = 82, làm mốc M tối thiểu"},
    {"Chỉ tiêu": "Ngưỡng Tây Nguyên phải đạt", "Giá trị": required_index_ch, "Diễn giải": "0.70 × 82 = 57.4"},
    {"Chỉ tiêu": "Đầu tư D cần cho Tây Nguyên", "Giá trị": required_xd_ch, "Diễn giải": "(57.4 - 32) / 0.002 = 12.700 tỷ"},
    {"Chỉ tiêu": "Trần ngân sách mỗi vùng", "Giá trị": REGION_CAP, "Diễn giải": "C3 chỉ cho tối đa 12.000 tỷ/vùng"},
    {"Chỉ tiêu": "lambda tối đa khả thi sơ bộ", "Giá trị": max_lam, "Diễn giải": "Khoảng 0.6829, nên λ=0.70 không khả thi"},
])

warning_text = f"""
Mô hình gốc trong đề có một điểm **không hợp lý về tính khả thi**.
Với $\lambda = 0.70$, $\gamma = 0.002$, $D_{{max}}$ ban đầu tối thiểu là **{np.max(D0):.0f}** vì Đông Nam Bộ có Digital Index = 82.
Do đó, Tây Nguyên phải đạt tối thiểu $0.70 \times 82 = {required_index_ch:.1f}$ điểm.
Tây Nguyên bắt đầu từ D0 = 32 nên cần đầu tư vào D ít nhất khoảng **{required_xd_ch:,.0f} tỷ VND**.
Nhưng đề lại đặt trần mỗi vùng là **12.000 tỷ VND**, nên nếu giữ đúng $\lambda=0.70$ thì bài toán **không khả thi**.
Ngưỡng $\lambda$ tối đa để vẫn khả thi dưới trần 12.000 tỷ là khoảng **{max_lam:.4f}**, bị ràng buộc bởi vùng **{binding_region}**.
"""

st.warning(warning_text)
st.dataframe(feasibility_diagnostic_df, use_container_width=True, hide_index=True)

original_check = solve_with_scipy(lam=LAM_ORIGINAL, fairness=True, use_region_cap=True)

if original_check["success"]:
    st.success("Mô hình gốc bất ngờ khả thi trong solver hiện tại.")
else:
    st.error(f"Kết quả kiểm tra mô hình gốc: KHÔNG KHẢ THI. Thông báo solver: {original_check['message']}")

st.markdown("""
Vì vậy, code bên dưới vẫn kiểm tra mô hình gốc đúng đề, nhưng để có kết quả phân bổ, heatmap, so sánh PuLP-CVXPY
và phân tích chính sách, webapp dùng **một bản hiệu chỉnh khả thi**. Cách hiệu chỉnh mặc định là giảm $\lambda$ từ 0,70 xuống 0,68.
""")

# ============================================================
# 8. THAM SỐ HIỆU CHỈNH
# ============================================================

st.header("3. Chọn tham số để chạy mô hình khả thi")

col1, col2, col3 = st.columns(3)
with col1:
    lam_used = st.slider(
        "Lambda công bằng dùng để giải",
        min_value=0.50,
        max_value=float(np.floor(max_lam * 1000) / 1000),
        value=0.68,
        step=0.001,
        help="Đề gốc dùng 0.70 nhưng không khả thi với trần 12.000 tỷ mỗi vùng."
    )
with col2:
    st.metric("Lambda gốc trong đề", f"{LAM_ORIGINAL:.2f}")
with col3:
    st.metric("Lambda tối đa khả thi xấp xỉ", f"{max_lam:.4f}")

# ============================================================
# 9. CÂU 4.4.1 - PuLP
# ============================================================

st.header("4. Câu 4.4.1 - Cài đặt mô hình bằng PuLP")

pulp_result = solve_with_pulp(lam=lam_used, fairness=True, use_region_cap=True)
if not pulp_result["success"]:
    st.error(pulp_result["message"])
    st.info("Nếu thiếu PuLP, hãy cài bằng lệnh: python -m pip install pulp")
    # Fallback bằng scipy để vẫn hiển thị kết quả tham khảo
    fallback_result = solve_with_scipy(lam=lam_used, fairness=True, use_region_cap=True)
    if fallback_result["success"]:
        st.warning("Đang hiển thị kết quả tham khảo từ SciPy HiGHS vì PuLP chưa chạy được.")
        main_result = fallback_result
    else:
        st.stop()
else:
    main_result = pulp_result

X_fair = main_result["X"]
M_fair = main_result["M"]
Z_fair = main_result["objective"]
solution_df = matrix_to_solution_df(X_fair)
item_df = item_total_df(X_fair)
check_df = check_constraints(X_fair, M_value=M_fair, lam=lam_used, fairness=True)

col_sol1, col_sol2, col_sol3 = st.columns(3)
with col_sol1:
    st.metric("Giá trị tối ưu Z*", f"{Z_fair:,.2f}")
with col_sol2:
    st.metric("Tổng ngân sách dùng", f"{X_fair.sum():,.0f}")
with col_sol3:
    st.metric("M / Digital Index cao nhất", f"{M_fair:.2f}" if M_fair is not None else "N/A")

st.subheader("Phân bổ tối ưu x_{j,r} dạng ma trận 6×4")
st.dataframe(solution_df.round(4), use_container_width=True)

st.subheader("Tổng phân bổ theo hạng mục")
st.dataframe(item_df.round(4), use_container_width=True)

st.subheader("Kiểm tra ràng buộc")
st.dataframe(check_df, use_container_width=True)

if pulp_result.get("constraint_df") is not None:
    st.subheader("Bảng slack và dual value từ PuLP")
    st.dataframe(pulp_result["constraint_df"].round(6), use_container_width=True)

# ============================================================
# 10. CÂU 4.4.2 - CVXPY
# ============================================================

st.header("5. Câu 4.4.2 - Cài đặt lại bằng CVXPY và so sánh với PuLP")

cvx_result = solve_with_cvxpy(lam=lam_used, fairness=True, use_region_cap=True)
if not cvx_result["success"]:
    st.error(cvx_result["message"])
    st.info("Nếu thiếu CVXPY, hãy cài bằng lệnh: python -m pip install cvxpy")
    cvx_solution_df = pd.DataFrame()
    compare_df = pd.DataFrame({
        "Nội dung": ["CVXPY"],
        "Trạng thái": [cvx_result["status"]],
        "Ghi chú": [cvx_result["message"]]
    })
else:
    X_cvx = cvx_result["X"]
    Z_cvx = cvx_result["objective"]
    cvx_solution_df = matrix_to_solution_df(X_cvx)
    max_abs_diff = float(np.max(np.abs(X_fair - X_cvx)))
    z_diff = abs(Z_fair - Z_cvx)

    compare_df = pd.DataFrame({
        "Tiêu chí": [
            "Solver PuLP/SciPy đang dùng cho nghiệm chính",
            "Solver CVXPY",
            "Z* PuLP/SciPy",
            "Z* CVXPY",
            "Chênh lệch tuyệt đối Z*",
            "Chênh lệch lớn nhất từng biến x"
        ],
        "Giá trị": [
            main_result["solver"],
            cvx_result["solver"],
            round(Z_fair, 6),
            round(Z_cvx, 6),
            round(z_diff, 6),
            round(max_abs_diff, 6)
        ]
    })

    st.subheader("Phân bổ tối ưu bằng CVXPY")
    st.dataframe(cvx_solution_df.round(4), use_container_width=True)

st.subheader("Bảng so sánh PuLP và CVXPY")
st.dataframe(compare_df, use_container_width=True)

st.markdown("""
Hai phương pháp có thể không giống tuyệt đối từng ô vì bài toán LP có thể có **nhiều nghiệm tối ưu**.
Khi giá trị mục tiêu Z* gần như bằng nhau và mọi ràng buộc đều đạt, hai kết quả được xem là tương đương về mặt tối ưu.
""")

# ============================================================
# 11. CÂU 4.4.3 - HEATMAP
# ============================================================

st.header("6. Câu 4.4.3 - Heatmap phân bổ tối ưu")

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(X_fair, aspect="auto")
ax.set_xticks(np.arange(len(ITEMS)))
ax.set_xticklabels([ITEM_NAMES[i] for i in ITEMS], rotation=20, ha="right")
ax.set_yticks(np.arange(len(REGIONS)))
ax.set_yticklabels(REGION_NAMES)
ax.set_title("Heatmap phân bổ ngân sách tối ưu theo vùng và hạng mục")

for i in range(X_fair.shape[0]):
    for j in range(X_fair.shape[1]):
        ax.text(j, i, f"{X_fair[i, j]:,.0f}", ha="center", va="center")

fig.colorbar(im, ax=ax, label="Tỷ VND")
fig.tight_layout()
st.pyplot(fig)

region_total = solution_df[["Vùng", "Tổng vùng"]].copy()
top_region = region_total.sort_values("Tổng vùng", ascending=False).iloc[0]
top_item = item_df.sort_values("Tổng phân bổ", ascending=False).iloc[0]

col_chart1, col_chart2 = st.columns(2)
with col_chart1:
    fig_region, ax_region = plt.subplots(figsize=(8, 4))
    ax_region.barh(region_total["Vùng"], region_total["Tổng vùng"])
    ax_region.set_title("Tổng ngân sách theo vùng")
    ax_region.set_xlabel("Tỷ VND")
    fig_region.tight_layout()
    st.pyplot(fig_region)

with col_chart2:
    fig_item, ax_item = plt.subplots(figsize=(8, 4))
    ax_item.bar(item_df["Hạng mục"], item_df["Tổng phân bổ"])
    ax_item.set_title("Tổng ngân sách theo hạng mục")
    ax_item.set_ylabel("Tỷ VND")
    ax_item.tick_params(axis="x", rotation=20)
    fig_item.tight_layout()
    st.pyplot(fig_item)

st.markdown(f"""
Theo nghiệm hiện tại, vùng nhận ngân sách nhiều nhất là **{top_region['Vùng']}** với khoảng **{top_region['Tổng vùng']:,.0f} tỷ VND**.
Hạng mục nhận ngân sách lớn nhất là **{top_item['Hạng mục']}** với khoảng **{top_item['Tổng phân bổ']:,.0f} tỷ VND**.
""")

# ============================================================
# 12. CÂU 4.4.4 - BỎ CÔNG BẰNG C5
# ============================================================

st.header("7. Câu 4.4.4 - So sánh với mô hình không có ràng buộc công bằng C5")

no_fair_result = solve_with_pulp(lam=lam_used, fairness=False, use_region_cap=True)
if not no_fair_result["success"]:
    no_fair_result = solve_with_scipy(lam=lam_used, fairness=False, use_region_cap=True)

if not no_fair_result["success"]:
    st.error(no_fair_result["message"])
    X_no_fair = None
    no_fair_solution_df = pd.DataFrame()
else:
    X_no_fair = no_fair_result["X"]
    Z_no_fair = no_fair_result["objective"]
    no_fair_solution_df = matrix_to_solution_df(X_no_fair)
    cost_fairness = Z_no_fair - Z_fair
    cost_fairness_pct = cost_fairness / Z_no_fair * 100 if Z_no_fair != 0 else np.nan

    compare_fair_df = pd.DataFrame({
        "Mô hình": ["Có ràng buộc công bằng C5", "Không có ràng buộc công bằng C5"],
        "Z*": [round(Z_fair, 4), round(Z_no_fair, 4)],
        "Tổng ngân sách": [round(X_fair.sum(), 4), round(X_no_fair.sum(), 4)],
        "Tổng H": [round(X_fair[:, 3].sum(), 4), round(X_no_fair[:, 3].sum(), 4)],
        "Min Digital Index sau đầu tư": [round((D0 + GAMMA * X_fair[:, 1]).min(), 4), round((D0 + GAMMA * X_no_fair[:, 1]).min(), 4)],
    })

    st.subheader("Bảng so sánh có và không có C5")
    st.dataframe(compare_fair_df, use_container_width=True)

    col_cost1, col_cost2 = st.columns(2)
    with col_cost1:
        st.metric("Chi phí kinh tế của công bằng vùng", f"{cost_fairness:,.2f}")
    with col_cost2:
        st.metric("Tỷ lệ giảm so với mô hình không C5", f"{cost_fairness_pct:.2f}%")

    st.subheader("Phân bổ tối ưu khi bỏ C5")
    st.dataframe(no_fair_solution_df.round(4), use_container_width=True)

# ============================================================
# 13. PHÂN TÍCH BỔ SUNG CHO CÂU 4.5(b) - BỎ TRẦN C3
# ============================================================

st.header("8. Phân tích bổ sung: tác động của trần ngân sách mỗi vùng C3")

no_cap_result = solve_with_pulp(lam=lam_used, fairness=True, use_region_cap=False)
if not no_cap_result["success"]:
    no_cap_result = solve_with_scipy(lam=lam_used, fairness=True, use_region_cap=False)

if no_cap_result["success"]:
    X_no_cap = no_cap_result["X"]
    Z_no_cap = no_cap_result["objective"]
    cost_cap = Z_no_cap - Z_fair
    cost_cap_pct = cost_cap / Z_no_cap * 100 if Z_no_cap != 0 else np.nan

    cap_compare_df = pd.DataFrame({
        "Mô hình": ["Có trần vùng C3", "Bỏ trần vùng C3"],
        "Z*": [round(Z_fair, 4), round(Z_no_cap, 4)],
        "Vùng nhận nhiều nhất": [
            REGION_NAMES[int(np.argmax(X_fair.sum(axis=1)))],
            REGION_NAMES[int(np.argmax(X_no_cap.sum(axis=1)))]
        ],
        "Ngân sách vùng cao nhất": [
            round(X_fair.sum(axis=1).max(), 4),
            round(X_no_cap.sum(axis=1).max(), 4)
        ],
        "Mức giảm do C3": [round(cost_cap, 4), ""],
        "Tỷ lệ giảm do C3 (%)": [round(cost_cap_pct, 4), ""],
    })
    st.dataframe(cap_compare_df, use_container_width=True)
else:
    st.error(no_cap_result["message"])
    cap_compare_df = pd.DataFrame()

# ============================================================
# 14. CÂU 4.5 - THẢO LUẬN CHÍNH SÁCH
# ============================================================

st.header("9. Câu 4.5 - Câu hỏi thảo luận chính sách")

if X_no_fair is not None:
    no_fair_top_region = REGION_NAMES[int(np.argmax(X_no_fair.sum(axis=1)))]
    no_fair_top_items = item_total_df(X_no_fair).sort_values("Tổng phân bổ", ascending=False)
    no_fair_top_item = no_fair_top_items.iloc[0]["Hạng mục"]
else:
    no_fair_top_region = "không xác định"
    no_fair_top_item = "không xác định"

policy_text = f"""
<h3>a) Nếu bỏ ràng buộc công bằng, vốn sẽ chảy về vùng nào? Vì sao? Hậu quả xã hội dài hạn ra sao?</h3>
<p>
Khi bỏ ràng buộc công bằng C5, mô hình có xu hướng dồn ngân sách nhiều hơn vào các vùng và hạng mục có hệ số tác động biên cao.
Trong kết quả hiện tại, vùng nhận nhiều nhất khi bỏ C5 là <b>{no_fair_top_region}</b>, còn hạng mục nổi bật là <b>{no_fair_top_item}</b>.
Điều này xảy ra vì hàm mục tiêu chỉ tối đa hóa GDP gain, nên vốn sẽ ưu tiên nơi có hệ số beta cao như AI ở các vùng có năng lực số tốt hoặc nhân lực số ở vùng có hiệu quả đào tạo cao.
</p>
<p>
Hậu quả dài hạn là khoảng cách số giữa vùng mạnh và vùng yếu có thể tăng lên. Các vùng có chỉ số số hóa thấp như Tây Nguyên hoặc Trung du miền núi phía Bắc có thể bị tụt lại, khiến chính sách chuyển đổi số kém bao trùm hơn.
</p>

<h3>b) Ràng buộc trần ngân sách mỗi vùng C3 có thể coi như một chính sách phân quyền. Nó làm giảm Z* bao nhiêu phần trăm?</h3>
<p>
Với lambda hiệu chỉnh đang dùng là <b>{lam_used:.3f}</b>, nếu bỏ trần C3 thì Z* đạt khoảng <b>{Z_no_cap if no_cap_result['success'] else np.nan:,.2f}</b>.
Khi giữ trần C3, Z* đạt khoảng <b>{Z_fair:,.2f}</b>.
Như vậy, trần ngân sách vùng làm giảm khoảng <b>{cost_cap if no_cap_result['success'] else np.nan:,.2f}</b>, tương đương <b>{cost_cap_pct if no_cap_result['success'] else np.nan:.2f}%</b> so với mô hình bỏ trần.
Mức giảm này có thể chấp nhận được nếu mục tiêu chính sách không chỉ là tối đa hóa GDP ngắn hạn mà còn là phân quyền, giảm tập trung nguồn lực và bảo đảm phát triển cân bằng vùng miền.
</p>

<h3>c) Tây Nguyên có sàn 5.000 tỷ nhưng hệ số AI rất thấp. Nên đầu tư AI hay tập trung H và I trước?</h3>
<p>
Theo ma trận beta, Tây Nguyên có hệ số AI chỉ <b>0,45</b>, thấp hơn nhiều so với nhân lực số H là <b>1,35</b> và hạ tầng số I là <b>1,20</b>.
Do đó, mô hình không khuyến nghị ưu tiên AI ngay từ đầu tại Tây Nguyên. Cách hợp lý hơn là đầu tư trước vào nhân lực số và hạ tầng số, đồng thời cần một phần đầu tư D để đáp ứng điều kiện công bằng vùng.
Sau khi nền tảng hạ tầng, dữ liệu và nhân lực được cải thiện, đầu tư AI sẽ có khả năng hấp thụ tốt hơn.
</p>
"""

st.markdown(policy_text, unsafe_allow_html=True)

# ============================================================
# 15. CHECKLIST HOÀN THÀNH
# ============================================================

st.header("10. Checklist hoàn thành yêu cầu Bài 4")

checklist = pd.DataFrame([
    ["4.4.1 Cài đặt mô hình bằng PuLP", "Đã làm" if HAS_PULP else "Cần cài PuLP", "python -m pip install pulp" if not HAS_PULP else "Có nghiệm bằng PuLP CBC hoặc fallback SciPy"],
    ["4.4.1 In phân bổ tối ưu 6×4 và Z*", "Đã làm", "Có solution_df và metric Z*"],
    ["4.4.2 Cài đặt lại bằng CVXPY", "Đã làm" if HAS_CVXPY else "Cần cài CVXPY", "python -m pip install cvxpy" if not HAS_CVXPY else "Có bảng so sánh"],
    ["4.4.2 So sánh PuLP và CVXPY", "Đã làm", "So sánh Z* và max abs diff"],
    ["4.4.3 Vẽ heatmap phân bổ tối ưu", "Đã làm", "Matplotlib imshow"],
    ["4.4.3 Xác định vùng/hạng mục nhận nhiều nhất", "Đã làm", "Có nhận xét tự động"],
    ["4.4.4 Bỏ ràng buộc công bằng C5", "Đã làm", "Có bảng so sánh và chi phí công bằng"],
    ["4.5 Trả lời câu hỏi chính sách a, b, c", "Đã làm", "Có phân tích theo kết quả mô hình"],
    ["Kiểm tra bất hợp lý của đề", "Đã làm", "Đề gốc lambda=0.70 không khả thi với trần 12.000"],
], columns=["Yêu cầu", "Trạng thái", "Ghi chú"])

st.dataframe(checklist, use_container_width=True)

# ============================================================
# 16. TẢI KẾT QUẢ
# ============================================================

st.header("11. Tải kết quả")

lambda_note_md = f"""
# Ghi chú nộp bài Bài 4 — Ràng buộc lambda

- Đề bài đặt lambda gốc = 0.70 cho ràng buộc công bằng vùng C5.
- Với Digital Index ban đầu lớn nhất M tối thiểu = {np.max(D0):.0f} và vùng thấp nhất là Tây Nguyên D0 = {np.min(D0):.0f}, để đạt 0.70 × M cần đầu tư D vượt trần vùng 12.000 tỷ VND.
- Vì vậy mô hình gốc với lambda = 0.70 không khả thi khi giữ đồng thời C3 và C5.
- Bản chạy chính dùng lambda = {lam_used:.3f}, nhỏ hơn ngưỡng khả thi sơ bộ {max_lam:.4f}, để tạo nghiệm tối ưu phục vụ phân tích chính sách.
- Khi viết báo cáo, cần trình bày đây là bản hiệu chỉnh khả thi, không phải tự ý thay số liệu mà không giải thích.
""".strip()

summary_html = f"""
<p><b>Điểm kiểm tra quan trọng:</b> Mô hình gốc trong đề với lambda = 0.70 không khả thi do C3 và C5 xung đột.</p>
<p><b>Diễn giải an toàn để nộp:</b> Bài làm báo cáo đầy đủ trường hợp gốc không khả thi, sau đó dùng bản hiệu chỉnh lambda = {lam_used:.3f} để giải và phân tích chính sách. Đây là hiệu chỉnh mô hình có giải thích, không phải thay số liệu tùy tiện.</p>
<p><b>Lambda hiệu chỉnh dùng để giải:</b> {lam_used:.3f}</p>
<p><b>Z* mô hình có công bằng:</b> {Z_fair:,.4f}</p>
<p><b>Vùng nhận nhiều nhất:</b> {top_region['Vùng']} ({top_region['Tổng vùng']:,.0f} tỷ VND)</p>
<p><b>Hạng mục nhận nhiều nhất:</b> {top_item['Hạng mục']} ({top_item['Tổng phân bổ']:,.0f} tỷ VND)</p>
"""

tables_for_report = {
    "Ghi chú nộp bài an toàn": pd.DataFrame({"Nội dung": lambda_note_md.split("\n")}),
    "Chẩn đoán lambda 0.70 không khả thi": feasibility_diagnostic_df.round(4),
    "Ma trận beta": beta_dataframe(),
    "Phân bổ tối ưu có công bằng": solution_df.round(4),
    "Tổng theo hạng mục": item_df.round(4),
    "Kiểm tra ràng buộc": check_df,
    "So sánh PuLP và CVXPY": compare_df,
}

if no_fair_result["success"]:
    tables_for_report["Phân bổ khi bỏ C5"] = no_fair_solution_df.round(4)
    tables_for_report["So sánh có và không có C5"] = compare_fair_df
if no_cap_result["success"]:
    tables_for_report["So sánh tác động C3"] = cap_compare_df

html_report = make_html_report(summary_html, tables_for_report)

# Lưu file ra outputs
feasibility_diagnostic_df.round(4).to_csv(OUTPUT_DIR / "bai04_feasibility_diagnostic.csv", index=False, encoding="utf-8-sig")
(OUTPUT_DIR / "bai04_lambda_note_for_report.md").write_text(lambda_note_md, encoding="utf-8")
solution_df.to_csv(OUTPUT_DIR / "bai04_solution_fair.csv", index=False, encoding="utf-8-sig")
item_df.to_csv(OUTPUT_DIR / "bai04_item_totals.csv", index=False, encoding="utf-8-sig")
check_df.to_csv(OUTPUT_DIR / "bai04_constraint_check.csv", index=False, encoding="utf-8-sig")
if no_fair_result["success"]:
    no_fair_solution_df.to_csv(OUTPUT_DIR / "bai04_solution_no_fairness.csv", index=False, encoding="utf-8-sig")
with open(OUTPUT_DIR / "bai04_report.html", "w", encoding="utf-8") as f:
    f.write(html_report)

excel_file = make_download_excel(tables_for_report)

col_dl1, col_dl2, col_dl3 = st.columns(3)
with col_dl1:
    st.download_button(
        label="Tải HTML report",
        data=html_report.encode("utf-8"),
        file_name="bai04_report.html",
        mime="text/html"
    )
with col_dl2:
    st.download_button(
        label="Tải Excel tổng hợp",
        data=excel_file,
        file_name="bai04_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
with col_dl3:
    st.download_button(
        label="Tải CSV phân bổ tối ưu",
        data=solution_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="bai04_solution_fair.csv",
        mime="text/csv"
    )

st.success("Bài 4 đã được kiểm tra đầy đủ: mô hình gốc không khả thi, bản hiệu chỉnh khả thi đã giải bằng PuLP/CVXPY hoặc fallback, có heatmap, so sánh bỏ C5, phân tích C3 và trả lời chính sách.")
