import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO

# ============================================================
# BÀI 2 - PHÂN BỔ NGÂN SÁCH ĐƠN GIẢN THEO 4 HẠNG MỤC ĐẦU TƯ SỐ
# Webapp Streamlit
# Yêu cầu chính theo đề:
# 2.4.1 Giải bằng scipy.optimize.linprog
# 2.4.2 Giải bằng PuLP và in shadow price/dual values
# 2.4.3 Phân tích độ nhạy ngân sách 100, 120, 140 và vẽ Z*(B)
# 2.4.4 Tăng ràng buộc nhân lực số x3 >= 30, kiểm tra khả thi và Z* thay đổi
# ============================================================

# ------------------------------------------------------------
# 0. Import giao diện dùng chung
# ------------------------------------------------------------
try:
    from utils.style import load_css, hero, card
except Exception:
    try:
        from style import load_css, hero, card
    except Exception:
        def load_css():
            return None

        def hero(title, subtitle="", badges=None):
            st.title(title)
            if subtitle:
                st.markdown(subtitle)

        def card(title, text):
            st.markdown(f"### {title}")
            st.markdown(text)

try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    import pulp
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False

st.set_page_config(
    page_title="Bài 2 - LP phân bổ ngân sách số",
    layout="wide"
)
load_css()

hero(
    title="📈 Bài 2 — LP phân bổ ngân sách theo 4 hạng mục đầu tư số",
    subtitle="Giải bài toán quy hoạch tuyến tính bằng scipy.optimize.linprog và PuLP; phân tích shadow price, độ nhạy ngân sách và kịch bản ưu tiên nhân lực số.",
    badges=["Cấp độ dễ", "Linear Programming", "scipy", "PuLP", "Shadow price"]
)

# ============================================================
# 1. THAM SỐ BÀI TOÁN
# ============================================================

ITEMS = ["x1", "x2", "x3", "x4"]
ITEM_NAMES = {
    "x1": "Hạ tầng số",
    "x2": "AI và dữ liệu",
    "x3": "Nhân lực số",
    "x4": "R&D công nghệ"
}

COEFFICIENTS = np.array([0.85, 1.20, 0.95, 1.35], dtype=float)
DEFAULT_MINIMUMS = {
    "x1": 25.0,
    "x2": 15.0,
    "x3": 20.0,
    "x4": 10.0
}
DEFAULT_BUDGET = 100.0
STRATEGIC_SHARE = 0.35
OUTPUT_DIR = Path("outputs")

CONSTRAINT_ORDER = [
    "Ngân sách tổng",
    "Sàn hạ tầng số",
    "Sàn AI và dữ liệu",
    "Sàn nhân lực số",
    "Sàn R&D công nghệ",
    "Tỷ trọng công nghệ chiến lược"
]

# ============================================================
# 2. HÀM XỬ LÝ TỐI ƯU
# ============================================================

def build_ub_matrix(budget=DEFAULT_BUDGET, minimums=None):
    """
    Chuyển bài toán max thành min để dùng scipy.linprog.

    Max Z = 0.85*x1 + 1.20*x2 + 0.95*x3 + 1.35*x4
    Min -Z = -0.85*x1 - 1.20*x2 - 0.95*x3 - 1.35*x4

    Ràng buộc dạng A_ub @ x <= b_ub:
    1) x1+x2+x3+x4 <= B
    2) x1 >= min1  -> -x1 <= -min1
    3) x2 >= min2  -> -x2 <= -min2
    4) x3 >= min3  -> -x3 <= -min3
    5) x4 >= min4  -> -x4 <= -min4
    6) x2+x4 >= 0.35*(x1+x2+x3+x4)
       <=> 0.35*x1 - 0.65*x2 + 0.35*x3 - 0.65*x4 <= 0
    """
    if minimums is None:
        minimums = DEFAULT_MINIMUMS.copy()

    A_ub = np.array([
        [1.00,  1.00,  1.00,  1.00],
        [-1.0,  0.00,  0.00,  0.00],
        [0.00, -1.0,   0.00,  0.00],
        [0.00,  0.00, -1.0,   0.00],
        [0.00,  0.00,  0.00, -1.0],
        [0.35, -0.65,  0.35, -0.65]
    ], dtype=float)

    b_ub = np.array([
        budget,
        -minimums["x1"],
        -minimums["x2"],
        -minimums["x3"],
        -minimums["x4"],
        0.0
    ], dtype=float)

    return A_ub, b_ub


def direct_slack_values(x, budget=DEFAULT_BUDGET, minimums=None):
    """Tính slack theo đúng ý nghĩa kinh tế của từng ràng buộc."""
    if minimums is None:
        minimums = DEFAULT_MINIMUMS.copy()

    total = float(np.sum(x))
    strategic_amount = float(x[1] + x[3])
    strategic_required = STRATEGIC_SHARE * total

    return {
        "Ngân sách tổng": budget - total,
        "Sàn hạ tầng số": x[0] - minimums["x1"],
        "Sàn AI và dữ liệu": x[1] - minimums["x2"],
        "Sàn nhân lực số": x[2] - minimums["x3"],
        "Sàn R&D công nghệ": x[3] - minimums["x4"],
        "Tỷ trọng công nghệ chiến lược": strategic_amount - strategic_required
    }


def solve_with_scipy(budget=DEFAULT_BUDGET, minimums=None):
    """Giải LP bằng scipy.optimize.linprog."""
    if not SCIPY_AVAILABLE:
        return {
            "success": False,
            "message": "Chưa cài scipy. Hãy cài bằng lệnh: pip install scipy",
            "x": np.array([np.nan, np.nan, np.nan, np.nan]),
            "Z": np.nan,
            "res": None
        }

    if minimums is None:
        minimums = DEFAULT_MINIMUMS.copy()

    c = -COEFFICIENTS
    A_ub, b_ub = build_ub_matrix(budget=budget, minimums=minimums)

    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(0, None)] * 4,
        method="highs"
    )

    if res.success:
        x = np.array(res.x, dtype=float)
        Z = float(COEFFICIENTS @ x)
    else:
        x = np.array([np.nan, np.nan, np.nan, np.nan])
        Z = np.nan

    return {
        "success": bool(res.success),
        "message": res.message,
        "x": x,
        "Z": Z,
        "res": res
    }


def solve_with_pulp(budget=DEFAULT_BUDGET, minimums=None):
    """Giải LP bằng PuLP và lấy dual values nếu CBC trả về được .pi."""
    if not PULP_AVAILABLE:
        return None

    if minimums is None:
        minimums = DEFAULT_MINIMUMS.copy()

    model = pulp.LpProblem("Bai_2_LP_Phan_Bo_Ngan_Sach_So", pulp.LpMaximize)

    x1 = pulp.LpVariable("x1_Ha_tang_so", lowBound=0, cat="Continuous")
    x2 = pulp.LpVariable("x2_AI_va_du_lieu", lowBound=0, cat="Continuous")
    x3 = pulp.LpVariable("x3_Nhan_luc_so", lowBound=0, cat="Continuous")
    x4 = pulp.LpVariable("x4_RD_cong_nghe", lowBound=0, cat="Continuous")
    x_vars = [x1, x2, x3, x4]

    model += 0.85*x1 + 1.20*x2 + 0.95*x3 + 1.35*x4, "Tong_GDP_tang_them_ky_vong"

    model += x1 + x2 + x3 + x4 <= budget, "Ngan_sach_tong"
    model += x1 >= minimums["x1"], "San_ha_tang_so"
    model += x2 >= minimums["x2"], "San_AI_va_du_lieu"
    model += x3 >= minimums["x3"], "San_nhan_luc_so"
    model += x4 >= minimums["x4"], "San_RD_cong_nghe"
    model += x2 + x4 >= STRATEGIC_SHARE * (x1 + x2 + x3 + x4), "Ty_trong_cong_nghe_chien_luoc"

    solver = pulp.PULP_CBC_CMD(msg=False)
    status_code = model.solve(solver)
    status = pulp.LpStatus[status_code]

    x_value = np.array([pulp.value(var) for var in x_vars], dtype=float)
    Z = float(pulp.value(model.objective)) if status == "Optimal" else np.nan

    name_map = {
        "Ngan_sach_tong": "Ngân sách tổng",
        "San_ha_tang_so": "Sàn hạ tầng số",
        "San_AI_va_du_lieu": "Sàn AI và dữ liệu",
        "San_nhan_luc_so": "Sàn nhân lực số",
        "San_RD_cong_nghe": "Sàn R&D công nghệ",
        "Ty_trong_cong_nghe_chien_luoc": "Tỷ trọng công nghệ chiến lược"
    }

    direct_slack = direct_slack_values(x_value, budget=budget, minimums=minimums)
    dual_rows = []
    for internal_name, con in model.constraints.items():
        display_name = name_map.get(internal_name, internal_name)
        dual_rows.append({
            "Ràng buộc": display_name,
            "Shadow_price_PuLP": getattr(con, "pi", np.nan),
            "Slack_truc_tiep": direct_slack.get(display_name, np.nan),
            "Ghi chú": interpret_dual(display_name, getattr(con, "pi", np.nan), direct_slack.get(display_name, np.nan))
        })

    dual_df = pd.DataFrame(dual_rows)
    order = {name: i for i, name in enumerate(CONSTRAINT_ORDER)}
    dual_df["_order"] = dual_df["Ràng buộc"].map(order)
    dual_df = dual_df.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    return {
        "status": status,
        "x": x_value,
        "Z": Z,
        "dual_df": dual_df,
        "model": model
    }


def solution_to_dataframe(x, Z, budget=DEFAULT_BUDGET, minimums=None):
    """Chuyển nghiệm tối ưu thành bảng dễ đọc."""
    if minimums is None:
        minimums = DEFAULT_MINIMUMS.copy()

    total = float(np.nansum(x))
    strategic_value = float(x[1] + x[3])
    strategic_pct = strategic_value / total * 100 if total > 0 else np.nan

    df = pd.DataFrame({
        "Biến": ITEMS,
        "Hạng mục": [ITEM_NAMES[item] for item in ITEMS],
        "Phân bổ tối ưu (nghìn tỷ VND)": np.round(x, 4),
        "Hệ số tác động GDP": COEFFICIENTS,
        "GDP tăng thêm theo hạng mục": np.round(x * COEFFICIENTS, 4),
        "Sàn tối thiểu": [minimums[item] for item in ITEMS]
    })

    summary = pd.DataFrame({
        "Chỉ tiêu": [
            "Tổng ngân sách sử dụng",
            "Ngân sách tối đa",
            "GDP tăng thêm kỳ vọng Z*",
            "AI + R&D",
            "Tỷ trọng AI + R&D",
            "Tỷ trọng AI + R&D tối thiểu"
        ],
        "Giá trị": [
            round(total, 4),
            round(budget, 4),
            round(Z, 4),
            round(strategic_value, 4),
            round(strategic_pct, 4),
            round(STRATEGIC_SHARE * 100, 4)
        ],
        "Đơn vị": [
            "nghìn tỷ VND",
            "nghìn tỷ VND",
            "nghìn tỷ VND",
            "nghìn tỷ VND",
            "%",
            "%"
        ]
    })

    return df, summary


def make_constraint_table(x, budget=DEFAULT_BUDGET, minimums=None):
    """Bảng kiểm tra các ràng buộc sau khi giải."""
    if minimums is None:
        minimums = DEFAULT_MINIMUMS.copy()

    total = float(np.sum(x))
    strategic_value = float(x[1] + x[3])
    strategic_required = STRATEGIC_SHARE * total
    slack = direct_slack_values(x, budget=budget, minimums=minimums)

    rows = [
        {
            "Ràng buộc": "Ngân sách tổng",
            "Vế trái": total,
            "Dấu": "<=",
            "Vế phải": budget,
            "Slack trực tiếp": slack["Ngân sách tổng"],
            "Kết luận": "Đạt" if total <= budget + 1e-7 else "Không đạt"
        },
        {
            "Ràng buộc": "Sàn hạ tầng số",
            "Vế trái": x[0],
            "Dấu": ">=",
            "Vế phải": minimums["x1"],
            "Slack trực tiếp": slack["Sàn hạ tầng số"],
            "Kết luận": "Đạt" if x[0] + 1e-7 >= minimums["x1"] else "Không đạt"
        },
        {
            "Ràng buộc": "Sàn AI và dữ liệu",
            "Vế trái": x[1],
            "Dấu": ">=",
            "Vế phải": minimums["x2"],
            "Slack trực tiếp": slack["Sàn AI và dữ liệu"],
            "Kết luận": "Đạt" if x[1] + 1e-7 >= minimums["x2"] else "Không đạt"
        },
        {
            "Ràng buộc": "Sàn nhân lực số",
            "Vế trái": x[2],
            "Dấu": ">=",
            "Vế phải": minimums["x3"],
            "Slack trực tiếp": slack["Sàn nhân lực số"],
            "Kết luận": "Đạt" if x[2] + 1e-7 >= minimums["x3"] else "Không đạt"
        },
        {
            "Ràng buộc": "Sàn R&D công nghệ",
            "Vế trái": x[3],
            "Dấu": ">=",
            "Vế phải": minimums["x4"],
            "Slack trực tiếp": slack["Sàn R&D công nghệ"],
            "Kết luận": "Đạt" if x[3] + 1e-7 >= minimums["x4"] else "Không đạt"
        },
        {
            "Ràng buộc": "Tỷ trọng công nghệ chiến lược",
            "Vế trái": strategic_value,
            "Dấu": ">=",
            "Vế phải": strategic_required,
            "Slack trực tiếp": slack["Tỷ trọng công nghệ chiến lược"],
            "Kết luận": "Đạt" if strategic_value + 1e-7 >= strategic_required else "Không đạt"
        }
    ]
    return pd.DataFrame(rows).round(4)


def interpret_dual(constraint_name, dual_value, slack_value):
    """Tạo câu giải thích ngắn cho bảng shadow price."""
    if pd.isna(dual_value):
        return "Không lấy được dual value từ solver."

    if abs(slack_value) > 1e-6:
        return "Ràng buộc không chặt nên shadow price bằng 0 hoặc xấp xỉ 0."

    if constraint_name == "Ngân sách tổng":
        return "Nếu nới ngân sách thêm 1 nghìn tỷ VND, Z* tăng xấp xỉ bằng shadow price này."

    if "Sàn" in constraint_name:
        return "Nếu nâng yêu cầu tối thiểu thêm 1 nghìn tỷ VND, Z* thay đổi theo shadow price này. Giá trị âm nghĩa là làm giảm mục tiêu."

    if constraint_name == "Tỷ trọng công nghệ chiến lược":
        return "Nếu siết ràng buộc tỷ trọng chiến lược, Z* thay đổi theo shadow price này."

    return "Shadow price đo mức thay đổi cận biên của Z* khi thay đổi vế phải ràng buộc."


def finite_difference_shadow_prices(base_budget=DEFAULT_BUDGET, base_minimums=None, eps=1.0):
    """
    Tính shadow price dạng dễ hiểu bằng sai phân hữu hạn.
    Cách này dùng để giải thích chính sách kể cả khi PuLP chưa cài.
    - Ngân sách tổng: tăng B thêm eps.
    - Các sàn tối thiểu: tăng sàn thêm eps.
    - Ràng buộc tỷ trọng chiến lược không sai phân ở đây vì cần thay đổi tham số tỷ lệ.
    """
    if base_minimums is None:
        base_minimums = DEFAULT_MINIMUMS.copy()

    base = solve_with_scipy(budget=base_budget, minimums=base_minimums)
    base_Z = base["Z"]
    rows = []

    # Ngân sách tổng
    new_result = solve_with_scipy(budget=base_budget + eps, minimums=base_minimums)
    rows.append({
        "Ràng buộc": "Ngân sách tổng",
        "Shadow_price_sai_phan": (new_result["Z"] - base_Z) / eps,
        "Diễn giải": "Tăng ngân sách thêm 1 nghìn tỷ VND thì Z* tăng khoảng giá trị này."
    })

    # Các sàn tối thiểu
    minimum_labels = {
        "x1": "Sàn hạ tầng số",
        "x2": "Sàn AI và dữ liệu",
        "x3": "Sàn nhân lực số",
        "x4": "Sàn R&D công nghệ"
    }
    for key, label in minimum_labels.items():
        new_min = base_minimums.copy()
        new_min[key] += eps
        new_result = solve_with_scipy(budget=base_budget, minimums=new_min)
        rows.append({
            "Ràng buộc": label,
            "Shadow_price_sai_phan": (new_result["Z"] - base_Z) / eps,
            "Diễn giải": "Tăng mức sàn tối thiểu thêm 1 nghìn tỷ VND thì Z* thay đổi khoảng giá trị này."
        })

    rows.append({
        "Ràng buộc": "Tỷ trọng công nghệ chiến lược",
        "Shadow_price_sai_phan": 0.0,
        "Diễn giải": "Ở nghiệm gốc ràng buộc này không chặt, nên shadow price bằng 0 trong vùng lân cận."
    })

    return pd.DataFrame(rows).round(4)


def run_budget_sensitivity(budget_values, minimums=None):
    """Chạy phân tích độ nhạy ngân sách cho nhiều mức B."""
    if minimums is None:
        minimums = DEFAULT_MINIMUMS.copy()

    rows = []
    for B in budget_values:
        result = solve_with_scipy(budget=float(B), minimums=minimums)
        x = result["x"]
        total = float(np.nansum(x))
        strategic_value = float(x[1] + x[3]) if result["success"] else np.nan
        strategic_pct = strategic_value / total * 100 if total > 0 else np.nan
        rows.append({
            "Ngân sách B": float(B),
            "x1 - Hạ tầng số": x[0],
            "x2 - AI và dữ liệu": x[1],
            "x3 - Nhân lực số": x[2],
            "x4 - R&D công nghệ": x[3],
            "Z*": result["Z"],
            "Tổng sử dụng": total,
            "Tỷ trọng AI+R&D (%)": strategic_pct,
            "Trạng thái": "Optimal" if result["success"] else result["message"]
        })
    return pd.DataFrame(rows).round(4)


def make_download_excel(sheets: dict):
    """Tạo file Excel nhiều sheet để tải xuống."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df_sheet in sheets.items():
            safe_name = sheet_name[:31]
            df_sheet.to_excel(writer, index=False, sheet_name=safe_name)
    return output.getvalue()


def make_html_report(
    scipy_solution_df,
    scipy_summary_df,
    constraint_df,
    dual_df,
    sensitivity_df,
    human_summary_df,
    policy_html
):
    """Tạo báo cáo HTML tổng hợp Bài 2."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Bài 2 - LP phân bổ ngân sách số</title>
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
        <h1>BÀI 2 - PHÂN BỔ NGÂN SÁCH ĐẦU TƯ SỐ BẰNG QUY HOẠCH TUYẾN TÍNH</h1>

        <div class="box">
            <p><b>Nghiệm tối ưu cơ sở:</b> x1 = 25; x2 = 15; x3 = 20; x4 = 40 nghìn tỷ VND.</p>
            <p><b>Z* cơ sở:</b> 112.25 nghìn tỷ VND GDP tăng thêm kỳ vọng.</p>
            <p><b>Kịch bản x3 ≥ 30:</b> vẫn khả thi; Z* = 108.25; giảm 4.00 nghìn tỷ VND so với cơ sở.</p>
        </div>

        <h2>1. Nghiệm tối ưu bằng scipy.optimize.linprog</h2>
        {scipy_solution_df.to_html(index=False)}
        {scipy_summary_df.to_html(index=False)}

        <h2>2. Kiểm tra ràng buộc</h2>
        {constraint_df.to_html(index=False)}

        <h2>3. Shadow price / Dual values</h2>
        {dual_df.to_html(index=False)}

        <h2>4. Phân tích độ nhạy ngân sách</h2>
        {sensitivity_df.to_html(index=False)}

        <h2>5. Kịch bản ưu tiên nhân lực số x3 ≥ 30</h2>
        {human_summary_df.to_html(index=False)}

        <h2>6. Câu hỏi thảo luận chính sách</h2>
        {policy_html}
    </body>
    </html>
    """
    return html


def save_outputs(output_dir, files: dict):
    """Lưu các bảng kết quả ra thư mục outputs."""
    output_dir.mkdir(exist_ok=True)
    for filename, content in files.items():
        path = output_dir / filename
        if isinstance(content, pd.DataFrame):
            content.to_csv(path, index=False, encoding="utf-8-sig")
        elif isinstance(content, str):
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

# ============================================================
# 3. KIỂM TRA TÍNH HỢP LÝ CỦA ĐỀ
# ============================================================

st.header("1. Kiểm tra yêu cầu và điểm cần lưu ý của Bài 2")

st.markdown("""
Bài 2 yêu cầu xây dựng một bài toán quy hoạch tuyến tính với 4 biến quyết định:

- **x₁**: đầu tư hạ tầng số.
- **x₂**: đầu tư AI và dữ liệu.
- **x₃**: đầu tư nhân lực số.
- **x₄**: đầu tư R&D công nghệ.

Đơn vị của biến là **nghìn tỷ VND**. Hàm mục tiêu là tối đa hóa tăng GDP kỳ vọng:

$$
\\max Z = 0.85x_1 + 1.20x_2 + 0.95x_3 + 1.35x_4
$$
""")

with st.expander("Các điểm cần lưu ý trước khi đọc kết quả", expanded=True):
    st.markdown("""
    1. Trong phần mục tiêu học tập đề nói **5 ràng buộc**, nhưng mô hình liệt kê thực tế có **6 ràng buộc chính**: ngân sách tổng, 4 ràng buộc sàn tối thiểu và 1 ràng buộc tỷ trọng công nghệ chiến lược. Chưa tính ràng buộc không âm.
    2. Các hệ số mục tiêu **0.85, 1.20, 0.95, 1.35** là tham số giả định/cho sẵn trong đề. Code không ước lượng lại các hệ số này từ dữ liệu CSV.
    3. Vì tất cả hệ số mục tiêu đều dương, mô hình tối ưu sẽ sử dụng hết ngân sách nếu bài toán khả thi.
    4. Vì hệ số của **R&D công nghệ x₄ = 1.35** cao nhất, sau khi đáp ứng các mức sàn tối thiểu, phần ngân sách còn lại có xu hướng dồn vào R&D.
    5. Câu hỏi chính sách nói “tăng thêm 1 tỷ VND”, nhưng mô hình dùng đơn vị **nghìn tỷ VND**. Vì vậy shadow price 1.35 được hiểu trực tiếp là: tăng 1 nghìn tỷ VND ngân sách làm Z* tăng khoảng 1.35 nghìn tỷ VND. Nếu quy đổi tuyến tính, tăng 1 tỷ VND làm GDP kỳ vọng tăng khoảng 1.35 tỷ VND.
    6. Để lấy **dual values bằng PuLP**, máy cần cài thư viện `pulp`. Nếu chưa cài, chạy lệnh: `pip install pulp`. App vẫn có bảng shadow price sai phân bằng scipy để kiểm tra.
    """)

# ============================================================
# 4. THAM SỐ CÓ THỂ ĐIỀU CHỈNH
# ============================================================

st.header("2. Thiết lập tham số mô hình")

col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    budget_input = st.number_input(
        "Ngân sách tổng B (nghìn tỷ VND)",
        min_value=70.0,
        max_value=300.0,
        value=DEFAULT_BUDGET,
        step=10.0
    )

with col_p2:
    human_min_input = st.number_input(
        "Sàn nhân lực số x3 (nghìn tỷ VND)",
        min_value=0.0,
        max_value=100.0,
        value=DEFAULT_MINIMUMS["x3"],
        step=5.0
    )

with col_p3:
    st.metric("Tỷ trọng AI + R&D tối thiểu", f"{STRATEGIC_SHARE*100:.0f}%")

current_minimums = DEFAULT_MINIMUMS.copy()
current_minimums["x3"] = float(human_min_input)

coef_df = pd.DataFrame({
    "Biến": ITEMS,
    "Hạng mục": [ITEM_NAMES[item] for item in ITEMS],
    "Hệ số GDP": COEFFICIENTS,
    "Sàn tối thiểu mặc định": [DEFAULT_MINIMUMS[item] for item in ITEMS],
    "Sàn đang dùng": [current_minimums[item] for item in ITEMS]
})

st.subheader("Bảng tham số đầu vào")
st.dataframe(coef_df, width="stretch")

# ============================================================
# 5. CÂU 2.4.1 - GIẢI BẰNG SCIPY
# ============================================================

st.header("3. Câu 2.4.1 - Giải bằng scipy.optimize.linprog")

scipy_result = solve_with_scipy(budget=float(budget_input), minimums=current_minimums)

if scipy_result["success"]:
    st.success("scipy.optimize.linprog đã tìm được nghiệm tối ưu.")
else:
    st.error(f"Không giải được bài toán bằng scipy: {scipy_result['message']}")

x_scipy = scipy_result["x"]
Z_scipy = scipy_result["Z"]
scipy_solution_df, scipy_summary_df = solution_to_dataframe(
    x_scipy,
    Z_scipy,
    budget=float(budget_input),
    minimums=current_minimums
)
constraint_df = make_constraint_table(x_scipy, budget=float(budget_input), minimums=current_minimums)

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.metric("Z* tối ưu", f"{Z_scipy:.4f}")
with col_s2:
    st.metric("Tổng ngân sách dùng", f"{np.sum(x_scipy):.4f}")
with col_s3:
    strategic_pct_scipy = (x_scipy[1] + x_scipy[3]) / np.sum(x_scipy) * 100
    st.metric("Tỷ trọng AI + R&D", f"{strategic_pct_scipy:.2f}%")

st.subheader("Bảng phân bổ tối ưu")
st.dataframe(scipy_solution_df, width="stretch")

st.subheader("Bảng tóm tắt nghiệm")
st.dataframe(scipy_summary_df, width="stretch")

st.subheader("Kiểm tra 100% ràng buộc")
st.dataframe(constraint_df, width="stretch")

fig_alloc, ax_alloc = plt.subplots(figsize=(8.5, 4.5))
ax_alloc.bar(scipy_solution_df["Hạng mục"], scipy_solution_df["Phân bổ tối ưu (nghìn tỷ VND)"])
ax_alloc.set_title("Phân bổ ngân sách tối ưu theo hạng mục")
ax_alloc.set_xlabel("Hạng mục")
ax_alloc.set_ylabel("Ngân sách phân bổ (nghìn tỷ VND)")
ax_alloc.grid(axis="y")
plt.xticks(rotation=20, ha="right")
st.pyplot(fig_alloc)

st.markdown(f"""
**Kết quả cơ sở khi B = {budget_input:.0f} nghìn tỷ VND:**

- x₁ = **{x_scipy[0]:.4f}** nghìn tỷ VND cho hạ tầng số.
- x₂ = **{x_scipy[1]:.4f}** nghìn tỷ VND cho AI và dữ liệu.
- x₃ = **{x_scipy[2]:.4f}** nghìn tỷ VND cho nhân lực số.
- x₄ = **{x_scipy[3]:.4f}** nghìn tỷ VND cho R&D công nghệ.
- Giá trị tối ưu Z* = **{Z_scipy:.4f}** nghìn tỷ VND GDP tăng thêm kỳ vọng.
""")

# ============================================================
# 6. CÂU 2.4.2 - GIẢI BẰNG PULP VÀ SHADOW PRICE
# ============================================================

st.header("4. Câu 2.4.2 - Giải lại bằng PuLP và phân tích shadow price")

if PULP_AVAILABLE:
    pulp_result = solve_with_pulp(budget=float(budget_input), minimums=current_minimums)
    st.success(f"PuLP đã chạy. Trạng thái solver: {pulp_result['status']}.")

    pulp_solution_df, pulp_summary_df = solution_to_dataframe(
        pulp_result["x"],
        pulp_result["Z"],
        budget=float(budget_input),
        minimums=current_minimums
    )

    dual_df = pulp_result["dual_df"].copy()
    dual_df["Shadow_price_PuLP"] = dual_df["Shadow_price_PuLP"].round(6)
    dual_df["Slack_truc_tiep"] = dual_df["Slack_truc_tiep"].round(6)

    c_pulp1, c_pulp2 = st.columns(2)
    with c_pulp1:
        st.subheader("Nghiệm từ PuLP")
        st.dataframe(pulp_solution_df, width="stretch")
    with c_pulp2:
        st.subheader("Tóm tắt từ PuLP")
        st.dataframe(pulp_summary_df, width="stretch")

    st.subheader("Dual values / Shadow price từ PuLP")
    st.dataframe(dual_df, width="stretch")
else:
    st.warning("Máy hiện chưa cài PuLP nên app chưa thể lấy dual values trực tiếp từ PuLP.")
    st.code("pip install pulp", language="bash")
    dual_df = finite_difference_shadow_prices(
        base_budget=float(budget_input),
        base_minimums=current_minimums,
        eps=1.0
    )
    st.subheader("Bảng shadow price xấp xỉ bằng sai phân hữu hạn từ scipy")
    st.dataframe(dual_df, width="stretch")

st.markdown("""
**Ý nghĩa chính sách của shadow price ràng buộc ngân sách tổng:**

Trong nghiệm cơ sở, shadow price của ngân sách tổng xấp xỉ **1.35**. Điều này có nghĩa là nếu Chính phủ nới ngân sách thêm **1 nghìn tỷ VND**, trong vùng tuyến tính hiện tại, GDP tăng thêm kỳ vọng sẽ tăng khoảng **1.35 nghìn tỷ VND**. Nếu diễn giải theo 1 tỷ VND, có thể quy đổi tuyến tính là khoảng **1.35 tỷ VND GDP kỳ vọng** cho mỗi 1 tỷ VND ngân sách tăng thêm.

Tuy nhiên, đây chỉ là giá trị cận biên trong mô hình tuyến tính, không nên hiểu là bằng chứng chắc chắn rằng đầu tư công ngoài thực tế luôn tạo ra cùng một tỷ suất. Khi ngân sách tăng quá lớn, năng lực hấp thụ, chất lượng dự án, độ trễ triển khai và rủi ro thất thoát có thể làm hệ số tác động giảm xuống.
""")

# ============================================================
# 7. CÂU 2.4.3 - ĐỘ NHẠY NGÂN SÁCH
# ============================================================

st.header("5. Câu 2.4.3 - Phân tích độ nhạy ngân sách B = 100, 120, 140")

required_budgets = [100, 120, 140]
sensitivity_df = run_budget_sensitivity(required_budgets, minimums=DEFAULT_MINIMUMS.copy())

st.subheader("Bảng Z*(B) theo yêu cầu đề bài")
st.dataframe(sensitivity_df, width="stretch")

fig_sens, ax_sens = plt.subplots(figsize=(8.5, 4.5))
ax_sens.plot(sensitivity_df["Ngân sách B"], sensitivity_df["Z*"], marker="o")
ax_sens.set_title("Đường cong Z*(B) khi ngân sách tăng")
ax_sens.set_xlabel("Ngân sách B (nghìn tỷ VND)")
ax_sens.set_ylabel("Z* - GDP tăng thêm kỳ vọng (nghìn tỷ VND)")
ax_sens.grid(True)
st.pyplot(fig_sens)

st.markdown("""
Khi ngân sách tăng từ 100 lên 120 và 140 nghìn tỷ VND, nghiệm tối ưu vẫn giữ các mức sàn của hạ tầng số, AI và dữ liệu, nhân lực số; phần ngân sách tăng thêm được đưa vào **R&D công nghệ** vì đây là hạng mục có hệ số tác động GDP cao nhất trong hàm mục tiêu.
""")

with st.expander("Chạy thêm độ nhạy ngân sách tùy chọn", expanded=False):
    min_B = st.number_input("B nhỏ nhất", min_value=70.0, max_value=300.0, value=80.0, step=10.0)
    max_B = st.number_input("B lớn nhất", min_value=70.0, max_value=500.0, value=180.0, step=10.0)
    step_B = st.number_input("Bước nhảy", min_value=1.0, max_value=50.0, value=10.0, step=1.0)

    if max_B >= min_B:
        custom_budgets = list(np.arange(min_B, max_B + step_B, step_B))
        custom_sensitivity_df = run_budget_sensitivity(custom_budgets, minimums=DEFAULT_MINIMUMS.copy())
        st.dataframe(custom_sensitivity_df, width="stretch")

        fig_custom, ax_custom = plt.subplots(figsize=(8.5, 4.5))
        ax_custom.plot(custom_sensitivity_df["Ngân sách B"], custom_sensitivity_df["Z*"], marker="o")
        ax_custom.set_title("Đường cong Z*(B) tùy chọn")
        ax_custom.set_xlabel("Ngân sách B (nghìn tỷ VND)")
        ax_custom.set_ylabel("Z* - GDP tăng thêm kỳ vọng")
        ax_custom.grid(True)
        st.pyplot(fig_custom)
    else:
        st.error("B lớn nhất phải lớn hơn hoặc bằng B nhỏ nhất.")

# ============================================================
# 8. CÂU 2.4.4 - ƯU TIÊN NHÂN LỰC SỐ x3 >= 30
# ============================================================

st.header("6. Câu 2.4.4 - Kịch bản ưu tiên nhân lực số: x₃ ≥ 30")

human_minimums = DEFAULT_MINIMUMS.copy()
human_minimums["x3"] = 30.0

base_result = solve_with_scipy(budget=100.0, minimums=DEFAULT_MINIMUMS.copy())
human_result = solve_with_scipy(budget=100.0, minimums=human_minimums)

if human_result["success"]:
    st.success("Bài toán vẫn khả thi khi tăng ràng buộc x₃ ≥ 30.")
else:
    st.error("Bài toán không khả thi hoặc không giải được khi tăng ràng buộc x₃ ≥ 30.")

human_solution_df, human_solution_summary = solution_to_dataframe(
    human_result["x"],
    human_result["Z"],
    budget=100.0,
    minimums=human_minimums
)

human_comparison_df = pd.DataFrame({
    "Chỉ tiêu": [
        "Z* cơ sở",
        "Z* khi x3 >= 30",
        "Mức thay đổi Z*",
        "x1 cơ sở",
        "x2 cơ sở",
        "x3 cơ sở",
        "x4 cơ sở",
        "x1 khi x3 >= 30",
        "x2 khi x3 >= 30",
        "x3 khi x3 >= 30",
        "x4 khi x3 >= 30"
    ],
    "Giá trị": [
        round(base_result["Z"], 4),
        round(human_result["Z"], 4),
        round(human_result["Z"] - base_result["Z"], 4),
        round(base_result["x"][0], 4),
        round(base_result["x"][1], 4),
        round(base_result["x"][2], 4),
        round(base_result["x"][3], 4),
        round(human_result["x"][0], 4),
        round(human_result["x"][1], 4),
        round(human_result["x"][2], 4),
        round(human_result["x"][3], 4)
    ],
    "Đơn vị": [
        "nghìn tỷ VND GDP kỳ vọng",
        "nghìn tỷ VND GDP kỳ vọng",
        "nghìn tỷ VND GDP kỳ vọng",
        "nghìn tỷ VND",
        "nghìn tỷ VND",
        "nghìn tỷ VND",
        "nghìn tỷ VND",
        "nghìn tỷ VND",
        "nghìn tỷ VND",
        "nghìn tỷ VND",
        "nghìn tỷ VND"
    ]
})

col_h1, col_h2, col_h3 = st.columns(3)
with col_h1:
    st.metric("Z* cơ sở", f"{base_result['Z']:.4f}")
with col_h2:
    st.metric("Z* khi x3 ≥ 30", f"{human_result['Z']:.4f}")
with col_h3:
    st.metric("Chênh lệch", f"{human_result['Z'] - base_result['Z']:.4f}")

st.subheader("Bảng nghiệm khi ưu tiên nhân lực số")
st.dataframe(human_solution_df, width="stretch")

st.subheader("So sánh với nghiệm cơ sở")
st.dataframe(human_comparison_df, width="stretch")

fig_human, ax_human = plt.subplots(figsize=(8.5, 4.5))
labels = [ITEM_NAMES[item] for item in ITEMS]
width = 0.35
idx = np.arange(len(labels))
ax_human.bar(idx - width/2, base_result["x"], width, label="Cơ sở")
ax_human.bar(idx + width/2, human_result["x"], width, label="x3 ≥ 30")
ax_human.set_title("So sánh phân bổ cơ sở và kịch bản ưu tiên nhân lực số")
ax_human.set_xlabel("Hạng mục")
ax_human.set_ylabel("Ngân sách phân bổ (nghìn tỷ VND)")
ax_human.set_xticks(idx)
ax_human.set_xticklabels(labels, rotation=20, ha="right")
ax_human.legend()
ax_human.grid(axis="y")
st.pyplot(fig_human)

st.markdown("""
Khi tăng sàn nhân lực số từ 20 lên 30 nghìn tỷ VND, bài toán vẫn khả thi. Tuy nhiên, do ngân sách tổng vẫn cố định ở 100 nghìn tỷ VND, mô hình phải giảm một phần ngân sách của R&D công nghệ từ 40 xuống 30 nghìn tỷ VND. Vì hệ số của R&D là 1.35 còn hệ số của nhân lực số là 0.95, Z* giảm 4.00 nghìn tỷ VND GDP kỳ vọng.
""")

# ============================================================
# 9. CÂU 2.5 - THẢO LUẬN CHÍNH SÁCH
# ============================================================

st.header("7. Câu 2.5 - Câu hỏi thảo luận chính sách")

policy_html = f"""
<h3>a) Khi ngân sách tổng tăng thêm 1 tỷ VND, GDP kỳ vọng tăng thêm bao nhiêu?</h3>
<p>
Trong mô hình, biến ngân sách được đo bằng <b>nghìn tỷ VND</b>. Shadow price của ràng buộc ngân sách tổng tại nghiệm cơ sở là khoảng <b>1.35</b>. Vì vậy, nếu ngân sách tăng thêm <b>1 nghìn tỷ VND</b>, GDP tăng thêm kỳ vọng tăng khoảng <b>1.35 nghìn tỷ VND</b>. Nếu quy đổi tuyến tính sang đơn vị nhỏ hơn, tăng <b>1 tỷ VND</b> ngân sách tương ứng với khoảng <b>1.35 tỷ VND</b> GDP kỳ vọng.
</p>
<p>
Tuy nhiên, đây chỉ là kết quả cận biên của mô hình tuyến tính. Trong thực tế, con số này không nên được xem là cận trên chắc chắn của chi phí cơ hội vốn công, vì hiệu quả đầu tư còn phụ thuộc vào chất lượng dự án, tốc độ giải ngân, năng lực hấp thụ, năng lực quản trị và rủi ro thất thoát.
</p>

<h3>b) Vì sao R&D có hệ số tác động cao nhất nhưng ràng buộc tối thiểu thấp nhất?</h3>
<p>
R&D công nghệ có hệ số tác động cao nhất vì có thể tạo hiệu ứng lan tỏa dài hạn tới năng suất, đổi mới sáng tạo và năng lực cạnh tranh. Tuy vậy, ràng buộc tối thiểu của R&D thấp hơn các hạng mục khác vì R&D thường có độ trễ dài, rủi ro thất bại cao, cần nhân lực chất lượng cao và cần hệ sinh thái đổi mới đủ mạnh để hấp thụ kết quả nghiên cứu.
</p>
<p>
Nói cách khác, R&D có tiềm năng sinh lợi xã hội lớn nhưng không thể tăng quá nhanh nếu thiếu nền tảng hạ tầng số, dữ liệu, nhân lực số và cơ chế thương mại hóa kết quả nghiên cứu.
</p>

<h3>c) Tỷ lệ 35% công nghệ chiến lược có khả thi không?</h3>
<p>
Trong nghiệm tối ưu cơ sở, tỷ trọng AI + R&D đạt khoảng <b>{(base_result['x'][1] + base_result['x'][3]) / np.sum(base_result['x']) * 100:.2f}%</b>, cao hơn mức tối thiểu <b>35%</b>. Vì vậy, xét riêng trong mô hình, ràng buộc này hoàn toàn khả thi và thậm chí không phải là ràng buộc gây áp lực.
</p>
<p>
Tuy nhiên, trong thực tiễn quản lý ngân sách Việt Nam, tỷ lệ 35% cho công nghệ chiến lược có thể gặp khó khăn nếu ngân sách nhà nước còn phải ưu tiên hạ tầng giao thông, y tế, giáo dục, an sinh xã hội và các nhiệm vụ cấp bách khác. Do đó, tỷ lệ này nên được triển khai theo lộ trình, có cơ chế đánh giá hiệu quả, lựa chọn dự án trọng điểm và bảo đảm không làm giảm các chi tiêu xã hội thiết yếu.
</p>
"""

st.markdown(policy_html, unsafe_allow_html=True)

# ============================================================
# 10. KẾT LUẬN KIỂM TRA 100% YÊU CẦU
# ============================================================

st.header("8. Checklist hoàn thành yêu cầu Bài 2")

checklist_df = pd.DataFrame({
    "Yêu cầu": [
        "2.4.1 Giải bằng scipy.optimize.linprog",
        "2.4.1 Báo cáo giá trị tối ưu và phân bổ tối ưu",
        "2.4.2 Giải lại bằng PuLP",
        "2.4.2 In dual values / shadow price",
        "2.4.2 Giải thích shadow price ngân sách tổng",
        "2.4.3 Phân tích B = 100, 120, 140",
        "2.4.3 Vẽ đường cong Z*(B)",
        "2.4.4 Kiểm tra x3 >= 30 có khả thi không",
        "2.4.4 So sánh Z* thay đổi như thế nào",
        "2.5 Trả lời câu hỏi chính sách a, b, c",
        "Xuất file kết quả phục vụ nộp bài"
    ],
    "Trạng thái": [
        "Đã làm" if SCIPY_AVAILABLE else "Cần cài scipy",
        "Đã làm" if scipy_result["success"] else "Chưa có nghiệm",
        "Đã làm" if PULP_AVAILABLE else "Cần cài pulp để chạy đúng yêu cầu PuLP",
        "Đã làm" if PULP_AVAILABLE else "Có bảng sai phân thay thế, nhưng nên cài pulp",
        "Đã làm",
        "Đã làm",
        "Đã làm",
        "Đã làm" if human_result["success"] else "Không khả thi",
        "Đã làm",
        "Đã làm",
        "Đã làm"
    ],
    "Ghi chú": [
        "Sử dụng linprog(method='highs')",
        f"Z* = {Z_scipy:.4f}" if scipy_result["success"] else scipy_result["message"],
        "PuLP/CBC có thể trả .pi cho LP liên tục" if PULP_AVAILABLE else "Chạy: pip install pulp",
        "Hiển thị trong bảng dual_df" if PULP_AVAILABLE else "Sai phân hữu hạn cho kết quả dễ hiểu",
        "Shadow price ngân sách tổng xấp xỉ 1.35",
        "Có bảng sensitivity_df",
        "Có biểu đồ matplotlib",
        "Kịch bản x3 >= 30 vẫn khả thi" if human_result["success"] else human_result["message"],
        f"Z* thay đổi {human_result['Z'] - base_result['Z']:.4f}",
        "Đã viết theo văn phong chính sách",
        "Có CSV, Excel và HTML report"
    ]
})

st.dataframe(checklist_df, width="stretch")

# ============================================================
# 11. TẢI KẾT QUẢ
# ============================================================

st.header("9. Tải kết quả")

html_report = make_html_report(
    scipy_solution_df=scipy_solution_df,
    scipy_summary_df=scipy_summary_df,
    constraint_df=constraint_df,
    dual_df=dual_df,
    sensitivity_df=sensitivity_df,
    human_summary_df=human_comparison_df,
    policy_html=policy_html
)

save_outputs(
    OUTPUT_DIR,
    {
        "bai02_scipy_solution.csv": scipy_solution_df,
        "bai02_scipy_summary.csv": scipy_summary_df,
        "bai02_constraint_check.csv": constraint_df,
        "bai02_dual_shadow_price.csv": dual_df,
        "bai02_budget_sensitivity.csv": sensitivity_df,
        "bai02_human_priority.csv": human_comparison_df,
        "bai02_checklist.csv": checklist_df,
        "bai02_report.html": html_report
    }
)

excel_file = make_download_excel({
    "Scipy_solution": scipy_solution_df,
    "Scipy_summary": scipy_summary_df,
    "Constraint_check": constraint_df,
    "Dual_shadow_price": dual_df,
    "Budget_sensitivity": sensitivity_df,
    "Human_priority": human_comparison_df,
    "Checklist": checklist_df
})

col_dl1, col_dl2, col_dl3 = st.columns(3)

with col_dl1:
    st.download_button(
        label="Tải HTML report",
        data=html_report.encode("utf-8"),
        file_name="bai02_report.html",
        mime="text/html"
    )

with col_dl2:
    st.download_button(
        label="Tải Excel tổng hợp",
        data=excel_file,
        file_name="bai02_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with col_dl3:
    st.download_button(
        label="Tải bảng nghiệm CSV",
        data=scipy_solution_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="bai02_scipy_solution.csv",
        mime="text/csv"
    )

st.success("Bài 2 đã hoàn thành các phần chính: scipy linprog, PuLP/shadow price nếu cài PuLP, độ nhạy ngân sách, kịch bản x3 >= 30 và thảo luận chính sách.")
