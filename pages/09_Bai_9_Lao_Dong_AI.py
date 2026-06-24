# -*- coding: utf-8 -*-
"""
Bài 9 — Tác động AI tới thị trường lao động Việt Nam
Webapp Streamlit cho môn Các mô hình ra quyết định.

File gợi ý đặt trong thư mục pages/ của webapp Streamlit:
    pages/09_Bai_9_Lao_Dong_AI.py

Nội dung đã bao phủ theo đề:
- 9.4.1: Cài đặt mô hình LP bằng PuLP; nếu chưa cài PuLP thì tự động dùng SciPy HiGHS.
          In phân bổ tối ưu x_AI, x_H; tính NewJob, UpgradeJob, DisplacedJob,
          RetrainingCapacity, NetJob theo từng ngành và tổng cộng.
- 9.4.2: Tính ngưỡng đầu tư đào tạo lại x_H tối thiểu cho ngành CN chế biến chế tạo
          khi x_AI ngành 2 ở mức tối đa; giải thích cả ngưỡng NetJob >= 0 và
          ngưỡng DisplacedJob <= RetrainingCapacity.
- 9.4.3: Mô phỏng nhóm dễ bị tổn thương trong ngành 1, 3, 4; vẽ biểu đồ swimming lane
          và Sankey nếu môi trường có Plotly.
- 9.4.4: Thêm ràng buộc không ngành nào mất quá 5% lao động và kiểm tra bài toán
          có còn khả thi không.
- 9.5: Trả lời câu hỏi thảo luận chính sách a, b, c, d.

Lưu ý mô hình:
Đề bài có công thức NewJob_i = a1_i*xAI_i + a2_i*xD_i, nhưng bài toán tối ưu 9.2
và gợi ý CVXPY chỉ có hai biến quyết định x_AI và x_H. Vì vậy code mặc định đặt
x_D = 0 để bám đúng gợi ý và yêu cầu LP hai hạng mục. Nếu muốn mở rộng thêm x_D,
có thể thêm biến x_D và ngân sách tương ứng.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# 0. CẤU HÌNH GIAO DIỆN
# ============================================================

st.set_page_config(
    page_title="Bài 9 - Lao động và AI",
    page_icon="👷",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from utils.style import load_css, hero, card
except Exception:
    try:
        from style import load_css, hero, card
    except Exception:
        def load_css() -> None:
            st.markdown(
                """
                <style>
                .block-container {padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1200px;}
                .hero-card {
                    padding: 26px 30px; border-radius: 22px;
                    background: linear-gradient(135deg, rgba(79,70,229,.18), rgba(236,72,153,.10));
                    border: 1px solid rgba(148,163,184,.22); margin-bottom: 22px;
                }
                .pill {
                    display:inline-block; padding: 7px 13px; margin: 3px 5px 8px 0;
                    border-radius:999px; font-size:.82rem; font-weight:700;
                    background:linear-gradient(90deg,#2563eb,#7c3aed); color:white;
                }
                .note-box {
                    padding: 15px 18px; border-radius: 15px;
                    border-left: 5px solid #f59e0b;
                    background: rgba(245,158,11,.10); margin: 12px 0 16px 0;
                }
                .ok-box {
                    padding: 15px 18px; border-radius: 15px;
                    border-left: 5px solid #22c55e;
                    background: rgba(34,197,94,.10); margin: 12px 0 16px 0;
                }
                .bad-box {
                    padding: 15px 18px; border-radius: 15px;
                    border-left: 5px solid #ef4444;
                    background: rgba(239,68,68,.10); margin: 12px 0 16px 0;
                }
                div[data-testid="stMetric"] {
                    border: 1px solid rgba(148,163,184,.25);
                    border-radius: 16px; padding: 14px;
                }
                .small-muted {opacity: .76; font-size: .92rem;}
                </style>
                """,
                unsafe_allow_html=True,
            )

        def hero(title: str, subtitle: str = "", badges: Optional[List[str]] = None) -> None:
            badges = badges or []
            badge_html = "".join([f'<span class="pill">{b}</span>' for b in badges])
            st.markdown(
                f"""
                <div class="hero-card">
                    <div>{badge_html}</div>
                    <h1>{title}</h1>
                    <p style="font-size:1.04rem; line-height:1.72; margin-bottom:0;">{subtitle}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        def card(title: str, text: str) -> None:
            st.info(f"**{title}**\n\n{text}")

load_css()

hero(
    title="👷 Bài 9 — Tác động AI tới thị trường lao động Việt Nam",
    subtitle=(
        "Mô phỏng tác động của AI và tự động hóa tới việc làm theo ngành; "
        "tối ưu phân bổ ngân sách AI và đào tạo lại để NetJob ròng không âm, "
        "đồng thời kiểm tra năng lực đào tạo lại và ràng buộc an sinh lao động."
    ),
    badges=["Cấp độ khá khó", "Linear Programming", "PuLP/CVXPY", "NetJob", "Labor simulation"],
)

# ============================================================
# 1. THƯ VIỆN TỐI ƯU TÙY CHỌN
# ============================================================

try:
    import pulp
    HAS_PULP = True
except Exception:
    pulp = None
    HAS_PULP = False

try:
    from scipy.optimize import linprog
    HAS_SCIPY = True
except Exception:
    linprog = None
    HAS_SCIPY = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    go = None
    HAS_PLOTLY = False

# ============================================================
# 2. DỮ LIỆU VÀ THAM SỐ THEO ĐỀ
# ============================================================

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_CANDIDATES = [
    Path("data") / "vietnam_sectors_2024.csv",
    Path("vietnam_sectors_2024.csv"),
]

# 8 ngành theo đề: bỏ Khai khoáng và Y tế.
SECTOR_NAMES = [
    "Nông-Lâm-Thủy sản",
    "CN chế biến chế tạo",
    "Xây dựng",
    "Bán buôn-bán lẻ",
    "Tài chính-Ngân hàng",
    "Logistics-Vận tải",
    "CNTT-Truyền thông",
    "Giáo dục-Đào tạo",
]

# Bảng 9.3 trong đề.
LABOR_MILLION = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15], dtype=float)
RISK_PCT = np.array([18, 42, 25, 38, 52, 35, 28, 22], dtype=float)
A1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5], dtype=float)
A2 = np.array([12.0, 18.5, 8.5, 15.2, 12.5, 16.8, 15.0, 22.0], dtype=float)
B1 = np.array([45, 28, 35, 32, 22, 30, 20, 55], dtype=float)
C1 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5], dtype=float)
D1 = np.array([50, 32, 42, 38, 26, 36, 24, 62], dtype=float)

N = len(SECTOR_NAMES)
BUDGET_DEFAULT = 30000.0  # tỷ VND
RISK = RISK_PCT / 100.0
DISPLACED_PER_AI = C1 * RISK
NET_AI_COEF = A1 - DISPLACED_PER_AI

VULNERABLE_INDICES = [0, 2, 3]  # ngành 1, 3, 4 theo đề

# ============================================================
# 3. HÀM DỮ LIỆU VÀ BẢNG KẾT QUẢ
# ============================================================

def load_sector_reference() -> Tuple[pd.DataFrame, str]:
    """Đọc CSV nếu có để kiểm tra tên ngành, lao động và risk; nếu không có thì dùng bảng đề."""
    for path in DATA_CANDIDATES:
        if path.exists():
            try:
                df_csv = pd.read_csv(path)
                needed = {"sector_name_vi", "labor_million", "automation_risk_pct"}
                if needed.issubset(df_csv.columns):
                    df_work = df_csv.copy()
                    # Bỏ Khai khoáng và Y tế cho đúng đề.
                    mask_drop = (
                        df_work["sector_name_vi"].str.contains("Khai khoáng", case=False, regex=False)
                        | df_work["sector_name_vi"].str.contains("Y tế", case=False, regex=False)
                    )
                    df_work = df_work.loc[~mask_drop].copy().reset_index(drop=True)
                    if len(df_work) == N:
                        return df_work, f"Đọc tham chiếu từ {path}. Tham số a1, a2, b1, c1, d1 vẫn lấy theo bảng 9.3 trong đề."
            except Exception:
                pass

    df_default = pd.DataFrame({
        "sector_id": np.arange(1, N + 1),
        "sector_name_vi": SECTOR_NAMES,
        "labor_million": LABOR_MILLION,
        "automation_risk_pct": RISK_PCT,
    })
    return df_default, "Không tìm thấy CSV phù hợp; dùng trực tiếp bảng 9.3 trong đề."


def build_parameter_df() -> pd.DataFrame:
    """Bảng tham số đầy đủ của Bài 9."""
    return pd.DataFrame({
        "sector_id": np.arange(1, N + 1),
        "Ngành": SECTOR_NAMES,
        "Lao động (triệu)": LABOR_MILLION,
        "Risk (%)": RISK_PCT,
        "a1 - việc/tỷ x_AI": A1,
        "a2 - việc/tỷ x_D": A2,
        "b1 - việc/tỷ x_H": B1,
        "c1 - việc/tỷ x_AI": C1,
        "d1 - việc/tỷ x_H": D1,
        "Displaced/1 tỷ AI = c1*risk": DISPLACED_PER_AI,
        "NetJob/1 tỷ AI nếu chưa tính H": NET_AI_COEF,
        "x_H tối thiểu / 1 tỷ AI để đào tạo lại": np.divide(DISPLACED_PER_AI, D1, out=np.zeros_like(DISPLACED_PER_AI), where=D1 != 0),
    })


def evaluate_solution(x_ai: np.ndarray, x_h: np.ndarray, x_d: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Tính NewJob, UpgradeJob, DisplacedJob, RetrainingCapacity và NetJob."""
    if x_d is None:
        x_d = np.zeros(N, dtype=float)
    x_ai = np.asarray(x_ai, dtype=float)
    x_h = np.asarray(x_h, dtype=float)
    x_d = np.asarray(x_d, dtype=float)

    new_job_ai = A1 * x_ai
    new_job_d = A2 * x_d
    new_job = new_job_ai + new_job_d
    upgrade = B1 * x_h
    displaced = DISPLACED_PER_AI * x_ai
    retrain_cap = D1 * x_h
    net_job = new_job + upgrade - displaced
    budget_used = x_ai + x_h + x_d

    out = pd.DataFrame({
        "sector_id": np.arange(1, N + 1),
        "Ngành": SECTOR_NAMES,
        "x_AI (tỷ VND)": x_ai,
        "x_H đào tạo lại (tỷ VND)": x_h,
        "x_D (tỷ VND, mặc định 0)": x_d,
        "Tổng đầu tư ngành": budget_used,
        "NewJob_AI": new_job_ai,
        "NewJob_D": new_job_d,
        "NewJob_total": new_job,
        "UpgradeJob": upgrade,
        "DisplacedJob": displaced,
        "RetrainingCapacity": retrain_cap,
        "NetJob": net_job,
        "NetJob >= 0?": net_job >= -1e-6,
        "Đào tạo đủ cho displaced?": retrain_cap + 1e-3 >= displaced,
    })
    return out


def summarize_solution(result_df: pd.DataFrame, budget_limit: float) -> pd.DataFrame:
    """Bảng tóm tắt nghiệm tối ưu."""
    total_budget = float(result_df["Tổng đầu tư ngành"].sum())
    total_ai = float(result_df["x_AI (tỷ VND)"].sum())
    total_h = float(result_df["x_H đào tạo lại (tỷ VND)"].sum())
    total_displaced = float(result_df["DisplacedJob"].sum())
    total_retrain = float(result_df["RetrainingCapacity"].sum())
    total_net = float(result_df["NetJob"].sum())
    selected_h = result_df.sort_values("x_H đào tạo lại (tỷ VND)", ascending=False).iloc[0]
    selected_ai = result_df.sort_values("x_AI (tỷ VND)", ascending=False).iloc[0]

    return pd.DataFrame({
        "Chỉ tiêu": [
            "Ngân sách sử dụng",
            "Ngân sách tối đa",
            "Tổng đầu tư AI",
            "Tổng đầu tư đào tạo lại H",
            "Tổng DisplacedJob",
            "Tổng RetrainingCapacity",
            "Tổng NetJob",
            "Ngành nhận H nhiều nhất",
            "Ngành nhận AI nhiều nhất",
        ],
        "Giá trị": [
            total_budget,
            budget_limit,
            total_ai,
            total_h,
            total_displaced,
            total_retrain,
            total_net,
            selected_h["Ngành"],
            selected_ai["Ngành"],
        ],
        "Đơn vị/Ghi chú": [
            "tỷ VND",
            "tỷ VND",
            "tỷ VND",
            "tỷ VND",
            "việc làm",
            "việc làm",
            "việc làm",
            f"x_H = {selected_h['x_H đào tạo lại (tỷ VND)']:.2f} tỷ VND",
            f"x_AI = {selected_ai['x_AI (tỷ VND)']:.2f} tỷ VND",
        ],
    })


def make_constraint_check(result_df: pd.DataFrame, budget_limit: float, five_pct_cap: bool = False) -> pd.DataFrame:
    """Bảng kiểm tra ràng buộc."""
    total_budget = float(result_df["Tổng đầu tư ngành"].sum())
    labor_jobs = LABOR_MILLION * 1_000_000.0
    cap_5pct = 0.05 * labor_jobs

    rows = [
        {
            "Ràng buộc": "Tổng ngân sách Σ(x_AI + x_H) <= B",
            "Giá trị kiểm tra": total_budget,
            "Ngưỡng": budget_limit,
            "Đạt?": total_budget <= budget_limit + 1e-6,
            "Ghi chú": "Đơn vị: tỷ VND",
        },
        {
            "Ràng buộc": "NetJob_i >= 0 với mọi ngành",
            "Giá trị kiểm tra": float(result_df["NetJob"].min()),
            "Ngưỡng": 0.0,
            "Đạt?": bool((result_df["NetJob"] >= -1e-6).all()),
            "Ghi chú": "Giá trị là NetJob nhỏ nhất",
        },
        {
            "Ràng buộc": "DisplacedJob_i <= RetrainingCapacity_i",
            "Giá trị kiểm tra": float((result_df["RetrainingCapacity"] - result_df["DisplacedJob"]).min()),
            "Ngưỡng": 0.0,
            "Đạt?": bool((result_df["RetrainingCapacity"] + 1e-3 >= result_df["DisplacedJob"]).all()),
            "Ghi chú": "Giá trị là phần dư đào tạo nhỏ nhất",
        },
    ]

    if five_pct_cap:
        max_ratio = float((result_df["DisplacedJob"].to_numpy() / labor_jobs).max())
        rows.append({
            "Ràng buộc": "DisplacedJob_i <= 5% lao động ngành",
            "Giá trị kiểm tra": max_ratio,
            "Ngưỡng": 0.05,
            "Đạt?": max_ratio <= 0.05 + 1e-9,
            "Ghi chú": "L_i trong đề là triệu lao động, code đổi sang số việc làm = L_i*1.000.000",
        })

    return pd.DataFrame(rows)

# ============================================================
# 4. GIẢI MÔ HÌNH LP
# ============================================================

def solve_lp_with_pulp(
    budget_limit: float = BUDGET_DEFAULT,
    five_pct_cap: bool = False,
    min_ai_budget: float = 0.0,
    max_sector_budget: Optional[float] = None,
) -> Dict:
    """Giải LP bằng PuLP/CBC."""
    if not HAS_PULP:
        return solve_lp_with_scipy(
            budget_limit=budget_limit,
            five_pct_cap=five_pct_cap,
            min_ai_budget=min_ai_budget,
            max_sector_budget=max_sector_budget,
        )

    model = pulp.LpProblem("Bai9_AI_Labor_NetJob", pulp.LpMaximize)

    x_ai = pulp.LpVariable.dicts("x_AI", range(N), lowBound=0, cat="Continuous")
    x_h = pulp.LpVariable.dicts("x_H", range(N), lowBound=0, cat="Continuous")

    # NetJob_i = (a1_i - c1_i*risk_i)*x_AI_i + b1_i*x_H_i
    model += pulp.lpSum(NET_AI_COEF[i] * x_ai[i] + B1[i] * x_h[i] for i in range(N)), "Total_NetJob"

    model += pulp.lpSum(x_ai[i] + x_h[i] for i in range(N)) <= budget_limit, "C1_Total_budget"

    if min_ai_budget > 0:
        model += pulp.lpSum(x_ai[i] for i in range(N)) >= min_ai_budget, "C_optional_Min_AI_budget"

    for i in range(N):
        net_i = NET_AI_COEF[i] * x_ai[i] + B1[i] * x_h[i]
        displaced_i = DISPLACED_PER_AI[i] * x_ai[i]
        retrain_i = D1[i] * x_h[i]

        model += net_i >= 0, f"C2_NetJob_nonnegative_sector_{i+1}"
        model += displaced_i <= retrain_i, f"C3_Retrain_capacity_sector_{i+1}"

        if five_pct_cap:
            labor_jobs_i = LABOR_MILLION[i] * 1_000_000.0
            model += displaced_i <= 0.05 * labor_jobs_i, f"C4_Five_pct_labor_cap_sector_{i+1}"

        if max_sector_budget is not None and max_sector_budget > 0:
            model += x_ai[i] + x_h[i] <= max_sector_budget, f"C_optional_Max_sector_budget_{i+1}"

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)
    status = pulp.LpStatus.get(model.status, str(model.status))

    if status != "Optimal":
        return {
            "success": False,
            "status": status,
            "solver": "PuLP/CBC",
            "objective": np.nan,
            "x_ai": np.zeros(N),
            "x_h": np.zeros(N),
            "message": f"PuLP status: {status}",
            "model": model,
        }

    x_ai_val = np.array([float(pulp.value(x_ai[i])) for i in range(N)])
    x_h_val = np.array([float(pulp.value(x_h[i])) for i in range(N)])
    objective = float(pulp.value(model.objective))

    dual_rows = []
    for name, con in model.constraints.items():
        dual_rows.append({
            "Ràng buộc": name,
            "Slack": getattr(con, "slack", np.nan),
            "Shadow price / dual": getattr(con, "pi", np.nan),
        })
    dual_df = pd.DataFrame(dual_rows)

    return {
        "success": True,
        "status": status,
        "solver": "PuLP/CBC",
        "objective": objective,
        "x_ai": x_ai_val,
        "x_h": x_h_val,
        "message": f"PuLP status: {status}",
        "model": model,
        "dual_df": dual_df,
    }


def solve_lp_with_scipy(
    budget_limit: float = BUDGET_DEFAULT,
    five_pct_cap: bool = False,
    min_ai_budget: float = 0.0,
    max_sector_budget: Optional[float] = None,
) -> Dict:
    """Giải LP bằng scipy.optimize.linprog nếu PuLP chưa cài."""
    if not HAS_SCIPY:
        return {
            "success": False,
            "status": "not_available",
            "solver": "SciPy HiGHS",
            "objective": np.nan,
            "x_ai": np.zeros(N),
            "x_h": np.zeros(N),
            "message": "Chưa cài PuLP hoặc SciPy. Hãy cài: python -m pip install pulp scipy",
        }

    # Biến z = [x_AI_1..x_AI_8, x_H_1..x_H_8]
    c = np.r_[-NET_AI_COEF, -B1]
    A_ub = []
    b_ub = []

    # Ngân sách tổng
    A_ub.append(np.r_[np.ones(N), np.ones(N)])
    b_ub.append(budget_limit)

    # Min AI budget: sum x_AI >= min_ai_budget -> -sum x_AI <= -min_ai_budget
    if min_ai_budget > 0:
        row = np.zeros(2 * N)
        row[:N] = -1.0
        A_ub.append(row)
        b_ub.append(-min_ai_budget)

    for i in range(N):
        # NetJob_i >= 0 -> -NET_AI_COEF*xAI - B1*xH <= 0
        row = np.zeros(2 * N)
        row[i] = -NET_AI_COEF[i]
        row[N + i] = -B1[i]
        A_ub.append(row)
        b_ub.append(0.0)

        # Displaced <= RetrainCap -> DISPLACED_PER_AI*xAI - D1*xH <= 0
        row = np.zeros(2 * N)
        row[i] = DISPLACED_PER_AI[i]
        row[N + i] = -D1[i]
        A_ub.append(row)
        b_ub.append(0.0)

        if five_pct_cap:
            row = np.zeros(2 * N)
            row[i] = DISPLACED_PER_AI[i]
            A_ub.append(row)
            b_ub.append(0.05 * LABOR_MILLION[i] * 1_000_000.0)

        if max_sector_budget is not None and max_sector_budget > 0:
            row = np.zeros(2 * N)
            row[i] = 1.0
            row[N + i] = 1.0
            A_ub.append(row)
            b_ub.append(max_sector_budget)

    res = linprog(
        c,
        A_ub=np.asarray(A_ub, dtype=float),
        b_ub=np.asarray(b_ub, dtype=float),
        bounds=[(0, None)] * (2 * N),
        method="highs",
    )

    if not res.success:
        return {
            "success": False,
            "status": "infeasible_or_error",
            "solver": "SciPy HiGHS",
            "objective": np.nan,
            "x_ai": np.zeros(N),
            "x_h": np.zeros(N),
            "message": res.message,
        }

    return {
        "success": True,
        "status": "Optimal",
        "solver": "SciPy HiGHS",
        "objective": float(-res.fun),
        "x_ai": np.asarray(res.x[:N], dtype=float),
        "x_h": np.asarray(res.x[N:], dtype=float),
        "message": res.message,
    }

# ============================================================
# 5. CÂU 9.4.2 - NGƯỠNG ĐÀO TẠO NGÀNH 2
# ============================================================

def threshold_for_sector_2(max_ai_budget: float = BUDGET_DEFAULT) -> pd.DataFrame:
    """Tính ngưỡng x_H tối thiểu cho ngành 2 khi x_AI2 ở mức tối đa."""
    idx = 1  # ngành 2: CN chế biến chế tạo
    x_ai_max = float(max_ai_budget)

    # Điều kiện NetJob2 >= 0:
    # (a1 - c1*risk)*x_AI + b1*x_H >= 0
    # Nếu a1 - c1*risk >= 0 thì x_H_min_net = 0.
    numerator_net = (DISPLACED_PER_AI[idx] - A1[idx]) * x_ai_max
    x_h_min_net = max(0.0, numerator_net / B1[idx]) if B1[idx] > 0 else np.inf

    # Điều kiện đào tạo đủ: Displaced <= RetrainCap
    x_h_min_retrain = (DISPLACED_PER_AI[idx] * x_ai_max) / D1[idx] if D1[idx] > 0 else np.inf

    # Nếu x_AI và x_H cùng dùng chung ngân sách, x_AI tối đa khả thi dưới điều kiện đào tạo đủ là:
    # x_AI + (displaced_per_ai/d1)*x_AI <= B
    ratio_h_per_ai = DISPLACED_PER_AI[idx] / D1[idx]
    x_ai_max_feasible_shared_budget = x_ai_max / (1.0 + ratio_h_per_ai)
    x_h_needed_at_feasible_ai = ratio_h_per_ai * x_ai_max_feasible_shared_budget

    return pd.DataFrame([
        {
            "Cách hiểu": "Chỉ yêu cầu NetJob₂ >= 0, đặt x_AI₂ = ngân sách tối đa",
            "x_AI₂ giả định": x_ai_max,
            "x_H₂ tối thiểu": x_h_min_net,
            "Công thức": "max(0, (c1*risk - a1)*x_AI / b1)",
            "Nhận xét": "Với tham số đề, a1 - c1*risk > 0 nên NetJob₂ không âm ngay cả khi x_H₂ = 0.",
        },
        {
            "Cách hiểu": "Yêu cầu đào tạo đủ: DisplacedJob₂ <= RetrainingCapacity₂",
            "x_AI₂ giả định": x_ai_max,
            "x_H₂ tối thiểu": x_h_min_retrain,
            "Công thức": "c1*risk*x_AI / d1",
            "Nhận xét": "Đây là ngưỡng quan trọng hơn về an sinh: tự động hóa phải đi kèm năng lực đào tạo lại.",
        },
        {
            "Cách hiểu": "Nếu x_AI₂ và x_H₂ cùng chia sẻ ngân sách 30.000 tỷ",
            "x_AI₂ giả định": x_ai_max_feasible_shared_budget,
            "x_H₂ tối thiểu": x_h_needed_at_feasible_ai,
            "Công thức": "x_AI <= B / (1 + c1*risk/d1)",
            "Nhận xét": "Không thể vừa đầu tư AI₂ = 30.000 vừa thêm đào tạo lại nếu tổng ngân sách vẫn chỉ là 30.000.",
        },
    ])

# ============================================================
# 6. CÂU 9.4.3 - NHÓM DỄ BỊ TỔN THƯƠNG
# ============================================================

def build_vulnerable_flow_df(
    result_df: pd.DataFrame,
    vulnerable_share: float = 0.60,
    demo_ai_budget: float = 500.0,
    use_demo_when_zero: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """Tính các luồng lao động phổ thông ở ngành 1, 3, 4."""
    rows = []
    note = "Dùng trực tiếp nghiệm tối ưu."

    for i in VULNERABLE_INDICES:
        x_ai = float(result_df.loc[i, "x_AI (tỷ VND)"])
        x_h = float(result_df.loc[i, "x_H đào tạo lại (tỷ VND)"])

        if use_demo_when_zero and x_ai <= 1e-9:
            # Tạo kịch bản minh họa để swimming lane có ý nghĩa khi nghiệm tối ưu không đầu tư AI vào ngành này.
            x_ai = demo_ai_budget
            # Gán x_H vừa đủ để đào tạo lại displaced trong kịch bản minh họa.
            x_h = (DISPLACED_PER_AI[i] * x_ai) / D1[i] if D1[i] > 0 else 0.0
            note = (
                "Nghiệm tối ưu không tạo luồng dịch chuyển ở ngành 1, 3, 4 nên biểu đồ dùng kịch bản minh họa: "
                f"x_AI = {demo_ai_budget:,.0f} tỷ/ngành và x_H vừa đủ để đào tạo lại."
            )

        vulnerable_workers = LABOR_MILLION[i] * 1_000_000.0 * vulnerable_share
        displaced = min(DISPLACED_PER_AI[i] * x_ai, vulnerable_workers)
        retrain_capacity = D1[i] * x_h
        retrained = min(displaced, retrain_capacity)
        residual_at_risk = max(0.0, displaced - retrained)
        unaffected = max(0.0, vulnerable_workers - displaced)
        upgraded = B1[i] * x_h
        new_jobs_ai = A1[i] * x_ai

        rows.append({
            "sector_id": i + 1,
            "Ngành": SECTOR_NAMES[i],
            "Lao động phổ thông giả định": vulnerable_workers,
            "x_AI dùng mô phỏng": x_ai,
            "x_H dùng mô phỏng": x_h,
            "Không bị dịch chuyển": unaffected,
            "Bị tự động hóa tác động": displaced,
            "Được đào tạo lại/bảo vệ": retrained,
            "Còn rủi ro sau đào tạo": residual_at_risk,
            "UpgradeJob từ đào tạo": upgraded,
            "NewJob_AI": new_jobs_ai,
        })

    return pd.DataFrame(rows), note


def plot_swimming_lane(flow_df: pd.DataFrame):
    """Vẽ biểu đồ swimming lane bằng matplotlib."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    y_pos = np.arange(len(flow_df))

    unaffected = flow_df["Không bị dịch chuyển"].to_numpy(dtype=float)
    retrained = flow_df["Được đào tạo lại/bảo vệ"].to_numpy(dtype=float)
    residual = flow_df["Còn rủi ro sau đào tạo"].to_numpy(dtype=float)

    ax.barh(y_pos, unaffected, label="Không bị dịch chuyển")
    ax.barh(y_pos, retrained, left=unaffected, label="Được đào tạo lại/bảo vệ")
    ax.barh(y_pos, residual, left=unaffected + retrained, label="Còn rủi ro")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(flow_df["Ngành"].tolist())
    ax.set_xlabel("Số lao động/việc làm")
    ax.set_title("Swimming lane mô phỏng luồng lao động phổ thông ở ngành 1, 3, 4")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)

    for y, total in zip(y_pos, flow_df["Lao động phổ thông giả định"]):
        ax.text(total * 1.005, y, f"Tổng: {total:,.0f}", va="center", fontsize=8)

    fig.tight_layout()
    return fig


def plot_sankey_if_available(flow_df: pd.DataFrame):
    """Vẽ Sankey nếu plotly có sẵn."""
    if not HAS_PLOTLY:
        return None

    labels = ["Lao động phổ thông"]
    sources = []
    targets = []
    values = []

    for _, row in flow_df.iterrows():
        sector_label = str(row["Ngành"])
        displaced_label = f"{sector_label} - bị tác động"
        retrained_label = f"{sector_label} - đào tạo lại"
        residual_label = f"{sector_label} - còn rủi ro"
        unaffected_label = f"{sector_label} - không bị dịch chuyển"

        base_idx = len(labels)
        labels.extend([sector_label, displaced_label, retrained_label, residual_label, unaffected_label])

        # Lao động phổ thông -> ngành
        sources.append(0)
        targets.append(base_idx)
        values.append(float(row["Lao động phổ thông giả định"]))

        # Ngành -> không bị dịch chuyển
        sources.append(base_idx)
        targets.append(base_idx + 4)
        values.append(float(row["Không bị dịch chuyển"]))

        # Ngành -> bị tác động
        sources.append(base_idx)
        targets.append(base_idx + 1)
        values.append(float(row["Bị tự động hóa tác động"]))

        # Bị tác động -> đào tạo lại
        sources.append(base_idx + 1)
        targets.append(base_idx + 2)
        values.append(float(row["Được đào tạo lại/bảo vệ"]))

        # Bị tác động -> còn rủi ro
        sources.append(base_idx + 1)
        targets.append(base_idx + 3)
        values.append(float(row["Còn rủi ro sau đào tạo"]))

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=16, thickness=16),
        link=dict(source=sources, target=targets, value=values),
    )])
    fig.update_layout(title_text="Sankey mô phỏng luồng dịch chuyển lao động phổ thông", font_size=10)
    return fig

# ============================================================
# 7. BÁO CÁO, EXCEL, CHÍNH SÁCH
# ============================================================

def make_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df_sheet in sheets.items():
            df_sheet.to_excel(writer, index=False, sheet_name=name[:31])
    return buffer.getvalue()


def make_policy_html(
    base_result_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    cap_result_df: pd.DataFrame,
    budget_limit: float,
) -> str:
    """Tạo phần trả lời câu hỏi chính sách 9.5."""
    top_h = base_result_df.sort_values("x_H đào tạo lại (tỷ VND)", ascending=False).iloc[0]
    finance_row = base_result_df.loc[base_result_df["Ngành"] == "Tài chính-Ngân hàng"].iloc[0]
    agri_row = base_result_df.loc[base_result_df["Ngành"] == "Nông-Lâm-Thủy sản"].iloc[0]
    cap_feasible = bool((cap_result_df["NetJob"] >= -1e-6).all()) if not cap_result_df.empty else False

    html = f"""
    <h3>a) Ngành nào cần đầu tư đào tạo lại nhiều nhất?</h3>
    <p>
    Theo nghiệm tối ưu cơ sở, ngành nhận đầu tư đào tạo lại lớn nhất là
    <b>{top_h['Ngành']}</b>, với x_H khoảng <b>{float(top_h['x_H đào tạo lại (tỷ VND)']):,.2f} tỷ VND</b>.
    Kết quả này xuất hiện vì hệ số b1 của ngành này cao nhất trong bảng tham số, tức mỗi 1 tỷ VND đào tạo lại
    tạo ra nhiều UpgradeJob nhất. Về mặt thực tế, kết quả này cần được đọc cẩn trọng: đề chưa đặt trần năng lực hấp thụ
    hoặc trần ngân sách từng ngành, nên mô hình LP có xu hướng dồn vốn vào ngành có hiệu quả biên cao nhất.
    </p>

    <h3>b) Ngành Tài chính-Ngân hàng có risk 52% nhưng hệ số tạo việc làm mới cao, mô hình khuyến nghị gì?</h3>
    <p>
    Tài chính-Ngân hàng có risk cao nên mỗi khoản đầu tư AI tạo ra DisplacedJob đáng kể, nhưng a1 cũng cao.
    Trong nghiệm cơ sở, ngành này nhận x_AI = <b>{float(finance_row['x_AI (tỷ VND)']):,.2f}</b> và
    x_H = <b>{float(finance_row['x_H đào tạo lại (tỷ VND)']):,.2f}</b> tỷ VND.
    Hàm ý chính sách là không nên tự động hóa đơn thuần, mà phải đi kèm đào tạo lại kỹ năng phân tích dữ liệu,
    quản trị rủi ro, an toàn thông tin và vận hành hệ thống AI. Nói ngắn gọn: AI trong tài chính nên là chiến lược
    <b>AI + reskilling</b>, không phải thay thế lao động một chiều.
    </p>

    <h3>c) Có nên đầu tư x_AI vào Nông-Lâm-Thủy sản không?</h3>
    <p>
    Trong nghiệm cơ sở, Nông-Lâm-Thủy sản nhận x_AI = <b>{float(agri_row['x_AI (tỷ VND)']):,.2f}</b> tỷ VND.
    Ngành này có a1 thấp hơn nhiều ngành khác, nên nếu chỉ tối đa hóa NetJob ngắn hạn, mô hình thường không ưu tiên AI cho nông nghiệp.
    Tuy nhiên, đây không có nghĩa là không nên đầu tư AI cho nông nghiệp. Với ngành có lực lượng lao động lớn,
    chính sách nên ưu tiên AI hỗ trợ năng suất, khuyến nông số, truy xuất nguồn gốc, tự động hóa vừa phải và đào tạo kỹ năng số cơ bản,
    thay vì triển khai tự động hóa nhanh gây dịch chuyển lao động lớn.
    </p>

    <h3>d) “Tốc độ tự động hóa không nên vượt quá năng lực đào tạo lại” được biểu diễn bằng ràng buộc nào?</h3>
    <p>
    Phát biểu này được biểu diễn trực tiếp bằng ràng buộc:
    <b>DisplacedJobᵢ ≤ RetrainingCapacityᵢ</b>, tức <b>c1ᵢ × riskᵢ × x_AIᵢ ≤ d1ᵢ × x_Hᵢ</b>.
    Code cũng kiểm tra ràng buộc mở rộng <b>DisplacedJobᵢ ≤ 0,05 × Lᵢ</b>.
    Với cách đổi Lᵢ từ triệu lao động sang số việc làm, bài toán có ràng buộc 5% hiện tại
    <b>{'vẫn khả thi' if cap_feasible else 'không khả thi hoặc solver không tìm được nghiệm'}</b>.
    </p>

    <p>
    Có thể bổ sung thêm các ràng buộc an sinh xã hội: trần ngân sách AI ở ngành có nhiều lao động phổ thông,
    sàn đào tạo lại cho nhóm dễ bị tổn thương, trần tốc độ tăng x_AI theo năm, và ràng buộc mỗi ngành phải dành
    một tỷ lệ tối thiểu của đầu tư AI cho đào tạo lại.
    </p>
    """
    return html


def make_html_report(
    parameter_df: pd.DataFrame,
    base_result_df: pd.DataFrame,
    base_summary_df: pd.DataFrame,
    check_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    vulnerable_df: pd.DataFrame,
    cap_result_df: pd.DataFrame,
    cap_summary_df: pd.DataFrame,
    extended_result_df: pd.DataFrame,
    extended_summary_df: pd.DataFrame,
    checklist_df: pd.DataFrame,
    policy_html: str,
    note_html: str,
) -> str:
    css = """
    <style>
    body {font-family: Arial, sans-serif; line-height: 1.5; margin: 30px;}
    h1, h2, h3 {color: #1f3b66;}
    table {border-collapse: collapse; width: 100%; margin-bottom: 24px; font-size: 12.5px;}
    th {background-color: #1f3b66; color: white; padding: 6px; border: 1px solid #ccc;}
    td {padding: 6px; border: 1px solid #ccc; text-align: left;}
    .box {background: #f2f6ff; padding: 15px; border-left: 5px solid #1f3b66; margin-bottom: 20px;}
    </style>
    """
    return f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"><title>Bài 9 - Lao động và AI</title>{css}</head>
    <body>
        <h1>BÀI 9 - TÁC ĐỘNG AI TỚI THỊ TRƯỜNG LAO ĐỘNG VIỆT NAM</h1>
        <div class="box">{note_html}</div>
        <h2>1. Tham số mô hình</h2>
        {parameter_df.to_html(index=False)}
        <h2>2. Nghiệm tối ưu cơ sở</h2>
        {base_summary_df.to_html(index=False)}
        {base_result_df.to_html(index=False)}
        <h2>3. Kiểm tra ràng buộc</h2>
        {check_df.to_html(index=False)}
        <h2>4. Ngưỡng đào tạo ngành 2</h2>
        {threshold_df.to_html(index=False)}
        <h2>5. Nhóm dễ bị tổn thương</h2>
        {vulnerable_df.to_html(index=False)}
        <h2>6. Ràng buộc không ngành nào mất quá 5% lao động</h2>
        {cap_summary_df.to_html(index=False)}
        {cap_result_df.to_html(index=False)}
        <h2>7. Kịch bản mở rộng: bắt buộc AI tối thiểu và trần ngành</h2>
        {extended_summary_df.to_html(index=False)}
        {extended_result_df.to_html(index=False)}
        <h2>8. Câu hỏi thảo luận chính sách</h2>
        {policy_html}
        <h2>9. Checklist</h2>
        {checklist_df.to_html(index=False)}
    </body>
    </html>
    """

# ============================================================
# 8. SIDEBAR ĐIỀU KHIỂN
# ============================================================

st.sidebar.header("⚙️ Thiết lập Bài 9")

budget_limit = st.sidebar.number_input(
    "Ngân sách tổng B (tỷ VND)",
    min_value=1000.0,
    max_value=100000.0,
    value=float(BUDGET_DEFAULT),
    step=1000.0,
)

min_ai_share = st.sidebar.slider(
    "Ràng buộc thêm: AI tối thiểu (% ngân sách)",
    min_value=0,
    max_value=80,
    value=0,
    step=5,
    help="Đề không bắt buộc mục này. Mặc định 0% để bám đúng đề. Dùng để xem nếu Chính phủ bắt buộc có đầu tư AI thì phân bổ đổi thế nào.",
)
min_ai_budget = budget_limit * min_ai_share / 100.0

use_sector_cap = st.sidebar.checkbox(
    "Thêm trần ngân sách mỗi ngành để tránh dồn vốn quá mức",
    value=False,
    help="Đề không yêu cầu. Tùy chọn này chỉ dùng để kiểm tra độ nhạy chính sách.",
)
max_sector_budget = None
if use_sector_cap:
    max_sector_budget = st.sidebar.number_input(
        "Trần ngân sách mỗi ngành (tỷ VND)",
        min_value=500.0,
        max_value=float(budget_limit),
        value=float(budget_limit / 3),
        step=500.0,
    )

vulnerable_share = st.sidebar.slider(
    "Tỷ lệ lao động phổ thông trong ngành 1,3,4",
    min_value=0.10,
    max_value=0.90,
    value=0.60,
    step=0.05,
)

demo_ai_budget = st.sidebar.number_input(
    "AI minh họa cho ngành 1,3,4 nếu nghiệm tối ưu không đầu tư AI",
    min_value=0.0,
    max_value=10000.0,
    value=500.0,
    step=100.0,
)

use_demo_when_zero = st.sidebar.checkbox(
    "Dùng kịch bản minh họa cho swimming lane khi x_AI tối ưu = 0",
    value=True,
)

st.sidebar.markdown("---")
st.sidebar.caption("Lệnh cài thư viện nên có:")
st.sidebar.code("python -m pip install streamlit pandas numpy matplotlib pulp scipy openpyxl", language="bash")
if not HAS_PLOTLY:
    st.sidebar.caption("Muốn có Sankey tương tác thì cài thêm:")
    st.sidebar.code("python -m pip install plotly", language="bash")

# ============================================================
# 9. GIAO DIỆN CHÍNH
# ============================================================

st.markdown(
    """
    <div class="note-box">
    <b>Điểm cần lưu ý trước khi chạy:</b><br>
    (1) Công thức đề có a2*x_D nhưng bài toán tối ưu và gợi ý code chỉ dùng x_AI, x_H, nên app đặt x_D = 0 ở mô hình cơ sở.<br>
    (2) Đề không đặt trần đầu tư theo ngành, nên nghiệm LP cơ sở có thể dồn ngân sách vào ngành có hiệu quả biên cao nhất.<br>
    (3) Với ràng buộc 5% lao động, L_i trong bảng là triệu lao động; app đổi sang số việc làm bằng L_i × 1.000.000 để cùng đơn vị với DisplacedJob.
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# 9.1. Dữ liệu
# ------------------------------------------------------------

st.header("1. Dữ liệu và tham số Bài 9")
sector_ref_df, data_note = load_sector_reference()
parameter_df = build_parameter_df()

col_data1, col_data2 = st.columns([1.2, 1])
with col_data1:
    st.subheader("Bảng tham số 8 ngành theo đề")
    st.dataframe(parameter_df.round(6), use_container_width=True)
with col_data2:
    st.subheader("Tham chiếu CSV")
    st.caption(data_note)
    st.dataframe(sector_ref_df, use_container_width=True)

st.markdown(
    r"""
    Mô hình sử dụng:

    $$
    NetJob_i = NewJob_i + UpgradeJob_i - DisplacedJob_i
    $$

    Với mô hình cơ sở hai biến:

    $$
    NewJob_i = a_{1i}x^{AI}_i, \quad
    UpgradeJob_i = b_{1i}x^H_i, \quad
    DisplacedJob_i = c_{1i}risk_i x^{AI}_i, \quad
    RetrainingCapacity_i = d_{1i}x^H_i
    $$
    """
)

# ------------------------------------------------------------
# 9.4.1. Giải mô hình cơ sở
# ------------------------------------------------------------

st.header("2. Câu 9.4.1 - Giải LP và tính NetJob")

base_solution = solve_lp_with_pulp(
    budget_limit=budget_limit,
    five_pct_cap=False,
    min_ai_budget=min_ai_budget,
    max_sector_budget=max_sector_budget,
)

if not base_solution["success"]:
    st.error(f"Không giải được mô hình cơ sở. Solver: {base_solution['solver']}. Lý do: {base_solution['message']}")
    st.stop()

base_result_df = evaluate_solution(base_solution["x_ai"], base_solution["x_h"])
base_summary_df = summarize_solution(base_result_df, budget_limit=budget_limit)
base_check_df = make_constraint_check(base_result_df, budget_limit=budget_limit, five_pct_cap=False)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Solver", base_solution["solver"])
with m2:
    st.metric("Tổng NetJob", f"{base_solution['objective']:,.0f}")
with m3:
    st.metric("Ngân sách dùng", f"{base_result_df['Tổng đầu tư ngành'].sum():,.0f} tỷ")
with m4:
    st.metric("Tổng AI", f"{base_result_df['x_AI (tỷ VND)'].sum():,.0f} tỷ")

st.subheader("Bảng phân bổ tối ưu và NetJob từng ngành")
st.dataframe(base_result_df.round(4), use_container_width=True)

st.subheader("Tóm tắt nghiệm tối ưu")
st.dataframe(base_summary_df, use_container_width=True)

st.subheader("Kiểm tra ràng buộc mô hình cơ sở")
st.dataframe(base_check_df, use_container_width=True)

if "dual_df" in base_solution:
    with st.expander("Xem shadow price / dual values từ PuLP"):
        st.dataframe(base_solution["dual_df"].round(6), use_container_width=True)

# Biểu đồ phân bổ
st.subheader("Biểu đồ phân bổ x_AI và x_H theo ngành")
fig_alloc, ax_alloc = plt.subplots(figsize=(11, 5.5))
x = np.arange(N)
width = 0.36
ax_alloc.bar(x - width / 2, base_result_df["x_AI (tỷ VND)"], width, label="x_AI")
ax_alloc.bar(x + width / 2, base_result_df["x_H đào tạo lại (tỷ VND)"], width, label="x_H")
ax_alloc.set_xticks(x)
ax_alloc.set_xticklabels(SECTOR_NAMES, rotation=45, ha="right")
ax_alloc.set_ylabel("Tỷ VND")
ax_alloc.set_title("Phân bổ tối ưu ngân sách AI và đào tạo lại")
ax_alloc.legend()
ax_alloc.grid(axis="y", alpha=0.25)
fig_alloc.tight_layout()
st.pyplot(fig_alloc)

st.subheader("Biểu đồ NetJob từng ngành")
fig_net, ax_net = plt.subplots(figsize=(11, 5.2))
ax_net.bar(base_result_df["Ngành"], base_result_df["NetJob"])
ax_net.axhline(0, linewidth=1)
ax_net.set_ylabel("NetJob")
ax_net.set_title("NetJob ròng theo ngành")
ax_net.tick_params(axis="x", rotation=45)
ax_net.grid(axis="y", alpha=0.25)
fig_net.tight_layout()
st.pyplot(fig_net)

# ------------------------------------------------------------
# 9.4.2. Ngưỡng ngành 2
# ------------------------------------------------------------

st.header("3. Câu 9.4.2 - Ngưỡng đào tạo lại tối thiểu cho ngành 2")
threshold_df = threshold_for_sector_2(max_ai_budget=budget_limit)
st.dataframe(threshold_df.round(4), use_container_width=True)

st.markdown(
    """
    <div class="note-box">
    <b>Diễn giải nhanh:</b> Nếu chỉ xét điều kiện NetJob₂ ≥ 0 thì ngành chế biến chế tạo không cần x_H tối thiểu,
    vì phần việc làm mới do AI tạo ra lớn hơn phần mất việc trong công thức NetJob. Tuy nhiên, điều kiện an sinh
    quan trọng hơn là DisplacedJob₂ ≤ RetrainingCapacity₂. Điều kiện này buộc phải có đào tạo lại đủ lớn.
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# 9.4.3. Vulnerable groups
# ------------------------------------------------------------

st.header("4. Câu 9.4.3 - Mô phỏng lao động phổ thông ngành 1, 3, 4")
flow_df, flow_note = build_vulnerable_flow_df(
    base_result_df,
    vulnerable_share=vulnerable_share,
    demo_ai_budget=demo_ai_budget,
    use_demo_when_zero=use_demo_when_zero,
)

st.caption(flow_note)
st.dataframe(flow_df.round(4), use_container_width=True)

fig_lane = plot_swimming_lane(flow_df)
st.pyplot(fig_lane)

sankey_fig = plot_sankey_if_available(flow_df)
if sankey_fig is not None:
    st.plotly_chart(sankey_fig, use_container_width=True)
else:
    st.info("Plotly chưa được cài nên app hiển thị swimming lane bằng matplotlib. Nếu muốn Sankey tương tác, cài: python -m pip install plotly")

# ------------------------------------------------------------
# 9.4.4. Ràng buộc 5% lao động
# ------------------------------------------------------------

st.header("5. Câu 9.4.4 - Thêm ràng buộc không ngành nào mất quá 5% lao động")

cap_solution = solve_lp_with_pulp(
    budget_limit=budget_limit,
    five_pct_cap=True,
    min_ai_budget=min_ai_budget,
    max_sector_budget=max_sector_budget,
)

if not cap_solution["success"]:
    st.error(f"Bài toán có ràng buộc 5% không giải được. Solver: {cap_solution['solver']}. Lý do: {cap_solution['message']}")
    cap_result_df = pd.DataFrame()
    cap_summary_df = pd.DataFrame()
    cap_check_df = pd.DataFrame()
else:
    cap_result_df = evaluate_solution(cap_solution["x_ai"], cap_solution["x_h"])
    cap_summary_df = summarize_solution(cap_result_df, budget_limit=budget_limit)
    cap_check_df = make_constraint_check(cap_result_df, budget_limit=budget_limit, five_pct_cap=True)

    c1_col, c2_col, c3_col = st.columns(3)
    with c1_col:
        st.metric("Trạng thái", cap_solution["status"])
    with c2_col:
        st.metric("Tổng NetJob có ràng buộc 5%", f"{cap_solution['objective']:,.0f}")
    with c3_col:
        delta_obj = cap_solution["objective"] - base_solution["objective"]
        st.metric("Chênh lệch so với cơ sở", f"{delta_obj:,.0f}")

    st.subheader("Bảng nghiệm khi thêm ràng buộc 5%")
    st.dataframe(cap_result_df.round(4), use_container_width=True)

    st.subheader("Kiểm tra ràng buộc 5%")
    st.dataframe(cap_check_df, use_container_width=True)

# ------------------------------------------------------------
# 9.4 mở rộng - Kịch bản tránh nghiệm góc
# ------------------------------------------------------------

st.header("6. Kịch bản mở rộng - Bắt buộc có AI và trần ngân sách ngành")
extended_min_ai_budget = max(min_ai_budget, 0.20 * budget_limit)
extended_sector_cap = max_sector_budget if max_sector_budget is not None else max(3000.0, 0.25 * budget_limit)
extended_solution = solve_lp_with_pulp(
    budget_limit=budget_limit,
    five_pct_cap=True,
    min_ai_budget=extended_min_ai_budget,
    max_sector_budget=extended_sector_cap,
)
if not extended_solution["success"]:
    st.warning(f"Kịch bản mở rộng không khả thi với AI tối thiểu {extended_min_ai_budget:,.0f} và trần ngành {extended_sector_cap:,.0f}. Giữ kết quả rỗng để báo cáo giới hạn mô hình.")
    extended_result_df = pd.DataFrame()
    extended_summary_df = pd.DataFrame()
else:
    extended_result_df = evaluate_solution(extended_solution["x_ai"], extended_solution["x_h"])
    extended_summary_df = summarize_solution(extended_result_df, budget_limit=budget_limit)
    st.markdown(
        f"""
        <div class="note-box">
        <b>Mục đích:</b> Đây là bảng bổ sung cho báo cáo để tránh kết quả LP cơ sở dồn toàn bộ ngân sách vào đào tạo H và x_AI=0.
        Kịch bản này thêm ba ràng buộc chính sách: AI tối thiểu <b>{extended_min_ai_budget:,.0f} tỷ VND</b>,
        trần ngân sách mỗi ngành <b>{extended_sector_cap:,.0f} tỷ VND</b>, và ràng buộc không ngành nào mất quá 5% lao động.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(extended_summary_df, use_container_width=True)
    st.dataframe(extended_result_df.round(4), use_container_width=True)

# ------------------------------------------------------------
# 9.5. Chính sách
# ------------------------------------------------------------

st.header("7. Câu 9.5 - Câu hỏi thảo luận chính sách")

policy_html = make_policy_html(
    base_result_df=base_result_df,
    threshold_df=threshold_df,
    cap_result_df=cap_result_df,
    budget_limit=budget_limit,
)
st.markdown(policy_html, unsafe_allow_html=True)

# ------------------------------------------------------------
# Checklist
# ------------------------------------------------------------

st.header("8. Checklist hoàn thành yêu cầu Bài 9")

checklist_df = pd.DataFrame([
    ["9.4.1", "Cài đặt mô hình tuyến tính bằng PuLP hoặc CVXPY", "Đã làm", f"Solver đang dùng: {base_solution['solver']}"],
    ["9.4.1", "In phân bổ tối ưu x_AI, x_H cho mỗi ngành", "Đã làm", "Có bảng base_result_df"],
    ["9.4.1", "Tính NetJob từng ngành và tổng cộng", "Đã làm", "Có NewJob, UpgradeJob, DisplacedJob, NetJob"],
    ["9.4.2", "Tìm ngưỡng x_H ngành 2 để NetJob₂ >= 0 khi AI tối đa", "Đã làm", "Có bảng threshold_df"],
    ["9.4.2", "Phân tích kết quả ngành 2", "Đã làm", "Tách rõ NetJob >= 0 và đào tạo đủ displaced"],
    ["9.4.3", "Mô phỏng nhóm dễ bị tổn thương ngành 1, 3, 4", "Đã làm", "Có bảng flow_df"],
    ["9.4.3", "Vẽ swimming lane/Sankey", "Đã làm", "Matplotlib swimming lane; Plotly Sankey nếu có"],
    ["9.4.4", "Thêm ràng buộc không ngành nào mất quá 5% lao động", "Đã làm", "Có nghiệm cap_result_df và bảng kiểm tra"],
    ["Bổ sung báo cáo", "Kịch bản mở rộng có AI tối thiểu và trần ngân sách ngành", "Đã làm", "Giúp tránh nghiệm góc x_AI=0 khi giải thích chính sách"],
    ["9.5", "Trả lời câu hỏi thảo luận chính sách a, b, c, d", "Đã làm", "Có phần chính sách trong app và HTML report"],
], columns=["Mục", "Yêu cầu", "Trạng thái", "Ghi chú"])

st.dataframe(checklist_df, use_container_width=True)

# ------------------------------------------------------------
# Tải kết quả
# ------------------------------------------------------------

st.header("9. Tải kết quả")

output_manifest = pd.DataFrame([
    ["bai09_parameters.csv", "Bảng tham số 8 ngành theo đề"],
    ["bai09_base_solution.csv", "Nghiệm tối ưu cơ sở; có thể là nghiệm góc x_AI=0"],
    ["bai09_threshold_sector2.csv", "Ngưỡng đào tạo ngành 2"],
    ["bai09_vulnerable_flow.csv", "Mô phỏng nhóm dễ bị tổn thương ngành 1, 3, 4"],
    ["bai09_five_pct_solution.csv", "Nghiệm với ràng buộc không ngành nào mất quá 5% lao động"],
    ["bai09_policy_extended_solution.csv", "Kịch bản mở rộng có AI tối thiểu và trần ngành"],
    ["bai09_policy_extended_summary.csv", "Tóm tắt kịch bản mở rộng"],
    ["bai09_report.html", "Báo cáo HTML"],
], columns=["File output", "Ý nghĩa"])

note_html = """
<p><b>Lưu ý mô hình:</b> Đề có a2*x_D trong công thức NewJob, nhưng bài toán 9.2 và gợi ý code chỉ tối ưu x_AI và x_H.
Báo cáo này giữ x_D = 0 ở mô hình cơ sở để bám đúng yêu cầu.</p>
<p><b>Lưu ý đơn vị:</b> Ngân sách tính bằng tỷ VND; hệ số việc làm tính bằng việc/tỷ VND; lao động L_i trong ràng buộc 5% được đổi từ triệu người sang số việc làm.</p>
"""

html_report = make_html_report(
    parameter_df=parameter_df,
    base_result_df=base_result_df.round(4),
    base_summary_df=base_summary_df,
    check_df=base_check_df,
    threshold_df=threshold_df.round(4),
    vulnerable_df=flow_df.round(4),
    cap_result_df=cap_result_df.round(4) if not cap_result_df.empty else pd.DataFrame(),
    cap_summary_df=cap_summary_df if not cap_summary_df.empty else pd.DataFrame(),
    extended_result_df=extended_result_df.round(4) if not extended_result_df.empty else pd.DataFrame(),
    extended_summary_df=extended_summary_df if not extended_summary_df.empty else pd.DataFrame(),
    checklist_df=checklist_df,
    policy_html=policy_html,
    note_html=note_html,
)

excel_bytes = make_excel_bytes({
    "Parameters": parameter_df.round(6),
    "Base_solution": base_result_df.round(4),
    "Base_summary": base_summary_df,
    "Constraint_check": base_check_df,
    "Threshold_sector2": threshold_df.round(4),
    "Vulnerable_flow": flow_df.round(4),
    "Five_pct_solution": cap_result_df.round(4) if not cap_result_df.empty else pd.DataFrame(),
    "Policy_extended": extended_result_df.round(4) if not extended_result_df.empty else pd.DataFrame(),
    "Policy_extended_summary": extended_summary_df if not extended_summary_df.empty else pd.DataFrame(),
    "Output_manifest": output_manifest,
    "Checklist": checklist_df,
})

# Lưu ra thư mục outputs để tiện nộp bài.
parameter_df.round(6).to_csv(OUTPUT_DIR / "bai09_parameters.csv", index=False, encoding="utf-8-sig")
output_manifest.to_csv(OUTPUT_DIR / "bai09_output_manifest.csv", index=False, encoding="utf-8-sig")
base_result_df.round(4).to_csv(OUTPUT_DIR / "bai09_base_solution.csv", index=False, encoding="utf-8-sig")
threshold_df.round(4).to_csv(OUTPUT_DIR / "bai09_threshold_sector2.csv", index=False, encoding="utf-8-sig")
flow_df.round(4).to_csv(OUTPUT_DIR / "bai09_vulnerable_flow.csv", index=False, encoding="utf-8-sig")
if not cap_result_df.empty:
    cap_result_df.round(4).to_csv(OUTPUT_DIR / "bai09_five_pct_solution.csv", index=False, encoding="utf-8-sig")
if not extended_result_df.empty:
    extended_result_df.round(4).to_csv(OUTPUT_DIR / "bai09_policy_extended_solution.csv", index=False, encoding="utf-8-sig")
if not extended_summary_df.empty:
    extended_summary_df.to_csv(OUTPUT_DIR / "bai09_policy_extended_summary.csv", index=False, encoding="utf-8-sig")
with open(OUTPUT_DIR / "bai09_report.html", "w", encoding="utf-8") as f:
    f.write(html_report)

col_dl1, col_dl2, col_dl3 = st.columns(3)
with col_dl1:
    st.download_button(
        label="Tải HTML report",
        data=html_report.encode("utf-8"),
        file_name="bai09_report.html",
        mime="text/html",
    )
with col_dl2:
    st.download_button(
        label="Tải Excel tổng hợp",
        data=excel_bytes,
        file_name="bai09_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
with col_dl3:
    st.download_button(
        label="Tải CSV nghiệm cơ sở",
        data=base_result_df.round(4).to_csv(index=False).encode("utf-8-sig"),
        file_name="bai09_base_solution.csv",
        mime="text/csv",
    )

st.success("Bài 9 đã hoàn thành đầy đủ các yêu cầu 9.4.1 đến 9.4.4 và phần thảo luận chính sách 9.5.")
