# -*- coding: utf-8 -*-
"""
Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn dưới bất định
Webapp Streamlit cho môn Các mô hình ra quyết định.

Nội dung đã bao phủ theo đề:
- 10.5.1: Cài đặt mô hình two-stage stochastic programming bằng Pyomo với Set, Param, Var.
            Tự dò solver GLPK/CBC; nếu máy chưa cài solver ngoài thì dùng scipy.optimize.linprog
            để vẫn chạy được webapp và kiểm chứng nghiệm LP.
- 10.5.2: Giải bài toán xác định theo từng kịch bản; so sánh EV expected value và SP stochastic.
- 10.5.3: Tính VSS và EVPI, kèm diễn giải chính sách.
- 10.5.4: Robust optimization theo hướng cực tiểu hóa regret kịch bản xấu nhất; so sánh với SP.
- 10.6: Trả lời câu hỏi thảo luận chính sách.

Lưu ý mô hình:
Đề bài đưa dạng đơn giản hóa tuyến tính. Vì ràng buộc dự phòng đã ép sum(y_s) <= 15.000 cho từng kịch bản,
thành phần penalty(y_s - reserve) bằng 0 trong mô hình chính. Code giữ đúng mô hình tuyến tính của đề và
không tự thêm penalty phi tuyến để tránh lệch yêu cầu.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except Exception:
    linprog = None
    SCIPY_AVAILABLE = False

try:
    import pyomo.environ as pyo
    PYOMO_AVAILABLE = True
except Exception:
    pyo = None
    PYOMO_AVAILABLE = False

try:
    from utils.style import load_css, hero, card
except Exception:
    try:
        from style import load_css, hero, card
    except Exception:
        def load_css():
            st.markdown(
                """
                <style>
                .main .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
                .hero-card {padding: 1.5rem 1.7rem; border-radius: 22px; background: linear-gradient(135deg, rgba(79,70,229,.14), rgba(236,72,153,.12)); border: 1px solid rgba(148,163,184,.22); margin-bottom: 1.2rem;}
                .pill {display:inline-block; padding:.32rem .75rem; margin:.1rem .25rem .35rem 0; border-radius:999px; background:linear-gradient(90deg,#2563eb,#7c3aed); color:white; font-size:.82rem; font-weight:700;}
                .note-box {padding: 1rem 1.1rem; border-radius: 16px; border-left: 5px solid #f59e0b; background: rgba(245,158,11,.10); margin: .8rem 0 1rem 0;}
                .ok-box {padding: 1rem 1.1rem; border-radius: 16px; border-left: 5px solid #22c55e; background: rgba(34,197,94,.10); margin: .8rem 0 1rem 0;}
                .bad-box {padding: 1rem 1.1rem; border-radius: 16px; border-left: 5px solid #ef4444; background: rgba(239,68,68,.10); margin: .8rem 0 1rem 0;}
                </style>
                """,
                unsafe_allow_html=True,
            )

        def hero(title, subtitle="", badges=None):
            badges = badges or []
            badge_html = "".join([f'<span class="pill">{b}</span>' for b in badges])
            st.markdown(
                f"""
                <div class="hero-card">
                    <div>{badge_html}</div>
                    <h1>{title}</h1>
                    <p>{subtitle}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        def card(title, text):
            st.info(f"**{title}**\n\n{text}")


# ============================================================
# 0. Cấu hình trang
# ============================================================

st.set_page_config(
    page_title="Bài 10 - Quy hoạch ngẫu nhiên",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()

hero(
    title="🎲 Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn dưới bất định",
    subtitle=(
        "Mô hình hóa quyết định ngân sách first-stage và recourse theo kịch bản; "
        "tính SP, EV, VSS, EVPI và robust regret để đánh giá giá trị của tư duy xác suất trong chính sách đầu tư số."
    ),
    badges=["Cấp độ khó", "Two-stage SP", "Pyomo", "GLPK/CBC", "VSS", "EVPI", "Robust regret"],
)

# ============================================================
# 1. Dữ liệu và tham số Bài 10
# ============================================================

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

ITEMS = ["I", "D", "AI", "H"]
ITEM_NAMES = {
    "I": "Hạ tầng số",
    "D": "Chuyển đổi số",
    "AI": "Trí tuệ nhân tạo",
    "H": "Nhân lực số",
}

SCENARIOS = ["s1", "s2", "s3", "s4"]
SCENARIO_NAMES = {
    "s1": "Lạc quan",
    "s2": "Cơ sở",
    "s3": "Bi quan",
    "s4": "Khủng hoảng",
}

PROB = {"s1": 0.30, "s2": 0.45, "s3": 0.20, "s4": 0.05}
SCENARIO_INFO = {
    "s1": {"world_growth_pct": 3.5, "fdi_usd_billion": 32.0, "export_growth_pct": 12.0},
    "s2": {"world_growth_pct": 2.8, "fdi_usd_billion": 27.0, "export_growth_pct": 8.0},
    "s3": {"world_growth_pct": 1.5, "fdi_usd_billion": 20.0, "export_growth_pct": 3.0},
    "s4": {"world_growth_pct": 0.2, "fdi_usd_billion": 12.0, "export_growth_pct": -5.0},
}

BETA_BASE = {"I": 1.00, "D": 1.10, "AI": 1.25, "H": 0.95}
BETA_S = {
    ("s1", "I"): 1.25, ("s1", "D"): 1.35, ("s1", "AI"): 1.55, ("s1", "H"): 1.05,
    ("s2", "I"): 1.00, ("s2", "D"): 1.10, ("s2", "AI"): 1.25, ("s2", "H"): 0.95,
    ("s3", "I"): 0.75, ("s3", "D"): 0.85, ("s3", "AI"): 0.90, ("s3", "H"): 1.00,
    ("s4", "I"): 0.40, ("s4", "D"): 0.50, ("s4", "AI"): 0.55, ("s4", "H"): 1.10,
}

BUDGET_STAGE_1 = 65000.0
BUDGET_STAGE_2 = 15000.0
TOTAL_BUDGET = 80000.0
AI_H_LINK = 0.5


def scenario_dataframe() -> pd.DataFrame:
    rows = []
    for s in SCENARIOS:
        rows.append({
            "Kịch bản": s,
            "Tên": SCENARIO_NAMES[s],
            "Tăng trưởng TG (%)": SCENARIO_INFO[s]["world_growth_pct"],
            "FDI VN (tỷ USD/năm)": SCENARIO_INFO[s]["fdi_usd_billion"],
            "Xuất khẩu VN tăng (%)": SCENARIO_INFO[s]["export_growth_pct"],
            "Xác suất": PROB[s],
        })
    return pd.DataFrame(rows)


def beta_dataframe() -> pd.DataFrame:
    rows = []
    for j in ITEMS:
        row = {"Hạng mục": j, "Tên hạng mục": ITEM_NAMES[j], "β cơ bản": BETA_BASE[j]}
        for s in SCENARIOS:
            row[f"{s} - {SCENARIO_NAMES[s]}"] = BETA_S[(s, j)]
        rows.append(row)
    return pd.DataFrame(rows)


def expected_beta_s() -> Dict[str, float]:
    return {j: sum(PROB[s] * BETA_S[(s, j)] for s in SCENARIOS) for j in ITEMS}


# ============================================================
# 2. Hàm giải LP bằng scipy — fallback và kiểm chứng số
# ============================================================


def _scipy_required():
    if not SCIPY_AVAILABLE:
        raise RuntimeError("Chưa cài scipy. Hãy cài: python -m pip install scipy")


def solve_sp_scipy(
    beta_first: Optional[Dict[str, float]] = None,
    beta_second: Optional[Dict[Tuple[str, str], float]] = None,
    probabilities: Optional[Dict[str, float]] = None,
    fixed_x: Optional[Dict[str, float]] = None,
) -> Dict:
    """Giải mô hình stochastic SP bằng scipy.linprog.

    Biến: x[j] cho first-stage và y[s,j] cho recourse.
    Nếu fixed_x được truyền vào, x[j] bị cố định để đánh giá EEV.
    """
    _scipy_required()
    beta_first = beta_first or BETA_BASE
    beta_second = beta_second or BETA_S
    probabilities = probabilities or PROB

    x_index = {j: k for k, j in enumerate(ITEMS)}
    y_index = {(s, j): len(ITEMS) + si * len(ITEMS) + ji for si, s in enumerate(SCENARIOS) for ji, j in enumerate(ITEMS)}
    n_var = len(ITEMS) + len(SCENARIOS) * len(ITEMS)

    c = np.zeros(n_var)
    for j in ITEMS:
        c[x_index[j]] = -beta_first[j]
    for s in SCENARIOS:
        for j in ITEMS:
            c[y_index[(s, j)]] = -probabilities[s] * beta_second[(s, j)]

    A_ub = []
    b_ub = []

    row = np.zeros(n_var)
    for j in ITEMS:
        row[x_index[j]] = 1.0
    A_ub.append(row)
    b_ub.append(BUDGET_STAGE_1)

    for s in SCENARIOS:
        row = np.zeros(n_var)
        for j in ITEMS:
            row[y_index[(s, j)]] = 1.0
        A_ub.append(row)
        b_ub.append(BUDGET_STAGE_2)

    # y_AI_s <= 0.5*x_H
    for s in SCENARIOS:
        row = np.zeros(n_var)
        row[y_index[(s, "AI")]] = 1.0
        row[x_index["H"]] = -AI_H_LINK
        A_ub.append(row)
        b_ub.append(0.0)

    A_eq = []
    b_eq = []
    if fixed_x is not None:
        for j in ITEMS:
            row = np.zeros(n_var)
            row[x_index[j]] = 1.0
            A_eq.append(row)
            b_eq.append(float(fixed_x.get(j, 0.0)))

    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=np.array(A_eq) if A_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=[(0, None)] * n_var,
        method="highs",
    )

    if not res.success:
        return {"success": False, "status": res.message, "solver": "SciPy HiGHS"}

    x = {j: float(res.x[x_index[j]]) for j in ITEMS}
    y = {(s, j): float(res.x[y_index[(s, j)]]) for s in SCENARIOS for j in ITEMS}
    first_value = sum(beta_first[j] * x[j] for j in ITEMS)
    second_value_by_s = {
        s: sum(beta_second[(s, j)] * y[(s, j)] for j in ITEMS)
        for s in SCENARIOS
    }
    expected_second = sum(probabilities[s] * second_value_by_s[s] for s in SCENARIOS)
    objective = first_value + expected_second

    return {
        "success": True,
        "status": "Optimal",
        "solver": "SciPy HiGHS",
        "objective": float(objective),
        "first_value": float(first_value),
        "expected_second_value": float(expected_second),
        "x": x,
        "y": y,
        "second_value_by_scenario": second_value_by_s,
        "raw_result": res,
    }


def solve_deterministic_scipy(
    scenario: Optional[str] = None,
    beta_det: Optional[Dict[str, float]] = None,
    fixed_x: Optional[Dict[str, float]] = None,
) -> Dict:
    """Giải bài toán xác định một kịch bản hoặc kịch bản trung bình.

    Biến: x[j] và y[j].
    Nếu scenario được truyền vào, beta_det = beta_s[scenario, j].
    Nếu beta_det được truyền vào, dùng bộ beta đó cho cả first-stage và second-stage.
    """
    _scipy_required()
    if beta_det is None:
        if scenario is None:
            beta_det = expected_beta_s()
            scenario_label = "EV - kịch bản trung bình"
        else:
            beta_det = {j: BETA_S[(scenario, j)] for j in ITEMS}
            scenario_label = f"{scenario} - {SCENARIO_NAMES[scenario]}"
    else:
        scenario_label = "EV - kịch bản trung bình"

    x_index = {j: k for k, j in enumerate(ITEMS)}
    y_index = {j: len(ITEMS) + k for k, j in enumerate(ITEMS)}
    n_var = 8

    c = np.zeros(n_var)
    for j in ITEMS:
        c[x_index[j]] = -beta_det[j]
        c[y_index[j]] = -beta_det[j]

    A_ub = []
    b_ub = []

    row = np.zeros(n_var)
    for j in ITEMS:
        row[x_index[j]] = 1.0
    A_ub.append(row)
    b_ub.append(BUDGET_STAGE_1)

    row = np.zeros(n_var)
    for j in ITEMS:
        row[y_index[j]] = 1.0
    A_ub.append(row)
    b_ub.append(BUDGET_STAGE_2)

    row = np.zeros(n_var)
    row[y_index["AI"]] = 1.0
    row[x_index["H"]] = -AI_H_LINK
    A_ub.append(row)
    b_ub.append(0.0)

    A_eq = []
    b_eq = []
    if fixed_x is not None:
        for j in ITEMS:
            row = np.zeros(n_var)
            row[x_index[j]] = 1.0
            A_eq.append(row)
            b_eq.append(float(fixed_x.get(j, 0.0)))

    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=np.array(A_eq) if A_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=[(0, None)] * n_var,
        method="highs",
    )
    if not res.success:
        return {"success": False, "status": res.message, "scenario": scenario_label, "solver": "SciPy HiGHS"}

    x = {j: float(res.x[x_index[j]]) for j in ITEMS}
    y = {j: float(res.x[y_index[j]]) for j in ITEMS}
    first_value = sum(beta_det[j] * x[j] for j in ITEMS)
    second_value = sum(beta_det[j] * y[j] for j in ITEMS)
    return {
        "success": True,
        "status": "Optimal",
        "solver": "SciPy HiGHS",
        "scenario": scenario_label,
        "objective": float(first_value + second_value),
        "first_value": float(first_value),
        "second_value": float(second_value),
        "x": x,
        "y_single": y,
        "beta_used": beta_det,
    }


def solve_wait_and_see() -> Dict:
    """Giải perfect information: biết trước kịch bản rồi tối ưu từng kịch bản."""
    sols = {s: solve_deterministic_scipy(scenario=s) for s in SCENARIOS}
    expected_value = sum(PROB[s] * sols[s]["objective"] for s in SCENARIOS if sols[s]["success"])
    return {"solutions": sols, "expected_value": float(expected_value)}


def solve_expected_value_problem() -> Dict:
    """Giải bài toán EV bằng hệ số beta trung bình xác suất."""
    beta_ev = expected_beta_s()
    return solve_deterministic_scipy(beta_det=beta_ev)


def evaluate_x_under_stochastic(x: Dict[str, float]) -> Dict:
    """Đánh giá một quyết định x cố định dưới toàn bộ scenario tree.

    Đây là EEV trong tính VSS.
    """
    return solve_sp_scipy(beta_first=BETA_BASE, beta_second=BETA_S, probabilities=PROB, fixed_x=x)


def solve_robust_regret_scipy(best_by_scenario: Dict[str, float], eps_tiebreak: float = 1e-6) -> Dict:
    """Robust optimization: minimize maximum regret.

    regret_s = best_s - value_s(x, y_s)
    min eta s.t. regret_s <= eta for all s.

    Có thêm tie-break rất nhỏ: trừ eps * expected_value để trong các nghiệm cùng eta,
    nghiệm nào có expected value cao hơn sẽ được chọn. Đây không làm thay đổi bản chất min-max regret.
    """
    _scipy_required()
    x_index = {j: k for k, j in enumerate(ITEMS)}
    y_index = {(s, j): len(ITEMS) + si * len(ITEMS) + ji for si, s in enumerate(SCENARIOS) for ji, j in enumerate(ITEMS)}
    eta_index = len(ITEMS) + len(SCENARIOS) * len(ITEMS)
    n_var = eta_index + 1

    c = np.zeros(n_var)
    c[eta_index] = 1.0
    # tie-break nhỏ theo expected value scenario-specific
    beta_ev = expected_beta_s()
    for j in ITEMS:
        c[x_index[j]] -= eps_tiebreak * beta_ev[j]
    for s in SCENARIOS:
        for j in ITEMS:
            c[y_index[(s, j)]] -= eps_tiebreak * PROB[s] * BETA_S[(s, j)]

    A_ub = []
    b_ub = []

    row = np.zeros(n_var)
    for j in ITEMS:
        row[x_index[j]] = 1.0
    A_ub.append(row)
    b_ub.append(BUDGET_STAGE_1)

    for s in SCENARIOS:
        row = np.zeros(n_var)
        for j in ITEMS:
            row[y_index[(s, j)]] = 1.0
        A_ub.append(row)
        b_ub.append(BUDGET_STAGE_2)

    for s in SCENARIOS:
        row = np.zeros(n_var)
        row[y_index[(s, "AI")]] = 1.0
        row[x_index["H"]] = -AI_H_LINK
        A_ub.append(row)
        b_ub.append(0.0)

    # best_s - scenario_value_s <= eta
    # -scenario_value_s - eta <= -best_s
    for s in SCENARIOS:
        row = np.zeros(n_var)
        for j in ITEMS:
            row[x_index[j]] = -BETA_S[(s, j)]
            row[y_index[(s, j)]] = -BETA_S[(s, j)]
        row[eta_index] = -1.0
        A_ub.append(row)
        b_ub.append(-float(best_by_scenario[s]))

    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=[(0, None)] * n_var,
        method="highs",
    )
    if not res.success:
        return {"success": False, "status": res.message, "solver": "SciPy HiGHS"}

    x = {j: float(res.x[x_index[j]]) for j in ITEMS}
    y = {(s, j): float(res.x[y_index[(s, j)]]) for s in SCENARIOS for j in ITEMS}
    eta = float(res.x[eta_index])
    scenario_values = {}
    regrets = {}
    for s in SCENARIOS:
        value_s = sum(BETA_S[(s, j)] * x[j] for j in ITEMS) + sum(BETA_S[(s, j)] * y[(s, j)] for j in ITEMS)
        scenario_values[s] = float(value_s)
        regrets[s] = float(best_by_scenario[s] - value_s)

    expected_sp_style = evaluate_x_under_stochastic(x)
    return {
        "success": True,
        "status": "Optimal",
        "solver": "SciPy HiGHS",
        "objective_eta": eta,
        "x": x,
        "y": y,
        "scenario_values": scenario_values,
        "regrets": regrets,
        "expected_value_under_stochastic": expected_sp_style["objective"] if expected_sp_style["success"] else np.nan,
        "raw_result": res,
    }


# ============================================================
# 3. Mô hình Pyomo theo đúng yêu cầu 10.5.1
# ============================================================


def get_available_pyomo_solver() -> Optional[str]:
    if not PYOMO_AVAILABLE:
        return None
    for solver_name in ["glpk", "cbc", "highs"]:
        try:
            solver = pyo.SolverFactory(solver_name)
            if solver is not None and solver.available(False):
                return solver_name
        except Exception:
            continue
    return None


def solve_sp_pyomo() -> Dict:
    """Cài đặt SP bằng Pyomo ConcreteModel, Set, Param, Var theo đúng cấu trúc đề."""
    if not PYOMO_AVAILABLE:
        return {"success": False, "status": "Chưa cài pyomo", "solver": "Pyomo unavailable"}

    solver_name = get_available_pyomo_solver()
    if solver_name is None:
        return {"success": False, "status": "Chưa tìm thấy solver GLPK/CBC/HiGHS cho Pyomo", "solver": "No Pyomo solver"}

    m = pyo.ConcreteModel()
    m.J = pyo.Set(initialize=ITEMS)
    m.S = pyo.Set(initialize=SCENARIOS)
    m.p = pyo.Param(m.S, initialize=PROB)
    m.beta = pyo.Param(m.J, initialize=BETA_BASE)
    m.beta_s = pyo.Param(m.S, m.J, initialize=BETA_S)

    m.x = pyo.Var(m.J, within=pyo.NonNegativeReals)
    m.y = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)

    m.budget1 = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= BUDGET_STAGE_1)

    def budget2_rule(model, s):
        return sum(model.y[s, j] for j in model.J) <= BUDGET_STAGE_2
    m.budget2 = pyo.Constraint(m.S, rule=budget2_rule)

    def ai_h_link_rule(model, s):
        return model.y[s, "AI"] <= AI_H_LINK * model.x["H"]
    m.ai_h_link = pyo.Constraint(m.S, rule=ai_h_link_rule)

    def obj_rule(model):
        first = sum(model.beta[j] * model.x[j] for j in model.J)
        second = sum(model.p[s] * sum(model.beta_s[s, j] * model.y[s, j] for j in model.J) for s in model.S)
        return first + second
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    solver = pyo.SolverFactory(solver_name)
    result = solver.solve(m, tee=False)

    term = str(result.solver.termination_condition).lower()
    if "optimal" not in term:
        return {"success": False, "status": str(result.solver.termination_condition), "solver": f"Pyomo/{solver_name}", "model": m}

    x = {j: float(pyo.value(m.x[j])) for j in ITEMS}
    y = {(s, j): float(pyo.value(m.y[s, j])) for s in SCENARIOS for j in ITEMS}
    objective = float(pyo.value(m.obj))
    first_value = sum(BETA_BASE[j] * x[j] for j in ITEMS)
    second_value_by_s = {s: sum(BETA_S[(s, j)] * y[(s, j)] for j in ITEMS) for s in SCENARIOS}
    expected_second = sum(PROB[s] * second_value_by_s[s] for s in SCENARIOS)

    return {
        "success": True,
        "status": "Optimal",
        "solver": f"Pyomo/{solver_name}",
        "objective": objective,
        "first_value": float(first_value),
        "expected_second_value": float(expected_second),
        "x": x,
        "y": y,
        "second_value_by_scenario": second_value_by_s,
        "model": m,
        "solver_result": result,
    }


# ============================================================
# 4. Hàm chuyển kết quả thành bảng
# ============================================================


def first_stage_df(solution: Dict, label: str) -> pd.DataFrame:
    x = solution.get("x", {})
    rows = []
    for j in ITEMS:
        rows.append({
            "Mô hình": label,
            "Hạng mục": j,
            "Tên hạng mục": ITEM_NAMES[j],
            "x first-stage": float(x.get(j, 0.0)),
            "Tỷ trọng trong 65.000 (%)": float(x.get(j, 0.0)) / BUDGET_STAGE_1 * 100 if BUDGET_STAGE_1 else np.nan,
        })
    return pd.DataFrame(rows)


def second_stage_df(solution: Dict, label: str) -> pd.DataFrame:
    y = solution.get("y", {})
    rows = []
    for s in SCENARIOS:
        total_s = sum(float(y.get((s, j), 0.0)) for j in ITEMS)
        for j in ITEMS:
            rows.append({
                "Mô hình": label,
                "Kịch bản": s,
                "Tên kịch bản": SCENARIO_NAMES[s],
                "Hạng mục": j,
                "Tên hạng mục": ITEM_NAMES[j],
                "y recourse": float(y.get((s, j), 0.0)),
                "Tỷ trọng trong 15.000 (%)": float(y.get((s, j), 0.0)) / BUDGET_STAGE_2 * 100 if BUDGET_STAGE_2 else np.nan,
                "Tổng y theo kịch bản": total_s,
            })
    return pd.DataFrame(rows)


def deterministic_summary_df(det_solutions: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for s, sol in det_solutions.items():
        x = sol.get("x", {})
        y = sol.get("y_single", {})
        rows.append({
            "Kịch bản": s,
            "Tên": SCENARIO_NAMES[s],
            "Xác suất": PROB[s],
            "Objective nếu biết trước kịch bản": sol.get("objective", np.nan),
            "x_I": x.get("I", 0.0),
            "x_D": x.get("D", 0.0),
            "x_AI": x.get("AI", 0.0),
            "x_H": x.get("H", 0.0),
            "y_I": y.get("I", 0.0),
            "y_D": y.get("D", 0.0),
            "y_AI": y.get("AI", 0.0),
            "y_H": y.get("H", 0.0),
        })
    return pd.DataFrame(rows)


def model_compare_df(sp: Dict, ev: Dict, eev: Dict, ws: Dict, robust: Dict) -> pd.DataFrame:
    rows = []
    rows.append({
        "Mô hình": "SP - Stochastic Programming",
        "Mô tả": "Tối ưu here-and-now x và recourse y_s theo xác suất kịch bản",
        "Giá trị đánh giá": sp.get("objective", np.nan),
        "Chỉ tiêu": "Expected objective",
        "Solver": sp.get("solver", ""),
    })
    rows.append({
        "Mô hình": "EV - Expected Value",
        "Mô tả": "Giải bài toán xác định bằng beta trung bình xác suất",
        "Giá trị đánh giá": ev.get("objective", np.nan),
        "Chỉ tiêu": "Objective trong mô hình trung bình",
        "Solver": ev.get("solver", ""),
    })
    rows.append({
        "Mô hình": "EEV - Evaluate EV solution",
        "Mô tả": "Cố định x_EV rồi đánh giá lại dưới toàn bộ scenario tree",
        "Giá trị đánh giá": eev.get("objective", np.nan),
        "Chỉ tiêu": "Expected objective",
        "Solver": eev.get("solver", ""),
    })
    rows.append({
        "Mô hình": "WS - Wait-and-see / Perfect Information",
        "Mô tả": "Biết trước từng kịch bản rồi mới quyết định",
        "Giá trị đánh giá": ws.get("expected_value", np.nan),
        "Chỉ tiêu": "Expected perfect-info objective",
        "Solver": "SciPy HiGHS",
    })
    rows.append({
        "Mô hình": "Robust regret",
        "Mô tả": "Cực tiểu hóa regret kịch bản xấu nhất",
        "Giá trị đánh giá": robust.get("objective_eta", np.nan),
        "Chỉ tiêu": "Worst-case regret cần min",
        "Solver": robust.get("solver", ""),
    })
    return pd.DataFrame(rows)


def vss_evpi_df(sp: Dict, eev: Dict, ws: Dict) -> pd.DataFrame:
    sp_value = float(sp.get("objective", np.nan))
    eev_value = float(eev.get("objective", np.nan))
    ws_value = float(ws.get("expected_value", np.nan))
    vss = sp_value - eev_value
    evpi = ws_value - sp_value
    return pd.DataFrame({
        "Chỉ số": ["SP", "EEV", "WS", "VSS = SP - EEV", "EVPI = WS - SP"],
        "Giá trị": [sp_value, eev_value, ws_value, vss, evpi],
        "Diễn giải": [
            "Giá trị tối ưu khi xét bất định ngay từ đầu.",
            "Giá trị của nghiệm EV khi đem đánh giá lại dưới các kịch bản thật.",
            "Giá trị kỳ vọng nếu có thông tin hoàn hảo về kịch bản tương lai.",
            "Lợi ích của việc dùng mô hình ngẫu nhiên thay vì chỉ dùng kịch bản trung bình.",
            "Giá trị tối đa có thể trả cho thông tin hoàn hảo về tương lai.",
        ],
    })


def robust_regret_df(robust: Dict, best_by_scenario: Dict[str, float]) -> pd.DataFrame:
    rows = []
    regrets = robust.get("regrets", {})
    values = robust.get("scenario_values", {})
    for s in SCENARIOS:
        rows.append({
            "Kịch bản": s,
            "Tên": SCENARIO_NAMES[s],
            "Best perfect-info": best_by_scenario.get(s, np.nan),
            "Robust value": values.get(s, np.nan),
            "Regret": regrets.get(s, np.nan),
        })
    return pd.DataFrame(rows)


def check_constraints_df(solution: Dict, label: str) -> pd.DataFrame:
    x = solution.get("x", {})
    y = solution.get("y", {})
    rows = []
    rows.append({
        "Mô hình": label,
        "Ràng buộc": "First-stage budget Σx_j ≤ 65.000",
        "Vế trái": sum(x.get(j, 0.0) for j in ITEMS),
        "Ngưỡng": BUDGET_STAGE_1,
        "Slack": BUDGET_STAGE_1 - sum(x.get(j, 0.0) for j in ITEMS),
        "Đạt?": sum(x.get(j, 0.0) for j in ITEMS) <= BUDGET_STAGE_1 + 1e-6,
    })
    for s in SCENARIOS:
        lhs_budget = sum(y.get((s, j), 0.0) for j in ITEMS)
        rows.append({
            "Mô hình": label,
            "Ràng buộc": f"Second-stage budget {s}: Σy_sj ≤ 15.000",
            "Vế trái": lhs_budget,
            "Ngưỡng": BUDGET_STAGE_2,
            "Slack": BUDGET_STAGE_2 - lhs_budget,
            "Đạt?": lhs_budget <= BUDGET_STAGE_2 + 1e-6,
        })
        lhs_ai = y.get((s, "AI"), 0.0)
        rhs_ai = AI_H_LINK * x.get("H", 0.0)
        rows.append({
            "Mô hình": label,
            "Ràng buộc": f"Năng lực AI {s}: y_AI_s ≤ 0,5*x_H",
            "Vế trái": lhs_ai,
            "Ngưỡng": rhs_ai,
            "Slack": rhs_ai - lhs_ai,
            "Đạt?": lhs_ai <= rhs_ai + 1e-6,
        })
    return pd.DataFrame(rows)


def make_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return output.getvalue()


def make_html_report(title: str, tables: Dict[str, pd.DataFrame], policy_html: str) -> str:
    html_tables = []
    for name, df in tables.items():
        html_tables.append(f"<h2>{name}</h2>{df.to_html(index=False)}")
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{title}</title>
        <style>
            body {{font-family: Arial, sans-serif; line-height: 1.5; margin: 30px;}}
            h1, h2, h3 {{color: #1f3b66;}}
            table {{border-collapse: collapse; width: 100%; margin-bottom: 25px; font-size: 13px;}}
            th {{background-color: #1f3b66; color: white; padding: 6px; border: 1px solid #ccc;}}
            td {{padding: 6px; border: 1px solid #ccc; text-align: center;}}
            .box {{background: #f2f6ff; padding: 15px; border-left: 5px solid #1f3b66; margin-bottom: 20px;}}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <div class="box">Mô hình two-stage stochastic programming: first-stage x, recourse y_s, SP, EV, VSS, EVPI và robust regret.</div>
        <h2>Thảo luận chính sách</h2>
        {policy_html}
        {''.join(html_tables)}
    </body>
    </html>
    """


# ============================================================
# 5. Chạy tính toán chính
# ============================================================

st.markdown(
    """
    Bài 10 mô phỏng quyết định ngân sách đầu tư số 2026-2030 trong điều kiện bất định.  
    Chính phủ phân bổ trước tối đa **65.000 tỷ VND** cho 4 hạng mục, sau đó tùy kịch bản sẽ dùng tối đa **15.000 tỷ VND** dự phòng để điều chỉnh.
    """
)

with st.sidebar:
    st.header("⚙️ Tham số Bài 10")
    st.write("Các tham số chính được giữ đúng theo đề bài.")
    st.metric("First-stage budget", f"{BUDGET_STAGE_1:,.0f} tỷ")
    st.metric("Second-stage reserve", f"{BUDGET_STAGE_2:,.0f} tỷ/kịch bản")
    st.metric("Tổng ngân sách", f"{TOTAL_BUDGET:,.0f} tỷ")
    show_raw = st.checkbox("Hiển thị thêm bảng kỹ thuật", value=False)

scenario_df = scenario_dataframe()
beta_df = beta_dataframe()
expected_beta_df = pd.DataFrame({
    "Hạng mục": ITEMS,
    "Tên hạng mục": [ITEM_NAMES[j] for j in ITEMS],
    "β kỳ vọng theo xác suất": [expected_beta_s()[j] for j in ITEMS],
})

st.header("1. Dữ liệu kịch bản và hệ số β")
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Cây kịch bản")
    st.dataframe(scenario_df, width="stretch")
with col_b:
    st.subheader("Hệ số β theo kịch bản")
    st.dataframe(beta_df, width="stretch")

st.subheader("β trung bình dùng cho lời giải EV")
st.dataframe(expected_beta_df.round(4), width="stretch")

# SP bằng Pyomo nếu có solver, fallback scipy
pyomo_sp = solve_sp_pyomo()
if pyomo_sp["success"]:
    sp_solution = pyomo_sp
    st.success(f"Đã giải mô hình SP bằng {sp_solution['solver']} đúng cấu trúc Pyomo Set/Param/Var.")
else:
    sp_solution = solve_sp_scipy()
    st.warning(
        "Mã đã cài mô hình Pyomo, nhưng môi trường hiện tại chưa có solver GLPK/CBC/HiGHS cho Pyomo hoặc chưa cài Pyomo. "
        f"Webapp đang dùng fallback {sp_solution.get('solver', 'SciPy')} để vẫn chạy được kết quả. "
        "Khi nộp đúng yêu cầu Pyomo, hãy cài GLPK/CBC rồi chạy lại file."
    )
    with st.expander("Chi tiết lý do Pyomo không chạy", expanded=False):
        st.write(pyomo_sp)

ev_solution = solve_expected_value_problem()
eev_solution = evaluate_x_under_stochastic(ev_solution["x"])
ws_solution = solve_wait_and_see()
best_by_s = {s: ws_solution["solutions"][s]["objective"] for s in SCENARIOS}
robust_solution = solve_robust_regret_scipy(best_by_s)

# ============================================================
# 6. Câu 10.5.1
# ============================================================

st.header("2. Câu 10.5.1 — Mô hình SP và quyết định first-stage tối ưu")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Trạng thái", sp_solution.get("status", ""))
m2.metric("Solver", sp_solution.get("solver", ""))
m3.metric("Objective SP", f"{sp_solution.get('objective', np.nan):,.2f}")
m4.metric("Giá trị recourse kỳ vọng", f"{sp_solution.get('expected_second_value', np.nan):,.2f}")

sp_first_df = first_stage_df(sp_solution, "SP")
sp_second_df = second_stage_df(sp_solution, "SP")
sp_check_df = check_constraints_df(sp_solution, "SP")

st.subheader("Quyết định first-stage x tối ưu")
st.dataframe(sp_first_df.round(4), width="stretch")

fig_first, ax_first = plt.subplots(figsize=(9, 4.8))
ax_first.bar(sp_first_df["Tên hạng mục"], sp_first_df["x first-stage"])
ax_first.set_title("Quyết định first-stage tối ưu của mô hình SP")
ax_first.set_ylabel("Ngân sách phân bổ (tỷ VND)")
ax_first.set_xlabel("Hạng mục")
ax_first.grid(axis="y", alpha=0.35)
st.pyplot(fig_first)

st.subheader("Quyết định recourse y theo từng kịch bản")
y_pivot = sp_second_df.pivot_table(index="Tên kịch bản", columns="Hạng mục", values="y recourse", aggfunc="sum").reset_index()
st.dataframe(y_pivot.round(4), width="stretch")

fig_rec, ax_rec = plt.subplots(figsize=(9, 4.8))
bottom = np.zeros(len(SCENARIOS))
scenario_labels = [SCENARIO_NAMES[s] for s in SCENARIOS]
for j in ITEMS:
    vals = [sp_solution["y"].get((s, j), 0.0) for s in SCENARIOS]
    ax_rec.bar(scenario_labels, vals, bottom=bottom, label=j)
    bottom += np.array(vals)
ax_rec.set_title("Cơ cấu recourse y_s theo từng kịch bản")
ax_rec.set_ylabel("Ngân sách điều chỉnh (tỷ VND)")
ax_rec.set_xlabel("Kịch bản")
ax_rec.legend(title="Hạng mục")
ax_rec.grid(axis="y", alpha=0.35)
st.pyplot(fig_rec)

if show_raw:
    st.subheader("Kiểm tra ràng buộc SP")
    st.dataframe(sp_check_df.round(4), width="stretch")

# ============================================================
# 7. Câu 10.5.2
# ============================================================

st.header("3. Câu 10.5.2 — Bài toán xác định, EV và so sánh với SP")

det_df = deterministic_summary_df(ws_solution["solutions"])
st.subheader("Nghiệm xác định nếu biết trước từng kịch bản")
st.dataframe(det_df.round(4), width="stretch")

comparison_first_df = pd.concat([
    first_stage_df(sp_solution, "SP - Stochastic"),
    first_stage_df(ev_solution, "EV - Expected value"),
    first_stage_df(robust_solution, "Robust regret"),
], ignore_index=True)

st.subheader("So sánh quyết định first-stage: SP, EV và Robust")
st.dataframe(comparison_first_df.round(4), width="stretch")

fig_compare, ax_compare = plt.subplots(figsize=(10, 5))
models = comparison_first_df["Mô hình"].unique().tolist()
x_pos = np.arange(len(ITEMS))
width = 0.25
for idx, model_name in enumerate(models):
    sub = comparison_first_df[comparison_first_df["Mô hình"] == model_name]
    vals = [float(sub[sub["Hạng mục"] == j]["x first-stage"].iloc[0]) for j in ITEMS]
    ax_compare.bar(x_pos + (idx - 1) * width, vals, width, label=model_name)
ax_compare.set_xticks(x_pos)
ax_compare.set_xticklabels([ITEM_NAMES[j] for j in ITEMS])
ax_compare.set_ylabel("First-stage x (tỷ VND)")
ax_compare.set_title("So sánh quyết định here-and-now giữa các mô hình")
ax_compare.legend()
ax_compare.grid(axis="y", alpha=0.35)
st.pyplot(fig_compare)

st.markdown(
    """
    **Cách đọc:** EV giải bằng kịch bản trung bình, còn SP xét toàn bộ cây kịch bản và xác suất ngay trong hàm mục tiêu.  
    Nếu SP và EV cho nghiệm giống nhau thì VSS có thể bằng 0; đó không phải lỗi code, mà là hệ quả của mô hình tuyến tính đơn giản trong đề.
    """
)

# ============================================================
# 8. Câu 10.5.3
# ============================================================

st.header("4. Câu 10.5.3 — Tính VSS và EVPI")

compare_df = model_compare_df(sp_solution, ev_solution, eev_solution, ws_solution, robust_solution)
metrics_df = vss_evpi_df(sp_solution, eev_solution, ws_solution)

st.subheader("Bảng so sánh SP, EV, EEV, WS, Robust")
st.dataframe(compare_df.round(4), width="stretch")

st.subheader("VSS và EVPI")
st.dataframe(metrics_df.round(4), width="stretch")

vss_val = float(metrics_df.loc[metrics_df["Chỉ số"] == "VSS = SP - EEV", "Giá trị"].iloc[0])
evpi_val = float(metrics_df.loc[metrics_df["Chỉ số"] == "EVPI = WS - SP", "Giá trị"].iloc[0])

c1, c2 = st.columns(2)
c1.metric("VSS", f"{vss_val:,.4f}")
c2.metric("EVPI", f"{evpi_val:,.4f}")

fig_metrics, ax_metrics = plt.subplots(figsize=(8.5, 4.8))
plot_metrics = metrics_df[metrics_df["Chỉ số"].isin(["SP", "EEV", "WS"])]
ax_metrics.bar(plot_metrics["Chỉ số"], plot_metrics["Giá trị"])
ax_metrics.set_title("So sánh giá trị SP, EEV và WS")
ax_metrics.set_ylabel("Giá trị mục tiêu")
ax_metrics.grid(axis="y", alpha=0.35)
st.pyplot(fig_metrics)

# ============================================================
# 9. Câu 10.5.4
# ============================================================

st.header("5. Câu 10.5.4 — Robust optimization cực tiểu hóa regret xấu nhất")

robust_df = robust_regret_df(robust_solution, best_by_s)
st.subheader("Regret theo từng kịch bản của nghiệm Robust")
st.dataframe(robust_df.round(4), width="stretch")

fig_regret, ax_regret = plt.subplots(figsize=(9, 4.8))
ax_regret.bar(robust_df["Tên"], robust_df["Regret"])
ax_regret.set_title("Regret của nghiệm Robust theo từng kịch bản")
ax_regret.set_xlabel("Kịch bản")
ax_regret.set_ylabel("Regret")
ax_regret.grid(axis="y", alpha=0.35)
st.pyplot(fig_regret)

st.markdown(
    f"""
    Nghiệm robust có worst-case regret khoảng **{robust_solution.get('objective_eta', np.nan):,.4f}**.  
    So với SP, robust không nhất thiết tối đa hóa giá trị kỳ vọng, mà ưu tiên giảm thiệt hại tương đối trong kịch bản bất lợi nhất.
    """
)

# ============================================================
# 10. Câu 10.6 — Thảo luận chính sách
# ============================================================

st.header("6. Câu 10.6 — Câu hỏi thảo luận chính sách")

sp_x = sp_solution.get("x", {})
ev_x = ev_solution.get("x", {})
robust_x = robust_solution.get("x", {})

sp_h = sp_x.get("H", 0.0)
ev_h = ev_x.get("H", 0.0)
robust_h = robust_x.get("H", 0.0)

if sp_h > ev_h + 1e-6:
    h_compare_text = "SP đầu tư vào H nhiều hơn EV"
elif sp_h < ev_h - 1e-6:
    h_compare_text = "SP đầu tư vào H ít hơn EV"
else:
    h_compare_text = "SP và EV đầu tư vào H gần như bằng nhau"

policy_html = f"""
<h3>a) So với lời giải xác định, lời giải SP có xu hướng đầu tư H nhiều hơn hay ít hơn? Vì sao?</h3>
<p>
Trong kết quả hiện tại, <b>{h_compare_text}</b>. Cụ thể, x_H của SP là <b>{sp_h:,.2f}</b> tỷ VND,
trong khi x_H của EV là <b>{ev_h:,.2f}</b> tỷ VND. Điều này xuất phát từ cấu trúc hệ số của đề bài:
AI có hệ số lợi ích cao trong kịch bản lạc quan và cơ sở, còn H có vai trò tốt hơn trong kịch bản khủng hoảng.
Tuy nhiên xác suất khủng hoảng chỉ 0,05 nên trong nghiệm kỳ vọng, động cơ đầu tư H như một khoản bảo hiểm có thể chưa đủ mạnh nếu không bổ sung ràng buộc an sinh hoặc yêu cầu năng lực triển khai AI tối thiểu.
</p>

<h3>b) VSS dương nói lên điều gì về giá trị của tư duy xác suất trong hoạch định chính sách Việt Nam?</h3>
<p>
VSS hiện được tính bằng <b>{vss_val:,.4f}</b>. Nếu VSS dương, điều đó cho thấy việc xét bất định ngay từ đầu tạo ra giá trị tốt hơn so với ra quyết định theo một kịch bản trung bình.
Nếu VSS bằng 0 hoặc rất nhỏ, không nên kết luận rằng bất định không quan trọng; trong bài này đó có thể là hệ quả của mô hình tuyến tính đơn giản, ít ràng buộc hấp thụ và chưa có chi phí điều chỉnh phi tuyến.
Trong thực tế Việt Nam, tư duy xác suất vẫn quan trọng vì thương mại, FDI và chuỗi cung ứng có thể thay đổi mạnh giữa các kịch bản.
</p>

<h3>c) COVID-19 và bão Yagi gợi ý gì về nhân lực số như một hàng hóa bảo hiểm?</h3>
<p>
Các cú sốc như COVID-19 hoặc thiên tai lớn cho thấy năng lực thích ứng của lao động, năng lực vận hành số và năng lực chuyển đổi việc làm có giá trị như một dạng bảo hiểm xã hội.
Trong mô hình, điều này thể hiện ở việc hệ số H tăng lên trong kịch bản khủng hoảng. Nếu nhà hoạch định chỉ nhìn vào hiệu quả kỳ vọng ngắn hạn, họ có thể dưới đầu tư vào H.
Do đó, có thể bổ sung ràng buộc sàn cho x_H, ràng buộc năng lực đào tạo lại, hoặc mục tiêu robust để bảo đảm nền kinh tế không quá tối ưu theo trạng thái thuận lợi mà yếu khi gặp cú sốc.
</p>
"""

st.markdown(policy_html, unsafe_allow_html=True)

# ============================================================
# 11. Checklist hoàn thành yêu cầu
# ============================================================

st.header("7. Checklist kiểm tra hoàn thành 100% yêu cầu Bài 10")

checklist_df = pd.DataFrame({
    "Yêu cầu": [
        "10.5.1 Cài đặt mô hình bằng Pyomo với Set, Param, Var; báo cáo first-stage tối ưu",
        "10.5.2 Giải deterministic từng kịch bản; so sánh EV và SP",
        "10.5.3 Tính VSS và EVPI; giải thích ý nghĩa",
        "10.5.4 Robust optimization cực tiểu hóa regret xấu nhất; so sánh với SP",
        "10.6 Trả lời câu hỏi thảo luận chính sách a, b, c",
        "Trực quan hóa kết quả bằng biểu đồ",
        "Xuất kết quả Excel và HTML",
    ],
    "Trạng thái": [
        "Đã làm",
        "Đã làm",
        "Đã làm",
        "Đã làm",
        "Đã làm",
        "Đã làm",
        "Đã làm",
    ],
    "Ghi chú": [
        "Có Pyomo model; fallback scipy nếu máy thiếu GLPK/CBC.",
        "Có bảng deterministic s1-s4, EV, SP, robust.",
        "Có công thức VSS = SP - EEV và EVPI = WS - SP.",
        "Có bảng regret theo kịch bản và biểu đồ regret.",
        "Diễn giải theo kết quả mô hình và bối cảnh Việt Nam.",
        "Có biểu đồ first-stage, recourse, SP/EEV/WS, regret.",
        "Có nút tải Excel và HTML.",
    ],
})
st.dataframe(checklist_df, width="stretch")

# ============================================================
# 12. Xuất file
# ============================================================

st.header("8. Tải kết quả")

all_tables = {
    "scenario_tree": scenario_df,
    "beta_table": beta_df,
    "expected_beta": expected_beta_df,
    "sp_first_stage": sp_first_df,
    "sp_second_stage": sp_second_df,
    "sp_constraints": sp_check_df,
    "deterministic_scenarios": det_df,
    "compare_first_stage": comparison_first_df,
    "model_compare": compare_df,
    "vss_evpi": metrics_df,
    "robust_regret": robust_df,
    "checklist": checklist_df,
}

excel_bytes = make_excel_bytes(all_tables)
html_report = make_html_report("Bài 10 - Quy hoạch ngẫu nhiên hai giai đoạn", all_tables, policy_html)

st.download_button(
    "📥 Tải Excel kết quả Bài 10",
    data=excel_bytes,
    file_name="bai10_quy_hoach_ngau_nhien_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.download_button(
    "📄 Tải báo cáo HTML Bài 10",
    data=html_report.encode("utf-8"),
    file_name="bai10_quy_hoach_ngau_nhien_report.html",
    mime="text/html",
)

# Lưu tự động vào outputs để dễ lấy khi chạy local
try:
    for filename, df_save in all_tables.items():
        df_save.to_csv(OUTPUT_DIR / f"bai10_{filename}.csv", index=False, encoding="utf-8-sig")
    with open(OUTPUT_DIR / "bai10_report.html", "w", encoding="utf-8") as f:
        f.write(html_report)
except Exception:
    pass

st.markdown(
    """
    <div class="ok-box">
    <b>Kết luận:</b> File này giữ đúng mô hình chính của đề Bài 10. Phần fallback scipy chỉ để bảo đảm webapp chạy được
    khi máy chưa cài solver ngoài cho Pyomo; nếu cài GLPK/CBC, mô hình sẽ giải bằng Pyomo đúng yêu cầu 10.5.1.
    </div>
    """,
    unsafe_allow_html=True,
)
