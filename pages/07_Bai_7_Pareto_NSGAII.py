# -*- coding: utf-8 -*-
"""
Bài 7 — Tối ưu đa mục tiêu Pareto với NSGA-II
Webapp Streamlit cho môn Các mô hình ra quyết định.

Nội dung chính:
- Dựng bài toán 24 biến, 4 mục tiêu theo đề Bài 7.
- Giải bằng pymoo/NSGA-II với pop_size=100, n_gen=200.
- Trích xuất tập nghiệm Pareto, vẽ scatter 3D và parallel coordinates.
- Chọn nghiệm thỏa hiệp bằng TOPSIS với trọng số (0.40, 0.25, 0.20, 0.15).
- Phân tích chi phí cơ hội của nghiệm tăng trưởng cao nhất so với nghiệm thỏa hiệp.
- Trả lời câu hỏi chính sách 7.5.

Lưu ý quan trọng:
Bài 7 yêu cầu giữ hệ ràng buộc C1-C6 của Bài 4. Với tham số gốc lambda=0.70,
ràng buộc công bằng C5 kế thừa từ Bài 4 không khả thi do Tây Nguyên không thể đạt
0.70 * mức số hóa tối đa khi vẫn bị trần ngân sách vùng 12.000 tỷ. Vì vậy app có phần
kiểm tra khả thi và mặc định chạy NSGA-II ở lambda=0.68 để tạo được tập Pareto.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from dataclasses import dataclass
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")

try:
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    HAVE_PYMOO = True
except Exception:
    HAVE_PYMOO = False
    ElementwiseProblem = object
    NSGA2 = None
    minimize = None
    get_termination = None

try:
    st.set_page_config(
        page_title="Bài 7 - Pareto NSGA-II",
        page_icon="🧬",
        layout="wide",
    )
except Exception:
    pass

# ===============================
# Giao diện cơ bản
# ===============================

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1180px;}
    .hero-box {
        padding: 28px 32px; border-radius: 24px;
        background: linear-gradient(135deg, rgba(88,86,214,0.18), rgba(0,188,212,0.10));
        border: 1px solid rgba(255,255,255,0.12); margin-bottom: 24px;
    }
    .pill {
        display:inline-block; padding: 8px 14px; margin: 3px 5px 8px 0;
        border-radius: 999px; font-size: 0.86rem; font-weight: 700;
        background: linear-gradient(135deg, #ec4899, #7c3aed); color: white;
    }
    .note-box {
        padding: 16px 18px; border-radius: 16px;
        border-left: 5px solid #f59e0b;
        background: rgba(245, 158, 11, 0.10); margin: 12px 0;
    }
    .ok-box {
        padding: 16px 18px; border-radius: 16px;
        border-left: 5px solid #22c55e;
        background: rgba(34, 197, 94, 0.10); margin: 12px 0;
    }
    .bad-box {
        padding: 16px 18px; border-radius: 16px;
        border-left: 5px solid #ef4444;
        background: rgba(239, 68, 68, 0.10); margin: 12px 0;
    }
    .small-muted {opacity: .78; font-size: .92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-box">
        <span class="pill">Cấp độ khá khó</span>
        <span class="pill">Pareto</span>
        <span class="pill">NSGA-II</span>
        <span class="pill">pymoo</span>
        <span class="pill">TOPSIS</span>
        <h1>🧬 Bài 7 — Tối ưu đa mục tiêu Pareto với NSGA-II</h1>
        <p style="font-size:1.05rem; line-height:1.75; margin-bottom:0;">
        Bài này phân bổ ngân sách số theo 6 vùng và 4 hạng mục đầu tư, nhưng không tối ưu một mục tiêu duy nhất.
        Mô hình đồng thời xét 4 mục tiêu: tăng trưởng GDP, bao trùm vùng miền, môi trường và an ninh dữ liệu.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===============================
# Dữ liệu và tham số
# ===============================

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

REGIONS = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
REGION_NAMES = {
    "NMM": "Trung du miền núi phía Bắc",
    "RRD": "Đồng bằng sông Hồng",
    "NCC": "Bắc Trung Bộ + DH Trung Bộ",
    "CH": "Tây Nguyên",
    "SE": "Đông Nam Bộ",
    "MD": "Đồng bằng sông Cửu Long",
}
ITEMS = ["I", "D", "AI", "H"]
ITEM_NAMES = {
    "I": "Hạ tầng số",
    "D": "Chuyển đổi số DN",
    "AI": "Năng lực AI",
    "H": "Nhân lực số",
}

# Beta matrix theo Bài 4: hàng là vùng, cột là I, D, AI, H
BETA = np.array(
    [
        [1.15, 0.85, 0.55, 1.30],
        [0.95, 1.25, 1.40, 1.05],
        [1.05, 0.95, 0.85, 1.15],
        [1.20, 0.75, 0.45, 1.35],
        [0.90, 1.30, 1.55, 1.00],
        [1.10, 0.85, 0.65, 1.25],
    ],
    dtype=float,
)

# D0 theo Bài 4
D0 = np.array([38, 78, 55, 32, 82, 48], dtype=float)

# Tham số bổ sung Bài 7
E_CO2 = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38], dtype=float)
RHO_RISK = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22], dtype=float)
SIGMA_H = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30], dtype=float)

BUDGET_TOTAL = 50000.0
REGION_MIN = 5000.0
REGION_MAX = 12000.0
H_MIN = 12000.0
GAMMA = 0.002
LAM_ORIGINAL = 0.70
LAM_SAFE = 0.68

POLICY_WEIGHTS = np.array([0.40, 0.25, 0.20, 0.15], dtype=float)


@dataclass
class RunResult:
    X: np.ndarray
    F_min: np.ndarray
    metrics: pd.DataFrame
    pareto_df: pd.DataFrame
    compromise_index: int
    growth_index: int
    compromise_alloc: pd.DataFrame
    growth_alloc: pd.DataFrame
    opportunity: pd.DataFrame


def lambda_feasibility_bound() -> float:
    """Cận lambda khả thi sơ bộ do vùng yếu nhất và trần ngân sách vùng tạo ra."""
    max_digital_possible_each_region = D0 + GAMMA * REGION_MAX
    m_min_required_base = D0.max()
    return float(max_digital_possible_each_region.min() / m_min_required_base)


def build_param_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    beta_df = pd.DataFrame(BETA, index=[REGION_NAMES[r] for r in REGIONS], columns=[ITEM_NAMES[i] for i in ITEMS])
    beta_df.index.name = "Vùng"
    d0_df = pd.DataFrame(
        {
            "Mã vùng": REGIONS,
            "Vùng": [REGION_NAMES[r] for r in REGIONS],
            "D0": D0,
        }
    )
    extra_df = pd.DataFrame(
        {
            "Mã vùng": REGIONS,
            "Vùng": [REGION_NAMES[r] for r in REGIONS],
            "eᵣ - CO₂/tỷ": E_CO2,
            "ρᵣ - rủi ro/AI": RHO_RISK,
            "σᵣ - giảm rủi ro/H": SIGMA_H,
        }
    )
    return beta_df, d0_df, extra_df


def evaluate_solution(x: np.ndarray, lam: float = LAM_SAFE) -> dict:
    """Tính 4 mục tiêu và mức vi phạm ràng buộc cho một nghiệm 24 biến."""
    X = np.asarray(x, dtype=float).reshape(6, 4)
    regional_budget = X.sum(axis=1)
    total_budget = X.sum()
    digital_after = D0 + GAMMA * X[:, 1]
    max_digital = digital_after.max()

    gdp_gain = float((BETA * X).sum())
    mean_budget = regional_budget.mean()
    inequality_mad = float(np.abs(regional_budget - mean_budget).mean() / (mean_budget + 1e-12))
    emission = float((E_CO2 * (X[:, 0] + X[:, 2])).sum())
    net_security_risk = float((RHO_RISK * X[:, 2]).sum() - (SIGMA_H * X[:, 3]).sum())

    constraint_violations = {
        "C1_total_budget": max(0.0, total_budget - BUDGET_TOTAL),
        "C2_region_floor": float(np.maximum(0.0, REGION_MIN - regional_budget).sum()),
        "C3_region_cap": float(np.maximum(0.0, regional_budget - REGION_MAX).sum()),
        "C4_H_floor": max(0.0, H_MIN - X[:, 3].sum()),
        "C5_fairness": float(np.maximum(0.0, lam * max_digital - digital_after).sum()),
    }
    return {
        "GDP_gain": gdp_gain,
        "Inequality_MAD": inequality_mad,
        "Emission": emission,
        "Net_security_risk": net_security_risk,
        "Total_budget": float(total_budget),
        "H_total": float(X[:, 3].sum()),
        "Max_digital_after": float(max_digital),
        "Min_digital_after": float(digital_after.min()),
        **constraint_violations,
    }


if HAVE_PYMOO:
    class VietnamDigitalParetoProblem(ElementwiseProblem):
        """Bài toán 24 biến, 4 mục tiêu theo Bài 7, giải dạng minimization cho pymoo."""

        def __init__(self, lam: float = LAM_SAFE):
            self.lam = float(lam)
            super().__init__(
                n_var=24,
                n_obj=4,
                n_ieq_constr=20,
                xl=np.zeros(24),
                xu=np.ones(24) * REGION_MAX,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            X = np.asarray(x, dtype=float).reshape(6, 4)
            regional_budget = X.sum(axis=1)
            total_budget = X.sum()
            digital_after = D0 + GAMMA * X[:, 1]
            max_digital = digital_after.max()

            # Mục tiêu 1: max GDP gain => minimize negative GDP gain
            f1 = -float((BETA * X).sum())

            # Mục tiêu 2: giảm bất bình đẳng vùng, dùng MAD chuẩn hóa
            mean_budget = regional_budget.mean()
            f2 = float(np.abs(regional_budget - mean_budget).mean() / (mean_budget + 1e-12))

            # Mục tiêu 3: giảm phát thải gián tiếp từ I và AI
            f3 = float((E_CO2 * (X[:, 0] + X[:, 2])).sum())

            # Mục tiêu 4: giảm rủi ro an ninh dữ liệu ròng
            f4 = float((RHO_RISK * X[:, 2]).sum() - (SIGMA_H * X[:, 3]).sum())

            g = []
            # C1: tổng ngân sách <= 50.000
            g.append(total_budget - BUDGET_TOTAL)
            # C2: ngân sách mỗi vùng >= 5.000
            g.extend((REGION_MIN - regional_budget).tolist())
            # C3: ngân sách mỗi vùng <= 12.000
            g.extend((regional_budget - REGION_MAX).tolist())
            # C4: tổng H >= 12.000
            g.append(H_MIN - X[:, 3].sum())
            # C5: công bằng số hóa D_r + gamma*x_D,r >= lambda*max_r(...)
            g.extend((self.lam * max_digital - digital_after).tolist())

            out["F"] = np.array([f1, f2, f3, f4], dtype=float)
            out["G"] = np.array(g, dtype=float)


def topsis_select(metrics_df: pd.DataFrame, weights: np.ndarray = POLICY_WEIGHTS) -> tuple[int, pd.DataFrame]:
    """Chọn nghiệm thỏa hiệp bằng TOPSIS trên tập Pareto."""
    criteria = ["GDP_gain", "Inequality_MAD", "Emission", "Net_security_risk"]
    X = metrics_df[criteria].to_numpy(dtype=float)
    is_benefit = np.array([True, False, False, False], dtype=bool)
    denom = np.sqrt((X ** 2).sum(axis=0))
    denom[denom == 0] = 1.0
    R = X / denom
    V = R * weights
    ideal = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
    anti = np.where(is_benefit, V.min(axis=0), V.max(axis=0))
    s_plus = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    s_minus = np.sqrt(((V - anti) ** 2).sum(axis=1))
    score = s_minus / (s_plus + s_minus + 1e-12)
    out = metrics_df.copy()
    out["TOPSIS_score"] = score
    out["TOPSIS_rank"] = out["TOPSIS_score"].rank(ascending=False, method="min").astype(int)
    return int(np.argmax(score)), out


def allocation_df(x: np.ndarray) -> pd.DataFrame:
    X = np.asarray(x, dtype=float).reshape(6, 4)
    df = pd.DataFrame(X, index=[REGION_NAMES[r] for r in REGIONS], columns=[ITEM_NAMES[i] for i in ITEMS])
    df.index.name = "Vùng"
    df["Tổng vùng"] = df.sum(axis=1)
    return df


def summarize_pareto(X: np.ndarray, F: np.ndarray, lam: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    for k, x in enumerate(X):
        m = evaluate_solution(x, lam=lam)
        m["solution_id"] = k
        metrics.append(m)
    metrics_df = pd.DataFrame(metrics)
    ordered = ["solution_id", "GDP_gain", "Inequality_MAD", "Emission", "Net_security_risk", "Total_budget", "H_total", "Max_digital_after", "Min_digital_after"]
    metrics_df = metrics_df[ordered + [c for c in metrics_df.columns if c not in ordered]]
    _, pareto_df = topsis_select(metrics_df)
    return metrics_df, pareto_df


@st.cache_data(show_spinner=False)
def run_nsga2_cached(lam: float, pop_size: int, n_gen: int, seed: int) -> dict:
    if not HAVE_PYMOO:
        return {"ok": False, "message": "Chưa cài pymoo", "X": None, "F": None}

    problem = VietnamDigitalParetoProblem(lam=lam)
    algorithm = NSGA2(pop_size=int(pop_size), eliminate_duplicates=True)
    termination = get_termination("n_gen", int(n_gen))
    res = minimize(problem, algorithm, termination, seed=int(seed), verbose=False, save_history=False)

    X = res.X
    F = res.F
    if X is None or F is None or len(np.atleast_2d(X)) == 0:
        return {"ok": False, "message": "NSGA-II không tìm thấy nghiệm khả thi", "X": None, "F": None}

    X = np.atleast_2d(X)
    F = np.atleast_2d(F)
    return {"ok": True, "message": "OK", "X": X, "F": F}


def prepare_result(X: np.ndarray, F: np.ndarray, lam: float) -> RunResult:
    metrics_df, pareto_df = summarize_pareto(X, F, lam)
    compromise_idx, pareto_df = topsis_select(metrics_df)
    growth_idx = int(metrics_df["GDP_gain"].idxmax())

    comp_alloc = allocation_df(X[compromise_idx])
    growth_alloc = allocation_df(X[growth_idx])

    comp = evaluate_solution(X[compromise_idx], lam=lam)
    growth = evaluate_solution(X[growth_idx], lam=lam)

    def pct_worse(growth_value: float, comp_value: float) -> float:
        if abs(comp_value) < 1e-12:
            return np.nan
        return (growth_value - comp_value) / abs(comp_value) * 100.0

    opportunity = pd.DataFrame(
        [
            {
                "Chỉ tiêu": "Bất bình đẳng vùng - Inequality_MAD",
                "Nghiệm thỏa hiệp": comp["Inequality_MAD"],
                "Nghiệm tăng trưởng cao nhất": growth["Inequality_MAD"],
                "% hi sinh / tăng thêm so với thỏa hiệp": pct_worse(growth["Inequality_MAD"], comp["Inequality_MAD"]),
            },
            {
                "Chỉ tiêu": "Phát thải - Emission",
                "Nghiệm thỏa hiệp": comp["Emission"],
                "Nghiệm tăng trưởng cao nhất": growth["Emission"],
                "% hi sinh / tăng thêm so với thỏa hiệp": pct_worse(growth["Emission"], comp["Emission"]),
            },
            {
                "Chỉ tiêu": "Rủi ro an ninh dữ liệu ròng",
                "Nghiệm thỏa hiệp": comp["Net_security_risk"],
                "Nghiệm tăng trưởng cao nhất": growth["Net_security_risk"],
                "% hi sinh / tăng thêm so với thỏa hiệp": pct_worse(growth["Net_security_risk"], comp["Net_security_risk"]),
            },
        ]
    )

    return RunResult(
        X=X,
        F_min=F,
        metrics=metrics_df,
        pareto_df=pareto_df,
        compromise_index=compromise_idx,
        growth_index=growth_idx,
        compromise_alloc=comp_alloc,
        growth_alloc=growth_alloc,
        opportunity=opportunity,
    )


def fig_3d_scatter(result: RunResult):
    df = result.pareto_df.copy()
    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df["GDP_gain"], df["Inequality_MAD"], df["Emission"], s=32, alpha=0.72)
    comp = df.loc[result.compromise_index]
    grow = df.loc[result.growth_index]
    ax.scatter([comp["GDP_gain"]], [comp["Inequality_MAD"]], [comp["Emission"]], s=140, marker="*", label="Thỏa hiệp TOPSIS")
    ax.scatter([grow["GDP_gain"]], [grow["Inequality_MAD"]], [grow["Emission"]], s=120, marker="^", label="Tăng trưởng cao nhất")
    ax.set_xlabel("GDP gain")
    ax.set_ylabel("Bất bình đẳng MAD")
    ax.set_zlabel("Phát thải")
    ax.set_title("Đường biên Pareto: tăng trưởng - bao trùm - môi trường")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def fig_parallel_coordinates(result: RunResult):
    df = result.pareto_df[["GDP_gain", "Inequality_MAD", "Emission", "Net_security_risk"]].copy()
    # Với GDP là benefit, đảo dấu các cost để đường càng cao càng tốt trên mọi trục.
    oriented = pd.DataFrame(
        {
            "Tăng trưởng": df["GDP_gain"],
            "Bao trùm tốt hơn": -df["Inequality_MAD"],
            "Môi trường tốt hơn": -df["Emission"],
            "An ninh tốt hơn": -df["Net_security_risk"],
        }
    )
    norm = (oriented - oriented.min()) / (oriented.max() - oriented.min() + 1e-12)

    fig, ax = plt.subplots(figsize=(9.5, 5.7))
    x_axis = np.arange(norm.shape[1])
    for _, row in norm.iterrows():
        ax.plot(x_axis, row.values, linewidth=0.8, alpha=0.20)

    comp_row = norm.iloc[result.compromise_index]
    grow_row = norm.iloc[result.growth_index]
    ax.plot(x_axis, comp_row.values, linewidth=3.0, marker="o", label="Thỏa hiệp TOPSIS")
    ax.plot(x_axis, grow_row.values, linewidth=2.6, marker="^", linestyle="--", label="Tăng trưởng cao nhất")
    ax.set_xticks(x_axis)
    ax.set_xticklabels(norm.columns, rotation=12)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Điểm chuẩn hóa 0-1, càng cao càng tốt")
    ax.set_title("Parallel coordinates cho 4 mục tiêu")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def fig_allocation_heatmap(alloc: pd.DataFrame, title: str):
    data = alloc[[ITEM_NAMES[i] for i in ITEMS]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.3, 5.5))
    im = ax.imshow(data, aspect="auto")
    ax.set_xticks(np.arange(len(ITEMS)))
    ax.set_xticklabels([ITEM_NAMES[i] for i in ITEMS], rotation=15)
    ax.set_yticks(np.arange(len(REGIONS)))
    ax.set_yticklabels([REGION_NAMES[r] for r in REGIONS])
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Tỷ VND")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:,.0f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    return fig


def make_excel_bytes(result: RunResult, lam: float) -> bytes:
    output = BytesIO()
    beta_df, d0_df, extra_df = build_param_tables()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame({"Tham số": ["lambda", "gamma", "budget_total", "region_min", "region_max", "H_min", "lambda_original", "lambda_bound"],
                      "Giá trị": [lam, GAMMA, BUDGET_TOTAL, REGION_MIN, REGION_MAX, H_MIN, LAM_ORIGINAL, lambda_feasibility_bound()]}).to_excel(writer, sheet_name="params", index=False)
        pd.DataFrame([
            {"Nội dung": "lambda gốc trong đề", "Giá trị": LAM_ORIGINAL, "Diễn giải": "không khả thi với trần 12.000 tỷ/vùng"},
            {"Nội dung": "lambda tối đa khả thi sơ bộ", "Giá trị": lambda_feasibility_bound(), "Diễn giải": "căn cứ D0 Tây Nguyên và D0 max 82"},
            {"Nội dung": "lambda chạy NSGA-II", "Giá trị": lam, "Diễn giải": "bản hiệu chỉnh khả thi"},
        ]).to_excel(writer, sheet_name="feasibility_note", index=False)
        beta_df.to_excel(writer, sheet_name="beta")
        d0_df.to_excel(writer, sheet_name="D0", index=False)
        extra_df.to_excel(writer, sheet_name="extra_params", index=False)
        result.pareto_df.to_excel(writer, sheet_name="pareto", index=False)
        result.compromise_alloc.to_excel(writer, sheet_name="compromise_alloc")
        result.growth_alloc.to_excel(writer, sheet_name="max_growth_alloc")
        result.opportunity.to_excel(writer, sheet_name="opportunity_cost", index=False)
    return output.getvalue()


def fig_to_png_bytes(fig) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    return buffer.getvalue()


def make_html_report(result: RunResult, lam: float) -> str:
    comp = result.pareto_df.loc[result.compromise_index]
    grow = result.pareto_df.loc[result.growth_index]
    html = f"""
    <html><head><meta charset='utf-8'><title>Bài 7 - Pareto NSGA-II</title>
    <style>body{{font-family:Arial, sans-serif; line-height:1.6; margin:32px;}} table{{border-collapse:collapse; width:100%; margin:12px 0;}} th,td{{border:1px solid #ddd; padding:8px;}} th{{background:#f2f2f2;}}</style>
    </head><body>
    <h1>Bài 7 — Tối ưu đa mục tiêu Pareto với NSGA-II</h1>
    <p><b>lambda dùng khi chạy:</b> {lam:.4f}</p>
    <div style="background:#fff7ed;border-left:5px solid #f59e0b;padding:12px;margin:12px 0;">
    <b>Ghi chú nộp bài:</b> Bài 7 kế thừa ràng buộc C1-C5 từ Bài 4. Với tham số gốc lambda=0.70, hệ C3 và C5 không khả thi. Vì vậy báo cáo cần nêu rõ mô hình gốc được kiểm tra infeasible, sau đó dùng lambda hiệu chỉnh khả thi để sinh tập Pareto bằng NSGA-II.
    </div>
    <h2>Nghiệm thỏa hiệp TOPSIS</h2>
    <ul>
      <li>GDP gain: {comp['GDP_gain']:,.2f}</li>
      <li>Inequality MAD: {comp['Inequality_MAD']:.6f}</li>
      <li>Emission: {comp['Emission']:,.2f}</li>
      <li>Net security risk: {comp['Net_security_risk']:,.2f}</li>
      <li>TOPSIS score: {comp['TOPSIS_score']:.6f}</li>
    </ul>
    <h2>Nghiệm tăng trưởng cao nhất</h2>
    <ul>
      <li>GDP gain: {grow['GDP_gain']:,.2f}</li>
      <li>Inequality MAD: {grow['Inequality_MAD']:.6f}</li>
      <li>Emission: {grow['Emission']:,.2f}</li>
      <li>Net security risk: {grow['Net_security_risk']:,.2f}</li>
    </ul>
    <h2>Chi phí cơ hội</h2>
    {result.opportunity.to_html(index=False)}
    <h2>Phân bổ nghiệm thỏa hiệp</h2>
    {result.compromise_alloc.to_html()}
    </body></html>
    """
    return html



# ===============================
# Sidebar điều khiển
# ===============================

st.sidebar.header("⚙️ Thiết lập chạy Bài 7")

lam_bound = lambda_feasibility_bound()
feasibility_diagnostic_df = pd.DataFrame([
    {
        "Nội dung": "Lambda gốc trong đề",
        "Giá trị": LAM_ORIGINAL,
        "Diễn giải": "Bài 7 kế thừa C5 từ Bài 4"
    },
    {
        "Nội dung": "Lambda tối đa khả thi sơ bộ",
        "Giá trị": lam_bound,
        "Diễn giải": "Do Tây Nguyên bị trần 12.000 tỷ/vùng nên chỉ đạt tối đa 32 + 0.002*12.000 = 56 điểm"
    },
    {
        "Nội dung": "Lambda dùng để chạy mặc định",
        "Giá trị": LAM_SAFE,
        "Diễn giải": "Bản hiệu chỉnh khả thi để tạo tập Pareto, không che giấu mô hình gốc không khả thi"
    },
])
st.sidebar.caption(f"Cận lambda khả thi sơ bộ do trần 12.000 tỷ/vùng: {lam_bound:.4f}")

use_safe_lam = st.sidebar.checkbox(
    "Dùng lambda khả thi 0.68 để chạy NSGA-II",
    value=True,
    help="Đề gốc dùng lambda=0.70 nhưng kế thừa mô hình không khả thi. Chọn mục này để chạy được tập Pareto.",
)

if use_safe_lam:
    lam_run = LAM_SAFE
else:
    lam_run = st.sidebar.number_input(
        "lambda công bằng vùng",
        min_value=0.10,
        max_value=0.95,
        value=LAM_ORIGINAL,
        step=0.01,
    )

pop_size = st.sidebar.slider("pop_size", min_value=40, max_value=200, value=100, step=20)
n_gen = st.sidebar.slider("n_gen", min_value=50, max_value=300, value=200, step=50)
seed = st.sidebar.number_input("seed", min_value=1, max_value=9999, value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.caption("Lệnh cài thư viện cần thiết:")
st.sidebar.code("python -m pip install pymoo pandas numpy matplotlib openpyxl streamlit", language="bash")


def make_checklist_df(status: str = "Đã làm") -> pd.DataFrame:
    """Checklist hoàn thành yêu cầu Bài 7 theo đúng các mục 7.4.1-7.4.4 và 7.5."""
    return pd.DataFrame(
        [
            ["7.4.1", "Cài đặt bài toán bằng pymoo", status, "Dùng ElementwiseProblem của pymoo"],
            ["7.4.1", "Định nghĩa 24 biến quyết định", status, "6 vùng × 4 hạng mục I, D, AI, H"],
            ["7.4.1", "Định nghĩa 4 mục tiêu", status, "Tăng trưởng, bao trùm, môi trường, an ninh dữ liệu"],
            ["7.4.1", "Chạy NSGA-II", status, f"pop_size={pop_size}, n_gen={n_gen}"],
            ["7.4.2", "Trích xuất quần thể Pareto cuối cùng", status, "Bảng Pareto có GDP gain, MAD, emission, risk"],
            ["7.4.2", "Vẽ scatter 3D giữa f1, f2, f3", status, "Biểu đồ 3D Pareto"],
            ["7.4.2", "Vẽ parallel coordinates cho 4 mục tiêu", status, "Chuẩn hóa 4 trục, càng cao càng tốt"],
            ["7.4.3", "Áp dụng TOPSIS lên tập Pareto", status, "Trọng số 0.40; 0.25; 0.20; 0.15"],
            ["7.4.3", "Chọn một nghiệm thỏa hiệp duy nhất", status, "Nghiệm có TOPSIS_score cao nhất"],
            ["7.4.4", "Phân tích chi phí cơ hội", status, "So sánh nghiệm tăng trưởng cao nhất với nghiệm thỏa hiệp"],
            ["7.5", "Trả lời câu hỏi chính sách a, b, c", status, "Có phần diễn giải riêng"],
            ["Tải kết quả", "Excel, CSV, HTML, PNG", status, "Có nút tải ở cuối trang"],
        ],
        columns=["Mục", "Yêu cầu của đề", "Trạng thái", "Ghi chú kiểm tra"],
    )


# ===============================
# 7.1. Bối cảnh và mục tiêu học tập
# ===============================

st.header("7.1. Bối cảnh và mục tiêu học tập")
st.markdown(
    """
    Bài 7 chuyển bài toán phân bổ ngân sách số từ dạng **một mục tiêu** sang **đa mục tiêu Pareto**.
    Thay vì chỉ tối đa hóa GDP gain, mô hình phải đồng thời xem xét tăng trưởng, bao trùm vùng miền,
    phát thải môi trường và rủi ro an ninh dữ liệu.

    Kết quả của Bài 7 không phải là một nghiệm tối ưu duy nhất ngay từ đầu, mà là **tập nghiệm Pareto**.
    Sau đó, ta dùng **TOPSIS** để chọn một nghiệm thỏa hiệp phù hợp với trọng số chính sách.
    """
)

# ===============================
# 7.2. Mô hình toán học
# ===============================

st.header("7.2. Mô hình toán học đa mục tiêu")
st.markdown(
    r"""
    Biến quyết định:

    $$x_{j,r} \ge 0, \quad j \in \{I,D,AI,H\}, \quad r \in \{1,...,6\}$$

    Trong đó có **6 vùng** và **4 hạng mục**, nên tổng số biến là:

    $$6 \times 4 = 24 \text{ biến quyết định}$$

    Bốn mục tiêu của mô hình:

    1. **Tối đa hóa tăng trưởng GDP kỳ vọng**

    $$\max f_1(x)=\sum_r \sum_j \beta_{j,r}x_{j,r}$$

    2. **Tối thiểu hóa bất bình đẳng phân bổ ngân sách giữa các vùng**

    $$\min f_2(x)=G(x)$$

    Trong code, Gini được xấp xỉ bằng **MAD chuẩn hóa** của tổng ngân sách vùng.

    3. **Tối thiểu hóa phát thải gián tiếp từ hạ tầng số và AI**

    $$\min f_3(x)=\sum_r e_r(x_{I,r}+x_{AI,r})$$

    4. **Tối thiểu hóa rủi ro an ninh dữ liệu ròng**

    $$\min f_4(x)=\sum_r \rho_r x_{AI,r}-\sum_r \sigma_r x_{H,r}$$
    """
)

st.markdown(
    """
    Các ràng buộc giữ theo Bài 4 gồm: ngân sách tổng, sàn/trần ngân sách mỗi vùng,
    sàn nhân lực số, công bằng vùng và không âm.
    """
)

# ===============================
# 7.3. Dữ liệu tham số
# ===============================

st.header("7.3. Dữ liệu tham số bổ sung")
beta_df, d0_df, extra_df = build_param_tables()

with st.expander("Bảng hệ số tác động biên β theo vùng và hạng mục", expanded=True):
    st.dataframe(beta_df.style.format("{:.2f}"), use_container_width=True)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Chỉ số số hóa ban đầu D0")
    st.dataframe(d0_df, use_container_width=True, hide_index=True)
with col_b:
    st.subheader("Tham số eᵣ, ρᵣ, σᵣ")
    st.dataframe(extra_df, use_container_width=True, hide_index=True)

# ===============================
# Kiểm tra điểm chưa hợp lý của đề
# ===============================

st.header("Kiểm tra tính hợp lý của ràng buộc công bằng C5")

if LAM_ORIGINAL > lam_bound:
    st.markdown(
        f"""
        <div class="bad-box">
        <b>Điểm chưa hợp lý trong đề:</b><br>
        Bài 7 yêu cầu kế thừa ràng buộc công bằng C5 từ Bài 4. Với tham số gốc λ = 0.70, γ = 0.002,
        D0 của Tây Nguyên = 32 và trần ngân sách mỗi vùng = 12.000 tỷ, mô hình không khả thi.<br><br>
        Đông Nam Bộ có D0 = 82 nên mức yêu cầu tối thiểu của Tây Nguyên là:<br>
        <b>0.70 × 82 = 57.4 điểm</b>.<br><br>
        Tây Nguyên tối đa chỉ đạt:<br>
        <b>32 + 0.002 × 12.000 = 56 điểm</b>.<br><br>
        Vì vậy λ = 0.70 vượt cận khả thi sơ bộ khoảng <b>{lam_bound:.4f}</b>. Code mặc định chạy λ = 0.68 để tạo được tập Pareto.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown("<div class='ok-box'>Bộ tham số gốc đạt kiểm tra khả thi sơ bộ.</div>", unsafe_allow_html=True)

st.dataframe(feasibility_diagnostic_df, use_container_width=True, hide_index=True)

if lam_run > lam_bound:
    st.warning(
        f"Bạn đang chọn lambda = {lam_run:.4f}, lớn hơn cận khả thi sơ bộ {lam_bound:.4f}. "
        "NSGA-II có thể không tìm thấy nghiệm khả thi."
    )
else:
    st.success(f"Lambda đang dùng để chạy: {lam_run:.4f}. Giá trị này nằm trong vùng khả thi sơ bộ.")

# ===============================
# Checklist đầu trang
# ===============================

st.header("Checklist yêu cầu Bài 7 trước khi chạy")
st.dataframe(make_checklist_df("Sẽ thực hiện trong app"), use_container_width=True, hide_index=True)

# ===============================
# 7.4.1. Cài đặt và chạy NSGA-II
# ===============================

st.header("7.4.1. Cài đặt bài toán bằng pymoo và chạy NSGA-II")
st.markdown(
    f"""
    Phần này thực hiện đúng yêu cầu **7.4.1**:

    - Dùng `pymoo`.
    - Định nghĩa class `VietnamDigitalParetoProblem(ElementwiseProblem)`.
    - Số biến: **24 biến**.
    - Số mục tiêu: **4 mục tiêu**.
    - Cận dưới biến: 0.
    - Cận trên biến: 12.000 tỷ cho từng biến để phù hợp trần vùng.
    - Thuật toán: **NSGA-II**.
    - Tham số đang chạy: `pop_size={pop_size}`, `n_gen={n_gen}`, `seed={seed}`.
    """
)

if not HAVE_PYMOO:
    st.error("Máy chưa cài pymoo nên chưa thể chạy đúng yêu cầu Bài 7.")
    st.code("python -m pip install pymoo", language="bash")
    st.stop()

run = st.button("🚀 Chạy / cập nhật NSGA-II", type="primary")
if "bai7_last_config" not in st.session_state:
    st.session_state["bai7_last_config"] = None
config = (float(lam_run), int(pop_size), int(n_gen), int(seed))
need_run = run or st.session_state.get("bai7_last_config") != config or "bai7_result" not in st.session_state

if need_run:
    with st.spinner("Đang chạy NSGA-II và trích xuất tập Pareto..."):
        raw = run_nsga2_cached(float(lam_run), int(pop_size), int(n_gen), int(seed))
    st.session_state["bai7_last_config"] = config
    st.session_state["bai7_raw"] = raw
    if raw["ok"]:
        st.session_state["bai7_result"] = prepare_result(raw["X"], raw["F"], float(lam_run))
    else:
        st.session_state.pop("bai7_result", None)

raw = st.session_state.get("bai7_raw", {"ok": False, "message": "Chưa chạy"})
if not raw.get("ok", False):
    st.error(raw.get("message", "Không tìm thấy nghiệm khả thi"))
    st.markdown(
        """
        Gợi ý sửa: bật tùy chọn **Dùng lambda khả thi 0.68** ở thanh bên trái, hoặc giảm lambda xuống dưới khoảng 0.6829.
        Nếu vẫn lỗi, tăng `pop_size` hoặc `n_gen`.
        """
    )
    st.stop()

result: RunResult = st.session_state["bai7_result"]

st.markdown(
    f"""
    <div class="ok-box">
    <b>Đã chạy xong 7.4.1.</b><br>
    NSGA-II tìm được <b>{len(result.pareto_df)}</b> nghiệm Pareto khả thi.
    Đây là tập phương án đánh đổi giữa tăng trưởng, bao trùm, môi trường và an ninh dữ liệu.
    </div>
    """,
    unsafe_allow_html=True,
)

# ===============================
# 7.4.2. Trích xuất Pareto và vẽ biểu đồ
# ===============================

st.header("7.4.2. Trích xuất quần thể Pareto cuối cùng và vẽ biểu đồ")
st.markdown(
    """
    Phần này thực hiện đúng yêu cầu **7.4.2**:

    - Trích xuất quần thể Pareto cuối cùng từ kết quả NSGA-II.
    - Lập bảng các nghiệm Pareto.
    - Vẽ biểu đồ **scatter 3D** giữa tăng trưởng GDP, bất bình đẳng MAD và phát thải.
    - Vẽ biểu đồ **parallel coordinates** cho cả 4 mục tiêu.
    """
)

show_cols = [
    "solution_id", "GDP_gain", "Inequality_MAD", "Emission", "Net_security_risk", "Total_budget", "H_total"
]
st.subheader("Bảng nghiệm Pareto")
st.dataframe(
    result.pareto_df[show_cols].head(20).style.format(
        {
            "GDP_gain": "{:,.2f}",
            "Inequality_MAD": "{:.5f}",
            "Emission": "{:,.2f}",
            "Net_security_risk": "{:,.2f}",
            "Total_budget": "{:,.2f}",
            "H_total": "{:,.2f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

c1, c2 = st.columns(2)
with c1:
    fig1 = fig_3d_scatter(result)
    st.pyplot(fig1, clear_figure=False)
with c2:
    fig2 = fig_parallel_coordinates(result)
    st.pyplot(fig2, clear_figure=False)

# ===============================
# 7.4.3. TOPSIS chọn nghiệm thỏa hiệp
# ===============================

st.header("7.4.3. Áp dụng TOPSIS lên tập Pareto để chọn nghiệm thỏa hiệp")
st.markdown(
    """
    Phần này thực hiện đúng yêu cầu **7.4.3**.

    Trọng số chính sách dùng trong TOPSIS:

    - Tăng trưởng: **0.40**.
    - Bao trùm: **0.25**.
    - Môi trường: **0.20**.
    - An ninh dữ liệu: **0.15**.

    Trong TOPSIS, GDP gain là tiêu chí lợi ích; bất bình đẳng, phát thải và rủi ro là tiêu chí chi phí.
    """
)

comp = result.pareto_df.loc[result.compromise_index]
grow = result.pareto_df.loc[result.growth_index]

m1, m2, m3, m4 = st.columns(4)
m1.metric("GDP gain - TOPSIS", f"{comp['GDP_gain']:,.0f}")
m2.metric("Bất bình đẳng MAD", f"{comp['Inequality_MAD']:.4f}")
m3.metric("Phát thải", f"{comp['Emission']:,.0f}")
m4.metric("Rủi ro ròng", f"{comp['Net_security_risk']:,.0f}")

st.subheader("Top 10 nghiệm theo điểm TOPSIS")
topsis_cols = [
    "solution_id", "TOPSIS_rank", "TOPSIS_score", "GDP_gain", "Inequality_MAD", "Emission", "Net_security_risk", "Total_budget", "H_total"
]
st.dataframe(
    result.pareto_df.sort_values("TOPSIS_rank")[topsis_cols].head(10).style.format(
        {
            "TOPSIS_score": "{:.4f}",
            "GDP_gain": "{:,.2f}",
            "Inequality_MAD": "{:.5f}",
            "Emission": "{:,.2f}",
            "Net_security_risk": "{:,.2f}",
            "Total_budget": "{:,.2f}",
            "H_total": "{:,.2f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Phân bổ nghiệm thỏa hiệp TOPSIS")
    st.dataframe(result.compromise_alloc.style.format("{:,.2f}"), use_container_width=True)
with col2:
    st.subheader("Phân bổ nghiệm tăng trưởng cao nhất")
    st.dataframe(result.growth_alloc.style.format("{:,.2f}"), use_container_width=True)

st.subheader("Heatmap phân bổ ngân sách của nghiệm thỏa hiệp")
fig3 = fig_allocation_heatmap(result.compromise_alloc, "Phân bổ ngân sách nghiệm thỏa hiệp TOPSIS")
st.pyplot(fig3, clear_figure=False)

# ===============================
# 7.4.4. Chi phí cơ hội
# ===============================

st.header("7.4.4. Phân tích chi phí cơ hội của nghiệm tăng trưởng cao nhất")
st.markdown(
    """
    Phần này thực hiện đúng yêu cầu **7.4.4**.

    Ta so sánh **nghiệm tăng trưởng cao nhất** với **nghiệm thỏa hiệp TOPSIS**.
    Nếu phần trăm dương, nghĩa là để đạt tăng trưởng cao nhất, mô hình phải chấp nhận mức bất bình đẳng,
    phát thải hoặc rủi ro cao hơn so với nghiệm thỏa hiệp.
    """
)

st.dataframe(
    result.opportunity.style.format(
        {
            "Nghiệm thỏa hiệp": "{:,.6f}",
            "Nghiệm tăng trưởng cao nhất": "{:,.6f}",
            "% hi sinh / tăng thêm so với thỏa hiệp": "{:,.2f}%",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

# ===============================
# 7.5. Chính sách
# ===============================

st.header("7.5. Câu hỏi thảo luận chính sách")

st.subheader("7.5.a. Đánh đổi giữa tăng trưởng và bao trùm có rõ ràng không?")
st.markdown(
    f"""
    Có. Tập Pareto cho thấy khi ưu tiên tối đa hóa GDP gain, mức phân bổ thường nghiêng về những vùng và hạng mục có hệ số β cao,
    ví dụ AI ở Đông Nam Bộ hoặc Đồng bằng sông Hồng. Khi đó chỉ tiêu bất bình đẳng vùng có xu hướng cao hơn nghiệm thỏa hiệp.

    Trong kết quả hiện tại, nghiệm tăng trưởng cao nhất đạt GDP gain khoảng **{grow['GDP_gain']:,.0f}**,
    còn nghiệm thỏa hiệp đạt khoảng **{comp['GDP_gain']:,.0f}**. Đổi lại, nghiệm tăng trưởng cao nhất có bất bình đẳng MAD là
    **{grow['Inequality_MAD']:.4f}**, so với **{comp['Inequality_MAD']:.4f}** ở nghiệm thỏa hiệp.

    Điều này phản ánh thực tế cơ cấu vùng của Việt Nam: những vùng có nền tảng số, công nghiệp và dịch vụ mạnh thường cho hiệu quả biên cao hơn,
    nhưng nếu chỉ chạy theo tăng trưởng thì khoảng cách vùng số có thể mở rộng.
    """
)

st.subheader("7.5.b. Trọng số 0.40; 0.25; 0.20; 0.15 có phù hợp không?")
st.markdown(
    """
    Bộ trọng số này tương đối hợp lý nếu coi tăng trưởng là mục tiêu trung tâm, nhưng vẫn giữ vai trò đáng kể cho bao trùm,
    môi trường và an ninh. Tuy nhiên, nếu muốn bám sát cam kết chuyển đổi xanh và phát thải ròng bằng 0, có thể tăng trọng số
    môi trường từ 0.20 lên khoảng 0.25-0.30. Nếu ưu tiên chủ quyền số và an ninh dữ liệu trong bối cảnh AI, có thể tăng trọng số
    an ninh từ 0.15 lên 0.20.

    Một phương án cân bằng hơn có thể là: tăng trưởng 0.35, bao trùm 0.25, môi trường 0.25, an ninh 0.15;
    hoặc tăng trưởng 0.35, bao trùm 0.25, môi trường 0.20, an ninh 0.20 nếu vấn đề dữ liệu và an ninh mạng được đặt cao hơn.
    """
)

st.subheader("7.5.c. Vai trò của NSGA-II khác gì LP đơn mục tiêu? Có thay thế quyết định chính trị không?")
st.markdown(
    """
    LP đơn mục tiêu thường trả về một nghiệm tối ưu duy nhất theo một hàm mục tiêu đã định trước, ví dụ tối đa hóa GDP gain.
    NSGA-II không ép các mục tiêu xung đột vào một chỉ tiêu duy nhất ngay từ đầu, mà tạo ra một tập nghiệm Pareto để người làm chính sách
    nhìn thấy các phương án đánh đổi khác nhau.

    Vì vậy, NSGA-II là công cụ hỗ trợ minh bạch hóa đánh đổi chính sách, chứ không thay thế quyết định chính trị - xã hội.
    Việc chọn nghiệm cuối cùng vẫn cần hội đồng chính sách, chuyên gia, địa phương, doanh nghiệp và cộng đồng tham gia thảo luận,
    đặc biệt khi lựa chọn đó liên quan đến công bằng vùng miền, sinh kế, môi trường và an ninh dữ liệu.
    """
)

# ===============================
# Checklist hoàn thành cuối trang
# ===============================

st.header("Checklist hoàn thành 100% yêu cầu Bài 7")
st.dataframe(make_checklist_df("Đã làm"), use_container_width=True, hide_index=True)

# ===============================
# Tải kết quả
# ===============================

st.header("Tải kết quả Bài 7")

output_manifest = pd.DataFrame([
    ["bai07_pareto.csv", "Tập nghiệm Pareto và điểm TOPSIS"],
    ["bai07_compromise_allocation.csv", "Phân bổ nghiệm thỏa hiệp TOPSIS"],
    ["bai07_max_growth_allocation.csv", "Phân bổ nghiệm tăng trưởng cao nhất"],
    ["bai07_opportunity_cost.csv", "Chi phí cơ hội giữa nghiệm tăng trưởng và nghiệm thỏa hiệp"],
    ["bai07_feasibility_diagnostic.csv", "Chẩn đoán lambda 0.70 không khả thi"],
    ["bai07_scatter_3d.png", "Scatter 3D Pareto"],
    ["bai07_parallel.png", "Parallel coordinates 4 mục tiêu"],
    ["bai07_heatmap.png", "Heatmap phân bổ nghiệm thỏa hiệp"],
    ["bai07_report.html", "Báo cáo HTML"],
], columns=["File output", "Ý nghĩa"])

excel_bytes = make_excel_bytes(result, float(lam_run))
html_report = make_html_report(result, float(lam_run)).encode("utf-8")
pareto_csv = result.pareto_df.to_csv(index=False).encode("utf-8-sig")
alloc_csv = result.compromise_alloc.to_csv().encode("utf-8-sig")
png_3d = fig_to_png_bytes(fig1)
png_parallel = fig_to_png_bytes(fig2)
png_heatmap = fig_to_png_bytes(fig3)

c1, c2, c3 = st.columns(3)
with c1:
    st.download_button(
        "📘 Tải Excel tổng hợp",
        excel_bytes,
        file_name="bai07_pareto_nsga2_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.download_button("📄 Tải HTML report", html_report, file_name="bai07_report.html", mime="text/html")
with c2:
    st.download_button("📊 Tải Pareto CSV", pareto_csv, file_name="bai07_pareto.csv", mime="text/csv")
    st.download_button("📊 Tải allocation CSV", alloc_csv, file_name="bai07_compromise_allocation.csv", mime="text/csv")
with c3:
    st.download_button("🖼️ Tải scatter 3D PNG", png_3d, file_name="bai07_scatter_3d.png", mime="image/png")
    st.download_button("🖼️ Tải parallel PNG", png_parallel, file_name="bai07_parallel.png", mime="image/png")
    st.download_button("🖼️ Tải heatmap PNG", png_heatmap, file_name="bai07_heatmap.png", mime="image/png")

# Tự lưu outputs để có bằng chứng chạy thật cho báo cáo cuối môn.
try:
    (OUTPUT_DIR / "bai07_report.html").write_bytes(html_report)
    result.pareto_df.to_csv(OUTPUT_DIR / "bai07_pareto.csv", index=False, encoding="utf-8-sig")
    result.compromise_alloc.to_csv(OUTPUT_DIR / "bai07_compromise_allocation.csv", encoding="utf-8-sig")
    result.growth_alloc.to_csv(OUTPUT_DIR / "bai07_max_growth_allocation.csv", encoding="utf-8-sig")
    result.opportunity.to_csv(OUTPUT_DIR / "bai07_opportunity_cost.csv", index=False, encoding="utf-8-sig")
    feasibility_diagnostic_df.to_csv(OUTPUT_DIR / "bai07_feasibility_diagnostic.csv", index=False, encoding="utf-8-sig")
    output_manifest.to_csv(OUTPUT_DIR / "bai07_output_manifest.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "bai07_scatter_3d.png").write_bytes(png_3d)
    (OUTPUT_DIR / "bai07_parallel.png").write_bytes(png_parallel)
    (OUTPUT_DIR / "bai07_heatmap.png").write_bytes(png_heatmap)
except Exception:
    pass

st.markdown("---")
st.caption("Bài 7 hoàn thành: 7.4.1, 7.4.2, 7.4.3, 7.4.4, 7.5 và phần tải kết quả.")
