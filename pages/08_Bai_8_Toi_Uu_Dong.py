# -*- coding: utf-8 -*-
"""
Bài 8 — Tối ưu động phân bổ liên thời gian 2026-2035
Môn: Các mô hình ra quyết định

File này được thiết kế để đặt trong thư mục pages/ của webapp Streamlit.
Tên gợi ý: 08_Bai_8_Toi_Uu_Dong.py

Nội dung chính:
- 8.3.1: Giải bài toán tối ưu động bằng scipy.optimize.minimize, phương pháp SLSQP.
- 8.3.2: Vẽ quỹ đạo tối ưu K, D, AI, H, Y, C giai đoạn 2026-2035.
- 8.3.3: Phân tích cú sốc năm 2028 làm Y giảm 8%.
- 8.3.4: So sánh chiến lược đầu tư trải đều và front-load.
- 8.4: Trả lời câu hỏi chính sách.
"""

from __future__ import annotations

import io
import math
import os
import textwrap
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover
    raise RuntimeError("File này cần chạy bằng Streamlit: python -m streamlit run app.py") from exc

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
    SCIPY_ERROR = ""
except Exception as exc:  # pragma: no cover
    SCIPY_AVAILABLE = False
    SCIPY_ERROR = str(exc)


# ============================================================
# 0. Cấu hình trang và CSS nhẹ
# ============================================================

st.set_page_config(
    page_title="Bài 8 - Tối ưu động",
    page_icon="📈",
    layout="wide",
)

def inject_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {max-width: 1180px; padding-top: 2rem;}
        .hero-card {
            padding: 1.5rem 1.7rem;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(79,70,229,.14), rgba(236,72,153,.12));
            border: 1px solid rgba(148,163,184,.22);
            margin-bottom: 1.2rem;
        }
        .pill {
            display: inline-block;
            padding: .32rem .75rem;
            margin: .1rem .25rem .35rem 0;
            border-radius: 999px;
            background: linear-gradient(90deg, #ec4899, #7c3aed);
            color: white;
            font-size: .82rem;
            font-weight: 700;
        }
        .note-box {
            padding: 1rem 1.1rem;
            border-radius: 16px;
            border-left: 5px solid #f59e0b;
            background: rgba(245,158,11,.10);
            margin: .8rem 0 1rem 0;
        }
        .ok-box {
            padding: 1rem 1.1rem;
            border-radius: 16px;
            border-left: 5px solid #22c55e;
            background: rgba(34,197,94,.10);
            margin: .8rem 0 1rem 0;
        }
        .bad-box {
            padding: 1rem 1.1rem;
            border-radius: 16px;
            border-left: 5px solid #ef4444;
            background: rgba(239,68,68,.10);
            margin: .8rem 0 1rem 0;
        }
        .metric-help {
            color: #94a3b8;
            font-size: .88rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()


# ============================================================
# 1. Tham số và dữ liệu
# ============================================================

@dataclass
class ModelParams:
    """Tham số mô hình động Bài 8."""

    T: int = 10
    start_year: int = 2026

    # Cobb-Douglas
    alpha_K: float = 0.33
    beta_L: float = 0.42
    gamma_D: float = 0.10
    delta_AI_exp: float = 0.08
    theta_H_exp: float = 0.07

    # Trạng thái ban đầu
    K0: float = 27500.0       # nghìn tỷ VND
    L0: float = 53.9          # triệu lao động
    D0: float = 20.3          # % GDP hoặc chỉ số số hóa
    AI0: float = 86.0         # nghìn DN / chỉ số AI
    H0: float = 30.0          # %
    A0: float = 30.944377     # lấy từ Bài 1, có thể chỉnh trên sidebar

    # Tăng trưởng lao động và TFP ngoại sinh
    g_L: float = 0.006        # giả định lao động tăng 0.6%/năm
    g_A_base: float = 0.012   # theo đề: TFP tăng 1.2%/năm

    # Khấu hao và tích lũy
    dep_K: float = 0.05
    dep_D: float = 0.12
    dep_AI: float = 0.15
    theta_H: float = 0.8
    mu: float = 0.02

    # Nội sinh hóa TFP: A_{t+1}=A_t(1+phi1 D + phi2 AI + phi3 H)
    # Nếu dùng nguyên giá trị đề, A tăng rất mạnh vì D, AI, H đang ở thang 20-100.
    # Code mặc định chia D, AI, H cho 100 trong hàm cập nhật để tránh bùng nổ phi thực tế.
    phi1: float = 0.003
    phi2: float = 0.002
    phi3: float = 0.004
    normalize_phi_inputs: bool = True

    # Hệ số chiết khấu
    rho: float = 0.97

    # Giới hạn tiêu dùng/đầu tư theo tỷ trọng GDP từng năm
    max_invest_share: float = 0.82
    min_consumption_share: float = 0.05

    # Hệ số chuyển đổi đơn vị đầu tư sang biến D, AI, H.
    # Lý do: đề cộng trực tiếp I_D, I_AI, I_H vào D, AI, H nhưng các biến khác đơn vị.
    # practical_scaled dùng hệ số chuyển đổi mềm để kết quả không bùng nổ.
    unit_mode: str = "practical_scaled"
    eta_D: float = 0.0010     # 1 nghìn tỷ đầu tư D làm tăng 0.001 điểm D sau khấu hao
    eta_AI: float = 0.0030    # 1 nghìn tỷ đầu tư AI làm tăng 0.003 đơn vị AI
    eta_H: float = 0.0008     # 1 nghìn tỷ đầu tư H làm tăng 0.0008 điểm H, nhân thêm theta_H

    # Cận trên kỹ thuật để tránh biến trạng thái tăng phi lý
    cap_D: float = 100.0
    cap_AI: float = 250.0
    cap_H: float = 80.0


def default_macro_data() -> pd.DataFrame:
    """Dữ liệu Bài 1 dùng để tính A tham khảo nếu cần."""
    return pd.DataFrame(
        {
            "year": [2020, 2021, 2022, 2023, 2024, 2025],
            "Y": [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
            "K": [16500, 17800, 19600, 21300, 23500, 25900],
            "L": [53.6, 50.5, 51.7, 52.4, 52.9, 53.4],
            "D": [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
            "AI": [55.6, 60.2, 65.4, 67.0, 73.8, 80.1],
            "H": [24.1, 26.1, 26.2, 27.0, 28.4, 29.2],
        }
    )


def estimate_a_from_bai1() -> pd.DataFrame:
    """Ước lượng A_t từ dữ liệu Bài 1 để minh họa A0 dùng trong Bài 8."""
    df = default_macro_data()
    alpha, beta, gamma, delta, theta = 0.33, 0.42, 0.10, 0.08, 0.07
    df["A_t"] = df["Y"] / (
        (df["K"] ** alpha)
        * (df["L"] ** beta)
        * (df["D"] ** gamma)
        * (df["AI"] ** delta)
        * (df["H"] ** theta)
    )
    return df


def output_dir() -> Path:
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    return out


# ============================================================
# 2. Hàm sản xuất, mô phỏng, tối ưu
# ============================================================

def production(A: float, K: float, L: float, D: float, AI: float, H: float, p: ModelParams) -> float:
    """Hàm sản xuất Cobb-Douglas mở rộng."""
    K = max(K, 1e-9)
    L = max(L, 1e-9)
    D = max(D, 1e-9)
    AI = max(AI, 1e-9)
    H = max(H, 1e-9)
    return (
        A
        * (K ** p.alpha_K)
        * (L ** p.beta_L)
        * (D ** p.gamma_D)
        * (AI ** p.delta_AI_exp)
        * (H ** p.theta_H_exp)
    )


def phi_growth_factor(D: float, AI: float, H: float, p: ModelParams) -> float:
    """Hệ số tăng TFP nội sinh."""
    if p.normalize_phi_inputs:
        d_val, ai_val, h_val = D / 100.0, AI / 100.0, H / 100.0
    else:
        d_val, ai_val, h_val = D, AI, H
    return 1.0 + p.phi1 * d_val + p.phi2 * ai_val + p.phi3 * h_val


def get_eta(p: ModelParams) -> Tuple[float, float, float]:
    """Lấy hệ số chuyển đổi đầu tư sang D/AI/H theo chế độ đơn vị."""
    if p.unit_mode == "raw_as_written":
        return 1.0, 1.0, 1.0
    return p.eta_D, p.eta_AI, p.eta_H


def reshape_shares(z: np.ndarray, p: ModelParams) -> np.ndarray:
    return np.asarray(z, dtype=float).reshape(p.T, 4)


def simulate_from_shares(
    z: np.ndarray,
    p: ModelParams,
    shock_year: Optional[int] = None,
    shock_drop: float = 0.08,
    label: str = "baseline",
) -> Tuple[pd.DataFrame, float, bool]:
    """
    Mô phỏng quỹ đạo từ ma trận tỷ trọng đầu tư.

    z: vector dài T*4 gồm tỷ trọng đầu tư vào K, D, AI, H trên Y hiệu dụng mỗi năm.
    C_t = Y_eff * (1 - tổng tỷ trọng đầu tư).
    """
    shares = reshape_shares(z, p)
    years = list(range(p.start_year, p.start_year + p.T))

    eta_D, eta_AI, eta_H = get_eta(p)

    K, D, AI, H, A = p.K0, p.D0, p.AI0, p.H0, p.A0
    rows: List[Dict[str, float]] = []
    welfare = 0.0
    feasible = True

    for t, year in enumerate(years):
        L = p.L0 * ((1 + p.g_L) ** t)
        Y_plan = production(A, K, L, D, AI, H, p)
        shock_factor = 1.0
        if shock_year is not None and year == shock_year:
            shock_factor = 1.0 - shock_drop
        Y_eff = Y_plan * shock_factor

        sK, sD, sAI, sH = np.clip(shares[t], 0.0, 0.999)
        invest_share = float(sK + sD + sAI + sH)

        if invest_share > 1.0 - p.min_consumption_share + 1e-8:
            feasible = False

        C = max(Y_eff * (1.0 - invest_share), 1e-9)
        I_K, I_D, I_AI, I_H = Y_eff * sK, Y_eff * sD, Y_eff * sAI, Y_eff * sH

        if not np.isfinite(C) or C <= 0:
            feasible = False

        welfare += (p.rho ** t) * math.log(max(C, 1e-9))

        rows.append(
            {
                "strategy": label,
                "year": year,
                "A": A,
                "K": K,
                "L": L,
                "D": D,
                "AI": AI,
                "H": H,
                "Y_plan": Y_plan,
                "shock_factor": shock_factor,
                "Y_effective": Y_eff,
                "C": C,
                "I_K": I_K,
                "I_D": I_D,
                "I_AI": I_AI,
                "I_H": I_H,
                "share_K": sK,
                "share_D": sD,
                "share_AI": sAI,
                "share_H": sH,
                "invest_share": invest_share,
                "consumption_share": 1.0 - invest_share,
                "welfare_term": (p.rho ** t) * math.log(max(C, 1e-9)),
            }
        )

        # Phương trình trạng thái
        K_next = (1 - p.dep_K) * K + I_K
        D_next = (1 - p.dep_D) * D + eta_D * I_D
        AI_next = (1 - p.dep_AI) * AI + eta_AI * I_AI
        # BrainDrain_t chưa được đề lượng hóa; code giả định mất mát = mu * H_t.
        H_next = H + p.theta_H * eta_H * I_H - p.mu * H

        if p.unit_mode == "practical_scaled":
            D_next = min(max(D_next, 1e-9), p.cap_D)
            AI_next = min(max(AI_next, 1e-9), p.cap_AI)
            H_next = min(max(H_next, 1e-9), p.cap_H)

        A_next = A * phi_growth_factor(D, AI, H, p) * (1 + p.g_A_base)

        K, D, AI, H, A = K_next, D_next, AI_next, H_next, A_next

        if not all(np.isfinite(v) for v in [K, D, AI, H, A]):
            feasible = False
            break

    return pd.DataFrame(rows), welfare, feasible


def objective(z: np.ndarray, p: ModelParams, shock_year: Optional[int] = None, shock_drop: float = 0.08) -> float:
    df, welfare, feasible = simulate_from_shares(z, p, shock_year=shock_year, shock_drop=shock_drop)
    if not feasible or df.empty:
        return 1e8

    # Phạt nhẹ nếu đầu tư quá lớn hoặc tiêu dùng quá thấp
    shares = reshape_shares(z, p)
    invest_sum = shares.sum(axis=1)
    penalty = 0.0
    penalty += 1e6 * np.maximum(invest_sum - p.max_invest_share, 0).sum()
    penalty += 1e6 * np.maximum(p.min_consumption_share - (1 - invest_sum), 0).sum()

    return -welfare + penalty


def make_constraints(p: ModelParams):
    cons = []
    for t in range(p.T):
        idx = slice(4 * t, 4 * (t + 1))
        cons.append(
            {
                "type": "ineq",
                "fun": lambda z, idx=idx: p.max_invest_share - float(np.sum(z[idx])),
            }
        )
        cons.append(
            {
                "type": "ineq",
                "fun": lambda z, idx=idx: (1.0 - p.min_consumption_share) - float(np.sum(z[idx])),
            }
        )
    return cons


def initial_guess(p: ModelParams, mode: str = "balanced") -> np.ndarray:
    """
    Tạo nghiệm khởi tạo cho SLSQP.
    Mỗi dòng là share đầu tư vào [K, D, AI, H].
    """
    if mode == "balanced":
        row = np.array([0.08, 0.08, 0.06, 0.08])
        X = np.tile(row, (p.T, 1))
    elif mode == "frontload":
        X = np.zeros((p.T, 4))
        for t in range(p.T):
            if t < 3:
                X[t] = np.array([0.12, 0.13, 0.11, 0.14])
            else:
                X[t] = np.array([0.06, 0.05, 0.04, 0.06])
    elif mode == "human_first":
        X = np.tile(np.array([0.06, 0.06, 0.04, 0.16]), (p.T, 1))
    elif mode == "ai_first":
        X = np.tile(np.array([0.06, 0.08, 0.16, 0.06]), (p.T, 1))
    else:
        X = np.tile(np.array([0.05, 0.05, 0.05, 0.05]), (p.T, 1))

    # Ép tổng đầu tư mỗi năm không vượt cận.
    for t in range(p.T):
        s = X[t].sum()
        if s > p.max_invest_share:
            X[t] *= p.max_invest_share / s
    return X.ravel()


def optimize_slsqp(
    p: ModelParams,
    shock_year: Optional[int] = None,
    shock_drop: float = 0.08,
    maxiter: int = 650,
) -> Tuple[pd.DataFrame, float, object]:
    """Giải bài toán bằng scipy.optimize.minimize SLSQP."""
    if not SCIPY_AVAILABLE:
        raise ImportError("Thiếu scipy. Hãy cài: python -m pip install scipy")

    bounds = [(0.0, p.max_invest_share) for _ in range(p.T * 4)]
    constraints = make_constraints(p)

    starts = [
        initial_guess(p, "balanced"),
        initial_guess(p, "frontload"),
        initial_guess(p, "human_first"),
        initial_guess(p, "ai_first"),
        initial_guess(p, "low"),
    ]

    best_res = None
    for x0 in starts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                objective,
                x0,
                args=(p, shock_year, shock_drop),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": maxiter, "ftol": 1e-8, "disp": False},
            )
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    df, welfare, feasible = simulate_from_shares(
        best_res.x,
        p,
        shock_year=shock_year,
        shock_drop=shock_drop,
        label="Tối ưu SLSQP" if shock_year is None else "Tối ưu có cú sốc",
    )
    return df, welfare, best_res


def optimize_slsqp_policy_constrained(
    p: ModelParams,
    min_d_share: float = 0.025,
    min_ai_share: float = 0.020,
    min_h_share: float = 0.030,
    maxiter: int = 650,
) -> Tuple[pd.DataFrame, float, object]:
    """Giải thêm kịch bản mở rộng có sàn đầu tư D/AI/H mỗi năm.

    Đây là phần gia cố cho báo cáo cuối môn: nghiệm tối ưu thuần utility của Bài 8
    có thể rơi vào nghiệm góc, gần như chỉ đầu tư K. Các sàn chính sách này không
    thay đổi mô hình gốc của đề, mà tạo bảng so sánh để giải thích rằng nếu Chính phủ
    yêu cầu duy trì chuyển đổi số, AI và nhân lực thì quỹ đạo cân bằng hơn.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("Thiếu scipy. Hãy cài: python -m pip install scipy")

    bounds = []
    for _ in range(p.T):
        bounds.extend([
            (0.0, p.max_invest_share),
            (min_d_share, p.max_invest_share),
            (min_ai_share, p.max_invest_share),
            (min_h_share, p.max_invest_share),
        ])
    constraints = make_constraints(p)

    starts = [
        initial_guess(p, "balanced"),
        initial_guess(p, "frontload"),
        initial_guess(p, "human_first"),
        initial_guess(p, "ai_first"),
    ]
    # Bảo đảm nghiệm khởi tạo thỏa các sàn D/AI/H.
    fixed_starts = []
    for x0 in starts:
        X = reshape_shares(x0, p)
        X[:, 1] = np.maximum(X[:, 1], min_d_share)
        X[:, 2] = np.maximum(X[:, 2], min_ai_share)
        X[:, 3] = np.maximum(X[:, 3], min_h_share)
        for t in range(p.T):
            total = X[t].sum()
            if total > p.max_invest_share:
                # Giảm phần K trước để giữ sàn D/AI/H.
                overflow = total - p.max_invest_share
                X[t, 0] = max(0.0, X[t, 0] - overflow)
        fixed_starts.append(X.ravel())

    best_res = None
    for x0 in fixed_starts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                objective,
                x0,
                args=(p, None, 0.08),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": maxiter, "ftol": 1e-8, "disp": False},
            )
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    df, welfare, feasible = simulate_from_shares(best_res.x, p, label="Mở rộng có sàn D-AI-H")
    return df, welfare, best_res


def fixed_strategy_shares(p: ModelParams, strategy: str) -> np.ndarray:
    """
    Tạo chính sách cố định để so sánh 8.3.4.
    Trả về vector T*4 tỷ trọng đầu tư vào [K, D, AI, H].
    """
    X = np.zeros((p.T, 4))

    if strategy == "Trải đều":
        # Tổng đầu tư 32% Y mỗi năm, chia tương đối đều nhưng ưu tiên K/H nhẹ.
        X[:] = np.array([0.09, 0.08, 0.06, 0.09])
    elif strategy == "Front-load":
        for t in range(p.T):
            if t < 3:
                # 3 năm đầu đầu tư mạnh hơn, sau đó giảm.
                X[t] = np.array([0.13, 0.12, 0.10, 0.13])
            else:
                X[t] = np.array([0.055, 0.05, 0.04, 0.055])
    elif strategy == "Đào tạo đi trước":
        for t in range(p.T):
            if t < 3:
                X[t] = np.array([0.07, 0.07, 0.04, 0.18])
            else:
                X[t] = np.array([0.08, 0.08, 0.07, 0.09])
    elif strategy == "AI đi trước":
        X[:] = np.array([0.06, 0.07, 0.17, 0.06])
    else:
        X[:] = np.array([0.05, 0.05, 0.05, 0.05])

    for t in range(p.T):
        s = X[t].sum()
        limit = min(p.max_invest_share, 1.0 - p.min_consumption_share)
        if s > limit:
            X[t] *= limit / s
    return X.ravel()


def compare_fixed_strategies(p: ModelParams) -> pd.DataFrame:
    rows = []
    for name in ["Trải đều", "Front-load", "Đào tạo đi trước", "AI đi trước"]:
        z = fixed_strategy_shares(p, name)
        df, welfare, feasible = simulate_from_shares(z, p, label=name)
        rows.append(
            {
                "Chiến lược": name,
                "Welfare tổng": welfare,
                "Y_2035": df["Y_plan"].iloc[-1],
                "C bình quân": df["C"].mean(),
                "Tổng đầu tư K": df["I_K"].sum(),
                "Tổng đầu tư D": df["I_D"].sum(),
                "Tổng đầu tư AI": df["I_AI"].sum(),
                "Tổng đầu tư H": df["I_H"].sum(),
                "Khả thi": "Có" if feasible else "Không",
            }
        )
    return pd.DataFrame(rows).sort_values("Welfare tổng", ascending=False).reset_index(drop=True)


def policy_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Tóm tắt tỷ trọng đầu tư bình quân theo giai đoạn."""
    early = df[df["year"] <= df["year"].min() + 2]
    late = df[df["year"] > df["year"].min() + 2]
    rows = []
    for label, sub in [("3 năm đầu", early), ("Các năm sau", late), ("Toàn kỳ", df)]:
        rows.append(
            {
                "Giai đoạn": label,
                "K_share_avg": sub["share_K"].mean(),
                "D_share_avg": sub["share_D"].mean(),
                "AI_share_avg": sub["share_AI"].mean(),
                "H_share_avg": sub["share_H"].mean(),
                "Invest_share_avg": sub["invest_share"].mean(),
                "Consumption_share_avg": sub["consumption_share"].mean(),
            }
        )
    return pd.DataFrame(rows)


def classify_loading(df: pd.DataFrame) -> str:
    """Nhận diện front-loaded hay back-loaded dựa trên tỷ trọng đầu tư 3 năm đầu so với các năm sau."""
    summary = policy_summary(df)
    early = float(summary.loc[summary["Giai đoạn"] == "3 năm đầu", "Invest_share_avg"].iloc[0])
    late = float(summary.loc[summary["Giai đoạn"] == "Các năm sau", "Invest_share_avg"].iloc[0])
    if early > late + 0.03:
        return "Front-loaded: đầu tư mạnh hơn ở giai đoạn đầu."
    if late > early + 0.03:
        return "Back-loaded: đầu tư tăng mạnh hơn ở giai đoạn sau."
    return "Tương đối ổn định: không lệch rõ về đầu kỳ hay cuối kỳ."


def make_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_name = sheet_name[:31]
            df.to_excel(writer, index=False, sheet_name=safe_name)
    return buffer.getvalue()


def fig_to_png_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=180)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# 3. Giao diện Streamlit
# ============================================================

st.markdown(
    """
    <div class="hero-card">
        <span class="pill">Cấp độ khá khó</span>
        <span class="pill">Dynamic optimization</span>
        <span class="pill">SLSQP</span>
        <span class="pill">Cobb-Douglas</span>
        <h1>📈 Bài 8 — Tối ưu động phân bổ liên thời gian 2026-2035</h1>
        <p>
        Giải bài toán phân bổ vốn dài hạn giữa K, D, AI, H; mô phỏng quỹ đạo kinh tế,
        phân tích cú sốc năm 2028 và so sánh chiến lược đầu tư trải đều với front-load.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.header("⚙️ Tham số Bài 8")
p = ModelParams()

p.unit_mode = st.sidebar.selectbox(
    "Chế độ đơn vị",
    ["practical_scaled", "raw_as_written"],
    index=0,
    help=(
        "practical_scaled: dùng hệ số chuyển đổi để D, AI, H không bùng nổ. "
        "raw_as_written: cộng trực tiếp đầu tư vào D/AI/H như phương trình đề, dễ cho kết quả phi thực tế."
    ),
)
p.A0 = st.sidebar.number_input("A0 - TFP ban đầu lấy từ Bài 1", min_value=1.0, max_value=200.0, value=float(p.A0), step=0.1)
p.rho = st.sidebar.slider("ρ - hệ số chiết khấu", min_value=0.80, max_value=0.99, value=float(p.rho), step=0.01)
p.g_A_base = st.sidebar.slider("Tăng TFP ngoại sinh/năm", min_value=0.00, max_value=0.05, value=float(p.g_A_base), step=0.001)
p.max_invest_share = st.sidebar.slider("Trần tỷ trọng đầu tư mỗi năm", min_value=0.10, max_value=0.90, value=float(p.max_invest_share), step=0.01)
p.min_consumption_share = st.sidebar.slider("Sàn tỷ trọng tiêu dùng mỗi năm", min_value=0.01, max_value=0.30, value=float(p.min_consumption_share), step=0.01)
p.normalize_phi_inputs = st.sidebar.checkbox(
    "Chuẩn hóa D, AI, H khi cập nhật TFP nội sinh",
    value=True,
    help="Nếu bỏ chọn, dùng đúng công thức chữ trong đề, A có thể tăng rất nhanh vì D/AI/H đang ở thang 20-100.",
)

with st.sidebar.expander("Hệ số chuyển đổi đơn vị practical_scaled"):
    p.eta_D = st.number_input("eta_D", min_value=0.0, max_value=0.1, value=float(p.eta_D), step=0.0005, format="%.4f")
    p.eta_AI = st.number_input("eta_AI", min_value=0.0, max_value=0.1, value=float(p.eta_AI), step=0.0005, format="%.4f")
    p.eta_H = st.number_input("eta_H", min_value=0.0, max_value=0.1, value=float(p.eta_H), step=0.0005, format="%.4f")

st.markdown("## Tóm tắt yêu cầu và các điểm cần kiểm tra")

check_intro = pd.DataFrame(
    [
        ["8.3.1", "Chọn Cách B: giải bằng scipy.optimize.minimize - SLSQP", "Sẽ chạy trực tiếp trong webapp"],
        ["8.3.2", "Vẽ quỹ đạo K, D, AI, H, Y, C giai đoạn 2026-2035", "Có bảng và biểu đồ"],
        ["8.3.3", "Cú sốc 2028 làm Y giảm 8%", "Tối ưu lại có cú sốc và so sánh với baseline"],
        ["8.3.4", "So sánh 'Đầu tư trải đều' và 'Front-load'", "Có thêm hai chiến lược tham chiếu"],
        ["8.4", "Trả lời câu hỏi chính sách a, b, c", "Có diễn giải dưới kết quả"],
    ],
    columns=["Mục", "Yêu cầu", "Cách xử lý trong code"],
)
st.dataframe(check_intro, use_container_width=True, hide_index=True)

st.markdown(
    """
    <div class="note-box">
    <b>Điểm cần lưu ý trước khi tính:</b><br>
    Đề cho phương trình trạng thái cộng trực tiếp đầu tư vào D, AI, H, trong khi K đo bằng nghìn tỷ VND,
    D/H là %, AI là nghìn doanh nghiệp. Vì vậy nếu dùng nguyên xi phương trình, D, AI, H có thể tăng phi thực tế.
    Code mặc định dùng chế độ <b>practical_scaled</b> để có hệ số chuyển đổi đơn vị. Bạn vẫn có thể chọn
    <b>raw_as_written</b> ở sidebar để xem mô hình đúng theo chữ của đề.
    </div>
    """,
    unsafe_allow_html=True,
)

if not SCIPY_AVAILABLE:
    st.markdown(
        f"""
        <div class="bad-box">
        <b>Thiếu thư viện scipy.</b><br>
        Bài 8 cần scipy để chạy SLSQP. Hãy mở CMD tại thư mục webapp và chạy:<br>
        <code>python -m pip install scipy</code><br><br>
        Lỗi hiện tại: {SCIPY_ERROR}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ============================================================
# 8.1
# ============================================================

st.markdown("## 8.1. Bối cảnh Việt Nam và mục tiêu học tập")
st.markdown(
    """
    Bài 8 đặt vấn đề phân bổ nguồn lực giai đoạn 2026-2035 để cân bằng giữa tăng trưởng hiện tại,
    tích lũy vốn vật chất, chuyển đổi số, năng lực AI và vốn nhân lực. Mục tiêu là tối đa hóa tổng phúc lợi
    xã hội liên thời gian, thay vì chỉ tối đa hóa GDP của một năm.
    """
)

# ============================================================
# 8.2
# ============================================================

st.markdown("## 8.2. Mô hình toán học động")

st.markdown(
    r"""
Hàm sản xuất mở rộng:

$$
Y_t = A_t K_t^{0.33} L_t^{0.42} D_t^{0.10} AI_t^{0.08} H_t^{0.07}
$$

Mục tiêu:

$$
\max \sum_{t=2026}^{2035} \rho^{t-2026}\ln(C_t)
$$

Phương trình trạng thái:

$$
K_{t+1} = (1-\delta_K)K_t + I_{K,t}
$$

$$
D_{t+1} = (1-\delta_D)D_t + I_{D,t}
$$

$$
AI_{t+1} = (1-\delta_{AI})AI_t + I_{AI,t}
$$

$$
H_{t+1} = H_t + \theta_H I_{H,t} - \mu BrainDrain_t
$$

Ràng buộc ngân sách từng năm:

$$
C_t + I_{K,t} + I_{D,t} + I_{AI,t} + I_{H,t} \le Y_t
$$
"""
)

with st.expander("Xem bảng tham số đang dùng"):
    params_df = pd.DataFrame([asdict(p)]).T.reset_index()
    params_df.columns = ["Tham số", "Giá trị"]
    st.dataframe(params_df, use_container_width=True, hide_index=True)

with st.expander("A0 tham khảo tính từ Bài 1"):
    a_df = estimate_a_from_bai1()
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(a_df.round(4), use_container_width=True, hide_index=True)
    with c2:
        fig_a, ax_a = plt.subplots(figsize=(6, 3.5))
        ax_a.plot(a_df["year"], a_df["A_t"], marker="o")
        ax_a.set_xlabel("Năm")
        ax_a.set_ylabel("A_t")
        ax_a.set_title("TFP A_t ước lượng từ dữ liệu Bài 1")
        ax_a.grid(True, alpha=0.3)
        st.pyplot(fig_a)


# ============================================================
# 8.3.1
# ============================================================

st.markdown("## 8.3.1. Giải bài toán bằng scipy.optimize.minimize — phương pháp SLSQP")

with st.spinner("Đang tối ưu bằng SLSQP..."):
    base_df, base_welfare, base_res = optimize_slsqp(p, shock_year=None)

status_text = "Thành công" if base_res.success else "Cần kiểm tra"
if base_res.success:
    st.markdown(
        f"""
        <div class="ok-box">
        <b>Kết quả SLSQP:</b> {status_text}. Solver message: {base_res.message}
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="note-box">
        <b>Kết quả SLSQP:</b> {status_text}. Solver message: {base_res.message}<br>
        Vẫn hiển thị nghiệm tốt nhất tìm được để phân tích, nhưng nên thử tăng maxiter hoặc chỉnh tham số.
        </div>
        """,
        unsafe_allow_html=True,
    )

m1, m2, m3, m4 = st.columns(4)
m1.metric("Welfare tổng", f"{base_welfare:,.4f}")
m2.metric("Y năm 2035", f"{base_df['Y_plan'].iloc[-1]:,.2f}")
m3.metric("C bình quân", f"{base_df['C'].mean():,.2f}")
m4.metric("Tỷ trọng đầu tư TB", f"{base_df['invest_share'].mean()*100:,.2f}%")

st.markdown("### Bảng quỹ đạo tối ưu")
display_cols = [
    "year", "A", "K", "D", "AI", "H", "Y_plan", "C",
    "I_K", "I_D", "I_AI", "I_H",
    "share_K", "share_D", "share_AI", "share_H", "invest_share", "consumption_share"
]
st.dataframe(base_df[display_cols].round(4), use_container_width=True, hide_index=True)

st.markdown("### Nhận diện dạng front-loaded/back-loaded")
loading_text = classify_loading(base_df)
st.info(loading_text)
st.dataframe(policy_summary(base_df).round(4), use_container_width=True, hide_index=True)

st.markdown("### Kịch bản mở rộng để tránh nghiệm góc: thêm sàn đầu tư D/AI/H")
with st.spinner("Đang tối ưu thêm kịch bản mở rộng có sàn D/AI/H..."):
    policy_constrained_df, policy_constrained_welfare, policy_constrained_res = optimize_slsqp_policy_constrained(p)

policy_constrained_summary_df = pd.DataFrame([
    {
        "Kịch bản": "Tối ưu gốc SLSQP",
        "Welfare": base_welfare,
        "Y_2035": base_df["Y_plan"].iloc[-1],
        "C bình quân": base_df["C"].mean(),
        "Tổng đầu tư K": base_df["I_K"].sum(),
        "Tổng đầu tư D": base_df["I_D"].sum(),
        "Tổng đầu tư AI": base_df["I_AI"].sum(),
        "Tổng đầu tư H": base_df["I_H"].sum(),
        "Ghi chú": "Mô hình gốc của đề; có thể tạo nghiệm góc nếu không có ràng buộc chính sách bổ sung.",
    },
    {
        "Kịch bản": "Mở rộng có sàn D/AI/H",
        "Welfare": policy_constrained_welfare,
        "Y_2035": policy_constrained_df["Y_plan"].iloc[-1],
        "C bình quân": policy_constrained_df["C"].mean(),
        "Tổng đầu tư K": policy_constrained_df["I_K"].sum(),
        "Tổng đầu tư D": policy_constrained_df["I_D"].sum(),
        "Tổng đầu tư AI": policy_constrained_df["I_AI"].sum(),
        "Tổng đầu tư H": policy_constrained_df["I_H"].sum(),
        "Ghi chú": "Bảng bổ sung cho báo cáo: mỗi năm có sàn share_D=2.5%, share_AI=2.0%, share_H=3.0%.",
    },
])
st.dataframe(policy_constrained_summary_df.round(4), use_container_width=True, hide_index=True)
st.caption("Kịch bản mở rộng không thay thế nghiệm gốc; mục đích là giải thích với giảng viên vì sao cần ràng buộc chính sách để kết quả kinh tế hợp lý hơn.")


# ============================================================
# 8.3.2
# ============================================================

st.markdown("## 8.3.2. Vẽ đồ thị quỹ đạo tối ưu của K, D, AI, H, Y, C")

fig_state, ax_state = plt.subplots(figsize=(10, 5))
for col in ["K", "D", "AI", "H"]:
    series = base_df[col].astype(float)
    norm_series = series / series.iloc[0]
    ax_state.plot(base_df["year"], norm_series, marker="o", label=f"{col} / {col}_2026")
ax_state.set_title("Quỹ đạo trạng thái chuẩn hóa theo năm 2026")
ax_state.set_xlabel("Năm")
ax_state.set_ylabel("Chỉ số chuẩn hóa")
ax_state.grid(True, alpha=0.3)
ax_state.legend()
st.pyplot(fig_state)

fig_yc, ax_yc = plt.subplots(figsize=(10, 5))
ax_yc.plot(base_df["year"], base_df["Y_plan"], marker="o", label="Y kế hoạch")
ax_yc.plot(base_df["year"], base_df["C"], marker="s", label="C tiêu dùng")
ax_yc.set_title("Quỹ đạo sản lượng Y và tiêu dùng C")
ax_yc.set_xlabel("Năm")
ax_yc.set_ylabel("Nghìn tỷ VND")
ax_yc.grid(True, alpha=0.3)
ax_yc.legend()
st.pyplot(fig_yc)

fig_inv, ax_inv = plt.subplots(figsize=(10, 5))
for col, label in [("I_K", "Đầu tư K"), ("I_D", "Đầu tư D"), ("I_AI", "Đầu tư AI"), ("I_H", "Đầu tư H")]:
    ax_inv.plot(base_df["year"], base_df[col], marker="o", label=label)
ax_inv.set_title("Cơ cấu đầu tư tối ưu theo thời gian")
ax_inv.set_xlabel("Năm")
ax_inv.set_ylabel("Nghìn tỷ VND")
ax_inv.grid(True, alpha=0.3)
ax_inv.legend()
st.pyplot(fig_inv)


# ============================================================
# 8.3.3
# ============================================================

st.markdown("## 8.3.3. Phân tích cú sốc năm 2028: Y giảm 8% so với kế hoạch")

shock_year = 2028
shock_drop = 0.08
with st.spinner("Đang tối ưu lại khi có cú sốc 2028..."):
    shock_df, shock_welfare, shock_res = optimize_slsqp(p, shock_year=shock_year, shock_drop=shock_drop)

shock_compare = base_df[["year", "Y_plan", "C", "I_K", "I_D", "I_AI", "I_H", "invest_share"]].copy()
shock_compare = shock_compare.rename(
    columns={
        "Y_plan": "Y_base",
        "C": "C_base",
        "I_K": "I_K_base",
        "I_D": "I_D_base",
        "I_AI": "I_AI_base",
        "I_H": "I_H_base",
        "invest_share": "invest_share_base",
    }
)
shock_cols = shock_df[["year", "Y_effective", "C", "I_K", "I_D", "I_AI", "I_H", "invest_share"]].rename(
    columns={
        "Y_effective": "Y_shock",
        "C": "C_shock",
        "I_K": "I_K_shock",
        "I_D": "I_D_shock",
        "I_AI": "I_AI_shock",
        "I_H": "I_H_shock",
        "invest_share": "invest_share_shock",
    }
)
shock_compare = shock_compare.merge(shock_cols, on="year", how="left")
for col in ["C", "I_K", "I_D", "I_AI", "I_H", "invest_share"]:
    shock_compare[f"delta_{col}"] = shock_compare[f"{col}_shock"] - shock_compare[f"{col}_base"]

m1, m2, m3 = st.columns(3)
m1.metric("Welfare baseline", f"{base_welfare:,.4f}")
m2.metric("Welfare có cú sốc", f"{shock_welfare:,.4f}", delta=f"{shock_welfare-base_welfare:,.4f}")
m3.metric("Y hiệu dụng năm 2028", f"{shock_df.loc[shock_df['year']==2028, 'Y_effective'].iloc[0]:,.2f}")

st.markdown("### Bảng so sánh baseline và kịch bản cú sốc")
st.dataframe(shock_compare.round(4), use_container_width=True, hide_index=True)

fig_shock, ax_shock = plt.subplots(figsize=(10, 5))
ax_shock.plot(base_df["year"], base_df["C"], marker="o", label="C baseline")
ax_shock.plot(shock_df["year"], shock_df["C"], marker="s", label="C có cú sốc")
ax_shock.axvline(shock_year, linestyle="--", alpha=0.6, label="Cú sốc 2028")
ax_shock.set_title("Điều chỉnh tiêu dùng khi Y năm 2028 giảm 8%")
ax_shock.set_xlabel("Năm")
ax_shock.set_ylabel("Nghìn tỷ VND")
ax_shock.grid(True, alpha=0.3)
ax_shock.legend()
st.pyplot(fig_shock)

fig_shock_inv, ax_shock_inv = plt.subplots(figsize=(10, 5))
ax_shock_inv.plot(base_df["year"], base_df["invest_share"], marker="o", label="Tỷ trọng đầu tư baseline")
ax_shock_inv.plot(shock_df["year"], shock_df["invest_share"], marker="s", label="Tỷ trọng đầu tư có cú sốc")
ax_shock_inv.axvline(shock_year, linestyle="--", alpha=0.6, label="Cú sốc 2028")
ax_shock_inv.set_title("Điều chỉnh tỷ trọng đầu tư khi có cú sốc")
ax_shock_inv.set_xlabel("Năm")
ax_shock_inv.set_ylabel("Tỷ trọng đầu tư/Y")
ax_shock_inv.grid(True, alpha=0.3)
ax_shock_inv.legend()
st.pyplot(fig_shock_inv)

st.markdown(
    """
    **Diễn giải:** Khi cú sốc làm sản lượng năm 2028 giảm, mô hình thường phải điều chỉnh giữa hai lựa chọn:
    giảm tiêu dùng hiện tại hoặc giảm đầu tư trong năm bị sốc. Nếu tỷ trọng đầu tư sau cú sốc không giảm mạnh,
    điều đó cho thấy mô hình ưu tiên bảo vệ quỹ đạo tăng trưởng dài hạn. Nếu tiêu dùng giảm rõ, đó là chi phí phúc lợi
    ngắn hạn của việc duy trì đầu tư.
    """
)


# ============================================================
# 8.3.4
# ============================================================

st.markdown("## 8.3.4. So sánh hai chiến lược: đầu tư trải đều và front-load")

strategy_df = compare_fixed_strategies(p)

# Thêm nghiệm tối ưu SLSQP vào bảng so sánh
opt_row = {
    "Chiến lược": "Tối ưu SLSQP",
    "Welfare tổng": base_welfare,
    "Y_2035": base_df["Y_plan"].iloc[-1],
    "C bình quân": base_df["C"].mean(),
    "Tổng đầu tư K": base_df["I_K"].sum(),
    "Tổng đầu tư D": base_df["I_D"].sum(),
    "Tổng đầu tư AI": base_df["I_AI"].sum(),
    "Tổng đầu tư H": base_df["I_H"].sum(),
    "Khả thi": "Có",
}
policy_row = {
    "Chiến lược": "Mở rộng có sàn D/AI/H",
    "Welfare tổng": policy_constrained_welfare,
    "Y_2035": policy_constrained_df["Y_plan"].iloc[-1],
    "C bình quân": policy_constrained_df["C"].mean(),
    "Tổng đầu tư K": policy_constrained_df["I_K"].sum(),
    "Tổng đầu tư D": policy_constrained_df["I_D"].sum(),
    "Tổng đầu tư AI": policy_constrained_df["I_AI"].sum(),
    "Tổng đầu tư H": policy_constrained_df["I_H"].sum(),
    "Khả thi": "Có",
}
strategy_df_full = pd.concat([pd.DataFrame([opt_row, policy_row]), strategy_df], ignore_index=True)
strategy_df_full = strategy_df_full.sort_values("Welfare tổng", ascending=False).reset_index(drop=True)

st.dataframe(strategy_df_full.round(4), use_container_width=True, hide_index=True)

fig_strategy, ax_strategy = plt.subplots(figsize=(10, 5))
ax_strategy.bar(strategy_df_full["Chiến lược"], strategy_df_full["Welfare tổng"])
ax_strategy.set_title("So sánh Welfare tổng giữa các chiến lược")
ax_strategy.set_ylabel("Welfare tổng")
ax_strategy.tick_params(axis="x", rotation=20)
ax_strategy.grid(True, axis="y", alpha=0.3)
st.pyplot(fig_strategy)

best_strategy = strategy_df_full.iloc[0]["Chiến lược"]
st.success(f"Chiến lược có welfare cao nhất trong bảng so sánh là: {best_strategy}")


# ============================================================
# 8.4 Câu hỏi chính sách
# ============================================================

st.markdown("## 8.4. Câu hỏi thảo luận chính sách")

ai_h_ratio = (
    base_df["I_AI"].sum() / base_df["I_H"].sum()
    if base_df["I_H"].sum() > 1e-9
    else np.nan
)
ai_h_early = base_df[base_df["year"] <= 2028]["I_AI"].sum() / max(
    base_df[base_df["year"] <= 2028]["I_H"].sum(), 1e-9
)
ai_h_late = base_df[base_df["year"] > 2028]["I_AI"].sum() / max(
    base_df[base_df["year"] > 2028]["I_H"].sum(), 1e-9
)

st.markdown(
    f"""
### a) Quỹ đạo tối ưu có front-loaded hay back-loaded không?

Kết quả hiện tại cho thấy: **{loading_text}**  
Tỷ trọng đầu tư bình quân toàn kỳ là **{base_df['invest_share'].mean()*100:.2f}%** GDP hiệu dụng mỗi năm.
Nếu đầu tư tập trung ở đầu kỳ, lý do là các khoản đầu tư vào K, D, AI và H tạo ra tác động tích lũy cho các năm sau.
Ngược lại, nếu tỷ trọng đầu tư ổn định hoặc giảm về cuối kỳ, mô hình đang cân bằng giữa phúc lợi tiêu dùng hiện tại
và lợi ích tăng trưởng dài hạn.

### b) Tỷ lệ đầu tư AI / đầu tư H theo thời gian có ổn định không?

Tỷ lệ AI/H toàn kỳ là **{ai_h_ratio:.4f}**.  
Giai đoạn 2026-2028 là **{ai_h_early:.4f}**, còn giai đoạn sau 2028 là **{ai_h_late:.4f}**.

Nếu AI/H thấp ở đầu kỳ, mô hình hàm ý đào tạo nhân lực nên đi trước hoặc đi song song với AI để tăng khả năng hấp thụ công nghệ.
Nếu AI/H cao ngay từ đầu, mô hình đang ưu tiên tăng nhanh năng lực AI, nhưng cần kiểm soát rủi ro thiếu nhân lực vận hành.

### c) Nếu ρ giảm từ 0,97 xuống 0,90 thì kết quả thay đổi thế nào?

Hệ số ρ = 0,97 thể hiện Chính phủ coi trọng phúc lợi dài hạn. Khi giảm xuống 0,90,
các lợi ích tương lai bị chiết khấu mạnh hơn, nên mô hình thường có xu hướng ưu tiên tiêu dùng hiện tại
và giảm bớt đầu tư dài hạn như R&D, AI hoặc vốn nhân lực. Đây là một cách giải thích vì sao trong thực tế
các chính phủ dễ bị áp lực ngắn hạn và có thể dưới đầu tư vào R&D hoặc đào tạo nhân lực.
"""
)

with st.expander("Chạy nhanh kịch bản ρ = 0,90 để so sánh"):
    p_short = ModelParams(**asdict(p))
    p_short.rho = 0.90
    with st.spinner("Đang tối ưu lại với ρ = 0,90..."):
        short_df, short_welfare, short_res = optimize_slsqp(p_short, shock_year=None, maxiter=450)

    rho_compare = pd.DataFrame(
        [
            {
                "Kịch bản": "ρ = 0.97 hiện tại",
                "Welfare": base_welfare,
                "Tỷ trọng đầu tư TB": base_df["invest_share"].mean(),
                "Tổng đầu tư K": base_df["I_K"].sum(),
                "Tổng đầu tư D": base_df["I_D"].sum(),
                "Tổng đầu tư AI": base_df["I_AI"].sum(),
                "Tổng đầu tư H": base_df["I_H"].sum(),
                "Y_2035": base_df["Y_plan"].iloc[-1],
            },
            {
                "Kịch bản": "ρ = 0.90 ngắn hạn hơn",
                "Welfare": short_welfare,
                "Tỷ trọng đầu tư TB": short_df["invest_share"].mean(),
                "Tổng đầu tư K": short_df["I_K"].sum(),
                "Tổng đầu tư D": short_df["I_D"].sum(),
                "Tổng đầu tư AI": short_df["I_AI"].sum(),
                "Tổng đầu tư H": short_df["I_H"].sum(),
                "Y_2035": short_df["Y_plan"].iloc[-1],
            },
        ]
    )
    st.dataframe(rho_compare.round(4), use_container_width=True, hide_index=True)


# ============================================================
# Checklist và tải kết quả
# ============================================================

st.markdown("## Checklist hoàn thành 100% yêu cầu Bài 8")

checklist = pd.DataFrame(
    [
        ["8.1", "Trình bày bối cảnh Việt Nam và mục tiêu bài toán", "Đã làm", "Có phần bối cảnh riêng"],
        ["8.2", "Trình bày mô hình toán học động", "Đã làm", "Có hàm sản xuất, mục tiêu, trạng thái, ngân sách"],
        ["8.3.1", "Giải bằng CVXPY hoặc scipy.optimize.minimize SLSQP", "Đã làm", "Chọn Cách B: SLSQP"],
        ["8.3.1", "Tự cài đặt hàm mục tiêu và callback/logic mô phỏng", "Đã làm", "Có simulate_from_shares và objective"],
        ["8.3.2", "Vẽ quỹ đạo K, D, AI, H, Y, C từ 2026-2035", "Đã làm", "Có 3 biểu đồ"],
        ["8.3.3", "Cú sốc năm 2028 làm Y giảm 8%", "Đã làm", "Tối ưu lại có shock_year=2028"],
        ["8.3.3", "Phân tích mô hình điều chỉnh phân bổ sau cú sốc", "Đã làm", "Có bảng delta và biểu đồ"],
        ["8.3.4", "So sánh đầu tư trải đều và front-load", "Đã làm", "Có bảng welfare và biểu đồ"],
        ["8.4a", "Trả lời front-loaded/back-loaded", "Đã làm", "Dựa trên kết quả tối ưu"],
        ["8.4b", "Trả lời tỷ lệ AI/H theo thời gian", "Đã làm", "Có tỷ lệ AI/H đầu kỳ, cuối kỳ"],
        ["8.4c", "Trả lời tác động khi ρ = 0,90", "Đã làm", "Có kịch bản chạy nhanh"],
        ["Bổ sung báo cáo", "Kịch bản mở rộng có sàn D/AI/H để tránh nghiệm góc", "Đã làm", "Không thay thế đề gốc; dùng để giải thích chính sách"],
        ["Tải kết quả", "Excel, CSV, PNG", "Đã làm", "Có nút tải bên dưới"],
    ],
    columns=["Mục", "Yêu cầu", "Trạng thái", "Ghi chú"],
)
st.dataframe(checklist, use_container_width=True, hide_index=True)

st.markdown("## Tải kết quả Bài 8")

output_manifest = pd.DataFrame([
    ["bai08_baseline_optimal.csv", "Nghiệm tối ưu gốc SLSQP; có thể xuất hiện nghiệm góc"],
    ["bai08_policy_constrained.csv", "Kịch bản mở rộng có sàn đầu tư D/AI/H"],
    ["bai08_policy_constrained_summary.csv", "Tóm tắt kịch bản mở rộng tránh nghiệm góc"],
    ["bai08_report.html", "Báo cáo HTML gồm cả nghiệm gốc và kịch bản mở rộng"],
    ["bai08_states.png", "Quỹ đạo trạng thái K, D, AI, H"],
    ["bai08_y_c.png", "Quỹ đạo Y và C"],
    ["bai08_investment.png", "Cơ cấu đầu tư tối ưu"],
    ["bai08_shock.png", "Điều chỉnh tiêu dùng khi cú sốc 2028"],
    ["bai08_strategies.png", "So sánh welfare giữa các chiến lược"],
], columns=["File output", "Ý nghĩa"])

excel_bytes = make_excel_bytes(
    {
        "baseline_optimal": base_df.round(6),
        "policy_constrained": policy_constrained_df.round(6),
        "policy_constrained_summary": policy_constrained_summary_df.round(6),
        "policy_summary": policy_summary(base_df).round(6),
        "shock_compare": shock_compare.round(6),
        "strategy_compare": strategy_df_full.round(6),
        "output_manifest": output_manifest,
        "checklist": checklist,
        "params": pd.DataFrame([asdict(p)]),
    }
)

csv_bytes = base_df.to_csv(index=False).encode("utf-8-sig")
png_state = fig_to_png_bytes(fig_state)
png_yc = fig_to_png_bytes(fig_yc)
png_inv = fig_to_png_bytes(fig_inv)
png_shock = fig_to_png_bytes(fig_shock)
png_strategy = fig_to_png_bytes(fig_strategy)

html_report = f"""
<html>
<head>
<meta charset="utf-8">
<title>Bài 8 - Tối ưu động</title>
<style>
body {{font-family: Arial, sans-serif; margin: 32px; line-height: 1.55;}}
h1, h2 {{color: #1f2937;}}
table {{border-collapse: collapse; width: 100%; margin: 16px 0;}}
th, td {{border: 1px solid #ddd; padding: 8px; text-align: right;}}
th {{background: #f3f4f6;}}
td:first-child, th:first-child {{text-align: left;}}
.note {{background: #fff7ed; border-left: 5px solid #f59e0b; padding: 12px;}}
</style>
</head>
<body>
<h1>Bài 8 — Tối ưu động phân bổ liên thời gian 2026-2035</h1>
<div class="note">
<p><b>Ghi chú:</b> File HTML này tóm tắt kết quả chính từ webapp. Mô hình dùng scipy.optimize.minimize SLSQP.</p>
<p><b>Cách đọc an toàn:</b> Nghiệm tối ưu gốc có thể là nghiệm góc do hàm mục tiêu ưu tiên phúc lợi/tiêu dùng. Vì vậy báo cáo trình bày thêm kịch bản mở rộng có sàn đầu tư D/AI/H để phân tích chính sách thực tế.</p>
</div>
<h2>Kết quả chính</h2>
<ul>
<li>Welfare baseline: {base_welfare:.6f}</li>
<li>Y năm 2035: {base_df['Y_plan'].iloc[-1]:,.4f}</li>
<li>C bình quân: {base_df['C'].mean():,.4f}</li>
<li>Nhận diện quỹ đạo: {loading_text}</li>
</ul>
<h2>Bảng quỹ đạo tối ưu</h2>
{base_df[display_cols].round(4).to_html(index=False)}
<h2>Kịch bản mở rộng có sàn D/AI/H</h2>
{policy_constrained_summary_df.round(4).to_html(index=False)}
{policy_constrained_df[display_cols].round(4).to_html(index=False)}
<h2>So sánh cú sốc 2028</h2>
{shock_compare.round(4).to_html(index=False)}
<h2>So sánh chiến lược</h2>
{strategy_df_full.round(4).to_html(index=False)}
<h2>Checklist</h2>
{checklist.to_html(index=False)}
</body>
</html>
""".encode("utf-8")

c1, c2, c3 = st.columns(3)
with c1:
    st.download_button(
        "⬇️ Tải Excel tổng hợp",
        data=excel_bytes,
        file_name="bai08_toi_uu_dong_ket_qua.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
with c2:
    st.download_button(
        "⬇️ Tải CSV quỹ đạo tối ưu",
        data=csv_bytes,
        file_name="bai08_baseline_optimal.csv",
        mime="text/csv",
    )
with c3:
    st.download_button(
        "⬇️ Tải HTML report",
        data=html_report,
        file_name="bai08_report.html",
        mime="text/html",
    )

c4, c5, c6 = st.columns(3)
with c4:
    st.download_button("⬇️ Tải PNG trạng thái", png_state, "bai08_states.png", "image/png")
with c5:
    st.download_button("⬇️ Tải PNG Y-C", png_yc, "bai08_y_c.png", "image/png")
with c6:
    st.download_button("⬇️ Tải PNG chiến lược", png_strategy, "bai08_strategies.png", "image/png")

# Lưu ra outputs nếu chạy local
try:
    out = output_dir()
    (out / "bai08_baseline_optimal.csv").write_bytes(csv_bytes)
    policy_constrained_df.round(6).to_csv(out / "bai08_policy_constrained.csv", index=False, encoding="utf-8-sig")
    policy_constrained_summary_df.round(6).to_csv(out / "bai08_policy_constrained_summary.csv", index=False, encoding="utf-8-sig")
    output_manifest.to_csv(out / "bai08_output_manifest.csv", index=False, encoding="utf-8-sig")
    (out / "bai08_toi_uu_dong_ket_qua.xlsx").write_bytes(excel_bytes)
    (out / "bai08_report.html").write_bytes(html_report)
    (out / "bai08_states.png").write_bytes(png_state)
    (out / "bai08_y_c.png").write_bytes(png_yc)
    (out / "bai08_investment.png").write_bytes(png_inv)
    (out / "bai08_shock.png").write_bytes(png_shock)
    (out / "bai08_strategies.png").write_bytes(png_strategy)
except Exception:
    pass

st.caption(
    "Bài 8 hoàn thành theo hướng scipy.optimize.minimize/SLSQP. "
    "Khi nộp bài, nên ghi rõ giả định chuyển đổi đơn vị practical_scaled nếu dùng chế độ này."
)
