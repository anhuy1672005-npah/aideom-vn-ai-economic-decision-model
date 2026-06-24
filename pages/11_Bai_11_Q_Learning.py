# -*- coding: utf-8 -*-
"""
Bài 11 — Học tăng cường Q-learning cho chính sách kinh tế thích nghi
Webapp Streamlit cho bộ bài tập AIDEOM-VN.

File gợi ý đặt trong thư mục pages/:
    pages/11_Bai_11_Q_Learning.py

Nội dung bao phủ yêu cầu đề:
- 11.3.1: Cài đặt môi trường gym/gymnasium kế thừa Env với reset, step,
          action_space, observation_space; mỗi episode mô phỏng 10 năm.
- 11.3.2: Cài đặt tabular Q-learning, alpha=0.1, gamma=0.95,
          epsilon-greedy giảm từ 1.0 xuống 0.05 qua 10.000 episodes.
- 11.3.3: Trích xuất chính sách pi*(s)=argmax_a Q(s,a) tại VN 2026
          và 4 trạng thái giả định.
- 11.3.4: So sánh reward tích lũy của pi* với 3 rule-based policies:
          luôn a1, luôn a3, random; vẽ learning curve.
- 11.3.5: Mở rộng DQN bằng stable-baselines3, 2 hidden layers 64 units
          nếu môi trường đã cài stable-baselines3 và torch.
- 11.4: Trả lời câu hỏi chính sách a, b, c.

Lưu ý logic:
Đề bài chỉ cho cấu trúc MDP, tập trạng thái, tập hành động và công thức reward tổng quát.
Đề không cho đầy đủ phương trình chuyển trạng thái. Vì vậy file này xây dựng hàm step
minh họa dựa trên Cobb-Douglas mở rộng, đúng tinh thần gợi ý code của đề.
"""

from __future__ import annotations

import io
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")

# ============================================================
# 0. Optional dependencies: gymnasium, stable-baselines3
# ============================================================
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except Exception:
    gym = None
    GYM_AVAILABLE = False

    class _FallbackEnv:
        metadata = {}

        def reset(self, seed=None, options=None):
            return None

    class _Discrete:
        def __init__(self, n):
            self.n = int(n)

        def sample(self):
            return int(np.random.randint(0, self.n))

    class _MultiDiscrete:
        def __init__(self, nvec):
            self.nvec = np.asarray(nvec, dtype=int)

        def sample(self):
            return np.array([np.random.randint(0, n) for n in self.nvec], dtype=int)

    class _Spaces:
        Discrete = _Discrete
        MultiDiscrete = _MultiDiscrete

    spaces = _Spaces()

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.env_checker import check_env
    SB3_AVAILABLE = True
except Exception:
    DQN = None
    check_env = None
    SB3_AVAILABLE = False

# ============================================================
# 1. Giao diện chung
# ============================================================
st.set_page_config(
    page_title="Bài 11 - Q-learning chính sách kinh tế",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
            p, li, span, div {color: #e5e7eb;}
            .card {background: rgba(18,26,47,0.95); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 20px 22px; box-shadow: 0 8px 24px rgba(0,0,0,0.25); margin-bottom: 18px;}
            .card-title {font-size: 1.1rem; font-weight: 750; color: #fff; margin-bottom: 8px;}
            .card-text {color: #cbd5e1; font-size: 0.95rem; line-height: 1.55;}
            .badge {display:inline-block; padding:6px 12px; margin-right:8px; margin-bottom:8px; border-radius:999px; font-size:0.78rem; font-weight:700; color:white; background:linear-gradient(90deg,#ff3b7f,#7c3aed);}
            div[data-testid="stMetric"] {background: rgba(18,26,47,0.95); border: 1px solid rgba(255,255,255,0.08); padding: 18px; border-radius: 16px;}
            .stDownloadButton > button, .stButton > button {border-radius:999px; font-weight:700;}
            </style>
            """,
            unsafe_allow_html=True,
        )

    def hero(title: str, subtitle: str = "", badges: Optional[List[str]] = None):
        badges = badges or []
        badge_html = "".join([f'<span class="badge">{b}</span>' for b in badges])
        st.markdown(
            f"""
            <div class="card">
                <div>{badge_html}</div>
                <h1>{title}</h1>
                <div class="card-text">{subtitle}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def card(title: str, text: str):
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">{title}</div>
                <div class="card-text">{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

load_css()

hero(
    title="🤖 Bài 11 — Q-learning cho chính sách kinh tế thích nghi",
    subtitle=(
        "Mô hình hóa nền kinh tế Việt Nam như một MDP đơn giản, huấn luyện tabular Q-learning, "
        "trích xuất chính sách π*, so sánh với rule-based policies và thử mở rộng DQN nếu đã cài stable-baselines3."
    ),
    badges=["Cấp độ khó", "Gymnasium", "Q-learning", "MDP", "DQN mở rộng"],
)

# ============================================================
# 2. Tham số, tập trạng thái, hành động
# ============================================================
STATE_LABELS = {
    0: "Thấp",
    1: "Trung bình",
    2: "Cao",
}

STATE_DIM_NAMES = [
    "GDP growth",
    "Digital index",
    "AI capacity",
    "Unemployment risk",
]

ACTION_NAMES = {
    0: "a0 - Truyền thống",
    1: "a1 - Cân bằng",
    2: "a2 - Số hóa nhanh",
    3: "a3 - AI dẫn dắt",
    4: "a4 - Bao trùm",
}

ACTION_SHORT = {
    0: "a0",
    1: "a1",
    2: "a2",
    3: "a3",
    4: "a4",
}

ACTION_DESCRIPTION = {
    0: "70% K + 10% D + 10% AI + 10% H",
    1: "40% K + 25% D + 15% AI + 20% H",
    2: "25% K + 45% D + 15% AI + 15% H",
    3: "20% K + 20% D + 45% AI + 15% H",
    4: "30% K + 20% D + 10% AI + 40% H",
}

ALLOCATION = {
    0: np.array([0.70, 0.10, 0.10, 0.10], dtype=float),
    1: np.array([0.40, 0.25, 0.15, 0.20], dtype=float),
    2: np.array([0.25, 0.45, 0.15, 0.15], dtype=float),
    3: np.array([0.20, 0.20, 0.45, 0.15], dtype=float),
    4: np.array([0.30, 0.20, 0.10, 0.40], dtype=float),
}

REWARD_WEIGHTS = np.array([0.40, 0.25, 0.20, 0.15], dtype=float)
VN2026_STATE = np.array([1, 1, 0, 1], dtype=int)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class EnvParams:
    """Tham số mô phỏng MDP Bài 11."""

    T: int = 10
    annual_budget: float = 1000.0  # nghìn tỷ VND/năm, theo gợi ý đề

    # Điều kiện ban đầu gần với Bài 8/gợi ý Bài 11
    K0: float = 27500.0
    D0: float = 20.3
    AI0: float = 86.0
    H0: float = 30.0
    L0: float = 54.0
    A0: float = 30.944377
    unemployment_risk0: float = 0.50

    # Cobb-Douglas mở rộng
    alpha_K: float = 0.33
    beta_L: float = 0.42
    gamma_D: float = 0.10
    delta_AI: float = 0.08
    theta_H: float = 0.07

    # Động học đơn giản hóa
    labor_growth: float = 0.005
    k_depreciation: float = 0.045
    h_depreciation: float = 0.010

    # Scale chuyển đầu tư sang D, AI, H theo gợi ý đề
    D_scale: float = 100.0
    AI_scale: float = 20.0
    H_scale: float = 200.0

    # Ngưỡng rời rạc hóa trạng thái
    gdp_low_threshold: float = 1.50
    gdp_high_threshold: float = 3.50
    D_low_threshold: float = 15.0
    D_high_threshold: float = 30.0
    AI_low_threshold: float = 100.0
    AI_high_threshold: float = 130.0
    U_low_threshold: float = 0.35
    U_high_threshold: float = 0.65

    # Hệ số reward và rủi ro
    shock_std: float = 0.00  # mặc định deterministic để tái lập kết quả


def action_table() -> pd.DataFrame:
    rows = []
    for a, alloc in ALLOCATION.items():
        rows.append(
            {
                "Mã hành động": ACTION_SHORT[a],
                "Tên hành động": ACTION_NAMES[a].replace(f"a{a} - ", ""),
                "K - Vốn vật chất": alloc[0],
                "D - Số hóa": alloc[1],
                "AI - Trí tuệ nhân tạo": alloc[2],
                "H - Nhân lực": alloc[3],
                "Đặc điểm phân bổ": ACTION_DESCRIPTION[a],
            }
        )
    return pd.DataFrame(rows)


def encode_state(state: np.ndarray | List[int] | Tuple[int, int, int, int]) -> int:
    s = np.asarray(state, dtype=int)
    return int(s[0] * 27 + s[1] * 9 + s[2] * 3 + s[3])


def decode_state(code: int) -> np.ndarray:
    code = int(code)
    g = code // 27
    rem = code % 27
    d = rem // 9
    rem = rem % 9
    ai = rem // 3
    u = rem % 3
    return np.array([g, d, ai, u], dtype=int)


def state_to_text(state: np.ndarray | List[int]) -> str:
    s = np.asarray(state, dtype=int)
    parts = [f"{name}: {STATE_LABELS[int(value)]}" for name, value in zip(STATE_DIM_NAMES, s)]
    return " | ".join(parts)


def state_to_initial_values(state: np.ndarray | List[int], p: EnvParams) -> Dict[str, float]:
    """Ánh xạ trạng thái rời rạc sang giá trị liên tục ban đầu để mô phỏng."""
    s = np.asarray(state, dtype=int)
    g, d, ai, u = s

    # GDP growth state chủ yếu phản ánh nền K ban đầu và mức năng động nền kinh tế.
    K_values = [25500.0, p.K0, 30000.0]
    D_values = [12.0, p.D0, 40.0]
    AI_values = [86.0, 115.0, 150.0]
    U_values = [0.25, p.unemployment_risk0, 0.75]

    return {
        "K": float(K_values[g]),
        "D": float(D_values[d]),
        "AI": float(AI_values[ai]),
        "H": p.H0,
        "L": p.L0,
        "A": p.A0,
        "unemployment_risk": float(U_values[u]),
    }


# ============================================================
# 3. Môi trường MDP theo Gymnasium
# ============================================================
BaseEnv = gym.Env if GYM_AVAILABLE else object


class VietnamEconomyEnv(BaseEnv):
    """
    Môi trường kinh tế Việt Nam dạng MDP rời rạc cho Bài 11.

    observation/state: [GDP_growth_level, Digital_level, AI_level, Unemployment_risk_level]
    action: 0..4 tương ứng 5 chính sách phân bổ ngân sách.
    episode: 10 bước = 10 năm.
    """

    metadata = {"render_modes": []}

    def __init__(self, params: Optional[EnvParams] = None, random_start: bool = False, seed: int = 42):
        if GYM_AVAILABLE:
            super().__init__()
        self.params = params or EnvParams()
        self.random_start = bool(random_start)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3])
        self.rng = np.random.default_rng(seed)
        self.seed_value = int(seed)
        self.reset()

    def production(self, K: float, L: float, D: float, AI: float, H: float, A: float) -> float:
        p = self.params
        K = max(float(K), 1e-9)
        L = max(float(L), 1e-9)
        D = max(float(D), 1e-9)
        AI = max(float(AI), 1e-9)
        H = max(float(H), 1e-9)
        return (
            A
            * (K ** p.alpha_K)
            * (L ** p.beta_L)
            * (D ** p.gamma_D)
            * (AI ** p.delta_AI)
            * (H ** p.theta_H)
        )

    def discretize_state(self, gdp_growth_pct: float, D: float, AI: float, unemployment_risk: float) -> np.ndarray:
        p = self.params
        if gdp_growth_pct < p.gdp_low_threshold:
            g = 0
        elif gdp_growth_pct < p.gdp_high_threshold:
            g = 1
        else:
            g = 2

        if D < p.D_low_threshold:
            d = 0
        elif D < p.D_high_threshold:
            d = 1
        else:
            d = 2

        if AI < p.AI_low_threshold:
            ai = 0
        elif AI < p.AI_high_threshold:
            ai = 1
        else:
            ai = 2

        if unemployment_risk < p.U_low_threshold:
            u = 0
        elif unemployment_risk < p.U_high_threshold:
            u = 1
        else:
            u = 2

        return np.array([g, d, ai, u], dtype=int)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if GYM_AVAILABLE:
            super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        options = options or {}
        random_start = bool(options.get("random_start", self.random_start))
        initial_state = options.get("initial_state", None)

        if initial_state is not None:
            self.state = np.asarray(initial_state, dtype=int)
        elif random_start:
            self.state = self.rng.integers(0, 3, size=4, dtype=int)
        else:
            self.state = VN2026_STATE.copy()

        init = state_to_initial_values(self.state, self.params)
        self.t = 0
        self.K = init["K"]
        self.D = init["D"]
        self.AI = init["AI"]
        self.H = init["H"]
        self.L = init["L"]
        self.A = init["A"]
        self.unemployment_risk = init["unemployment_risk"]
        self.prev_Y = self.production(self.K, self.L, self.D, self.AI, self.H, self.A)

        info = {
            "year_index": self.t,
            "Y": self.prev_Y,
            "K": self.K,
            "D": self.D,
            "AI": self.AI,
            "H": self.H,
            "L": self.L,
            "unemployment_risk": self.unemployment_risk,
            "state_text": state_to_text(self.state),
        }
        return self.state.copy(), info

    def step(self, action: int):
        p = self.params
        action = int(action)
        alloc = ALLOCATION[action]
        aK, aD, aAI, aH = alloc

        old_Y = float(self.prev_Y)
        old_u = float(self.unemployment_risk)

        # Cập nhật trạng thái liên tục theo gợi ý đề và Cobb-Douglas mở rộng.
        self.K = (1.0 - p.k_depreciation) * self.K + aK * p.annual_budget
        self.D = self.D + aD * p.annual_budget / p.D_scale
        self.AI = self.AI + aAI * p.annual_budget / p.AI_scale
        self.H = (1.0 - p.h_depreciation) * self.H + aH * p.annual_budget / p.H_scale
        self.L = self.L * (1.0 + p.labor_growth)

        # Tác động thất nghiệp: AI có thể tăng rủi ro ngắn hạn; H và tăng trưởng làm giảm rủi ro.
        # Công thức này là phần mô phỏng bổ sung do đề không cho đầy đủ transition function.
        new_Y_raw = self.production(self.K, self.L, self.D, self.AI, self.H, self.A)
        gdp_growth_pct_raw = (new_Y_raw - old_Y) / max(old_Y, 1e-9) * 100.0

        automation_pressure = 0.080 * aAI + 0.020 * aD
        human_buffer = 0.110 * aH + 0.010 * aK
        growth_buffer = 0.008 * max(gdp_growth_pct_raw, -5.0)
        shock = self.rng.normal(0.0, p.shock_std) if p.shock_std > 0 else 0.0
        self.unemployment_risk = float(np.clip(old_u + automation_pressure - human_buffer - growth_buffer + shock, 0.05, 0.95))

        # TFP cải thiện nhẹ nhờ D, AI, H để phản ánh học hỏi chính sách.
        self.A = self.A * (1.0 + 0.0010 * (self.D / 100.0) + 0.0008 * (self.AI / 100.0) + 0.0012 * (self.H / 100.0))
        new_Y = self.production(self.K, self.L, self.D, self.AI, self.H, self.A)
        gdp_growth_pct = (new_Y - old_Y) / max(old_Y, 1e-9) * 100.0

        # Các thành phần reward theo đề: GDP, unemployment, cyber risk, emission.
        delta_unemployment_pp = (self.unemployment_risk - old_u) * 100.0
        cyber_risk = max(0.0, 10.0 * (0.60 * aAI + 0.25 * aD - 0.30 * aH + 0.10 * (self.AI / 160.0)))
        emission = max(0.0, 10.0 * (0.50 * aK + 0.20 * aD + 0.35 * aAI + 0.10 * (self.K / 40000.0)))

        w1, w2, w3, w4 = REWARD_WEIGHTS
        reward = float(w1 * gdp_growth_pct - w2 * delta_unemployment_pp - w3 * cyber_risk - w4 * emission)

        self.t += 1
        terminated = self.t >= p.T
        truncated = False
        self.state = self.discretize_state(gdp_growth_pct, self.D, self.AI, self.unemployment_risk)
        self.prev_Y = new_Y

        info = {
            "year_index": self.t,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "K": self.K,
            "D": self.D,
            "AI": self.AI,
            "H": self.H,
            "L": self.L,
            "Y": new_Y,
            "GDP_growth_pct": gdp_growth_pct,
            "unemployment_risk": self.unemployment_risk,
            "delta_unemployment_pp": delta_unemployment_pp,
            "cyber_risk": cyber_risk,
            "emission": emission,
            "reward": reward,
            "state_code": encode_state(self.state),
            "state_text": state_to_text(self.state),
        }
        return self.state.copy(), reward, terminated, truncated, info


class VietnamEconomyDiscreteObsEnv(VietnamEconomyEnv):
    """Wrapper cho DQN: observation là số nguyên 0..80 thay vì MultiDiscrete."""

    def __init__(self, params: Optional[EnvParams] = None, random_start: bool = True, seed: int = 42):
        super().__init__(params=params, random_start=random_start, seed=seed)
        self.observation_space = spaces.Discrete(81)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        state, info = super().reset(seed=seed, options=options)
        return encode_state(state), info

    def step(self, action: int):
        state, reward, terminated, truncated, info = super().step(action)
        return encode_state(state), reward, terminated, truncated, info


# ============================================================
# 4. Q-learning và đánh giá chính sách
# ============================================================
def train_q_learning(
    n_episodes: int = 10000,
    alpha: float = 0.10,
    discount: float = 0.95,
    epsilon_start: float = 1.00,
    epsilon_min: float = 0.05,
    epsilon_decay_episodes: int = 5000,
    random_start: bool = True,
    seed: int = 42,
    params: Optional[EnvParams] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Huấn luyện tabular Q-learning đúng công thức đề."""
    rng = np.random.default_rng(seed)
    env = VietnamEconomyEnv(params=params or EnvParams(), random_start=random_start, seed=seed)
    Q = np.zeros((3, 3, 3, 3, 5), dtype=float)
    rows = []

    for ep in range(int(n_episodes)):
        eps = max(float(epsilon_min), float(epsilon_start) - ep / max(float(epsilon_decay_episodes), 1.0))
        state, _ = env.reset(options={"random_start": random_start})
        total_reward = 0.0
        steps = 0

        while True:
            state_tuple = tuple(state)
            if rng.random() < eps:
                action = int(rng.integers(0, 5))
            else:
                action = int(np.argmax(Q[state_tuple]))

            next_state, reward, terminated, truncated, info = env.step(action)
            next_tuple = tuple(next_state)
            old_q = Q[state_tuple + (action,)]
            if terminated or truncated:
                target = reward
            else:
                target = reward + discount * np.max(Q[next_tuple])
            Q[state_tuple + (action,)] = old_q + alpha * (target - old_q)

            total_reward += reward
            steps += 1
            state = next_state
            if terminated or truncated:
                break

        rows.append(
            {
                "episode": ep + 1,
                "epsilon": eps,
                "total_reward": total_reward,
                "steps": steps,
            }
        )

    learning_df = pd.DataFrame(rows)
    learning_df["rolling_reward_100"] = learning_df["total_reward"].rolling(100, min_periods=1).mean()
    return Q, learning_df


@st.cache_data(show_spinner=False)
def train_q_learning_cached(
    n_episodes: int,
    alpha: float,
    discount: float,
    epsilon_start: float,
    epsilon_min: float,
    epsilon_decay_episodes: int,
    random_start: bool,
    seed: int,
    shock_std: float,
) -> Tuple[np.ndarray, pd.DataFrame]:
    params = EnvParams(shock_std=float(shock_std))
    return train_q_learning(
        n_episodes=n_episodes,
        alpha=alpha,
        discount=discount,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay_episodes=epsilon_decay_episodes,
        random_start=random_start,
        seed=seed,
        params=params,
    )


def greedy_action(Q: np.ndarray, state: np.ndarray | List[int]) -> int:
    return int(np.argmax(Q[tuple(np.asarray(state, dtype=int))]))


def run_policy_episode(
    policy: str,
    Q: Optional[np.ndarray] = None,
    fixed_action: Optional[int] = None,
    initial_state: Optional[np.ndarray] = None,
    seed: int = 123,
    params: Optional[EnvParams] = None,
    dqn_model: Optional[Any] = None,
) -> Tuple[pd.DataFrame, float]:
    """Chạy 1 episode để đánh giá policy."""
    rng = np.random.default_rng(seed)
    env = VietnamEconomyEnv(params=params or EnvParams(), random_start=False, seed=seed)
    state, info0 = env.reset(options={"initial_state": initial_state if initial_state is not None else VN2026_STATE})
    total_reward = 0.0
    rows = []

    for _ in range(env.params.T):
        if policy == "q_learning":
            action = greedy_action(Q, state)
        elif policy == "fixed":
            action = int(fixed_action)
        elif policy == "random":
            action = int(rng.integers(0, 5))
        elif policy == "dqn":
            if dqn_model is None:
                action = 0
            else:
                obs_code = encode_state(state)
                action, _ = dqn_model.predict(obs_code, deterministic=True)
                action = int(action)
        else:
            action = 0

        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rows.append(
            {
                "Năm mô phỏng": int(info["year_index"]),
                "Trạng thái trước": state_to_text(state),
                "Hành động": ACTION_NAMES[action],
                "K": info["K"],
                "D": info["D"],
                "AI": info["AI"],
                "H": info["H"],
                "Y": info["Y"],
                "Tăng trưởng GDP (%)": info["GDP_growth_pct"],
                "Rủi ro thất nghiệp": info["unemployment_risk"],
                "Δ thất nghiệp (điểm %)": info["delta_unemployment_pp"],
                "CyberRisk": info["cyber_risk"],
                "Emission": info["emission"],
                "Reward": reward,
                "Trạng thái sau": state_to_text(next_state),
            }
        )
        state = next_state
        if terminated or truncated:
            break

    return pd.DataFrame(rows), float(total_reward)


def evaluate_policies(Q: np.ndarray, n_random_eval: int = 100, seed: int = 123, params: Optional[EnvParams] = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """So sánh pi* với rule-based a1, a3 và random."""
    params = params or EnvParams()
    policies = [
        ("Q-learning π*", "q_learning", None),
        ("Rule a1 - Cân bằng", "fixed", 1),
        ("Rule a3 - AI dẫn dắt", "fixed", 3),
    ]
    detail: Dict[str, pd.DataFrame] = {}
    rows = []

    for name, policy, fixed in policies:
        df_ep, total = run_policy_episode(policy, Q=Q, fixed_action=fixed, seed=seed, params=params)
        detail[name] = df_ep
        rows.append(
            {
                "Chính sách": name,
                "Reward tích lũy": total,
                "Reward bình quân/năm": df_ep["Reward"].mean(),
                "GDP growth bình quân (%)": df_ep["Tăng trưởng GDP (%)"].mean(),
                "D cuối kỳ": df_ep["D"].iloc[-1],
                "AI cuối kỳ": df_ep["AI"].iloc[-1],
                "H cuối kỳ": df_ep["H"].iloc[-1],
                "Rủi ro thất nghiệp cuối kỳ": df_ep["Rủi ro thất nghiệp"].iloc[-1],
                "Hành động đầu tiên": df_ep["Hành động"].iloc[0],
            }
        )

    # Random: lấy trung bình nhiều episode.
    random_totals = []
    random_final_rows = []
    for k in range(int(n_random_eval)):
        df_ep, total = run_policy_episode("random", Q=Q, seed=seed + k, params=params)
        random_totals.append(total)
        random_final_rows.append(df_ep.iloc[-1])
    random_df = pd.DataFrame(random_final_rows)
    detail["Random"] = random_df
    rows.append(
        {
            "Chính sách": f"Random trung bình {n_random_eval} lần",
            "Reward tích lũy": float(np.mean(random_totals)),
            "Reward bình quân/năm": float(random_df["Reward"].mean()),
            "GDP growth bình quân (%)": float(random_df["Tăng trưởng GDP (%)"].mean()),
            "D cuối kỳ": float(random_df["D"].mean()),
            "AI cuối kỳ": float(random_df["AI"].mean()),
            "H cuối kỳ": float(random_df["H"].mean()),
            "Rủi ro thất nghiệp cuối kỳ": float(random_df["Rủi ro thất nghiệp"].mean()),
            "Hành động đầu tiên": "Ngẫu nhiên",
        }
    )

    return pd.DataFrame(rows).sort_values("Reward tích lũy", ascending=False).reset_index(drop=True), detail


def all_state_policy_table(Q: np.ndarray) -> pd.DataFrame:
    rows = []
    for g in range(3):
        for d in range(3):
            for ai in range(3):
                for u in range(3):
                    s = np.array([g, d, ai, u], dtype=int)
                    a = greedy_action(Q, s)
                    rows.append(
                        {
                            "GDP growth": STATE_LABELS[g],
                            "Digital index": STATE_LABELS[d],
                            "AI capacity": STATE_LABELS[ai],
                            "Unemployment risk": STATE_LABELS[u],
                            "state_code": encode_state(s),
                            "action_code": ACTION_SHORT[a],
                            "Chính sách π*(s)": ACTION_NAMES[a],
                            "Phân bổ": ACTION_DESCRIPTION[a],
                            "Q_max": np.max(Q[tuple(s)]),
                        }
                    )
    return pd.DataFrame(rows)


def initial_state_policy_table(Q: np.ndarray) -> pd.DataFrame:
    cases = [
        ("Việt Nam 2026 thực tế", np.array([1, 1, 0, 1], dtype=int)),
        ("GDP thấp, D thấp, AI thấp, U cao", np.array([0, 0, 0, 2], dtype=int)),
        ("GDP cao, D cao, AI cao, U thấp", np.array([2, 2, 2, 0], dtype=int)),
        ("GDP trung bình, D thấp, AI thấp, U trung bình", np.array([1, 0, 0, 1], dtype=int)),
        ("GDP trung bình, D trung bình, AI cao, U cao", np.array([1, 1, 2, 2], dtype=int)),
    ]
    rows = []
    for name, s in cases:
        a = greedy_action(Q, s)
        df_ep, total = run_policy_episode("q_learning", Q=Q, initial_state=s, seed=777)
        rows.append(
            {
                "Trường hợp": name,
                "Trạng thái": state_to_text(s),
                "Hành động π*(s) đầu tiên": ACTION_NAMES[a],
                "Phân bổ": ACTION_DESCRIPTION[a],
                "Reward 10 năm nếu đi theo π*": total,
                "D cuối kỳ": df_ep["D"].iloc[-1],
                "AI cuối kỳ": df_ep["AI"].iloc[-1],
                "H cuối kỳ": df_ep["H"].iloc[-1],
                "U cuối kỳ": df_ep["Rủi ro thất nghiệp"].iloc[-1],
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# 5. Biểu đồ và báo cáo
# ============================================================
def fig_learning_curve(learning_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(learning_df["episode"], learning_df["rolling_reward_100"], label="Reward rolling mean 100 episodes")
    ax.set_title("Learning curve của tabular Q-learning")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward tích lũy / episode")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def fig_epsilon_decay(learning_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(learning_df["episode"], learning_df["epsilon"])
    ax.set_title("Epsilon-greedy giảm dần trong quá trình huấn luyện")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_policy_comparison(compare_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(compare_df))
    ax.bar(x, compare_df["Reward tích lũy"])
    ax.set_xticks(x)
    ax.set_xticklabels(compare_df["Chính sách"], rotation=20, ha="right")
    ax.set_title("So sánh reward tích lũy giữa π* và rule-based policies")
    ax.set_ylabel("Reward tích lũy")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_policy_grid(Q: np.ndarray, ai_level: int = 0, u_level: int = 1):
    grid = np.zeros((3, 3), dtype=int)
    for g in range(3):
        for d in range(3):
            s = np.array([g, d, ai_level, u_level], dtype=int)
            grid[g, d] = greedy_action(Q, s)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    im = ax.imshow(grid, aspect="auto")
    ax.set_title(f"Bản đồ hành động π*(s) khi AI={STATE_LABELS[ai_level]}, U={STATE_LABELS[u_level]}")
    ax.set_xlabel("Digital index")
    ax.set_ylabel("GDP growth")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([STATE_LABELS[i] for i in range(3)])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([STATE_LABELS[i] for i in range(3)])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, ACTION_SHORT[int(grid[i, j])], ha="center", va="center")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mã hành động")
    fig.tight_layout()
    return fig


def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe_name = name[:31]
            df.to_excel(writer, index=False, sheet_name=safe_name)
    return output.getvalue()


def make_html_report(
    action_df: pd.DataFrame,
    learning_tail_df: pd.DataFrame,
    initial_policy_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    policy_all_df: pd.DataFrame,
    policy_html: str,
) -> str:
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Bài 11 - Q-learning chính sách kinh tế</title>
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
        <h1>BÀI 11 - HỌC TĂNG CƯỜNG Q-LEARNING CHO CHÍNH SÁCH KINH TẾ THÍCH NGHI</h1>
        <div class="box">
            <p><b>Trạng thái:</b> 4 yếu tố, mỗi yếu tố 3 mức, tổng 3^4 = 81 trạng thái.</p>
            <p><b>Hành động:</b> 5 chính sách phân bổ ngân sách a0-a4.</p>
            <p><b>Thuật toán:</b> tabular Q-learning với epsilon-greedy.</p>
        </div>
        <h2>1. Tập hành động</h2>
        {action_df.to_html(index=False)}
        <h2>2. Learning curve - 10 episode cuối</h2>
        {learning_tail_df.to_html(index=False)}
        <h2>3. Chính sách tại VN 2026 và 4 trạng thái giả định</h2>
        {initial_policy_df.to_html(index=False)}
        <h2>4. So sánh với rule-based policies</h2>
        {compare_df.to_html(index=False)}
        <h2>5. Chính sách π*(s) cho toàn bộ 81 trạng thái</h2>
        {policy_all_df.to_html(index=False)}
        <h2>6. Câu hỏi thảo luận chính sách</h2>
        {policy_html}
    </body>
    </html>
    """
    return html


def policy_discussion_html(initial_policy_df: pd.DataFrame, compare_df: pd.DataFrame) -> str:
    # Lấy hành động ở các trạng thái cần trả lời chính sách.
    quick_win_row = initial_policy_df[initial_policy_df["Trường hợp"].str.contains("GDP thấp", regex=False)].iloc[0]
    consolidation_row = initial_policy_df[initial_policy_df["Trường hợp"].str.contains("GDP cao", regex=False)].iloc[0]
    best_policy = compare_df.iloc[0]["Chính sách"] if len(compare_df) else "không xác định"

    html = f"""
    <h3>11.4.a. Khi GDP growth thấp, D thấp, U cao, chính sách π*(s) chọn gì? Có khớp với quick win không?</h3>
    <p>
    Ở trạng thái GDP thấp, mức số hóa thấp, AI thấp và rủi ro thất nghiệp cao, chính sách học được chọn
    <b>{quick_win_row['Hành động π*(s) đầu tiên']}</b>, tức <b>{quick_win_row['Phân bổ']}</b>.
    Nếu hành động nghiêng về số hóa nhanh hoặc bao trùm, có thể hiểu là mô hình ưu tiên quick win theo hướng cải thiện nền tảng số
    và giảm rủi ro lao động. Nếu hành động nghiêng về cân bằng, mô hình đang tránh dồn quá mạnh vào AI khi nền tảng D và H còn yếu.
    </p>

    <h3>11.4.b. Khi GDP growth cao, AI cao, U thấp, chính sách chọn gì? Có phù hợp consolidation không?</h3>
    <p>
    Ở trạng thái GDP cao, số hóa cao, AI cao và thất nghiệp thấp, chính sách π*(s) chọn
    <b>{consolidation_row['Hành động π*(s) đầu tiên']}</b>, tức <b>{consolidation_row['Phân bổ']}</b>.
    Trạng thái này đã có nền tảng thuận lợi, nên nếu mô hình chọn cân bằng hoặc bao trùm thì phù hợp với logic consolidation:
    củng cố thành quả, giảm rủi ro xã hội và tránh để tự động hóa làm tăng thất nghiệp. Nếu mô hình vẫn chọn AI dẫn dắt,
    cần hiểu đó là chiến lược tiếp tục khai thác lợi thế công nghệ nhưng phải đi kèm kiểm soát CyberRisk và đào tạo lại.
    </p>

    <h3>11.4.c. Tích hợp π* vào quy trình hoạch định chính sách thế nào để không thay thế quyết định chính trị - xã hội?</h3>
    <p>
    Kết quả Q-learning nên được dùng như một công cụ hỗ trợ ra quyết định, không phải cơ chế tự động ban hành chính sách.
    Quy trình hợp lý là: dùng mô hình để tạo kịch bản định lượng, công khai giả định và reward, kiểm tra độ nhạy,
    sau đó trình kết quả cho hội đồng chính sách, chuyên gia ngành, địa phương và đại diện xã hội phản biện.
    Trong kết quả hiện tại, chính sách có reward tốt nhất là <b>{best_policy}</b>, nhưng lựa chọn cuối cùng vẫn cần cân nhắc
    mục tiêu công bằng, an sinh, an ninh dữ liệu và trách nhiệm giải trình của cơ quan nhà nước.
    </p>
    """
    return html


# ============================================================
# 6. Sidebar tham số
# ============================================================
st.sidebar.header("⚙️ Tham số Bài 11")

n_episodes = st.sidebar.number_input(
    "Số episode huấn luyện",
    min_value=500,
    max_value=50000,
    value=10000,
    step=500,
    help="Đề yêu cầu 10.000 episodes. Có thể giảm để test nhanh.",
)
alpha = st.sidebar.slider("Learning rate α", 0.01, 0.50, 0.10, 0.01)
discount = st.sidebar.slider("Discount γ", 0.50, 0.99, 0.95, 0.01)
epsilon_start = st.sidebar.slider("Epsilon ban đầu", 0.10, 1.00, 1.00, 0.05)
epsilon_min = st.sidebar.slider("Epsilon tối thiểu", 0.00, 0.20, 0.05, 0.01)
epsilon_decay_episodes = st.sidebar.number_input(
    "Số episode để epsilon giảm về mức tối thiểu",
    min_value=100,
    max_value=50000,
    value=5000,
    step=100,
)
random_start = st.sidebar.checkbox(
    "Huấn luyện trên nhiều trạng thái khởi đầu",
    value=True,
    help="Reset mặc định của môi trường vẫn là VN 2026. Tùy chọn này giúp Q-table học được nhiều trạng thái hơn để trả lời 4 trạng thái giả định.",
)
seed = st.sidebar.number_input("Seed tái lập", min_value=1, max_value=999999, value=42, step=1)
shock_std = st.sidebar.slider(
    "Nhiễu cú sốc trong transition",
    0.00,
    0.05,
    0.00,
    0.01,
    help="Mặc định 0 để kết quả tái lập. Tăng nhẹ để mô phỏng môi trường bất định.",
)
n_random_eval = st.sidebar.number_input("Số lần đánh giá random policy", 10, 1000, 100, 10)

params = EnvParams(shock_std=float(shock_std))

# ============================================================
# 7. Nội dung webapp
# ============================================================
st.markdown(
    """
Bài 11 mô hình hóa nền kinh tế Việt Nam như một **Markov Decision Process (MDP)**:
trạng thái gồm 4 yếu tố rời rạc, hành động là 5 chính sách phân bổ ngân sách,
và reward phản ánh đánh đổi giữa tăng trưởng GDP, thất nghiệp, rủi ro an ninh mạng và phát thải.
"""
)

if not GYM_AVAILABLE:
    st.warning(
        "Chưa cài gymnasium. Code vẫn chạy bằng lớp fallback để bạn xem kết quả, "
        "nhưng để đúng yêu cầu 11.3.1 hãy cài: python -m pip install gymnasium"
    )
else:
    st.success("Đã phát hiện gymnasium. Môi trường VietnamEconomyEnv kế thừa gym.Env đúng yêu cầu 11.3.1.")

st.header("1. Mô hình MDP: trạng thái, hành động và reward")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Số chiều trạng thái", "4")
with c2:
    st.metric("Số trạng thái", "3⁴ = 81")
with c3:
    st.metric("Số hành động", "5")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Tập trạng thái")
    state_df = pd.DataFrame(
        {
            "Thành phần trạng thái": STATE_DIM_NAMES,
            "Mức 0": ["Thấp"] * 4,
            "Mức 1": ["Trung bình"] * 4,
            "Mức 2": ["Cao"] * 4,
        }
    )
    st.dataframe(state_df, use_container_width=True, hide_index=True)

with col_b:
    st.subheader("Hàm thưởng")
    reward_df = pd.DataFrame(
        {
            "Thành phần": ["ΔGDP", "Δunemployment", "CyberRisk", "Emission"],
            "Vai trò": ["Lợi ích", "Chi phí", "Chi phí", "Chi phí"],
            "Trọng số": REWARD_WEIGHTS,
        }
    )
    st.dataframe(reward_df, use_container_width=True, hide_index=True)

st.subheader("5 hành động phân bổ ngân sách")
action_df = action_table()
st.dataframe(action_df, use_container_width=True, hide_index=True)

st.markdown(
    """
**Điểm cần hiểu đúng:** đề bài chỉ nêu reward tổng quát và gợi ý cập nhật K, D, AI, H; không cung cấp đầy đủ hàm chuyển trạng thái.
Vì vậy phần `step()` trong code dùng mô phỏng Cobb-Douglas mở rộng và các hệ số rủi ro minh họa để tạo MDP chạy được.
"""
)

# ============================================================
# 8. Huấn luyện Q-learning
# ============================================================
st.header("2. Câu 11.3.1 - 11.3.2: Cài đặt môi trường và huấn luyện Q-learning")

with st.spinner("Đang huấn luyện tabular Q-learning..."):
    Q, learning_df = train_q_learning_cached(
        int(n_episodes),
        float(alpha),
        float(discount),
        float(epsilon_start),
        float(epsilon_min),
        int(epsilon_decay_episodes),
        bool(random_start),
        int(seed),
        float(shock_std),
    )

st.success("Đã huấn luyện xong Q-learning.")

last_100 = learning_df.tail(100)
first_100 = learning_df.head(100)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Episode", f"{int(n_episodes):,}")
with col2:
    st.metric("Reward TB 100 episode đầu", f"{first_100['total_reward'].mean():.4f}")
with col3:
    st.metric("Reward TB 100 episode cuối", f"{last_100['total_reward'].mean():.4f}")

fig1 = fig_learning_curve(learning_df)
st.pyplot(fig1, clear_figure=False)

fig2 = fig_epsilon_decay(learning_df)
st.pyplot(fig2, clear_figure=False)

st.subheader("Bảng 10 episode cuối")
st.dataframe(learning_df.tail(10).round(6), use_container_width=True, hide_index=True)

# ============================================================
# 9. Trích xuất chính sách π*
# ============================================================
st.header("3. Câu 11.3.3: Trích xuất chính sách π*(s)")
initial_policy_df = initial_state_policy_table(Q)
st.dataframe(
    initial_policy_df.style.format(
        {
            "Reward 10 năm nếu đi theo π*": "{:,.4f}",
            "D cuối kỳ": "{:,.2f}",
            "AI cuối kỳ": "{:,.2f}",
            "H cuối kỳ": "{:,.2f}",
            "U cuối kỳ": "{:,.4f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

policy_all_df = all_state_policy_table(Q)

st.subheader("Toàn bộ chính sách π*(s) cho 81 trạng thái")
st.dataframe(policy_all_df, use_container_width=True, hide_index=True)

col_grid_1, col_grid_2 = st.columns(2)
with col_grid_1:
    ai_level = st.selectbox("Chọn mức AI để xem policy grid", [0, 1, 2], format_func=lambda x: STATE_LABELS[x], index=0)
with col_grid_2:
    u_level = st.selectbox("Chọn mức thất nghiệp để xem policy grid", [0, 1, 2], format_func=lambda x: STATE_LABELS[x], index=1)
fig_grid = fig_policy_grid(Q, int(ai_level), int(u_level))
st.pyplot(fig_grid, clear_figure=False)

# ============================================================
# 10. So sánh với rule-based policies
# ============================================================
st.header("4. Câu 11.3.4: So sánh π* với 3 chính sách rule-based")
compare_df, detail_dict = evaluate_policies(Q, int(n_random_eval), int(seed) + 1000, params=params)

st.dataframe(
    compare_df.style.format(
        {
            "Reward tích lũy": "{:,.4f}",
            "Reward bình quân/năm": "{:,.4f}",
            "GDP growth bình quân (%)": "{:,.4f}",
            "D cuối kỳ": "{:,.2f}",
            "AI cuối kỳ": "{:,.2f}",
            "H cuối kỳ": "{:,.2f}",
            "Rủi ro thất nghiệp cuối kỳ": "{:,.4f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

fig3 = fig_policy_comparison(compare_df)
st.pyplot(fig3, clear_figure=False)

st.subheader("Chi tiết quỹ đạo của chính sách Q-learning π*")
st.dataframe(
    detail_dict["Q-learning π*"].style.format(
        {
            "K": "{:,.2f}",
            "D": "{:,.2f}",
            "AI": "{:,.2f}",
            "H": "{:,.2f}",
            "Y": "{:,.2f}",
            "Tăng trưởng GDP (%)": "{:,.4f}",
            "Rủi ro thất nghiệp": "{:,.4f}",
            "Δ thất nghiệp (điểm %)": "{:,.4f}",
            "CyberRisk": "{:,.4f}",
            "Emission": "{:,.4f}",
            "Reward": "{:,.4f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# 11. Mở rộng DQN
# ============================================================
st.header("5. Câu 11.3.5 mở rộng: DQN bằng stable-baselines3")

st.markdown(
    """
Phần này là mở rộng. Code dùng `stable-baselines3` với neural network 2 hidden layers `[64, 64]`.
Mặc định không tự chạy để tránh làm webapp chậm; bạn bấm nút bên dưới nếu máy đã cài đủ thư viện.
"""
)

if not SB3_AVAILABLE:
    st.warning(
        "Chưa cài stable-baselines3 hoặc torch. Muốn chạy DQN, cài: python -m pip install stable-baselines3 torch"
    )
else:
    dqn_timesteps = st.number_input("Số timesteps train DQN", 1000, 100000, 10000, 1000)
    run_dqn = st.button("Chạy DQN mở rộng")
    if run_dqn:
        with st.spinner("Đang huấn luyện DQN..."):
            dqn_env = VietnamEconomyDiscreteObsEnv(params=params, random_start=True, seed=int(seed))
            try:
                check_env(dqn_env, warn=True)
            except Exception:
                pass
            model = DQN(
                "MlpPolicy",
                dqn_env,
                learning_rate=1e-3,
                buffer_size=5000,
                learning_starts=200,
                batch_size=64,
                gamma=float(discount),
                train_freq=1,
                target_update_interval=250,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                policy_kwargs={"net_arch": [64, 64]},
                verbose=0,
                seed=int(seed),
            )
            model.learn(total_timesteps=int(dqn_timesteps), progress_bar=False)

            dqn_df, dqn_reward = run_policy_episode("dqn", Q=Q, seed=int(seed) + 2000, params=params, dqn_model=model)
            q_df, q_reward = run_policy_episode("q_learning", Q=Q, seed=int(seed) + 2000, params=params)
            dqn_compare = pd.DataFrame(
                [
                    {"Mô hình": "Tabular Q-learning", "Reward tích lũy": q_reward, "Hành động đầu tiên": q_df["Hành động"].iloc[0]},
                    {"Mô hình": "DQN mở rộng", "Reward tích lũy": dqn_reward, "Hành động đầu tiên": dqn_df["Hành động"].iloc[0]},
                ]
            )
            st.dataframe(dqn_compare, use_container_width=True, hide_index=True)
            if dqn_reward > q_reward:
                st.success("Trong lần chạy này, DQN cho reward cao hơn tabular Q-learning.")
            else:
                st.info("Trong lần chạy này, DQN chưa cải thiện so với tabular Q-learning. Điều này có thể do số trạng thái nhỏ nên tabular Q-learning đã đủ tốt, hoặc DQN cần thêm timesteps.")

# ============================================================
# 12. Câu hỏi thảo luận chính sách
# ============================================================
st.header("6. Câu 11.4 - Thảo luận chính sách")
policy_html = policy_discussion_html(initial_policy_df, compare_df)
st.markdown(policy_html, unsafe_allow_html=True)

# ============================================================
# 13. Tải file kết quả
# ============================================================
st.header("7. Xuất kết quả")

learning_tail_df = learning_df.tail(200).reset_index(drop=True)
excel_bytes = to_excel_bytes(
    {
        "actions": action_df,
        "learning_tail": learning_tail_df,
        "initial_policy": initial_policy_df,
        "compare_policies": compare_df,
        "policy_81_states": policy_all_df,
        "q_policy_trajectory": detail_dict["Q-learning π*"],
    }
)

html_report = make_html_report(
    action_df=action_df,
    learning_tail_df=learning_tail_df.tail(20),
    initial_policy_df=initial_policy_df,
    compare_df=compare_df,
    policy_all_df=policy_all_df,
    policy_html=policy_html,
)

col_dl1, col_dl2, col_dl3 = st.columns(3)
with col_dl1:
    st.download_button(
        "⬇️ Tải Excel kết quả Bài 11",
        data=excel_bytes,
        file_name="bai11_q_learning_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
with col_dl2:
    st.download_button(
        "⬇️ Tải HTML báo cáo Bài 11",
        data=html_report.encode("utf-8"),
        file_name="bai11_q_learning_report.html",
        mime="text/html",
    )
with col_dl3:
    st.download_button(
        "⬇️ Tải CSV so sánh policy",
        data=compare_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="bai11_policy_comparison.csv",
        mime="text/csv",
    )

# Tự lưu outputs để khi viết báo cáo có đủ bảng/chạy thật, không cần bấm nút thủ công.
try:
    (OUTPUT_DIR / "bai11_q_learning_report.html").write_text(html_report, encoding="utf-8")
    learning_df.to_csv(OUTPUT_DIR / "bai11_learning_curve.csv", index=False, encoding="utf-8-sig")
    initial_policy_df.to_csv(OUTPUT_DIR / "bai11_initial_policy.csv", index=False, encoding="utf-8-sig")
    compare_df.to_csv(OUTPUT_DIR / "bai11_policy_comparison.csv", index=False, encoding="utf-8-sig")
    policy_all_df.to_csv(OUTPUT_DIR / "bai11_policy_81_states.csv", index=False, encoding="utf-8-sig")
    detail_dict["Q-learning π*"].to_csv(OUTPUT_DIR / "bai11_q_policy_trajectory.csv", index=False, encoding="utf-8-sig")
except Exception:
    pass

st.markdown("---")
st.caption(
    "Bài 11 hoàn thành: gymnasium Env, tabular Q-learning, trích xuất π*, so sánh rule-based, learning curve, DQN mở rộng và thảo luận chính sách."
)
