import streamlit as st

def load_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #0b1020 0%, #0f172a 100%);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        h1 {
            font-size: 2.4rem !important;
            font-weight: 800 !important;
            color: #ffffff !important;
        }

        h2 {
            font-size: 1.55rem !important;
            font-weight: 750 !important;
            color: #ffffff !important;
            margin-top: 2rem !important;
        }

        h3 {
            color: #f5f7fb !important;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111827 0%, #0b1020 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        section[data-testid="stSidebar"] * {
            color: #e5e7eb !important;
        }

        .card {
            background: rgba(18, 26, 47, 0.95);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 20px 22px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.25);
            margin-bottom: 18px;
        }

        .card-title {
            font-size: 1.1rem;
            font-weight: 750;
            color: #ffffff;
            margin-bottom: 8px;
        }

        .card-text {
            color: #cbd5e1;
            font-size: 0.95rem;
            line-height: 1.55;
        }

        .badge {
            display: inline-block;
            padding: 6px 12px;
            margin-right: 8px;
            margin-bottom: 8px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            color: white;
            background: linear-gradient(90deg, #ff3b7f, #7c3aed);
        }

        div[data-testid="stMetric"] {
            background: rgba(18, 26, 47, 0.95);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 18px;
            border-radius: 16px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.22);
        }

        div[data-testid="stMetricValue"] {
            color: #ffffff;
            font-weight: 800;
        }

        .stButton > button {
            border-radius: 999px;
            border: 0;
            padding: 0.6rem 1.2rem;
            background: linear-gradient(90deg, #ff3b7f, #7c3aed);
            color: white;
            font-weight: 700;
        }

        .stDownloadButton > button {
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.15);
            padding: 0.6rem 1.2rem;
            background: linear-gradient(90deg, #2563eb, #7c3aed);
            color: white;
            font-weight: 700;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.08);
        }

        button[data-baseweb="tab"] {
            background: rgba(255,255,255,0.04);
            border-radius: 999px;
            margin-right: 8px;
            padding: 8px 16px;
            color: #e5e7eb;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(90deg, #ff3b7f, #7c3aed);
            color: white;
        }

        div[data-testid="stAlert"] {
            border-radius: 16px;
        }

        hr {
            border: none;
            border-top: 1px solid rgba(255,255,255,0.1);
            margin: 2rem 0;
        }
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