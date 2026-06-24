# AIDEOM-VN: AI Economic Decision Optimization Model for Vietnam

## Overview

AIDEOM-VN is a Streamlit-based decision-support prototype for Vietnam’s AI-driven economic development. The project combines economic modeling, optimization, multi-criteria decision analysis, scenario simulation, and policy recommendation.

## Methods

- Extended Cobb-Douglas production function
- Linear Programming
- Mixed Integer Programming
- TOPSIS / AHP / Entropy weighting
- NSGA-II multi-objective optimization
- Dynamic optimization
- Two-stage stochastic programming
- Q-learning for adaptive policy recommendation

## Tools

Python, Streamlit, pandas, NumPy, SciPy, PuLP, CVXPY, pymoo, Pyomo, Plotly, Matplotlib

## Key Outputs

- GDP forecast and TFP decomposition
- Sector priority ranking
- Regional AI readiness ranking
- Digital budget allocation
- Project portfolio selection
- Labor market impact simulation
- Risk dashboard
- Integrated AIDEOM-VN policy dashboard

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py