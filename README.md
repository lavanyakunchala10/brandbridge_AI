# BrandBridge AI

An independent academic prototype inspired by creator/brand collaboration workflows. It is not affiliated with or integrated into BrandBridge. No official public BrandBridge developer API was verified during project planning, so this app uses local synthetic data and its own SQLite database.

## Features
- 500 reproducible synthetic creator profiles
- Campaign form with niche, audience, language, and budget criteria
- Hybrid TF-IDF/content similarity + profile feature scoring
- Explainable score components and ranking
- Template-based outreach pitch draft
- Streamlit UI, Plotly chart, CSV export
- SQLite campaign and recommendation history

## Run locally
Python 3.10+ recommended.
open terminal or command prompt 
move to project directory

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python generate_dataset.py
streamlit run app.py
```

The database `brandbridge_ai.db` is created automatically when the app starts. `data/creators.csv` is included; regenerate it with `python generate_dataset.py` for the same seeded dataset.

## Project structure
- `app.py`: Streamlit UI
- `matching.py`: ranking engine
- `database.py`: SQLite persistence
- `generate_dataset.py`: reproducible synthetic data generator
- `data/creators.csv`: 500 sample profiles
- `tests/test_matching.py`: basic unit tests

## Research protocol
The current score is a heuristic baseline, not a trained predictive model. To make the project publication-ready:
1. Create a labelled dataset of campaign–creator pairs using a transparent annotation protocol.
2. Compare popularity, engagement, TF-IDF similarity, and hybrid methods.
3. Use a held-out test set and report Precision@K, Recall@K, NDCG@K, budget-feasibility rate, confidence intervals, and ablation results.
4. Avoid claiming real-world campaign success from synthetic data.
5. Use only authorized creator data and comply with platform/API terms.

## Core matching formula
`MatchScore = 100 × Σ(weight_i × feature_score_i)`

Default weights: niche 0.30, audience 0.20, language 0.15, engagement 0.15, budget 0.10, views 0.10. Scores are designed to be interpretable and must be validated against labelled data before any performance claims.

## Limitations
- All bundled creator records are synthetic.
- The pitch generator is a template; it does not call an LLM API.
- No live social media metrics or BrandBridge API integration is included.
- For a real deployment, add authentication, authorization, validation, rate limits, logging, and secure secrets management.

