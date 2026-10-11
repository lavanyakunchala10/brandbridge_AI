"""Hybrid explainable creator matching engine."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

DEFAULT_WEIGHTS = {"niche": 0.30, "audience": 0.20, "language": 0.15, "engagement": 0.15, "budget": 0.10, "views": 0.10}

def recommend_creators(creators: pd.DataFrame, campaign: dict, top_n: int = 10, weights: dict | None = None) -> pd.DataFrame:
    """Return ranked creators with component scores and human-readable explanations.

    Scores are heuristic research baselines, not calibrated probabilities of campaign success.
    """
    if creators.empty:
        return creators.copy()
    weights = weights or DEFAULT_WEIGHTS
    df = creators.copy()
    for col in ["niche", "audience_segment", "language", "content_description"]:
        df[col] = df[col].fillna("").astype(str)

    requested_text = " ".join(str(campaign.get(k, "")) for k in ["niche", "audience_segment", "campaign_description"])
    creator_text = (df["niche"] + " " + df["audience_segment"] + " " + df["content_description"]).tolist()
    corpus = creator_text + [requested_text]
    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    try:
        matrix = tfidf.fit_transform(corpus)
        semantic = cosine_similarity(matrix[:-1], matrix[-1]).ravel()
    except ValueError:
        semantic = np.zeros(len(df))

    requested_niche = str(campaign.get("niche", "Any")).lower()
    niche_exact = (df["niche"].str.lower() == requested_niche).astype(float) if requested_niche not in ("", "any") else np.ones(len(df))
    audience = str(campaign.get("audience_segment", "Any")).lower()
    audience_score = (df["audience_segment"].str.lower() == audience).astype(float) if audience not in ("", "any") else np.ones(len(df))
    language = str(campaign.get("language", "Any")).lower()
    language_score = (df["language"].str.lower() == language).astype(float) if language not in ("", "any") else np.ones(len(df))

    max_budget = float(campaign.get("budget_inr", 0) or 0)
    if max_budget > 0:
        budget_score = np.clip(1 - (df["rate_inr"].astype(float) / max_budget), 0, 1)
        # An affordable creator receives a non-zero baseline; over-budget creators are excluded below.
        affordable = df["rate_inr"].astype(float) <= max_budget
    else:
        budget_score = np.ones(len(df))
        affordable = np.ones(len(df), dtype=bool)

    scaler = MinMaxScaler()
    engagement_score = scaler.fit_transform(df[["engagement_rate"]].astype(float)).ravel() if df["engagement_rate"].nunique() > 1 else np.ones(len(df))
    views_score = scaler.fit_transform(np.log1p(df[["avg_views"]].astype(float))).ravel() if df["avg_views"].nunique() > 1 else np.ones(len(df))
    df["niche_score"] = np.maximum(niche_exact.to_numpy(), semantic)
    df["audience_score"] = audience_score.to_numpy()
    df["language_score"] = language_score.to_numpy()
    df["engagement_score"] = engagement_score
    df["budget_score"] = budget_score
    df["views_score"] = views_score
    df["match_score"] = 100 * sum(float(weights[k]) * df[f"{k}_score"] for k in weights)
    df["match_score"] = df["match_score"].clip(0, 100).round(2)
    df["within_budget"] = affordable

    def explain(row):
        reasons = []
        if requested_niche not in ("", "any") and row["niche"].lower() == requested_niche:
            reasons.append("niche match")
        if audience not in ("", "any") and row["audience_segment"].lower() == audience:
            reasons.append("target audience match")
        if language not in ("", "any") and row["language"].lower() == language:
            reasons.append("language match")
        if max_budget and row["rate_inr"] <= max_budget:
            reasons.append("within budget")
        if row["engagement_rate"] >= df["engagement_rate"].median():
            reasons.append("above-median engagement")
        return ", ".join(reasons) if reasons else "partial content/profile similarity"
    df["explanation"] = df.apply(explain, axis=1)
    df = df[df["within_budget"]].sort_values("match_score", ascending=False)
    return df.head(top_n).reset_index(drop=True)
