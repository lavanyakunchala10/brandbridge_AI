"""Generate a reproducible synthetic creator dataset for academic experiments."""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N = 500
OUT = Path(__file__).parent / "data" / "creators.csv"

rng = np.random.default_rng(SEED)
niches = ["education", "technology", "beauty", "fitness", "food", "travel", "gaming", "finance", "lifestyle", "parenting"]
languages = ["Tamil", "English", "Hindi", "Telugu", "Malayalam", "Kannada"]
platforms = ["YouTube", "Instagram", "TikTok"]
locations = ["India", "United States", "United Kingdom", "Singapore", "Malaysia"]
rows = []
for i in range(1, N + 1):
    followers = int(np.clip(rng.lognormal(mean=10.1, sigma=1.15), 800, 2_500_000))
    niche = rng.choice(niches)
    engagement = float(np.clip(rng.normal(4.2 if followers < 100_000 else 2.8, 1.5), 0.2, 14.0))
    views = int(max(100, followers * rng.uniform(0.08, 0.65)))
    rate = int(max(500, followers * rng.uniform(0.003, 0.025) + views * 0.015))
    audience = rng.choice(["students", "young professionals", "parents", "general", "fitness enthusiasts", "tech buyers", "beauty consumers"])
    language = rng.choice(languages, p=[.28, .27, .17, .12, .08, .08])
    platform = rng.choice(platforms, p=[.42, .43, .15])
    location = rng.choice(locations, p=[.68, .12, .08, .06, .06])
    rows.append({
        "creator_id": f"CR{i:04d}", "creator_name": f"Creator {i:03d}", "platform": platform,
        "niche": niche, "language": language, "location": location, "audience_segment": audience,
        "followers": followers, "avg_views": views, "engagement_rate": round(engagement, 2),
        "rate_inr": rate, "content_description": f"{niche} creator making {language}-language content for {audience}"
    })
OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"Wrote {N} synthetic creators to {OUT}")
