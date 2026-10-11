import pandas as pd
from matching import recommend_creators

def sample_data():
    return pd.DataFrame([
        {"creator_id":"A","creator_name":"A","platform":"YouTube","niche":"education","language":"Tamil","location":"India","audience_segment":"students","followers":10000,"avg_views":3000,"engagement_rate":5.0,"rate_inr":5000,"content_description":"education Tamil students Python"},
        {"creator_id":"B","creator_name":"B","platform":"Instagram","niche":"beauty","language":"English","location":"India","audience_segment":"beauty consumers","followers":50000,"avg_views":10000,"engagement_rate":3.0,"rate_inr":10000,"content_description":"beauty English beauty consumers"},
    ])

def test_matching_prioritizes_relevant_creator():
    campaign = {"niche":"education","audience_segment":"students","language":"Tamil","budget_inr":6000,"campaign_description":"Python course for students"}
    result = recommend_creators(sample_data(), campaign, top_n=2)
    assert result.iloc[0].creator_id == "A"

def test_budget_filter():
    campaign = {"niche":"education","audience_segment":"students","language":"Tamil","budget_inr":6000,"campaign_description":"Python course"}
    result = recommend_creators(sample_data(), campaign, top_n=10)
    assert all(result.rate_inr <= 6000)
