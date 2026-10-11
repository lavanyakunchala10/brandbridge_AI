from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from matching import recommend_creators
from database import init_db, save_campaign, campaign_history, get_recommendations

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "creators.csv"
st.set_page_config(page_title="BrandBridge AI Research Prototype", page_icon="📊", layout="wide")
init_db()

@st.cache_data
def load_creators():
    return pd.read_csv(DATA_PATH)

creators = load_creators()
st.title("BrandBridge AI — Research Prototype")
st.caption("Independent academic prototype for explainable brand–creator matching. Not affiliated with BrandBridge.")
page = st.sidebar.radio("Navigate", ["Campaign Matcher", "Creator Explorer", "Campaign History", "Research Notes"])

if page == "Campaign Matcher":
    st.subheader("Create a campaign and rank suitable creators")
    with st.form("campaign_form"):
        c1, c2 = st.columns(2)
        with c1:
            brand_name = st.text_input("Brand / organization", "Tamil Tech Academy")
            campaign_name = st.text_input("Campaign name", "Python course launch")
            niche = st.selectbox("Creator niche", ["Any"] + sorted(creators.niche.unique().tolist()), index=3 if "education" in creators.niche.unique() else 0)
            language = st.selectbox("Preferred language", ["Any"] + sorted(creators.language.unique().tolist()), index=1 if "Tamil" in creators.language.unique() else 0)
        with c2:
            audience = st.selectbox("Target audience", ["Any"] + sorted(creators.audience_segment.unique().tolist()))
            budget = st.number_input("Maximum creator fee (INR)", min_value=0, value=15000, step=1000)
            top_n = st.slider("Number of recommendations", 5, 30, 10)
            description = st.text_area("Campaign description", "Promote an affordable beginner-friendly Python course to students and early-career learners.")
        submitted = st.form_submit_button("Find creators", type="primary")
    if submitted:
        campaign = {"brand_name": brand_name.strip() or "Unnamed brand", "campaign_name": campaign_name.strip() or "Unnamed campaign", "niche": niche, "audience_segment": audience, "language": language, "budget_inr": float(budget), "campaign_description": description}
        results = recommend_creators(creators, campaign, top_n=top_n)
        st.session_state["last_campaign"] = campaign
        st.session_state["last_results"] = results
    if "last_results" in st.session_state:
        results = st.session_state["last_results"]
        campaign = st.session_state["last_campaign"]
        if results.empty:
            st.warning("No creators meet the budget. Increase the budget or change the campaign criteria.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Creators recommended", len(results))
            m2.metric("Average match score", f"{results.match_score.mean():.1f}/100")
            m3.metric("Median engagement", f"{results.engagement_rate.median():.2f}%")
            st.dataframe(results[["creator_id","creator_name","platform","niche","language","followers","engagement_rate","avg_views","rate_inr","match_score","explanation"]], use_container_width=True, hide_index=True)
            fig = px.bar(results, x="creator_name", y="match_score", color="niche", title="Explainable creator match scores")
            st.plotly_chart(fig, use_container_width=True)
            pitch_creator = st.selectbox("Draft a sample outreach pitch for", results.creator_id.tolist(), format_func=lambda x: f"{x} — {results.loc[results.creator_id == x, 'creator_name'].iloc[0]}")
            row = results.loc[results.creator_id == pitch_creator].iloc[0]
            pitch = f"Subject: Collaboration opportunity — {campaign['campaign_name']}\n\nHi {row['creator_name']},\n\nWe are {campaign['brand_name']}, and we are planning a campaign about {campaign.get('niche', 'your content area')}. Your content profile and audience appear relevant to this campaign. We would like to explore a collaboration for {campaign.get('campaign_description', '')}\n\nCould you share your current media kit, audience insights, availability, and proposed fee? We will review the details together.\n\nBest regards,\n{campaign['brand_name']} team"
            st.text_area("Editable pitch draft (template-based; no external LLM used)", pitch, height=220)
            if st.button("Save campaign and recommendations"):
                cid = save_campaign(campaign, results)
                st.success(f"Saved campaign #{cid} and {len(results)} recommendations to SQLite.")
            st.download_button("Download recommendations CSV", results.to_csv(index=False).encode("utf-8"), "creator_recommendations.csv", "text/csv")

elif page == "Creator Explorer":
    st.subheader("Explore the synthetic creator dataset")
    f1, f2, f3 = st.columns(3)
    with f1: niche_filter = st.multiselect("Niche", sorted(creators.niche.unique()), default=[])
    with f2: platform_filter = st.multiselect("Platform", sorted(creators.platform.unique()), default=[])
    with f3: lang_filter = st.multiselect("Language", sorted(creators.language.unique()), default=[])
    view = creators.copy()
    if niche_filter: view = view[view.niche.isin(niche_filter)]
    if platform_filter: view = view[view.platform.isin(platform_filter)]
    if lang_filter: view = view[view.language.isin(lang_filter)]
    a,b,c = st.columns(3)
    a.metric("Creators", len(view)); b.metric("Median followers", f"{view.followers.median():,.0f}" if len(view) else "—"); c.metric("Median engagement", f"{view.engagement_rate.median():.2f}%" if len(view) else "—")
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("Download filtered creators", view.to_csv(index=False).encode(), "filtered_creators.csv", "text/csv")

elif page == "Campaign History":
    st.subheader("Saved campaigns")
    history = campaign_history()
    if history.empty:
        st.info("No campaigns saved yet. Use Campaign Matcher and select Save campaign.")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)
        selected = st.selectbox("View saved recommendations", history.campaign_id.tolist())
        st.dataframe(get_recommendations(int(selected)), use_container_width=True, hide_index=True)

else:
    st.subheader("Academic research notes")
    st.markdown("""
**Research objectives**
1. Build a creator–campaign dataset and baseline ranking system.
2. Compare popularity-only ranking with the hybrid score.
3. Evaluate recommendation relevance using a manually labelled test set.
4. Explain ranking components to improve transparency.

**Important limitations**
- The included dataset is synthetic and must not be presented as real creator data.
- Match scores are heuristic relevance scores, not probabilities of campaign success.
- The current pitch writer is a deterministic template, not an LLM.
- A defensible ML evaluation requires labelled brand–creator fit pairs or observed campaign outcomes.

**Suggested experiments**
- Baseline A: rank by followers.
- Baseline B: rank by engagement rate.
- Proposed: weighted hybrid score.
- Metrics: Precision@K, Recall@K, NDCG@K, budget-feasibility rate, and ranking stability.
- Ablation: remove one feature group at a time and compare NDCG@K.
""")
