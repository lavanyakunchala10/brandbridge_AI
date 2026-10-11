"""SQLite persistence layer for campaign recommendations."""
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).parent
DB_PATH = ROOT / "brandbridge_ai.db"

def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with connect() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS campaigns (
            campaign_id INTEGER PRIMARY KEY AUTOINCREMENT, brand_name TEXT NOT NULL,
            campaign_name TEXT NOT NULL, niche TEXT, audience_segment TEXT, language TEXT,
            budget_inr REAL, campaign_description TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, campaign_id INTEGER NOT NULL,
            creator_id TEXT NOT NULL, match_score REAL, explanation TEXT,
            FOREIGN KEY(campaign_id) REFERENCES campaigns(campaign_id))""")

def save_campaign(campaign: dict, recommendations: pd.DataFrame) -> int:
    with connect() as c:
        cur = c.execute("INSERT INTO campaigns (brand_name,campaign_name,niche,audience_segment,language,budget_inr,campaign_description) VALUES (?,?,?,?,?,?,?)",
            (campaign["brand_name"], campaign["campaign_name"], campaign.get("niche"), campaign.get("audience_segment"), campaign.get("language"), campaign.get("budget_inr"), campaign.get("campaign_description")))
        campaign_id = cur.lastrowid
        c.executemany("INSERT INTO recommendations (campaign_id,creator_id,match_score,explanation) VALUES (?,?,?,?)",
            [(campaign_id, str(r.creator_id), float(r.match_score), str(r.explanation)) for r in recommendations.itertuples()])
        return int(campaign_id)

def campaign_history() -> pd.DataFrame:
    with connect() as c:
        return pd.read_sql_query("SELECT campaign_id,brand_name,campaign_name,niche,budget_inr,created_at FROM campaigns ORDER BY campaign_id DESC", c)

def get_recommendations(campaign_id: int) -> pd.DataFrame:
    with connect() as c:
        return pd.read_sql_query("SELECT creator_id,match_score,explanation FROM recommendations WHERE campaign_id=? ORDER BY match_score DESC", c, params=(campaign_id,))
