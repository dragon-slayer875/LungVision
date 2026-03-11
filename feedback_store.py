"""
feedback_store.py — Pluggable feedback storage backend.

Two backends are supported:
  - CSVStore   : appends rows to feedback/feedback_log.csv  (default)
  - SQLiteStore: persists rows in feedback/feedback.db

Both expose the same interface:
    store.save(row: dict)
    store.load() -> pd.DataFrame
"""

import os
import csv
import uuid
import sqlite3
import datetime
import pandas as pd


# ── Shared config ─────────────────────────────────────────────────────────────
FEEDBACK_DIR  = "feedback"
FEEDBACK_CSV  = os.path.join(FEEDBACK_DIR, "feedback_log.csv")
FEEDBACK_DB   = os.path.join(FEEDBACK_DIR, "feedback.db")

COLS = [
    "id", "timestamp", "filename",
    "predicted_label", "predicted_confidence",
    "correct_label", "user_agrees", "notes",
    "prob_bacterial", "prob_covid19", "prob_normal", "prob_viral",
    "resnet_bacterial", "resnet_covid19", "resnet_normal", "resnet_viral",
    "densenet_bacterial", "densenet_covid19", "densenet_normal", "densenet_viral",
]


def build_row(filename, result, user_agrees, correct_label, notes) -> dict:
    """Construct a feedback row dict from inference result + user input."""
    return {
        "id":                   str(uuid.uuid4())[:8],
        "timestamp":            datetime.datetime.now().isoformat(timespec="seconds"),
        "filename":             filename,
        "predicted_label":      result["label"],
        "predicted_confidence": round(result["confidence"], 6),
        "correct_label":        correct_label,
        "user_agrees":          user_agrees,
        "notes":                notes,
        "prob_bacterial":       round(result["probs"]["bacterial_and_other"], 6),
        "prob_covid19":         round(result["probs"]["covid19"], 6),
        "prob_normal":          round(result["probs"]["normal"], 6),
        "prob_viral":           round(result["probs"]["viral_pneumonia"], 6),
        "resnet_bacterial":     round(result["resnet_probs"]["bacterial_and_other"], 6),
        "resnet_covid19":       round(result["resnet_probs"]["covid19"], 6),
        "resnet_normal":        round(result["resnet_probs"]["normal"], 6),
        "resnet_viral":         round(result["resnet_probs"]["viral_pneumonia"], 6),
        "densenet_bacterial":   round(result["densenet_probs"]["bacterial_and_other"], 6),
        "densenet_covid19":     round(result["densenet_probs"]["covid19"], 6),
        "densenet_normal":      round(result["densenet_probs"]["normal"], 6),
        "densenet_viral":       round(result["densenet_probs"]["viral_pneumonia"], 6),
    }


# ── CSV backend ───────────────────────────────────────────────────────────────
class CSVStore:
    def __init__(self):
        os.makedirs(FEEDBACK_DIR, exist_ok=True)
        if not os.path.isfile(FEEDBACK_CSV):
            with open(FEEDBACK_CSV, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=COLS).writeheader()

    def save(self, row: dict):
        with open(FEEDBACK_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=COLS).writerow(row)

    def load(self) -> pd.DataFrame:
        try:
            return pd.read_csv(FEEDBACK_CSV)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=COLS)

    def export_csv(self) -> bytes:
        return open(FEEDBACK_CSV, "rb").read()


# ── SQLite backend ────────────────────────────────────────────────────────────
class SQLiteStore:
    _CREATE = f"""
    CREATE TABLE IF NOT EXISTS feedback (
        id                   TEXT PRIMARY KEY,
        timestamp            TEXT,
        filename             TEXT,
        predicted_label      TEXT,
        predicted_confidence REAL,
        correct_label        TEXT,
        user_agrees          INTEGER,
        notes                TEXT,
        prob_bacterial       REAL,
        prob_covid19         REAL,
        prob_normal          REAL,
        prob_viral           REAL,
        resnet_bacterial     REAL,
        resnet_covid19       REAL,
        resnet_normal        REAL,
        resnet_viral         REAL,
        densenet_bacterial   REAL,
        densenet_covid19     REAL,
        densenet_normal      REAL,
        densenet_viral       REAL
    )"""

    def __init__(self):
        os.makedirs(FEEDBACK_DIR, exist_ok=True)
        con = sqlite3.connect(FEEDBACK_DB)
        con.execute(self._CREATE)
        con.commit()
        con.close()

    def _connect(self):
        return sqlite3.connect(FEEDBACK_DB)

    def save(self, row: dict):
        # SQLite stores booleans as 0/1
        row = {**row, "user_agrees": int(row["user_agrees"])}
        placeholders = ", ".join("?" * len(COLS))
        values = [row[c] for c in COLS]
        con = self._connect()
        con.execute(f"INSERT INTO feedback VALUES ({placeholders})", values)
        con.commit()
        con.close()

    def load(self) -> pd.DataFrame:
        con = self._connect()
        df = pd.read_sql_query(
            "SELECT * FROM feedback ORDER BY timestamp DESC", con
        )
        con.close()
        # Restore bool dtype
        if "user_agrees" in df.columns:
            df["user_agrees"] = df["user_agrees"].astype(bool)
        return df

    def export_csv(self) -> bytes:
        return self.load().to_csv(index=False).encode("utf-8")


# ── Factory ───────────────────────────────────────────────────────────────────
def get_store(backend: str):
    """Return the appropriate store instance. backend: 'CSV' or 'SQLite'"""
    if backend == "SQLite":
        return SQLiteStore()
    return CSVStore()
