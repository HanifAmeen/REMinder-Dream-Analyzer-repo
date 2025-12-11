# scripts/migrate_add_fields.py
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dreams.db")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# List of new fields to add
fields = {
    "events": "TEXT",
    "entities": "TEXT",
    "people": "TEXT",
    "locations": "TEXT",
    "objects": "TEXT",
    "cause_effect": "TEXT",
    "conflicts": "TEXT",
    "desires": "TEXT",
    "emotional_arc": "TEXT",
    "narrative": "TEXT",
    "analysis_version": "TEXT"
}

for col, coltype in fields.items():
    try:
        cursor.execute(f"ALTER TABLE dream ADD COLUMN {col} {coltype};")
        print(f"Added column: {col}")
    except sqlite3.OperationalError:
        print(f"Column already exists: {col}")

conn.commit()
conn.close()

print("Migration completed.")
