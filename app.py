# app.py (Hardened + Auto-migrate missing columns + Structured interpretation support)

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime, timedelta
import jwt
import bcrypt
from functools import wraps
import traceback
import json
import os
import sqlite3

# --- AI analysis utilities ---
from utils.analyzer_upgraded import analyze_dream

# ---------------------------------------
# CONFIG
# ---------------------------------------
SECRET_KEY = os.environ.get("REMINDER_SECRET_KEY", "CHANGE_THIS_TO_A_RANDOM_SECRET")

app = Flask(__name__)
CORS(app)

# Database setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "dreams.db")
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{db_path}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


# ---------------------------------------
# DB MIGRATION HELPERS (lightweight)
# ---------------------------------------
def add_sqlite_column_if_missing(db_file: str, table: str, column: str, column_type: str, default_sql: str = "NULL"):
    """
    Adds a column to an sqlite table if it doesn't exist.
    Uses PRAGMA table_info to check existence and then ALTER TABLE.
    column_type should be e.g. 'TEXT' or 'INTEGER'.
    default_sql is the column default expression (as SQL string, e.g. "''" or "NULL").
    """
    try:
        con = sqlite3.connect(db_file)
        cur = con.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        cols = [r[1] for r in cur.fetchall()]  # name is at index 1
        if column not in cols:
            # SQLite supports limited ALTER TABLE: only ADD COLUMN
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type} DEFAULT {default_sql}")
            con.commit()
            print(f"[migrate] Added column `{column}` to `{table}`")
        cur.close()
        con.close()
    except Exception as e:
        print(f"[migrate] Failed adding column {column}: {e}")
        traceback.print_exc()


def ensure_dream_table_columns():
    """
    Ensure all expected columns exist in `dream` table. If not, add them.
    Run inside app context before using the model.
    """
    expected = {
        "structured_interpretation": ("TEXT", "NULL"),
        "structured_text": ("TEXT", "''"),
        "psychological_interpretation": ("TEXT", "NULL"),
        "combined_insights": ("TEXT", "NULL"),
        "themes": ("TEXT", "NULL"),
        "symbols": ("TEXT", "NULL"),
        "analysis_version": ("TEXT", "NULL"),
        "events": ("TEXT", "NULL"),
        "entities": ("TEXT", "NULL"),
        "people": ("TEXT", "NULL"),
        "locations": ("TEXT", "NULL"),
        "objects": ("TEXT", "NULL"),
        "cause_effect": ("TEXT", "NULL"),
        "conflicts": ("TEXT", "NULL"),
        "desires": ("TEXT", "NULL"),
        "emotional_arc": ("TEXT", "NULL"),
        "narrative": ("TEXT", "NULL"),
    }

    for col, (ctype, default) in expected.items():
        add_sqlite_column_if_missing(db_path, "dream", col, ctype, default)


# ---------------------------------------
# MODELS
# ---------------------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Dream(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    title = db.Column(db.String(200))
    content = db.Column(db.Text)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    mood = db.Column(db.String(50))
    summary = db.Column(db.Text)

    themes = db.Column(db.Text)
    symbols = db.Column(db.Text)
    combined_insights = db.Column(db.Text)

    psychological_interpretation = db.Column(db.Text)

    # NEW STRUCTURED INTERPRETATION FIELDS (TEXT storing JSON or plain text)
    structured_interpretation = db.Column(db.Text)
    structured_text = db.Column(db.Text)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    # Structured fields (existing)
    events = db.Column(db.Text)
    entities = db.Column(db.Text)
    people = db.Column(db.Text)
    locations = db.Column(db.Text)
    objects = db.Column(db.Text)
    cause_effect = db.Column(db.Text)
    conflicts = db.Column(db.Text)
    desires = db.Column(db.Text)
    emotional_arc = db.Column(db.Text)
    narrative = db.Column(db.Text)
    analysis_version = db.Column(db.String(80))


# ---------------------------------------
# STARTUP: create tables and ensure columns exist
# ---------------------------------------
with app.app_context():
    # create table if not present
    db.create_all()
    # ensure missing columns are added (ALTER TABLE)
    ensure_dream_table_columns()


# ---------------------------------------
# HELPERS: tokens / auth / json safe
# ---------------------------------------
def make_token(user_id):
    payload = {"user_id": user_id, "exp": datetime.utcnow() + timedelta(hours=24)}
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    # pyjwt v1 returns bytes, v2 returns str
    return token.decode("utf-8") if isinstance(token, bytes) else token


def decode_token(token):
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return decoded.get("user_id")
    except Exception:
        return None


def auth_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        header = request.headers.get("Authorization", "") or request.headers.get("authorization", "")
        if not header or not header.startswith("Bearer "):
            return jsonify({"error": "Missing token"}), 401

        token = header.split(" ", 1)[1].strip()
        user_id = decode_token(token)
        if not user_id:
            return jsonify({"error": "Invalid or expired token"}), 401

        request.user_id = user_id
        return f(*args, **kwargs)

    return wrapper


def safe_json_load(val):
    """Try to decode JSON string; if already a dict/list return it; if None return None."""
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except Exception:
        return val


# ---------------------------------------
# AUTH ROUTES
# ---------------------------------------
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not email or not username or not password:
        return jsonify({"error": "All fields required"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already exists"}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 400

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    user = User(email=email, username=username, password_hash=hashed)
    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "User created"}), 201


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    user = User.query.filter_by(email=email).first()
    if not user or not bcrypt.checkpw(password.encode(), user.password_hash.encode()):
        return jsonify({"error": "Invalid credentials"}), 401

    token = make_token(user.id)

    return jsonify({
        "token": token,
        "user": {"id": user.id, "email": user.email, "username": user.username}
    })


# ---------------------------------------
# ADD DREAM
# ---------------------------------------
@app.route('/add_dream', methods=['POST'])
@auth_required
def add_dream():
    data = request.get_json() or {}

    title = data.get('title') or "Untitled Dream"
    content = data.get('content') or ""
    mood_input = data.get('mood', '')

    if not content.strip():
        return jsonify({"error": "Title and content required"}), 400

    # Previous dreams for recurring symbols (safe)
    previous = []
    try:
        rows = Dream.query.filter_by(user_id=request.user_id).all()
        for d in rows:
            try:
                prev_symbols = safe_json_load(d.symbols) or []
            except Exception:
                prev_symbols = []
            previous.append({"content": d.content or "", "symbols": prev_symbols})
    except Exception:
        previous = []

    # Run analyzer (capture exceptions)
    try:
        analysis = analyze_dream(content, previous_dreams=previous)
        if not isinstance(analysis, dict):
            analysis = {}
    except Exception:
        traceback.print_exc()
        analysis = {}

    # Extract fields with safe defaults
    summary = analysis.get("summary", "") or ""
    emotions = analysis.get("emotions", {}) or {}
    dominant_emotion = (emotions.get("dominant") or mood_input or "neutral")

    structured_interpretation = analysis.get("structured_interpretation") or {}
    structured_text = analysis.get("structured_text") or ""

    themes_list = analysis.get("themes") or []
    symbols_list = analysis.get("symbols") or []
    combined_insights_list = analysis.get("combined_insights") or []
    psychological_interpretation = analysis.get("psychological_interpretation") or {}

    events = analysis.get("events") or []
    entities = analysis.get("entities") or []
    people = analysis.get("people") or []
    locations = analysis.get("locations") or []
    objects = analysis.get("objects") or []
    cause_effect = analysis.get("cause_effect") or []
    conflicts = analysis.get("conflicts") or []
    desires = analysis.get("desires") or []
    emotional_arc = analysis.get("emotional_arc") or {}
    narrative = analysis.get("narrative") or {}
    analysis_version = analysis.get("analysis_version") or "analyzer_v5"

    # Save dream — store JSON-able fields as strings
    try:
        dream = Dream(
            title=title,
            content=content,
            mood=dominant_emotion,
            summary=summary,

            themes=json.dumps(themes_list),
            symbols=json.dumps(symbols_list),
            combined_insights=json.dumps(combined_insights_list),

            psychological_interpretation=json.dumps(psychological_interpretation) if psychological_interpretation else None,

            structured_interpretation=json.dumps(structured_interpretation) if structured_interpretation else None,
            structured_text=structured_text or None,

            user_id=request.user_id,

            events=json.dumps(events) if events else None,
            entities=json.dumps(entities) if entities else None,
            people=json.dumps(people) if people else None,
            locations=json.dumps(locations) if locations else None,
            objects=json.dumps(objects) if objects else None,
            cause_effect=json.dumps(cause_effect) if cause_effect else None,
            conflicts=json.dumps(conflicts) if conflicts else None,
            desires=json.dumps(desires) if desires else None,
            emotional_arc=json.dumps(emotional_arc) if emotional_arc else None,
            narrative=json.dumps(narrative) if narrative else None,
            analysis_version=analysis_version
        )

        db.session.add(dream)
        db.session.commit()
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Failed saving dream"}), 500

    # Return saved analysis back to client (structured fields included)
    return jsonify({
        "message": "Dream saved",
        "summary": summary,
        "emotions": emotions,
        "themes": themes_list,
        "symbols": symbols_list,
        "combined_insights": combined_insights_list,
        "psychological_interpretation": psychological_interpretation,
        "structured_interpretation": structured_interpretation,
        "structured_text": structured_text,
        "analysis_version": analysis_version
    }), 201


# ---------------------------------------
# GET DREAMS
# ---------------------------------------
@app.route('/get_dreams', methods=['GET'])
@auth_required
def get_dreams():
    try:
        dreams = Dream.query.filter_by(user_id=request.user_id).order_by(Dream.date.desc()).all()
    except Exception:
        traceback.print_exc()
        return jsonify([])

    result = []

    for d in dreams:
        result.append({
            "id": d.id,
            "title": d.title,
            "content": d.content,
            "mood": d.mood,
            "summary": d.summary,
            "date": d.date.strftime("%Y-%m-%d %H:%M:%S") if d.date else None,

            "themes": safe_json_load(d.themes) or [],
            "symbols": safe_json_load(d.symbols) or [],
            "combined_insights": safe_json_load(d.combined_insights) or [],

            "psychological_interpretation": safe_json_load(d.psychological_interpretation) or {},

            "structured_interpretation": safe_json_load(d.structured_interpretation) or {},
            "structured_text": d.structured_text or "",

            "events": safe_json_load(d.events) or [],
            "entities": safe_json_load(d.entities) or [],
            "people": safe_json_load(d.people) or [],
            "locations": safe_json_load(d.locations) or [],
            "objects": safe_json_load(d.objects) or [],
            "cause_effect": safe_json_load(d.cause_effect) or [],
            "conflicts": safe_json_load(d.conflicts) or [],
            "desires": safe_json_load(d.desires) or [],
            "emotional_arc": safe_json_load(d.emotional_arc) or {},
            "narrative": safe_json_load(d.narrative) or {},

            "analysis_version": d.analysis_version,
        })

    return jsonify(result)


# ---------------------------------------
# DELETE DREAM
# ---------------------------------------
@app.route('/delete_dream/<int:id>', methods=['DELETE'])
@auth_required
def delete_dream(id):
    dream = Dream.query.get_or_404(id)
    if dream.user_id != request.user_id:
        return jsonify({"error": "Unauthorized"}), 403

    try:
        db.session.delete(dream)
        db.session.commit()
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Failed to delete"}), 500

    return jsonify({"message": "Dream deleted"}), 200


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
