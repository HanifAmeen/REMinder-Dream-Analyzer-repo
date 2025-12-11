from transformers import pipeline
from keybert import KeyBERT
import pandas as pd
import re
import string
from itertools import combinations
import random
import traceback

# --- Load dream dictionary ---
dict_path = r"C:\Users\amjad\Downloads\Research Papers 2025\Dream Journal\Datasets\cleaned_dream_interpretations.csv"
dream_dict_df = pd.read_csv(dict_path)
dream_dict_df = dream_dict_df.loc[:, ~dream_dict_df.columns.str.contains('^Unnamed|^$', case=False)]
dream_dict_df.columns = [c.strip().lower() for c in dream_dict_df.columns]

# --- Initialize NLP models (loaded once) ---
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception as e:
    print("[analyzer] Summarizer load failed:", e)
    summarizer = None

try:
    emotion_classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )
except Exception as e:
    print("[analyzer] Emotion classifier load failed:", e)
    emotion_classifier = None

try:
    kw_model = KeyBERT()
except Exception as e:
    print("[analyzer] KeyBERT load failed:", e)
    kw_model = None

# -------------------------
# Helper utilities
# -------------------------
def safe_first_sentence(text):
    """Return the first complete sentence from text, or a cleaned truncated fallback."""
    if not text:
        return ""
    # Replace newlines with spaces, trim
    t = " ".join(text.split())
    # Find sentence-ending punctuation
    m = re.search(r'([^.?!]*[.?!])', t)
    if m:
        return m.group(0).strip()
    # fallback: return up to 200 chars without breaking words
    if len(t) <= 200:
        return t
    else:
        part = t[:200]
        # don't cut in the middle of a word
        return re.sub(r'\s+\S+$', '', part).strip() + "..."

# -------------------------
# NLP helper functions with safe fallbacks
# -------------------------
def summarize_dream(text):
    """Generate a short summary of the dream; fallback to truncation if summarizer fails."""
    try:
        if summarizer is None:
            raise RuntimeError("summarizer not available")
        summary = summarizer(text, max_length=60, min_length=20, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print("[analyzer] summarize_dream error:", e)
        # fallback: return first 120 chars (clean)
        clean = " ".join(text.strip().split())
        return (clean[:120] + ("..." if len(clean) > 120 else ""))

def detect_emotion(text):
    """Detect dominant emotion from dream text; returns dict with dominant and scores."""
    try:
        if emotion_classifier is None:
            raise RuntimeError("emotion_classifier not available")
        results = emotion_classifier(text)[0]
        top_emotion = max(results, key=lambda x: x.get('score', 0))
        return {"dominant": top_emotion['label'], "scores": results}
    except Exception as e:
        print("[analyzer] detect_emotion error:", e)
        # fallback: neutral structure
        return {"dominant": "neutral", "scores": []}

def extract_themes(text):
    """Extract key themes or topics from the dream; fallback to simple heuristics."""
    try:
        if kw_model is None:
            raise RuntimeError("kw_model not available")
        keywords = kw_model.extract_keywords(text, top_n=5)
        return [kw[0] for kw in keywords]
    except Exception as e:
        print("[analyzer] extract_themes error:", e)
        # fallback: simple heuristic using common words (very light)
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:5]]

# -------------------------
# Dream symbol matching
# -------------------------
def interpret_symbols(text):
    """Identify dream symbols and their interpretations (whole-word matching).
       Returns list of dicts: {symbol, meaning} where meaning is the first sentence.
    """
    try:
        matches = []
        # lowercase and remove punctuation except keep internal hyphens (common in some symbols)
        translator = str.maketrans('', '', string.punctuation.replace('-', ''))
        text_clean = text.lower().translate(translator)
        # iterate dictionary
        for _, row in dream_dict_df.iterrows():
            symbol = str(row.get('word', '')).lower().strip()
            meaning_raw = str(row.get('interpretation', '')).strip()
            if not symbol:
                continue
            # boundary match to avoid partial matches (e.g., 'day' in 'yesterday')
            pattern = r'\b' + re.escape(symbol) + r'\b'
            if re.search(pattern, text_clean):
                meaning_short = safe_first_sentence(meaning_raw)
                matches.append({"symbol": symbol, "meaning": meaning_short})
        return matches
    except Exception as e:
        print("[analyzer] interpret_symbols error:", e)
        traceback.print_exc()
        return []

# -------------------------
# Ranking and context
# -------------------------
def rank_symbols_by_context(text, symbols):
    """Weight dream symbols based on whole-word frequency and emotional proximity."""
    try:
        text_lower = text.lower()
        ranked = []
        for s in symbols:
            sym = s.get("symbol", "")
            # count whole-word occurrences
            count = len(re.findall(rf'\b{re.escape(sym)}\b', text_lower))
            score = count
            # proximity heuristic: if emotion words nearby, boost weight
            if re.search(rf"(fear|love|death|pain|joy|anger|sad|happy).*?{re.escape(sym)}", text_lower) or \
               re.search(rf"{re.escape(sym)}.*?(fear|love|death|pain|joy|anger|sad|happy)", text_lower):
                score += 2
            ranked.append({**s, "weight": score})
        return sorted(ranked, key=lambda x: x.get("weight", 0), reverse=True)
    except Exception as e:
        print("[analyzer] rank_symbols_by_context error:", e)
        return symbols

# -------------------------
# Archetypes, coherence, compare
# -------------------------
archetype_map = {
    "shadow": ["snake", "darkness", "monster", "mirror"],
    "anima": ["woman", "water", "moon", "emotion"],
    "animus": ["man", "fire", "war", "control"],
    "self": ["circle", "mandala", "sun", "unity"],
    "persona": ["mask", "clothes", "actor", "crowd"]
}

def detect_archetype(symbols):
    """Identify dominant Jungian archetype from matched symbols."""
    try:
        counts = {a: 0 for a in archetype_map}
        for sym in symbols:
            sname = sym.get("symbol")
            for arch, words in archetype_map.items():
                if sname in words:
                    counts[arch] += 1
        dominant = max(counts, key=counts.get)
        return dominant if counts[dominant] > 0 else None
    except Exception as e:
        print("[analyzer] detect_archetype error:", e)
        return None

def compute_coherence_score(themes, symbols, emotion_scores):
    """Quantify dream coherence by density and emotional intensity."""
    try:
        if not emotion_scores:
            return 0
        density = len(symbols) / (len(themes) + 1)
        intensity = max((e.get('score', 0) for e in emotion_scores), default=0)
        return round((density * 0.5 + intensity * 0.5) * 100, 2)
    except Exception as e:
        print("[analyzer] compute_coherence_score error:", e)
        return 0

def compare_with_previous(text, previous_dreams):
    """Find recurring symbols from previous dreams. previous_dreams should be list of dicts with 'symbols'."""
    try:
        if not previous_dreams:
            return []
        prev_symbols = set()
        for d in previous_dreams:
            if isinstance(d, dict):
                for s in d.get('symbols', []):
                    if isinstance(s, dict):
                        prev_symbols.add(s.get('symbol'))
                    elif isinstance(s, str):
                        prev_symbols.add(s)
        current_symbols = set([s['symbol'] for s in interpret_symbols(text)])
        overlap = prev_symbols.intersection(current_symbols)
        return list(overlap)
    except Exception as e:
        print("[analyzer] compare_with_previous error:", e)
        return []

# -------------------------
# NEW: Unified Combined Insight
# -------------------------
def generate_combined_insights_unified(symbols, dominant_emotion=None):
    """
    Produce one cohesive Jung/Freud-style paragraph that synthesizes the set of symbols.
    Returns a single-element list containing a dict: {"symbols": [...], "insight": "..."}
    """
    try:
        if not symbols:
            return []

        # Use top N weighted symbols (already ranked if you call with ranked list)
        top_symbols = [s["symbol"] for s in symbols[:8]]  # limit to 8 to keep paragraph focused
        # Gather short meanings (first-sentence) and build a small context sentence list
        short_meanings = [s.get("meaning", "") for s in symbols[:8]]

        # Compose combined narrative ensuring no truncation of words
        title_part = " + ".join(top_symbols)
        meanings_joined = " ".join([m if m.endswith(('.', '?', '!')) else m + "." for m in short_meanings])
        meanings_joined = re.sub(r'\s+', ' ', meanings_joined).strip()

        # Choose interpretive style
        mood = (dominant_emotion or "").lower()
        if mood in ['fear', 'anger', 'disgust', 'sadness']:
            school = "Freudian"
            framing = "This pattern suggests repressed or unresolved emotional material."
        else:
            school = "Jungian"
            framing = "This pattern suggests deeper archetypal or individuation processes."

        # Build the paragraph: 3 sentences (context, interpretation, suggestion)
        sentence1 = f"You dreamed of {', '.join(top_symbols)} — symbols that appear together in this scene."
        sentence2 = (
            f"In context: {meanings_joined} From a {school} perspective, {framing.lower()}"
        )
        # Practical suggestion (actionable, not prescriptive)
        sentence3 = "Consider reflecting on which of these symbols best maps to an active concern in your waking life — that link often points to where change or attention is needed."

        insight_text = " ".join([sentence1, sentence2, sentence3])
        # Clean whitespace
        insight_text = re.sub(r'\s+', ' ', insight_text).strip()

        return [{"symbols": top_symbols, "insight": insight_text}]
    except Exception as e:
        print("[analyzer] generate_combined_insights_unified error:", e)
        traceback.print_exc()
        return []

# -------------------------
# Master Dream Analyzer
# -------------------------
def analyze_dream(text, previous_dreams=None):
    """Analyze a dream holistically with robust fallbacks and a single combined insight."""
    result = {
        "summary": "",
        "emotions": {"dominant": "neutral", "scores": []},
        "themes": [],
        "symbols": [],
        "combined_insights": [],
        "archetype": None,
        "coherence_score": 0,
        "recurring_symbols": []
    }

    if not text or not text.strip():
        return result

    try:
        result["summary"] = summarize_dream(text)
    except Exception as e:
        print("[analyzer] summarize fallback:", e)

    try:
        result["emotions"] = detect_emotion(text)
    except Exception as e:
        print("[analyzer] detect_emotion fallback:", e)

    try:
        result["themes"] = extract_themes(text) or []
    except Exception as e:
        print("[analyzer] extract_themes fallback:", e)
        result["themes"] = []

    try:
        raw_symbols = interpret_symbols(text) or []
        result["symbols"] = raw_symbols
    except Exception as e:
        print("[analyzer] interpret_symbols fallback:", e)
        result["symbols"] = []

    try:
        # Rank symbols by context (adds weight field)
        result["symbols"] = rank_symbols_by_context(text, result["symbols"])
    except Exception as e:
        print("[analyzer] rank_symbols_by_context fallback:", e)

    try:
        # Generate one unified combined insight paragraph
        result["combined_insights"] = generate_combined_insights_unified(
            result["symbols"], result["emotions"].get("dominant")
        )
    except Exception as e:
        print("[analyzer] combined_insights fallback:", e)
        result["combined_insights"] = []

    try:
        result["archetype"] = detect_archetype(result["symbols"])
    except Exception as e:
        print("[analyzer] archetype fallback:", e)
        result["archetype"] = None

    try:
        result["coherence_score"] = compute_coherence_score(
            result["themes"], result["symbols"], result["emotions"].get("scores", [])
        )
    except Exception as e:
        print("[analyzer] coherence_score fallback:", e)
        result["coherence_score"] = 0

    try:
        result["recurring_symbols"] = compare_with_previous(text, previous_dreams)
    except Exception as e:
        print("[analyzer] recurring_symbols fallback:", e)
        result["recurring_symbols"] = []

    return result
