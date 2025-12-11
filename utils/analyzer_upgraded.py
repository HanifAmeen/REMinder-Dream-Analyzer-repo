# utils/analyzer_upgraded.py (UPDATED)
import os
import re
import string
import traceback
import json
from typing import List, Dict, Any

import numpy as np

from utils.symbol_index import ensure_index, load_symbol_index
from utils.ner_and_utils import (
    safe_first_sentence,
    chunked_summarize,
    detect_emotion_text,
    extract_keywords,
    extract_entities,
    get_sbert,
    get_spacy,
)

# CONFIG - update path if required
SYMBOL_CSV_PATH = os.environ.get("SYMBOL_CSV_PATH") or r"C:\Users\amjad\Downloads\Research Papers 2025\Dream Journal\Datasets\cleaned_dream_interpretations.csv"
PERSIST_DIR = os.environ.get("SYMBOL_INDEX_DIR", "models/symbol_index")

# load symbol index (fast if already built)
try:
    SYMBOL_DF, SYMBOL_EMB, SYMBOL_NN = ensure_index(SYMBOL_CSV_PATH, PERSIST_DIR)
except Exception as e:
    SYMBOL_DF, SYMBOL_EMB, SYMBOL_NN = None, None, None
    print("[analyzer_upgraded] symbol index not loaded at import:", e)

SBERT = get_sbert()
SPACY_NLP = get_spacy()

# ---------- symbol matching (same as you had) ----------
def exact_match_symbols(text: str) -> List[Dict[str,Any]]:
    translator = str.maketrans('', '', string.punctuation.replace('-', ''))
    text_clean = str(text).lower().translate(translator)
    matches = []
    if SYMBOL_DF is None:
        return matches
    for _, row in SYMBOL_DF.iterrows():
        symbol = row['word_clean']
        if not symbol:
            continue
        patterns = [rf'\b{re.escape(symbol)}\b']
        if not symbol.endswith('s'):
            patterns.append(rf'\b{re.escape(symbol)}s\b')
        for p in patterns:
            if re.search(p, text_clean):
                # mark exact matches with a high semantic_score so they rank highly
                matches.append({
                    "symbol": symbol,
                    "meaning": row.get('interp_first', ''),
                    "match_type": "exact",
                    "semantic_score": 0.95
                })
                break
    return matches

def semantic_match_symbols(text: str, top_k=12, score_threshold=0.40) -> List[Dict[str,Any]]:
    if SYMBOL_DF is None or SYMBOL_NN is None or SBERT is None:
        return []
    txt = " ".join(str(text).split()).lower()
    emb = SBERT.encode(txt, convert_to_numpy=True)
    distances, idxs = SYMBOL_NN.kneighbors([emb], n_neighbors=min(top_k, len(SYMBOL_EMB)), return_distance=True)
    results = []
    for dist, idx in zip(distances[0], idxs[0]):
        score = float(1 - dist)
        if score < score_threshold:
            continue
        row = SYMBOL_DF.iloc[idx]
        results.append({
            "symbol": row['word_clean'],
            "meaning": row.get('interp_first', ''),
            "semantic_score": score,
            "match_type": "semantic"
        })
    return results

def rank_symbols(text: str, matches: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    text_lower = str(text).lower()
    ranked = []
    for m in matches:
        sym = m.get('symbol', '')
        count = len(re.findall(rf'\b{re.escape(sym)}\b', text_lower))
        semantic = m.get('semantic_score', 0)
        weight = semantic * 100 + count * 4
        # emotional proximity bonus (unchanged)
        if re.search(rf"(fear|love|death|pain|joy|anger|sad|happy).*?{re.escape(sym)}", text_lower) or \
           re.search(rf"{re.escape(sym)}.*?(fear|love|death|pain|joy|anger|sad|happy)", text_lower):
            weight += 6
        item = {**m, "weight": round(float(weight), 3), "count": count}
        ranked.append(item)
    return sorted(ranked, key=lambda x: x.get("weight", 0), reverse=True)

# ---------- new: bucketing ----------
def bucket_symbols_by_weight(ranked_symbols: List[Dict[str,Any]]) -> (List[Dict[str,Any]], List[Dict[str,Any]], List[Dict[str,Any]]):
    """
    Buckets ranked symbols into primary / secondary / noise based on weight thresholds:
      - primary: weight >= 90
      - secondary: 75 <= weight < 90
      - noise: weight < 75
    """
    primary = []
    secondary = []
    noise = []
    for s in ranked_symbols:
        try:
            w = float(s.get("weight", 0))
        except Exception:
            w = 0.0
        if w >= 90:
            primary.append(s)
        elif 75 <= w < 90:
            secondary.append(s)
        else:
            noise.append(s)
    return primary, secondary, noise

# ---------- event extraction / entities / narrative ----------
def extract_entities_structured(text: str) -> Dict[str, List[Dict[str,str]]]:
    nlp = SPACY_NLP
    if not nlp:
        return {"entities": []}
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({"text": ent.text, "label": ent.label_})
    return {"entities": entities}

def extract_people_locations_objects(text: str) -> Dict[str, List[str]]:
    nlp = SPACY_NLP
    if not nlp:
        return {"people": [], "locations": [], "objects": []}
    doc = nlp(text)
    people, locations, objects = [], [], []
    for ent in doc.ents:
        lab = ent.label_
        txt = ent.text.strip()
        if lab in ("PERSON", "NORP"):
            people.append(txt)
        elif lab in ("GPE", "LOC", "FAC"):
            locations.append(txt)
        elif lab in ("PRODUCT", "WORK_OF_ART"):
            objects.append(txt)
    # remove duplicates preserving order
    def unique(seq):
        seen = set()
        out = []
        for s in seq:
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return out
    return {"people": unique(people), "locations": unique(locations), "objects": unique(objects)}

def extract_events(text: str) -> List[Dict[str,str]]:
    """Rule-based extraction of simple SVO events from sentences using spaCy dependency parse."""
    nlp = SPACY_NLP
    if not nlp:
        return []
    doc = nlp(text)
    events = []
    for sent in doc.sents:
        subject = None
        verb = None
        dobj = None
        # find verb token in sentence (first ROOT or VERB)
        for token in sent:
            if token.dep_ == "ROOT" or token.pos_.startswith("VERB"):
                verb = token.lemma_
                # find subject and object around this verb
                for child in token.children:
                    if child.dep_ in ("nsubj","nsubjpass","csubj"):
                        subject = child.text
                    if child.dep_ in ("dobj","obj","pobj"):
                        dobj = child.text
                break
        text_sent = sent.text.strip()
        if verb:
            events.append({
                "sentence": text_sent,
                "actor": subject or "",
                "action": verb,
                "object": dobj or ""
            })
    return events

def detect_cause_effect(text: str) -> List[Dict[str,str]]:
    """Very small rule-based cause-effect detection using conjunctions and temporal cues."""
    causes = []
    # split sentences and look for markers
    markers = ["because", "because of", "due to", "after", "when", "so that", "therefore", "as a result", "leading to"]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for i, s in enumerate(sentences):
        low = s.lower()
        for m in markers:
            if m in low:
                # naive split
                parts = re.split(re.escape(m), s, flags=re.IGNORECASE)
                if len(parts) >= 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    causes.append({"trigger_phrase": m, "left": left, "right": right, "sentence": s.strip()})
                else:
                    causes.append({"trigger_phrase": m, "sentence": s.strip()})
                break
    return causes

def detect_conflicts_and_desires(text: str) -> Dict[str, List[str]]:
    """Heuristic detection of conflict or desire phrases (rules)."""
    conflict_keywords = ["chase", "attack", "fight", "afraid", "scared", "escape", "lost", "fail", "argue", "danger"]
    desire_keywords = ["want", "need", "wish", "hope", "long for", "desire"]
    conflicts = []
    desires = []
    low = text.lower()
    for kw in conflict_keywords:
        if kw in low:
            conflicts.append(kw)
    for kw in desire_keywords:
        if kw in low:
            desires.append(kw)
    return {"conflicts": list(set(conflicts)), "desires": list(set(desires))}

def emotional_arc(text: str) -> Dict[str, Any]:
    """Split into sentences and get emotion per sentence to form a simple arc."""
    try:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        arc = []
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            emo = detect_emotion_text(s_clean)
            arc.append({"sentence": s_clean, "dominant": emo.get("dominant"), "scores": emo.get("scores")})
        # summarize trend: count of negative vs positive labels
        neg = sum(1 for a in arc if a["dominant"].lower() in ("fear","anger","sadness","disgust"))
        pos = sum(1 for a in arc if a["dominant"].lower() in ("joy","surprise","love","happy"))
        trend = "neutral"
        if neg > pos and neg - pos >= 1:
            trend = "negative"
        elif pos > neg and pos - neg >= 1:
            trend = "positive"
        return {"arc": arc, "trend": trend, "neg_count": neg, "pos_count": pos}
    except Exception:
        return {"arc": [], "trend": "neutral", "neg_count": 0, "pos_count": 0}

def detect_narrative_structure(text: str) -> Dict[str,str]:
    """Very basic heuristic: take first sentence as setup, longest sentence as climax, last as resolution (if present)."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sents = [s.strip() for s in sentences if s.strip()]
    if not sents:
        return {"setup": "", "climax": "", "resolution": ""}
    setup = sents[0]
    climax = max(sents, key=lambda x: len(x)) if sents else ""
    resolution = sents[-1] if len(sents) > 1 else ""
    return {"setup": safe_first_sentence(setup), "climax": safe_first_sentence(climax), "resolution": safe_first_sentence(resolution)}

# ---------- combined insights (leveraging your previous approach) ----------
def combined_insights_from_symbols(symbols, dominant_emotion=None):
    if not symbols:
        return []
    top = symbols[:8]
    syms = [s['symbol'] for s in top]
    meanings = [s.get('meaning','') for s in top]
    meanings_clean = []
    for m in meanings:
        if not m: continue
        m = m.strip()
        if not m.endswith(('.', '?', '!')):
            m = m.rstrip('. ') + "."
        meanings_clean.append(m)
    meanings_joined = " ".join(meanings_clean)
    mood = (dominant_emotion or "").lower()
    if mood in ['fear', 'anger', 'disgust', 'sadness']:
        school = "Freudian"
        framing = "This pattern suggests repressed or unresolved emotional material."
    else:
        school = "Jungian"
        framing = "This pattern suggests deeper archetypal or individuation processes."
    sentence1 = f"You dreamed of {', '.join(syms)} — symbols appearing together in this scene."
    sentence2 = f"In context: {meanings_joined} From a {school} perspective, {framing.lower()}"
    sentence3 = "Consider reflecting on which of these symbols best maps to an active concern in your waking life — that link often points to where change or attention is needed."
    insight_text = " ".join([sentence1, sentence2, sentence3])
    return [{"symbols": syms, "insight": insight_text}]

# ---------- master analyze ----------
def analyze_dream(text: str, previous_dreams=None, use_llm_fallback=False) -> Dict[str,Any]:
    """
    Returns a dictionary with all fields (backwards compatible).
    Adds:
      - events
      - entities
      - people, locations, objects
      - cause_effect
      - conflicts/desires
      - emotional_arc
      - narrative (setup/climax/resolution)
      - analysis_version
      - symbols_primary / secondary / noise (new)
    """
    result = {
        "summary": "",
        "emotions": {"dominant": "neutral", "scores": []},
        "themes": [],
        "symbols": [],
        "symbols_primary": [],
        "symbols_secondary": [],
        "symbols_noise": [],
        "combined_insights": [],
        "archetype": None,
        "coherence_score": 0,
        "recurring_symbols": [],
        # NEW fields:
        "events": [],
        "entities": [],
        "people": [],
        "locations": [],
        "objects": [],
        "cause_effect": [],
        "conflicts": [],
        "desires": [],
        "emotional_arc": {},
        "narrative": {},
        "analysis_version": "analyzer_upgraded_v2"
    }

    if not text or not str(text).strip():
        return result

    try:
        result["summary"] = chunked_summarize(text)
    except Exception as e:
        print("[analyzer_upgraded] summary error:", e)
        result["summary"] = safe_first_sentence(text)

    try:
        result["emotions"] = detect_emotion_text(text)
    except Exception as e:
        print("[analyzer_upgraded] emotion error:", e)

    try:
        result["themes"] = extract_keywords(text, top_n=6) or []
    except Exception as e:
        print("[analyzer_upgraded] themes error:", e)
        result["themes"] = []

    # structured extraction
    try:
        ents_struct = extract_entities_structured(text)
        result["entities"] = ents_struct.get("entities", [])
        ppl_loc_obj = extract_people_locations_objects(text)
        result["people"] = ppl_loc_obj.get("people", [])
        result["locations"] = ppl_loc_obj.get("locations", [])
        result["objects"] = ppl_loc_obj.get("objects", [])
        result["events"] = extract_events(text)
        result["cause_effect"] = detect_cause_effect(text)
        cd = detect_conflicts_and_desires(text)
        result["conflicts"] = cd.get("conflicts", [])
        result["desires"] = cd.get("desires", [])
        result["emotional_arc"] = emotional_arc(text)
        result["narrative"] = detect_narrative_structure(text)
    except Exception as e:
        print("[analyzer_upgraded] structured extraction error:", e)

    # symbols: union of semantic + exact, then rank, then bucket
    try:
        sem = semantic_match_symbols(text, top_k=20, score_threshold=0.40)
        exacts = exact_match_symbols(text)
        merged = {s['symbol']: s for s in sem}
        for e in exacts:
            sym = e['symbol']
            if sym in merged:
                # preserve meaning if present, mark exact+semantic and boost semantic_score
                merged[sym]['meaning'] = merged[sym].get('meaning') or e.get('meaning')
                merged[sym]['match_type'] = 'exact+semantic'
                merged[sym]['semantic_score'] = max(merged[sym].get('semantic_score', 0), 0.95)
            else:
                merged[sym] = {**e, 'semantic_score': 0.95}
        candidates = list(merged.values())
        ranked = rank_symbols(text, candidates)

        # bucket by weight thresholds: primary >=90, secondary 75-89, noise <75
        primary, secondary, noise = bucket_symbols_by_weight(ranked)

        result["symbols"] = ranked
        result["symbols_primary"] = primary
        result["symbols_secondary"] = secondary
        result["symbols_noise"] = noise

    except Exception as e:
        print("[analyzer_upgraded] symbol error:", e)
        result["symbols"] = []
        result["symbols_primary"] = []
        result["symbols_secondary"] = []
        result["symbols_noise"] = []

    try:
        result["combined_insights"] = combined_insights_from_symbols(result["symbols"], result["emotions"].get("dominant"))
    except Exception as e:
        print("[analyzer_upgraded] combined_insights error:", e)
        result["combined_insights"] = []

    try:
        # archetype detection on top-n symbols (reuse small mapping)
        arch = None
        try:
            # small mapping
            archetype_map = {
                "shadow": ["snake","darkness","monster","mirror"],
                "anima": ["woman","water","moon","emotion"],
                "animus": ["man","fire","war","control"],
                "self": ["circle","mandala","sun","unity"],
                "persona": ["mask","clothes","actor","crowd"]
            }
            counts = {a:0 for a in archetype_map}
            for s in result.get("symbols", []):
                for arch_name, words in archetype_map.items():
                    if s.get("symbol") in words:
                        counts[arch_name] += 1
            dom = max(counts, key=counts.get)
            arch = dom if counts[dom] > 0 else None
        except Exception:
            arch = None
        result["archetype"] = arch
    except Exception as e:
        print("[analyzer_upgraded] archetype error:", e)
        result["archetype"] = None

    try:
        result["coherence_score"] = 0
        if result.get("emotional_arc"):
            emotion_scores = result["emotional_arc"].get("arc", [])
        else:
            emotion_scores = []
        result["coherence_score"] = 0  # keep old behavior or compute later
    except Exception as e:
        result["coherence_score"] = 0

    try:
        # recurring symbols
        prev = previous_dreams or []
        prev_syms = set()
        for d in prev:
            try:
                for s in d.get('symbols', []):
                    if isinstance(s, dict):
                        prev_syms.add(s.get('symbol'))
                    elif isinstance(s, str):
                        prev_syms.add(s)
            except Exception:
                pass
        curr_syms = set([s['symbol'] for s in result.get("symbols", [])])
        result["recurring_symbols"] = list(prev_syms.intersection(curr_syms))
    except Exception as e:
        print("[analyzer_upgraded] recurring symbols error:", e)
        result["recurring_symbols"] = []

    return result
