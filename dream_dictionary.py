import pandas as pd
import re
import os
import string

# --- Load CSV dynamically relative to this file ---
dict_path = r"C:\Users\amjad\Downloads\Research Papers 2025\Dream Journal\Datasets\cleaned_dream_interpretations.csv"

dream_dict_df = pd.read_csv(dict_path)

# Drop empty / unnamed columns
dream_dict_df = dream_dict_df.loc[:, ~dream_dict_df.columns.str.contains('^Unnamed|^$', case=False)]

# Normalize column names
dream_dict_df.columns = [c.strip().lower() for c in dream_dict_df.columns]

def interpret_dream_text(text, dream_dict_df):
    """
    Return a list of symbols found in `text` with their meanings from the CSV.
    """
    interpretations = []

    # Normalize text: lowercase, remove punctuation
    text_lower = text.lower()
    text_lower = text_lower.translate(str.maketrans('', '', string.punctuation))

    for _, row in dream_dict_df.iterrows():
        symbol = str(row['word']).lower().strip()
        meaning = str(row['interpretation']).strip()

        # Normalize symbol
        symbol_clean = symbol.translate(str.maketrans('', '', string.punctuation))

        if re.search(rf'\b{re.escape(symbol_clean)}\b', text_lower):
            interpretations.append({"symbol": symbol, "meaning": meaning})

    return interpretations
