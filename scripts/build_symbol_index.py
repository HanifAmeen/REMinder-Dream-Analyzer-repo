import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.symbol_index import load_symbol_csv, build_symbol_index

CSV = r"C:\Users\amjad\Downloads\Research Papers 2025\Dream Journal\Datasets\cleaned_dream_interpretations.csv"

df = load_symbol_csv(CSV)
build_symbol_index(df, persist_dir="models/symbol_index")
print("Symbol index built.")
