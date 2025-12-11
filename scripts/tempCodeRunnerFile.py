# scripts/build_symbol_index.py
from utils.symbol_index import load_symbol_csv, build_symbol_index
CSV = r"C:\Users\amjad\Downloads\Research Papers 2025\Datasets\cleaned_dream_interpretations.csv"
df = load_symbol_csv(CSV)
build_symbol_index(df, persist_dir="models/symbol_index")
print("Symbol index built.")
