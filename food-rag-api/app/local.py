# local_check.py

import pandas as pd
import faiss
import os

# Path to files (adjust if needed)
parquet_path = r"D:\\FAST-NUCES\Data viz\AI\AI-Powered-Recipe-System\\food-rag-api\\app\\assets\\recipes_with_ids.parquet"
faiss_index_path = r"D:\\FAST-NUCES\Data viz\AI\AI-Powered-Recipe-System\\food-rag-api\\app\\assets\\recipe_index.faiss"

# Check Parquet file
try:
    df = pd.read_parquet(parquet_path)
    print(f"✅ Parquet file loaded: {len(df)} recipes found.")
except Exception as e:
    print(f"❌ Error reading Parquet file: {e}")

# Check FAISS index
try:
    if not os.path.exists(faiss_index_path):
        raise FileNotFoundError(f"File not found: {faiss_index_path}")
    
    index = faiss.read_index(faiss_index_path)
    print(f"✅ FAISS index loaded: {index.ntotal} vectors stored.")
except Exception as e:
    print(f"❌ Error reading FAISS index: {e}")
