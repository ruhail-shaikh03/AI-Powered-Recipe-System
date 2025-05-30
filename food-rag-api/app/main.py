# app/main.py
import os
import ast
import time
import gc
import logging

import pandas as pd
import numpy as np
from tqdm.auto import tqdm # Optional: for progress during initial processing if run

from sentence_transformers import SentenceTransformer
import faiss
from ctransformers import AutoModelForCausalLM
from huggingface_hub import hf_hub_download

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# --- Configuration ---
# Default MAX_RECIPES. If your pre-downloaded parquet/faiss used a different number,
# ensure this matches or the app might try to re-process.
# Can be overridden by environment variable MAX_RECIPES.
DEFAULT_MAX_RECIPES = 200000 # Set to the number of recipes your .parquet and .faiss files correspond to.
                            # If you used 20000 in Kaggle, set this to 20000.
                            # Set to a small number (e.g., 5000) for faster local dev if generating from scratch.
MAX_RECIPES = int(os.getenv("MAX_RECIPES", DEFAULT_MAX_RECIPES))

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
# Default LLM file, can be overridden by environment variable LLM_MODEL_FILE
LLM_MODEL_FILE = os.getenv("LLM_MODEL_FILE", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")

# Paths relative to this main.py file (inside app/)
BASE_ASSET_PATH = "./assets"
BASE_DATA_PATH = "./data"

FAISS_INDEX_PATH = os.path.join(BASE_ASSET_PATH, "recipe_index.faiss")
RECIPE_DATA_PATH = os.path.join(BASE_ASSET_PATH, "recipes_with_ids.parquet")
RAW_RECIPES_CSV_PATH = os.path.join(BASE_DATA_PATH, "RAW_recipes.csv")

# Ensure asset directory exists (data directory should be mounted with RAW_recipes.csv if needed)
os.makedirs(BASE_ASSET_PATH, exist_ok=True)

# --- Global Variables for Models and Data ---
recipes_df: Optional[pd.DataFrame] = None
embedding_model: Optional[SentenceTransformer] = None
faiss_index: Optional[faiss.Index] = None
llm: Optional[AutoModelForCausalLM] = None
user_data: Dict[str, Dict[str, Any]] = {
    'user1': { # Example user
        'favorites': [],
        'preferences': {'dietary': [], 'cuisine': [], 'allergies': []}
    }
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

class RecipeResult(BaseModel):
    id: int
    name: str
    distance: Optional[float] = None
    ingredients: Optional[str] = None
    steps: Optional[str] = None
    combined_text: Optional[str] = None # For RAG context

class RagQuery(BaseModel):
    user_id: str
    user_query: str
    top_k_retrieval: int = 10

class RagResponse(BaseModel):
    recommendation: str
    generation_time_seconds: float

class GenerateQuery(BaseModel):
    ingredients_list_str: str

class GenerateResponse(BaseModel):
    generated_recipe: str
    generation_time_seconds: float

class UserPreferenceRequest(BaseModel):
    preferences: Dict[str, List[str]]

class FavoriteRequest(BaseModel):
    recipe_id: int

# --- Core Logic ---

def _load_raw_and_preprocess():
    """Loads RAW_recipes.csv and preprocesses it."""
    global recipes_df
    logger.info(f"Attempting to load and preprocess from {RAW_RECIPES_CSV_PATH} with MAX_RECIPES={MAX_RECIPES}")
    try:
        df_raw = pd.read_csv(RAW_RECIPES_CSV_PATH)
    except FileNotFoundError:
        logger.error(f"ERROR: {RAW_RECIPES_CSV_PATH} not found. Cannot generate processed data.")
        return False

    logger.info(f"Loaded {len(df_raw)} raw recipes.")
    df_raw['name'] = df_raw['name'].fillna('Unnamed Recipe')
    df_raw['description'] = df_raw['description'].fillna('No description available.')
    df_raw['ingredients'] = df_raw['ingredients'].fillna("[]")
    df_raw['steps'] = df_raw['steps'].fillna("[]")

    # Use the MAX_RECIPES from environment or default for subsetting
    current_max_recipes = MAX_RECIPES
    if current_max_recipes is not None and current_max_recipes < len(df_raw):
        logger.info(f"Subsetting to {current_max_recipes} recipes.")
        df_raw = df_raw.sample(n=current_max_recipes, random_state=42).reset_index(drop=True)
    elif current_max_recipes is None: # Process all if MAX_RECIPES is None
        logger.info("Processing all recipes from CSV.")


    processed_recipes = []
    # Use tqdm only if you want progress in logs, can be verbose for API
    # for _, row in tqdm(df_raw.iterrows(), total=df_raw.shape[0], desc="Preprocessing Recipes"):
    for _, row in df_raw.iterrows():
        try:
            ingredients_list = ast.literal_eval(row['ingredients'])
            steps_list = ast.literal_eval(row['steps'])
            ingredients_str = ", ".join(ingredients_list)
            steps_str = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps_list)])
            combined_text = f"Recipe Title: {row['name']}\n\nDescription: {row['description']}\n\nIngredients: {ingredients_str}\n\nInstructions:\n{steps_str}"
            processed_recipes.append({
                'id': row['id'], 'name': row['name'], 'ingredients_str': ingredients_str,
                'steps_str': steps_str, 'combined_text_for_embedding': combined_text,
                'minutes': row['minutes'], 'n_steps': row['n_steps'], 'n_ingredients': row['n_ingredients']
            })
        except Exception as e:
            logger.warning(f"Skipping recipe id {row.get('id', 'Unknown')} during preprocessing: {e}")
    
    recipes_df = pd.DataFrame(processed_recipes)
    if recipes_df.empty:
        logger.error("Preprocessing resulted in an empty DataFrame.")
        return False
        
    logger.info(f"Successfully processed {len(recipes_df)} recipes.")
    recipes_df.to_parquet(RECIPE_DATA_PATH, index=False)
    logger.info(f"Processed recipe data saved to {RECIPE_DATA_PATH}")
    return True

def _build_faiss_index_from_df(current_recipes_df, current_embedding_model):
    """Builds FAISS index from the provided DataFrame and embedding model."""
    global faiss_index
    logger.info("Building FAISS index...")
    texts_to_embed = current_recipes_df['combined_text_for_embedding'].tolist()
    
    logger.info(f"Generating embeddings for {len(texts_to_embed)} recipes...")
    embeddings = current_embedding_model.encode(texts_to_embed, show_progress_bar=False, batch_size=32) # Progress bar off for API
    embedding_dim = embeddings.shape[1]
    logger.info(f"Embeddings generated. Shape: {embeddings.shape}")

    index = faiss.IndexFlatL2(embedding_dim)
    logger.info("Using CPU for FAISS index.")
    index.add(np.array(embeddings, dtype=np.float32))
    
    logger.info(f"FAISS index built. Total vectors: {index.ntotal}")
    faiss.write_index(index, FAISS_INDEX_PATH)
    logger.info(f"FAISS index saved to {FAISS_INDEX_PATH}")
    faiss_index = index
    return True

def load_llm_on_startup():
    global llm
    logger.info(f"Attempting to load LLM: {LLM_MODEL_REPO}/{LLM_MODEL_FILE}")
    try:
        model_path = hf_hub_download(repo_id=LLM_MODEL_REPO, filename=LLM_MODEL_FILE)
        logger.info(f"LLM model file path: {model_path}")
        config = {
            'max_new_tokens': 256, 
            'repetition_penalty': 1.1, 
            'temperature': 0.3, 
            'stream': False, 
            'context_length': 2048 # Reduced for local efficiency
        }
        # Forcing CPU for LLM on laptop/basic Docker for predictability.
        # Set gpu_layers > 0 only if you have a local GPU setup and understand Docker NVIDIA toolkit.
        llm = AutoModelForCausalLM.from_pretrained(model_path, model_type='mistral', gpu_layers=0, **config)
        logger.info(f"LLM loaded successfully on CPU.")
    except Exception as e:
        logger.error(f"Error loading LLM: {e}", exc_info=True)
        llm = None # Ensure llm is None if loading fails

# User Preference Logic (same as before)
def add_favorite_recipe_logic(user_id, recipe_id):
    if user_id not in user_data: user_data[user_id] = {'favorites': [], 'preferences': {'dietary': [], 'cuisine': [], 'allergies': []}}
    if recipe_id not in user_data[user_id]['favorites']:
        user_data[user_id]['favorites'].append(recipe_id)
        return True
    return False

def remove_favorite_recipe_logic(user_id, recipe_id):
    if user_id in user_data and recipe_id in user_data[user_id]['favorites']:
        user_data[user_id]['favorites'].remove(recipe_id)
        return True
    return False

def set_user_preferences_logic(user_id, preferences_payload: Dict[str, List[str]]):
    if user_id not in user_data: user_data[user_id] = {'favorites': [], 'preferences': {'dietary': [], 'cuisine': [], 'allergies': []}}
    for key, value in preferences_payload.items():
        current_prefs = user_data[user_id]['preferences'].get(key, [])
        user_data[user_id]['preferences'][key] = list(set(current_prefs + value))
    return user_data[user_id]['preferences']

def get_user_data_logic(user_id):
    return user_data.get(user_id)


# Core Search, RAG, Generate Logic (same as before, just using the globals)
def search_recipes_logic(query: str, top_k: int = 5) -> List[RecipeResult]:
    if faiss_index is None or embedding_model is None or recipes_df is None:
        logger.error("Search components (FAISS, embedding model, or recipes_df) not ready.")
        return []
    try:
        query_embedding = embedding_model.encode([query], convert_to_tensor=False) # Using global embedding_model
        distances, indices = faiss_index.search(np.array(query_embedding, dtype=np.float32), top_k) # Using global faiss_index
    except Exception as e:
        logger.error(f"Error during recipe search: {e}", exc_info=True)
        return []
    
    results = []
    # Using global recipes_df
    valid_indices = [idx for idx in indices[0] if 0 <= idx < len(recipes_df)] 
    for i, recipe_df_index in enumerate(valid_indices):
        recipe_details = recipes_df.iloc[recipe_df_index]
        results.append(RecipeResult(
            id=int(recipe_details['id']), # Ensure id is int
            name=str(recipe_details['name']),
            distance=float(distances[0][i]), # Use index 'i' for distances
            ingredients=str(recipe_details['ingredients_str']),
            steps=str(recipe_details['steps_str']),
            combined_text=str(recipe_details['combined_text_for_embedding'])
        ))
    return results

def recommend_recipe_with_rag_logic(user_id: str, user_query: str, top_k_retrieval: int = 10):
    if llm is None: return "LLM not available.", 0.0 # Using global llm
    
    user_prefs_data = get_user_data_logic(user_id)
    user_preferences_str = ""
    if user_prefs_data and user_prefs_data.get('preferences'):
        prefs = user_prefs_data['preferences']
        pref_strings = [f"{key}: {', '.join(values)}" for key, values in prefs.items() if values]
        if pref_strings: user_preferences_str = "User's General Preferences: " + "; ".join(pref_strings) + "\n"
    
    retrieved_docs = search_recipes_logic(user_query, top_k=top_k_retrieval)
    if not retrieved_docs: return "No recipes found to provide context for RAG.", 0.0

    context_parts = [
        f"<RECIPE_START id={doc.id}> Recipe Title: {doc.name}\nIngredients: {doc.ingredients}\nSteps:\n{doc.steps}\n<RECIPE_END>\n"
        for doc in retrieved_docs
    ]
    context_str = "\n".join(context_parts)
    # Simple truncation if context is too long (adjust max_context_chars as needed for your LLM)
    max_context_chars = 3000 # Example: ~3500-4000 tokens for Mistral, chars are rough
    if len(context_str) > max_context_chars:
        logger.warning(f"RAG context string is too long ({len(context_str)} chars), truncating to {max_context_chars}.")
        context_str = context_str[:max_context_chars]


    prompt = (
        f"<s>[INST] You are a recipe recommendation assistant. Your task is to find the BEST recipe from the provided context that closely matches the user's request and any stated preferences.\n\n"
        f"User Query: \"{user_query}\"\n"
        f"{user_preferences_str}\n" # This will be empty if no preferences
        f"Provided Recipes Context (analyze these):\n{context_str}\n\n"
        f"Based on the query and preferences, select the most relevant recipe. Present its Title, Ingredients, and Instructions clearly. If no suitable recipe is found in the context, state that.\n[/INST]"
    )
    
    logger.info(f"RAG: Sending prompt (context {len(context_str)} chars) to LLM.")
    start_time_llm = time.time()
    try:
        response_str = llm(prompt)
    except Exception as e:
        logger.error(f"Error during RAG LLM generation: {e}", exc_info=True)
        return f"Sorry, an error occurred during RAG generation: {e}", 0.0
    end_time_llm = time.time()
    llm_gen_time = end_time_llm - start_time_llm
    
    gc.collect()
    logger.info(f"RAG LLM generation took {llm_gen_time:.2f}s.")
    return response_str, llm_gen_time


def generate_recipe_from_ingredients_logic(ingredients_list_str: str):
    if llm is None: return "LLM not available.", 0.0 # Using global llm
    
    prompt = (
        f"<s>[INST] You are a creative recipe chef. Generate a plausible recipe based *primarily* on the following available ingredients: \"{ingredients_list_str}\".\n"
        f"You can assume common pantry staples like salt, pepper, oil, and water are available.\n"
        f"Please provide a catchy recipe Title, a list of all Ingredients (including any you add from pantry staples), and step-by-step cooking Instructions. Make the recipe sound appealing.\n[/INST]\n"
        f"Here is a unique recipe using {ingredients_list_str}:\n"
    )
    logger.info("Generating new recipe from ingredients...")
    start_time = time.time()
    try:
        response_str = llm(prompt)
    except Exception as e:
        logger.error(f"Error during new recipe LLM generation: {e}", exc_info=True)
        return f"Sorry, an error occurred during recipe generation: {e}", 0.0

    end_time = time.time()
    gen_time = end_time - start_time
    logger.info(f"LLM generation for new recipe took {gen_time:.2f}s.")
    return response_str, gen_time

# --- FastAPI App Definition and Endpoints ---
app = FastAPI(title="Food RAG API")

@app.on_event("startup")
async def startup_event():
    global recipes_df, embedding_model, faiss_index, llm
    logger.info("Application startup: Initializing components...")

    # 1. Load Embedding Model (needed for FAISS build if index is missing)
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
        logger.info("Embedding model loaded successfully on CPU.")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}", exc_info=True)
        # Potentially raise an error or exit if this is critical and cannot be recovered
        raise RuntimeError("Could not load embedding model. Application cannot start.")


    # 2. Load Processed Recipe Data (recipes_df)
    # Prioritize loading pre-generated parquet. If not found, try to generate from CSV.
    processed_data_loaded_or_generated = False
    if os.path.exists(RECIPE_DATA_PATH):
        logger.info(f"Attempting to load processed recipe data from {RECIPE_DATA_PATH}")
        try:
            recipes_df = pd.read_parquet(RECIPE_DATA_PATH)
            # Validate against MAX_RECIPES
            # If MAX_RECIPES is set and parquet file size doesn't match, it *might* indicate inconsistency.
            # For simplicity, we'll assume if it exists and MAX_RECIPES is for subsetting, it's fine.
            # If MAX_RECIPES is None, we load all.
            if MAX_RECIPES is not None and len(recipes_df) > MAX_RECIPES:
                logger.info(f"Loaded {len(recipes_df)} recipes from parquet, subsetting to MAX_RECIPES={MAX_RECIPES}.")
                recipes_df = recipes_df.sample(n=MAX_RECIPES, random_state=42).reset_index(drop=True)
            elif MAX_RECIPES is None:
                 logger.info(f"Loaded all {len(recipes_df)} recipes from parquet.")
            else: # MAX_RECIPES is set, and parquet has <= MAX_RECIPES
                 logger.info(f"Loaded {len(recipes_df)} recipes from parquet (MAX_RECIPES={MAX_RECIPES}).")
            
            processed_data_loaded_or_generated = True
            logger.info("Processed recipe data loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load {RECIPE_DATA_PATH}: {e}. Will try to generate if RAW_recipes.csv exists.", exc_info=True)
            recipes_df = None # Ensure it's None so generation is attempted

    if not processed_data_loaded_or_generated:
        logger.info(f"{RECIPE_DATA_PATH} not found or failed to load. Attempting to generate from raw data.")
        if os.path.exists(RAW_RECIPES_CSV_PATH):
            if _load_raw_and_preprocess(): # This sets global recipes_df and saves parquet
                processed_data_loaded_or_generated = True
            else:
                logger.error("Failed to generate processed data from CSV.")
        else:
            logger.warning(f"{RAW_RECIPES_CSV_PATH} not found. Cannot generate processed data.")

    if not processed_data_loaded_or_generated or recipes_df is None or recipes_df.empty:
        logger.critical("Recipe data (recipes_df) could not be loaded or generated. API cannot function correctly.")
        raise RuntimeError("Recipe data initialization failed.")

    # 3. Load FAISS Index
    # Prioritize loading pre-generated index. If not found, try to build.
    faiss_index_ready = False
    if os.path.exists(FAISS_INDEX_PATH):
        logger.info(f"Attempting to load FAISS index from {FAISS_INDEX_PATH}")
        try:
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            # Validate index size against recipes_df size
            if faiss_index.ntotal == len(recipes_df):
                logger.info(f"FAISS index loaded successfully with {faiss_index.ntotal} vectors.")
                faiss_index_ready = True
            else:
                logger.warning(f"FAISS index size ({faiss_index.ntotal}) mismatch with recipes_df ({len(recipes_df)}). Will attempt to rebuild.")
                faiss_index = None # Force rebuild
        except Exception as e:
            logger.error(f"Failed to load {FAISS_INDEX_PATH}: {e}. Will try to build.", exc_info=True)
            faiss_index = None

    if not faiss_index_ready:
        logger.info(f"FAISS index not loaded or needs rebuild. Attempting to build...")
        if embedding_model and recipes_df is not None and not recipes_df.empty:
            if _build_faiss_index_from_df(recipes_df, embedding_model): # This sets global faiss_index
                faiss_index_ready = True
            else:
                logger.error("Failed to build FAISS index.")
        else:
            logger.error("Cannot build FAISS index: embedding model or recipe data missing/empty.")
            
    if not faiss_index_ready:
        logger.critical("FAISS index could not be loaded or built. API search/RAG functionality will be impaired.")
        raise RuntimeError("FAISS index initialization failed.")

    # 4. Load LLM
    load_llm_on_startup() # This sets global llm
    if llm is None:
        logger.warning("LLM could not be loaded. RAG and generation endpoints will not work.")
        # Decide if this is critical enough to stop startup. For now, allow startup but endpoints will fail.

    logger.info("Application startup sequence complete.")
    if llm is None:
        logger.warning("Note: LLM is not available. Endpoints requiring the LLM will return errors.")


@app.post("/search", response_model=List[RecipeResult])
async def search_recipes_api(payload: SearchQuery):
    if recipes_df is None or faiss_index is None or embedding_model is None:
        raise HTTPException(status_code=503, detail="Core search components not ready.")
    results = search_recipes_logic(payload.query, payload.top_k)
    if not results and payload.query: # if query was non-empty but no results
        logger.info(f"No search results found for query: {payload.query}")
        # It's not an error to find no results, so return empty list.
    return results

@app.post("/recommend-rag", response_model=RagResponse)
async def recommend_recipe_rag_api(payload: RagQuery):
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM not available for RAG.")
    if recipes_df is None or faiss_index is None or embedding_model is None:
        raise HTTPException(status_code=503, detail="RAG components (data, index, or embedder) not ready.")
        
    recommendation, gen_time = recommend_recipe_with_rag_logic(payload.user_id, payload.user_query, payload.top_k_retrieval)
    if "LLM not available" in recommendation or "error occurred" in recommendation: # Check for internal errors
        raise HTTPException(status_code=500, detail=recommendation)
    if "No recipes found to provide context" in recommendation: # Specific case of no context
         raise HTTPException(status_code=404, detail=recommendation)
    return RagResponse(recommendation=recommendation, generation_time_seconds=gen_time)

@app.post("/generate-recipe", response_model=GenerateResponse)
async def generate_recipe_api(payload: GenerateQuery):
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM not available for recipe generation.")
    generated_text, gen_time = generate_recipe_from_ingredients_logic(payload.ingredients_list_str)
    if "LLM not available" in generated_text or "error occurred" in generated_text:
        raise HTTPException(status_code=500, detail=generated_text)
    return GenerateResponse(generated_recipe=generated_text, generation_time_seconds=gen_time)

# User Preference Endpoints (same as before)
@app.post("/users/{user_id}/preferences", response_model=Dict[str, List[str]])
async def set_user_preferences_api(user_id: str, payload: UserPreferenceRequest):
    if not payload.preferences:
        raise HTTPException(status_code=400, detail="Preferences payload cannot be empty.")
    updated_prefs = set_user_preferences_logic(user_id, payload.preferences)
    return updated_prefs

@app.post("/users/{user_id}/favorites", status_code=201)
async def add_favorite_api(user_id: str, payload: FavoriteRequest):
    if add_favorite_recipe_logic(user_id, payload.recipe_id):
        return {"message": f"Recipe {payload.recipe_id} added to {user_id}'s favorites."}
    # Check if user exists but recipe is already favorite
    if user_id in user_data and payload.recipe_id in user_data[user_id]['favorites']:
        raise HTTPException(status_code=409, detail=f"Recipe {payload.recipe_id} already in {user_id}'s favorites.")
    # Generic failure if user doesn't exist and it's not added
    raise HTTPException(status_code=404, detail=f"Could not add favorite. User {user_id} might not exist or other issue.")


@app.delete("/users/{user_id}/favorites/{recipe_id}", status_code=200)
async def remove_favorite_api(user_id: str, recipe_id: int):
    if remove_favorite_recipe_logic(user_id, recipe_id):
        return {"message": f"Recipe {recipe_id} removed from {user_id}'s favorites."}
    raise HTTPException(status_code=404, detail=f"Recipe {recipe_id} not in {user_id}'s favorites or user {user_id} not found.")

@app.get("/users/{user_id}")
async def get_user_data_api(user_id: str):
    data = get_user_data_logic(user_id)
    if data:
        return data
    # Create user if not exists on GET? Or require explicit creation?
    # For now, 404 if not found.
    # user_data[user_id] = {'favorites': [], 'preferences': {'dietary': [], 'cuisine': [], 'allergies': []}}
    # return user_data[user_id]
    raise HTTPException(status_code=404, detail=f"User {user_id} not found.")

@app.get("/")
async def root():
    return {"message": "Food RAG API is running. Visit /docs for API documentation."}

# To run locally (from the directory containing this app/ folder):
# uvicorn app.main:app --reload --port 8000