# AI-Powered Recipe System (RAG Food API)

This project implements an AI-powered recipe system featuring:
- Recipe search using semantic embeddings and FAISS.
- Recipe recommendations using a Retrieval Augmented Generation (RAG) approach with a Large Language Model (LLM).
- Direct recipe generation from a list of ingredients using an LLM.
- Basic user preference management (favorites).

The application is containerized using Docker and exposes a FastAPI interface.

## Features

- **Semantic Recipe Search:** Find recipes based on natural language queries or ingredient lists.
- **RAG-Based Recommendations:** Get intelligent recipe suggestions based on your query, with context provided by retrieved recipes and considering (basic) user preferences.
- **LLM-Powered Recipe Generation:** Create new recipes from a given set of ingredients.
- **User Profile (In-Memory):**
    - Add/remove favorite recipes.
    - Set basic dietary/cuisine preferences.
- **Dockerized:** Easy to set up and run in a containerized environment.
- **FastAPI Backend:** Provides a robust and well-documented API (Swagger UI).

## Project Structure

```
food-rag-api/
├── app/
│   ├── main.py             # FastAPI application code
│   ├── assets/             # Stores pre-generated recipes_with_ids.parquet and recipe_index.faiss
│   └── data/               # Stores RAW_recipes.csv
├── persistent_assets/      # (Local only, gitignored) For runtime model caches & generated assets
│   ├── app_assets/
│   ├── hf_cache/
│   └── st_cache/
├── Dockerfile              # Defines the Docker image build process
├── docker-compose.yml      # For easy local development and running
├── requirements.txt        # Python dependencies
├── .gitignore              # Specifies intentionally untracked files by Git
└── README.md               # This file
```

## Prerequisites

- Docker and Docker Compose installed.
- A Docker Hub account (only if you plan to push/pull images from Docker Hub).
- Git for cloning the repository.

## Setup and Running the Application

**1. Clone the Repository:**
   ```bash
   git clone <your-repository-url>
   cd food-rag-api
   ```

**2. Prepare Data and Assets:**

   *   **Raw Data:**
       - Download `RAW_recipes.csv` (e.g., from the Kaggle dataset: [Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)).
       - Place it in the `food-rag-api/app/data/` directory.
   *   **Pre-generated Assets (Recommended for faster startup):**
       - This repository includes pre-generated `recipes_with_ids.parquet` and `recipe_index.faiss` in the `food-rag-api/app/assets/` directory. These were generated based on `MAX_RECIPES = 50000` (or update this value if different).
       - If these files are not present, the application will attempt to generate them on first startup if `RAW_recipes.csv` is available. This process can take a significant amount of time.
   *   **Modify `DEFAULT_MAX_RECIPES` (Important if using different pre-generated assets):**
        - In `app/main.py`, ensure the `DEFAULT_MAX_RECIPES` variable matches the number of recipes your `.parquet` and `.faiss` files in `app/assets/` correspond to. This helps the application validate the assets correctly. The `MAX_RECIPES` environment variable will override this.

**3. Create Local Persistent Directories (for caches and runtime generated assets):**
   Run these commands in the `food-rag-api` root:
   ```bash
   mkdir -p ./persistent_assets/app_assets
   mkdir -p ./persistent_assets/hf_cache
   mkdir -p ./persistent_assets/st_cache
   ```
   These directories are gitignored and will store downloaded models and any assets generated or modified at runtime if you mount `app/assets` to `persistent_assets/app_assets`.

**4. Build and Run with Docker Compose (Recommended):**

   *   Navigate to the `food-rag-api` root directory.
   *   Adjust environment variables in `docker-compose.yml` if needed (e.g., `MAX_RECIPES`, `LLM_MODEL_FILE`). For example, to use the pre-generated assets for 50,000 recipes:
     ```yaml
     # In docker-compose.yml
     services:
       app:
         # ...
         environment:
           - MAX_RECIPES=50000
           - LLM_MODEL_FILE=mistral-7b-instruct-v0.2.Q3_K_S.gguf # Or Q2_K for lighter
     ```
   *   Run:
     ```bash
     docker-compose up --build
     ```
     The first build might take time to download base images and install Python packages. Subsequent builds when only app code changes will be much faster. The first run will also download LLM and embedding models, storing them in `./persistent_assets/`.

**5. (Alternative) Build and Run with Docker CLI:**

   *   Build the image:
     ```bash
     docker build -t food-rag-api-app .
     ```
   *   Run the container (adjust `MAX_RECIPES` and `LLM_MODEL_FILE`):
     ```bash
     docker run -p 8000:8000 --name my-food-api ^
         -v "${PWD}/app/data:/app/data:ro" ^
         -v "${PWD}/app/assets:/app/assets" ^
         -v "${PWD}/persistent_assets/hf_cache:/root/.cache/huggingface" ^
         -v "${PWD}/persistent_assets/st_cache:/root/.cache/torch/sentence_transformers" ^
         -e MAX_RECIPES="50000" ^
         -e LLM_MODEL_FILE="mistral-7b-instruct-v0.2.Q3_K_S.gguf" ^
         food-rag-api-app
     ```
     *(Use `\` for line continuation in Bash/Linux, `^` for PowerShell)*

**6. Access the API:**

   Once the application is running, open your browser and navigate to:
   `http://localhost:8000/docs`

   You will find the FastAPI Swagger UI to interact with the API endpoints.

## API Endpoints

- **`GET /health`**: Basic health check.
- **`POST /search`**: Semantic search for recipes.
  - Request Body: `{"query": "your search term", "top_k": 5}`
- **`POST /recommend-rag`**: Get RAG-based recipe recommendations.
  - Request Body: `{"user_id": "some_user", "user_query": "e.g., easy chicken dinner", "top_k_retrieval": 5}`
- **`POST /generate-recipe`**: Generate a new recipe from ingredients.
  - Request Body: `{"ingredients_list_str": "e.g., chicken, rice, broccoli"}`
- **User Preferences:**
    - `POST /users/{user_id}/preferences`: Set user preferences.
      - Request Body: `{"preferences": {"dietary": ["vegan"], "cuisine": ["italian"]}}`
    - `POST /users/{user_id}/favorites`: Add a recipe to favorites.
      - Request Body: `{"recipe_id": 12345}`
    - `DELETE /users/{user_id}/favorites/{recipe_id}`: Remove a recipe from favorites.
    - `GET /users/{user_id}`: Get user data (favorites and preferences).

## Configuration

Key configurations are managed via environment variables (see `docker-compose.yml` or `docker run -e` flags):

- `MAX_RECIPES`: Number of recipes to process from the dataset. Set this to match your pre-generated `.parquet` and `.faiss` files for optimal startup.
- `LLM_MODEL_FILE`: The GGUF model file to use (e.g., `mistral-7b-instruct-v0.2.Q3_K_S.gguf`, `mistral-7b-instruct-v0.2.Q2_K.gguf`). Lighter quantizations (Q3, Q2) are faster on CPU but may have slightly lower quality.
- `EMBEDDING_MODEL_NAME` (in `app/main.py`): `all-MiniLM-L6-v2`
- `LLM_MODEL_REPO` (in `app/main.py`): `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`

LLM context length and max new tokens are configured within `app/main.py`.

## Performance Notes

- **LLM Inference on CPU is Slow:** Endpoints using the LLM (`/recommend-rag`, `/generate-recipe`) will take a significant amount of time (tens of seconds to minutes) to respond when running on CPU. This is expected.
- **First Run:** The very first run will be longer due to model downloads and potential asset generation if pre-generated files are not found or mismatched. Subsequent runs with volume-mounted caches will be faster.
- **Resource Usage:** This application, especially with a 7B parameter LLM, is resource-intensive (CPU and RAM). Ensure Docker has adequate resources allocated.

## Troubleshooting

- **Context Length Exceeded Warning:** If you see `ctransformers - WARNING - Number of tokens (...) exceeded maximum context length (...)` in the logs for RAG requests:
    - Reduce `max_context_chars` in `app/main.py` (inside `recommend_recipe_with_rag_logic`).
    - Reduce `top_k_retrieval` in your API request.
    - Consider slightly increasing `context_length` for the LLM in `app/main.py` (inside `load_llm_on_startup`) if your system RAM allows.
- **Slow Rebuilds:** Ensure your `Dockerfile` copies `requirements.txt` and runs `pip install` *before* copying the `./app` directory to leverage Docker layer caching effectively.
- **Missing Parquet Engine:** Ensure `pyarrow` is in `requirements.txt`.

## TODO / Future Enhancements

- Persistent user data storage (e.g., using a database instead of in-memory).
- More sophisticated user preference modeling.
- GPU support for LLM and embedding inference for significant speedup.
- More robust error handling and input validation.
- Unit and integration tests.
- CI/CD pipeline for automated builds and deployments.
