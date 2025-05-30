
# AI-Powered Recipe System (RAG Food API)

This project implements an AI-powered recipe system featuring:
- Recipe search using semantic embeddings and FAISS.
- Recipe recommendations using a Retrieval Augmented Generation (RAG) approach with a Large Language Model (LLM).
- Direct recipe generation from a list of ingredients using an LLM.
- Basic user preference management (favorites).

The application is containerized using Docker and exposes a FastAPI interface. The Docker image is available on Docker Hub at `ruhailshaikh/food-api-app`.

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
│   ├── assets/             # Stores pre-generated recipes_with_ids.parquet and recipe_index.faiss (via Git LFS)
│   └── data/               # Stores RAW_recipes.csv
├── persistent_assets/      # (Local only, gitignored) For runtime model caches
│   ├── hf_cache/
│   └── st_cache/
├── Dockerfile              # Defines the Docker image build process
├── requirements.txt        # Python dependencies
├── .gitattributes          # For Git LFS configuration
├── .gitignore              # Specifies intentionally untracked files by Git
└── README.md               # This file
```

## Prerequisites

- Docker installed.
- Git installed.
- Git LFS (Large File Storage) installed. (Run `git lfs install` once after installing Git LFS).

## Setup and Running the Application

There are two main ways to run the application: by pulling the pre-built image from Docker Hub (recommended for ease of use) or by building the image locally from the source code.

**Method 1: Pulling and Running from Docker Hub (Recommended)**

1.  **Create Local Directories for Data and Caches:**
    Even when using the pre-built image, you need to provide the raw data and create directories for persistent model caches.
    ```bash
    # Create these in a local directory where you want to manage the app's runtime data
    mkdir -p my_food_app_runtime/app/data
    mkdir -p my_food_app_runtime/app/assets # Will be populated by the image if needed, or use your LFS cloned assets
    mkdir -p my_food_app_runtime/persistent_assets/hf_cache
    mkdir -p my_food_app_runtime/persistent_assets/st_cache

    # Download RAW_recipes.csv and place it in my_food_app_runtime/app/data/
    # (e.g., from Kaggle: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
    ```
    *Note: The `app/assets` directory is crucial. If the Docker image expects assets and they aren't volume-mounted, it might try to regenerate them. For consistency, it's best to ensure your Git LFS assets are available locally and mounted.*

2.  **Pull the Docker Image:**
    ```bash
    docker pull ruhailshaikh/food-api-app:latest # Or a specific version tag if available
    ```

3.  **Run the Docker Container:**
    Navigate to your `my_food_app_runtime` directory (or wherever you created the subdirectories).
    ```bash
    # Adjust MAX_RECIPES if your assets correspond to a different number
    # (e.g., if your LFS assets were for 50000 recipes, use MAX_RECIPES="50000")
    # The Docker image ruhailshaikh/food-api-app:latest was built with ENV MAX_RECIPES="200000"
    # If your local assets (from Git LFS) are for 200,000 recipes, this should align.
    # If they are for a different count (e.g. 50,000), set MAX_RECIPES accordingly.

    docker run -p 8000:8000 --name my-food-api ^
        -v "${PWD}/app/data:/app/data:ro" ^
        -v "${PWD}/app/assets:/app/assets" ^
        -v "${PWD}/persistent_assets/hf_cache:/root/.cache/huggingface" ^
        -v "${PWD}/persistent_assets/st_cache:/root/.cache/torch/sentence_transformers" ^
        -e MAX_RECIPES="200000" ^  # IMPORTANT: Match this to your LFS assets or desired processing size
        -e LLM_MODEL_FILE="mistral-7b-instruct-v0.2.Q3_K_S.gguf" ^
        ruhailshaikh/food-api-app:latest
    ```
    *(Use `\` for line continuation in Bash/Linux, `^` for PowerShell/Windows CMD)*

**Method 2: Building and Running from Source Code**

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url> # Replace with your repository URL
    cd food-rag-api
    ```

2.  **Initialize Git LFS and Pull Assets:**
    If you haven't already, initialize Git LFS and pull the large asset files:
    ```bash
    git lfs install     # Run once per machine
    git lfs pull
    ```
    This will download the actual `recipes_with_ids.parquet` and `recipe_index.faiss` files into `app/assets/`.

3.  **Prepare Raw Data:**
    - Download `RAW_recipes.csv` (e.g., from Kaggle).
    - Place it in the `food-rag-api/app/data/` directory.

4.  **Create Local Persistent Cache Directories:**
    Run these commands in the `food-rag-api` root:
    ```bash
    mkdir -p ./persistent_assets/hf_cache
    mkdir -p ./persistent_assets/st_cache
    ```

5.  **Build the Docker Image:**
    ```bash
    docker build -t food-rag-api-app .
    ```
    The first build might take time to download base images and install Python packages. The `Dockerfile` is structured to cache Python dependencies, so subsequent builds (if only app code changes) will be faster.

6.  **Run the Docker Container:**
    (Same `docker run` command as in Method 1, Step 3, but using your locally built `food-rag-api-app` image name)
    ```bash
    docker run -p 8000:8000 --name my-food-api ^
        -v "${PWD}/app/data:/app/data:ro" ^
        -v "${PWD}/app/assets:/app/assets" ^
        -v "${PWD}/persistent_assets/hf_cache:/root/.cache/huggingface" ^
        -v "${PWD}/persistent_assets/st_cache:/root/.cache/torch/sentence_transformers" ^
        -e MAX_RECIPES="200000" ^ # IMPORTANT: Match this to your LFS assets or desired processing size
        -e LLM_MODEL_FILE="mistral-7b-instruct-v0.2.Q3_K_S.gguf" ^
        food-rag-api-app
    ```

**7. Access the API:**

   Once the application is running (from either method), open your browser and navigate to:
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

Key configurations are managed via environment variables (see `docker run -e` flags):

- `MAX_RECIPES`: Number of recipes to process from the dataset.
    - The Docker image `ruhailshaikh/food-api-app:latest` was built with a default `ENV MAX_RECIPES="200000"`.
    - When running, set this to match the size of your assets in `app/assets/` (from Git LFS) to avoid asset regeneration. If your LFS assets are for 50,000 recipes, use `-e MAX_RECIPES="50000"`.
- `LLM_MODEL_FILE`: The GGUF model file to use (e.g., `mistral-7b-instruct-v0.2.Q3_K_S.gguf`, `mistral-7b-instruct-v0.2.Q2_K.gguf`). Lighter quantizations (Q3, Q2) are faster on CPU but may have slightly lower quality. The image default is `mistral-7b-instruct-v0.2.Q3_K_S.gguf`.

Other configurations (embedding model, LLM repo, context lengths) are set within `app/main.py`.

## Git LFS Setup (for contributors)

The pre-generated assets (`*.parquet`, `*.faiss`) in `app/assets/` are tracked using Git LFS.

1.  Install Git LFS: [https://git-lfs.com](https://git-lfs.com)
2.  Initialize Git LFS for your local repository (run once):
    ```bash
    git lfs install
    ```
3.  Track large file types (if adding new ones):
    ```bash
    git lfs track "*.parquet"
    git lfs track "*.faiss"
    git add .gitattributes
    ```
    (The `.gitattributes` file should already be configured in this repository to track these.)
4.  When cloning, `git lfs pull` will download the actual large files.

## Performance Notes

- **LLM Inference on CPU is Slow:** Endpoints using the LLM (`/recommend-rag`, `/generate-recipe`) will take a significant amount of time (tens of seconds to minutes) to respond when running on CPU. This is expected.
- **First Run:** The very first run (especially if not using cached models from `persistent_assets`) will be longer due to model downloads. If assets are mismatched or not provided, generation will also take time.
- **Resource Usage:** This application, especially with a 7B parameter LLM, is resource-intensive (CPU and RAM). Ensure Docker has adequate resources allocated.

## Troubleshooting

- **Context Length Exceeded Warning:** If you see `ctransformers - WARNING - Number of tokens (...) exceeded maximum context length (...)` in the logs for RAG requests:
    - Reduce `max_context_chars` in `app/main.py` (inside `recommend_recipe_with_rag_logic`).
    - Reduce `top_k_retrieval` in your API request.
    - Consider slightly increasing `context_length` for the LLM in `app/main.py` (inside `load_llm_on_startup`) if your system RAM allows (requires image rebuild if not using the Docker Hub image).
- **Slow Image Rebuilds (if building from source):** The `Dockerfile` is structured to cache Python dependencies. Rebuilds should be fast if only `app/` code changes. If `requirements.txt` changes, `pip install` will re-run.
- **"File not found" for assets/data:** Ensure your volume mounts in the `docker run` command correctly point to your local directories containing `RAW_recipes.csv` and the LFS-pulled assets.

## TODO / Future Enhancements

- Persistent user data storage (e.g., using a database instead of in-memory).
- More sophisticated user preference modeling.
- GPU support for LLM and embedding inference for significant speedup.
- More robust error handling and input validation.
- Unit and integration tests.
- CI/CD pipeline for automated builds and deployments.
```

**Key Changes in this README:**

*   **Docker Hub Image:** Prominently mentions your `ruhailshaikh/food-api-app` image and provides instructions for pulling and running it.
*   **Git LFS:** Explains that assets are on Git LFS and how to set it up for cloning/contributing.
*   **Setup Sections:** Clearly separates "Pulling from Docker Hub" and "Building from Source."
*   **`MAX_RECIPES` Clarification:** Emphasizes matching the `MAX_RECIPES` environment variable to the actual size of the assets being used (whether from LFS or if the image's default baked-in assets are intended).
*   **Removed `docker-compose.yml` references** and focused on `docker build` and `docker run`.
*   Added `.gitattributes` to the project structure as it's essential for Git LFS.

**Next Steps for You:**

1.  **Create `.gitattributes` file (if it doesn't exist):**
    If you haven't already told Git LFS which files to track, create a `.gitattributes` file in your project root:
    ```
    *.parquet filter=lfs diff=lfs merge=lfs -text
    *.faiss filter=lfs diff=lfs merge=lfs -text
    # Add other large file patterns if necessary
    ```
    Then add and commit it:
    ```bash
    git add .gitattributes
    git commit -m "Configure Git LFS for parquet and faiss files"
    ```
    If you already pushed your `.parquet` and `.faiss` files to GitHub *without* LFS tracking them properly, you might need to retroactively convert them to LFS. See Git LFS documentation for "migrating existing data." It's easier if you set up LFS tracking *before* committing the large files for the first time.

2.  **Ensure your LFS assets are actually pushed to the LFS storage:**
    After `git lfs track` and committing the files, a `git push` should upload the large files to the LFS storage associated with your GitHub repository.

3.  **Update your local `README.md`** with the content above.

4.  **Commit and push** your `README.md` and `.gitattributes` (if new/changed) to GitHub:
    ```bash
    git add README.md .gitattributes
    git commit -m "Update README for Dockerfile usage, Docker Hub, and Git LFS"
    git push origin main
    ```
