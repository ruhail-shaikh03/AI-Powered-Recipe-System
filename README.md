# Generative AI Project: AI-Powered Recipe System 🍳🤖

This project, developed by Saim Mubarak, Qazi Mohib Ul Nabi, and Ruhail Rizwan for Session 2025 under the supervision of Sir Akhtar Jamil (Department of Computer Science, National University of Computer and Emerging Sciences, Islamabad), focuses on creating an intelligent, user-centric recipe generation and recommendation system. It leverages Large Language Models (LLMs) and advanced information retrieval techniques to address the common challenge of meal planning.

The core of the system uses **Retrieval-Augmented Generation (RAG)** with the **Mistral-7B** model and a **FAISS** index for semantic search over the Food.com dataset. The project also includes a baseline LLM generation approach for comparison.

**For a comprehensive understanding of the project's background, motivation, architecture, detailed methodology, evaluation, and future work, please refer to the accompanying PDF report: `Generative_AI_Project_Report.pdf` (derived from the provided images).**

## 📖 Table of Contents

*   [Overview](#-overview)
*   [Features](#-features)
*   [Technologies Used](#-technologies-used)
*   [Dataset](#-dataset)
*   [Project Components & Functionality](#-project-components--functionality)
*   [RAG vs. Baseline LLM Generation Comparison](#-rag-vs-baseline-llm-generation-comparison)
*   [Setup & Running the Notebook](#-setup--running-the-notebook)
*   [Team Members](#-team-members)

## 🌟 Overview

Deciding what to cook can be challenging given personal tastes, available ingredients, health goals, and time constraints. Traditional recipe search often relies on basic keyword matching, lacking personalization and contextual understanding. This project aims to solve this by building an AI-driven recipe assistant that:

1.  Understands user input (ingredients, preferences, constraints).
2.  Recommends existing recipes from a large dataset using semantic search.
3.  Generates new, creative recipe ideas using a generative AI model (Mistral-7B).
4.  Enhances recommendations with Retrieval-Augmented Generation (RAG).
5.  Compares the RAG approach with a baseline LLM generation approach.

## ✨ Features

*   **Ingredient-Based Recipe Search:** Finds recipes from the Food.com dataset based on user-provided ingredients using FAISS for semantic similarity.
*   **RAG-Enhanced Recipe Recommendation:** Uses Mistral-7B with context retrieved from the FAISS index to provide more relevant and grounded recipe recommendations. It also considers (simulated) user preferences.
*   **Baseline LLM Recipe Generation:** Generates new recipes directly from a list of ingredients using Mistral-7B without external retrieval context.
*   **User Preference Simulation:** Basic in-memory simulation for storing user's favorite recipes and dietary/cuisine preferences.
*   **Comparative Analysis:** Qualitatively and quantitatively compares the RAG-based recommendation against the baseline LLM generation.

## 🛠️ Technologies Used

*   **Programming Language:** Python 3.11
*   **Core AI & ML:**
    *   `sentence-transformers` (all-MiniLM-L6-v2 for embeddings)
    *   `faiss-cpu` (or `faiss-gpu` if available) for efficient similarity search
    *   `ctransformers` for running GGUF quantized LLMs (Mistral-7B-Instruct-v0.2.Q4_K_M.gguf)
    *   `huggingface_hub` for model downloading
*   **Data Handling & Processing:**
    *   `pandas`
    *   `numpy`
    *   `ast` (for parsing string representations of lists)
*   **Development Environment:**
    *   Jupyter Notebook (via Kaggle Notebooks in this instance)
    *   `tqdm` for progress bars
    *   `matplotlib` for plotting comparison results
*   **Deployment (as per report):**
    *   Free-tier tools like Kaggle Notebooks, Hugging Face Spaces.

## 📊 Dataset

*   **Name:** Food.com Recipes and User Interactions
*   **Source:** Kaggle (`/kaggle/input/food-com-recipes-and-user-interactions/RAW_recipes.csv`)
*   **Content:** Contains 230K+ recipes and over 1.2M user interactions, including titles, ingredients, instructions, tags, nutrition info, and ratings.
*   **Preprocessing:** The notebook preprocesses this data by:
    *   Handling missing values.
    *   Parsing string-formatted ingredient and step lists.
    *   Creating a combined text field for embedding (`combined_text_for_embedding`).
    *   Subsetting the dataset (e.g., to 200,000 recipes) for manageable processing in the notebook environment.

## 🧩 Project Components & Functionality (as in the Notebook)

The Jupyter Notebook (`genai-project-food-rag-vs-llm.ipynb`) demonstrates the end-to-end pipeline:

1.  **Setup and Library Installation:** Installs necessary Python packages.
2.  **Data Loading and Preprocessing (`load_and_preprocess_data`):**
    *   Loads the `RAW_recipes.csv` dataset.
    *   Cleans and transforms the data (handling NaNs, parsing ingredients/steps).
    *   Creates a `combined_text_for_embedding` field for each recipe.
    *   Saves the processed DataFrame to `recipes_with_ids.parquet`.
3.  **Embedding and FAISS Indexing (`build_faiss_index`):**
    *   Loads a sentence transformer model (e.g., `all-MiniLM-L6-v2`).
    *   Generates embeddings for the `combined_text_for_embedding` of all recipes.
    *   Builds a FAISS index (`IndexFlatL2`) using these embeddings.
    *   Saves the FAISS index to `recipe_index.faiss`.
    *   Loads pre-built index and data if available to save time.
4.  **LLM Loading (`load_llm`):**
    *   Downloads and loads a GGUF quantized version of Mistral-7B (e.g., `mistral-7b-instruct-v0.2.Q4_K_M.gguf`) using `ctransformers`.
    *   Configures the LLM with parameters like `max_new_tokens`, `temperature`, `gpu_layers`.
5.  **User Preference Simulation (`add_favorite_recipe`, `set_user_preferences`, etc.):**
    *   Basic functions to simulate storing and retrieving user favorites and preferences in an in-memory dictionary.
6.  **Core Functions:**
    *   **`search_recipes(query, top_k)`:** Embeds a user query and uses the FAISS index to retrieve the `top_k` most semantically similar recipes.
    *   **`recommend_recipe_with_rag_improved(user_id, user_query, top_k_retrieval)`:**
        *   Retrieves relevant recipes using `search_recipes`.
        *   Constructs a detailed prompt for Mistral-7B, including the user query, (simulated) user preferences, and the context of retrieved recipes.
        *   Generates a recipe recommendation based on this augmented context.
    *   **`generate_recipe_from_ingredients(ingredients_list_str)`:**
        *   Takes a string of ingredients.
        *   Prompts Mistral-7B to generate a new recipe from scratch based primarily on these ingredients.
7.  **Comparison Implementation:**
    *   Defines qualitative metrics (Relevance, Adherence, Faithfulness, Structure, Creativity, Completeness).
    *   Runs test cases using both the RAG approach and the baseline LLM generation.
    *   Records LLM generation time for quantitative comparison.
    *   Plots the timing results using `matplotlib`.

## ⚖️ RAG vs. Baseline LLM Generation Comparison

The project includes a direct comparison between the RAG-enhanced recommendation and a baseline LLM generation approach.

**Qualitative Findings (summarized from the report):**

*   **Relevance to Query/Ingredients:** Both generally relevant. RAG shows stronger alignment due to grounded retrieval.
*   **Adherence to Constraints/Preferences:** RAG excels by using recipe metadata and user preferences. Baseline LLM sometimes misses constraints.
*   **Faithfulness/Groundedness:** RAG outputs are based on real retrieved recipes, reducing hallucinations. Baseline LLM occasionally introduces exotic/implausible elements.
*   **Recipe Structure & Format:** Both standard. Baseline LLM is more verbose; RAG is more concise and closer to real-world recipes.
*   **Creativity/Uniqueness:** Baseline LLM is more creative and novel. RAG is more practical and grounded in existing data.
*   **Completeness:** Both cover essential steps. RAG might truncate details based on context length; baseline might omit finer details unless well-prompted.

**Quantitative Findings (summarized from the report & notebook):**

*   **Average Generation Time:**
    *   RAG-Based Recommendation: ~15.6 seconds (LLM generation part, varies per query)
    *   Baseline LLM Generation: ~20.8 seconds (varies per query)
*   RAG has slightly lower latency for the LLM generation step, benefiting from precomputed context and a more deterministic prompt.

**Summary:** RAG offers higher faithfulness and preference alignment, ideal for trust and specificity. Baseline LLM excels in novel recipe generation where creativity is prioritized.

## ⚙️ Setup & Running the Notebook

This project is implemented as a Jupyter Notebook, designed to be run in an environment like Kaggle Notebooks (which provides GPU access and pre-installed libraries).

**Prerequisites:**

*   Python 3.9+ (3.11 used in the notebook)
*   Access to a GPU is highly recommended for reasonable performance with sentence transformers and the LLM.
*   Sufficient RAM (especially if running the LLM on CPU).

**Steps (as per the notebook):**

1.  **Environment:**
    *   The notebook is best run on Kaggle with a GPU accelerator (e.g., Tesla T4).
    *   Ensure internet connectivity is enabled in the Kaggle environment for downloading models.
2.  **Data:**
    *   The Food.com dataset should be available in the `/kaggle/input/food-com-recipes-and-user-interactions/` directory if using Kaggle.
    *   If running locally, download the dataset and adjust the path in `load_and_preprocess_data()`.
3.  **Install Dependencies:**
    The notebook includes cells to install necessary packages:
    ```bash
    !pip install -q faiss-cpu # or faiss-gpu if you have the CUDA toolkit setup
    !pip install -q ctransformers[cuda] sentence-transformers pandas numpy huggingface_hub matplotlib
    ```
4.  **Run Notebook Cells:**
    *   Execute the cells in the Jupyter Notebook sequentially.
    *   **Cell 1-2:** Initial setup and library imports/installations.
    *   **Cell 3 (`load_and_preprocess_data`):** Loads and preprocesses the recipe data. This will create `recipes_with_ids.parquet`.
    *   **Cell 4 (`build_faiss_index`):** Generates embeddings and builds/saves the FAISS index (`recipe_index.faiss`). This step can be time-consuming on the first run. Subsequent runs will load the saved index and data.
    *   **Cell 5 (`load_llm`):** Downloads and loads the Mistral-7B GGUF model. This also takes time on the first download.
    *   **Cell 6 (User Preferences):** Sets up and tests the basic user preference simulation.
    *   **Cell 7-9 (Core Functions):** Define and test `search_recipes`, `recommend_recipe_with_rag_improved`, and `generate_recipe_from_ingredients`.
    *   **Cell 10-12 (Comparison):** Define comparison metrics, run test cases for RAG vs. Baseline, summarize quantitative results, and plot timings.

**Note on GGUF Models:**
The notebook uses a GGUF (GPT-Generated Unified Format) version of Mistral-7B. These are quantized models designed to run efficiently on CPUs and can be partially offloaded to GPUs. The `gpu_layers` parameter in `AutoModelForCausalLM.from_pretrained()` controls how many layers are offloaded.

## 🧑‍💻 Team Members

*   **Saim Mubarak** (21i-0720)
*   **Qazi Mohib Ul Nabi** (21i-2532)
*   **Ruhail Rizwan** (21i-2462)

**Session:** 2025
**Submitted to:** Dr. Akhtar Jamil
**Department of Computer Science, National University of Computer and Emerging Sciences, Islamabad, Pakistan.**
