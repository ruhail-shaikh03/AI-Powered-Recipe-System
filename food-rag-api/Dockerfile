# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing pyc files to disc (optional)
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to reduce image size and --upgrade pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (app directory) into the container at /app
# This assumes your FastAPI app code is in a subdirectory named 'app'
COPY ./app /app

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variables for application configuration
# These can be overridden at runtime using 'docker run -e VAR=value'
# Set DEFAULT_MAX_RECIPES in main.py to your Kaggle-generated parquet/faiss size
# MAX_RECIPES here is an example override if you want to force a smaller set in Docker
ENV MAX_RECIPES="200000" 
ENV LLM_MODEL_FILE="mistral-7b-instruct-v0.2.Q3_K_S.gguf" 
# Command to run the application using uvicorn
# The application (main.py) will handle downloading/generating assets if not present
# Host 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]