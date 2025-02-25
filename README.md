# Data Analysis API

This repository contains a FastAPI application that provides endpoints for data clustering and similarity search operations.

## Features

1. **Clustering**: Performs DBSCAN clustering on CSV datasets.
2. **Similarity Search**: Conducts vector embedding and similarity search for a given query in a CSV dataset.

## Setup

1. Clone the repository: git clone https://github.com/yourusername/curotec_task.git cd curotec_task

2. Install the required dependencies: pip install -r requirements.txt

## Usage

Run the FastAPI application: python src/main.py


The server will start at `http://127.0.0.1:8000/`.

## API Endpoints

### Root
- **GET /** : Welcome message and basic API information.

### Clustering
- **POST /clustering** : Perform DBSCAN clustering on a CSV file.
  - Parameters:
    - `file`: CSV file (required)
    - `params`: JSON string of clustering parameters (optional)

### Similarity Search
- **POST /similarity** : Perform similarity search on text data in a CSV file.
  - Parameters:
    - `file`: CSV file (required)
    - `params`: JSON string of similarity search parameters (optional)

### Docs
- **/docs** : You can always access the FastAPI docs at `/docs` (e.g., http://127.0.0.1:8000/docs). This will provide a "friendly UI" for interacting with the API.

## Parameters

### Clustering Parameters
- `eps_range`: List of float values for DBSCAN epsilon parameter
- `min_samples_range`: List of integer values for DBSCAN min_samples parameter
- `label_column_index`: Index of the label column (if exists)
- `max_grid_search_combinations`: Maximum number of parameter combinations for grid search

### Similarity Parameters
- `text_column`: Column name containing text to embed
- `query_text`: Optional text to compare against
- `top_k`: Number of most similar items to return

## Dependencies

- FastAPI
- pandas
- numpy
- scikit-learn
- sentence-transformers
- uvicorn
- pydantic

For a complete list of dependencies, see `requirements.txt`.