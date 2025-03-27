import os
import logging
import requests
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Embedding URL from the .env file
embedding_url = os.getenv("EMBEDDING_URL")
if not embedding_url:
    raise ValueError("EMBEDDING_URL not found in .env file")

def get_embedding(text: str, url) -> list:
    # Prepare the payload with a list of prompts
    texts = [text]
    model = os.getenv("EMBEDDING_MODEL") or "llama2"
    payload = {
        "model": model,
        "input": texts  # Pass all texts in a batch
    }

    logging.debug(f"Making request to {url} with payload: {payload}")
    # Send the request with SSL verification disabled
    try:
        response = requests.post(url, json=payload, verify=False)
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            logging.debug(f"Response: {result}")
            if 'data' in result and result['data'] and 'embedding' in result['data'][0]:
                embeddings = np.array(result['data'][0]['embedding'])
                return embeddings
            else:
                logging.error(f"No embeddings found in response: {result}")
                return None
        else:
            logging.error(f"Error: {response.status_code}, {response.text}")
            return None  # If there was an error, return None
    except Exception as e:
        logging.error(f"Exception during request: {str(e)}")
        return None

def process_parquet_file(input_path: str, output_path: str, column_name: str, embedding_column: str) -> tuple[bool, str]:
    """
    Process a parquet file by adding embeddings to a specified column.
    
    Args:
        input_path (str): Path to the input Parquet file
        output_path (str): Path to the output Parquet file
        column_name (str): The name of the column containing the text to embed
        embedding_column (str): Name of the new column where embeddings will be saved
        
    Returns:
        tuple[bool, str]: A tuple containing success status and message
    """
    try:
        import pandas as pd
        
        # Read the parquet file into a pandas dataframe
        df = pd.read_parquet(input_path)
        
        # Check if the column exists
        if column_name not in df.columns:
            return False, f"Error: Column '{column_name}' not found in the parquet file."

        # Apply embedding to each row in the specified column
        embeddings = []
        for text in df[column_name]:
            embedding = get_embedding(str(text), embedding_url)  # Make sure to convert to string if it's not
            embeddings.append(embedding)

        # Add the embeddings as a new column
        df[embedding_column] = embeddings

        # Save the modified dataframe to a new Parquet file
        df.to_parquet(output_path)

        return True, f"Successfully added embedding to column '{embedding_column}' and saved the output to {output_path}"

    except Exception as e:
        return False, f"Error processing parquet file: {str(e)}"
