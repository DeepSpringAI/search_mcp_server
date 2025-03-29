import os
import logging
import requests
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Embedding URL from the .env file
embedding_url = os.getenv("EMBEDDING_URL")
if not embedding_url:
    raise ValueError("EMBEDDING_URL not found in .env file")

def get_embedding(texts: list, url) -> list:
    """
    Fetch embeddings for a batch of texts from the embedding server.

    Args:
        texts (list): A list of texts to generate embeddings for.
        url (str): The URL of the embedding server.

    Returns:
        list: A list of embeddings corresponding to the input texts.
    """
    model = os.getenv("EMBEDDING_MODEL") or "llama2"
    payload = {
        "model": model,
        "input": texts  # Pass all texts in a batch
    }

    logging.debug(f"Making request to {url} with payload: {payload}")
    try:
        response = requests.post(url, json=payload, verify=False)
        if response.status_code == 200:
            result = response.json()
            # logging.debug(f"Response: {result}")
            if 'embeddings' in result:
                embeddings = np.array(result['embeddings'])
                return embeddings
            else:
                logging.error(f"No embeddings found in response: {result}")
                return []
        else:
            logging.error(f"Error: {response.status_code}, {response.text}")
            return []  # Return an empty list on error
    except Exception as e:
        logging.error(f"Exception during request: {str(e)}")
        return []

def process_parquet_file(input_path: str, output_path: str, column_name: str, embedding_column: str, batch_size: int) -> tuple[bool, str]:
    """
    Process a parquet file by adding embeddings to a specified column with batch processing.

    Args:
        input_path (str): Path to the input Parquet file.
        output_path (str): Path to the output Parquet file.
        column_name (str): The name of the column containing the text to embed.
        embedding_column (str): Name of the new column where embeddings will be saved.
        batch_size (int): The size of each batch for embedding requests.

    Returns:
        tuple[bool, str]: A tuple containing success status and message.
    """
    try:
        import pandas as pd
        
        # Read the parquet file into a pandas dataframe
        df = pd.read_parquet(input_path)
        
        # Check if the column exists
        if column_name not in df.columns:
            return False, f"Error: Column '{column_name}' not found in the parquet file."

        # Prepare to process in batches
        embeddings = []
        texts = df[column_name].astype(str).tolist()  # Convert all column entries to string
        num_batches = len(texts) // batch_size + (1 if len(texts) % batch_size != 0 else 0)

        # logging.info(f"Processing {num_batches} batches of {batch_size} texts")
        logging.debug(f"Processing {num_batches} batches of {batch_size} texts")
        for i in range(num_batches):
            batch_texts = texts[i * batch_size: (i + 1) * batch_size]
            batch_embeddings = get_embedding(batch_texts, embedding_url)
            embeddings.extend(batch_embeddings)  # Add the batch embeddings to the list

        # Ensure the embeddings list matches the number of rows in the dataframe
        if len(embeddings) != len(df):
            return False, f"Error: The number of embeddings does not match the number of rows in the input file. {len(embeddings)} != {len(df)}"

        # Add the embeddings as a new column
        df[embedding_column] = embeddings

        # Save the modified dataframe to a new Parquet file
        df.to_parquet(output_path)

        return True, f"Successfully added embedding to column '{embedding_column}' and saved the output to {output_path}"

    except Exception as e:
        return False, f"Error processing parquet file: {str(e)}"
