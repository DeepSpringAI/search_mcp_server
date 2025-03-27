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
