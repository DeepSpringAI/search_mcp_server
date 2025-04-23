import os
from dotenv import load_dotenv
from supabase import create_client, Client
import numpy as np
from typing import Dict, List, Any, Optional

# Load environment variables
load_dotenv()

class SupabaseDB:
    def __init__(self):
        """Initialize Supabase client with environment variables."""
        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")
        self.supabase: Client = create_client(url, key)

    def add_new_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add new data to the web_search table.
        
        Args:
            data: Dictionary containing text, metadata, and embedding
            
        Returns:
            Dict containing the inserted data or error information
        """
        try:
            # Extract URL and search_id from metadata (assuming metadata is a dictionary)
            url_to_delete = data.get("metadata", {}).get("url")
            new_search_id = data.get("metadata", {}).get("search_id")

            if url_to_delete and new_search_id:
                # Step 1: Get rows with the same URL and check that search_id is different
                response = self.supabase.table('web_search').select("id, metadata").filter('metadata->>url', 'eq', url_to_delete).execute()

                rows_to_delete = []
                for row in response.data:
                    # Check if the search_id is different in the existing row
                    existing_search_id = row['metadata'].get('search_id')
                    if existing_search_id != new_search_id:
                        rows_to_delete.append(row['id'])

                if rows_to_delete:
                    # Step 2: Delete rows with the same URL and different search_id
                    delete_response = self.supabase.table('web_search').delete().in_('id', rows_to_delete).execute()

                    # If there's an error with deletion, raise an exception
                    if 'error' in delete_response:
                        raise Exception(f"Failed to delete old data: {delete_response['error']}")
                    
                # Step 3: Insert new data
                insert_response = self.supabase.table('web_search').insert(data).execute()

                # If there's an error with insertion, raise an exception
                if 'error' in insert_response:
                    raise Exception(f"Failed to insert new data: {insert_response['error']}")

                return {"success": True, "data": insert_response.data}
            else:
                raise Exception("URL or search_id not found in the provided data metadata.")

        except Exception as e:
            raise Exception(f"Failed to insert data into Supabase: {str(e)}")

    def get_top_10_results(self) -> Dict[str, Any]:
        """
        Get top 10 results from the web_search table.
        
        Returns:
            Dict containing the results or error information
        """
        try:
            response = self.supabase.table('web_search').select(
                "id, metadata, text, embedding"
            ).limit(10).execute()
            return {"success": True, "data": response.data}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search_results_by_similarity(
        self, 
        query_embedding: List[float], 
        threshold: float = 0.55, 
        match_count: int = 10
    ) -> Dict[str, Any]:
        """
        Search results by similarity using the match_web_search RPC function.
        
        Args:
            query_embedding: The embedding vector to search with
            threshold: Similarity threshold (default: 0.55)
            match_count: Number of matches to return (default: 10)
            
        Returns:
            Dict containing the search results or error information
        """
        try:
            response = self.supabase.rpc(
                'match_web_search',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': threshold,
                    'match_count': match_count
                }
            ).execute()
            return {"success": True, "data": response.data}
        except Exception as e:
            return {"success": False, "error": str(e)}

# Example usage
if __name__ == "__main__":
    # Initialize the database client
    db = SupabaseDB()
    
    # Example data
    sample_data = {
        "text": "This is a sample chunk of text from a web search result.",
        "metadata": {
            "title": "Sample Web Search Result",
            "search_id": "2",
            "url": "https://example.com",
            "description": "This is a sample web search result",
            "timestamp": "2024-03-20T12:00:00Z"
        },
        "embedding": np.random.rand(1024).tolist()
    }
    
    # Example query embedding
    query_embedding = np.random.rand(1024).tolist()
    
    # Add new data
    insert_result = db.add_new_data(sample_data)
    
    # # Get top 10 results
    # top_results = db.get_top_10_results()
    
    # # Search results by similarity
    # similarity_results = db.search_results_by_similarity(
    #     query_embedding, 
    #     threshold=0.55, 
    #     match_count=10
    # )

    # print(similarity_results)
