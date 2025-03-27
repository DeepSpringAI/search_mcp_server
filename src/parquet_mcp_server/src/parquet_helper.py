import pyarrow.parquet as pq
import os
import logging

def get_parquet_info(file_path: str) -> tuple[bool, str]:
    """
    Get information about a parquet file including column names, number of rows, and file size.
    
    Args:
        file_path (str): Path to the parquet file
        
    Returns:
        tuple[bool, str]: Success status and information message
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
            
        # Get file size
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)  # Convert to MB
        
        # Read parquet file metadata
        parquet_file = pq.ParquetFile(file_path)
        
        # Get number of rows
        num_rows = parquet_file.metadata.num_rows
        
        # Get column names
        column_names = parquet_file.schema.names
        
        # Create information message
        info_message = f"""
Parquet File Information:
------------------------
File Path: {file_path}
File Size: {file_size_mb:.2f} MB
Number of Rows: {num_rows:,}
Columns: {', '.join(column_names)}
"""
        
        return True, info_message
        
    except Exception as e:
        logging.error(f"Error getting parquet information: {str(e)}")
        return False, f"Error getting parquet information: {str(e)}" 