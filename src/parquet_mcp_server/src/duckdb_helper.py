import duckdb
import os
import logging

def convert_parquet_to_duckdb(parquet_path: str, output_dir: str = None) -> tuple[bool, str]:
    """
    Convert a Parquet file to a DuckDB database.
    
    Args:
        parquet_path (str): Path to the input Parquet file
        output_dir (str, optional): Directory to save the DuckDB database. If None, uses the same directory as parquet file.
    
    Returns:
        tuple[bool, str]: (success status, message or database path)
    """
    try:
        # If output_dir is not specified, use the same directory as the parquet file
        if output_dir is None:
            output_dir = os.path.dirname(parquet_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate database filename based on parquet filename
        db_filename = os.path.splitext(os.path.basename(parquet_path))[0] + '.duckdb'
        db_path = os.path.join(output_dir, db_filename)
        
        # Create a connection to the DuckDB database
        conn = duckdb.connect(db_path)
        
        # Load the Parquet file and create a table
        table_name = 'parquet_data'
        conn.execute(f"""
            CREATE TABLE {table_name} AS 
            SELECT * FROM '{parquet_path}'
        """)
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        logging.info(f"Successfully converted {parquet_path} to DuckDB database at {db_path}")
        return True, db_path
        
    except Exception as e:
        error_msg = f"Error converting Parquet to DuckDB: {str(e)}"
        logging.error(error_msg)
        return False, error_msg
