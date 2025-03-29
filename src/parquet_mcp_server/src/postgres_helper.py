import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import os
from dotenv import load_dotenv
import logging

load_dotenv()

def get_postgres_connection():
    """Create a connection to PostgreSQL database using environment variables."""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('POSTGRES_DB'),
            user=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD'),
            host=os.getenv('POSTGRES_HOST'),
            port=os.getenv('POSTGRES_PORT', '5432')
        )
        return conn
    except Exception as e:
        logging.error(f"Error connecting to PostgreSQL: {str(e)}")
        raise

def convert_parquet_to_postgres(parquet_path: str, table_name: str) -> tuple[bool, str]:
    """
    Convert a Parquet file to a PostgreSQL table with pgvector support.
    If the table doesn't exist, it will be created. If it exists, data will be appended.
    
    Args:
        parquet_path (str): Path to the input Parquet file
        table_name (str): Name of the PostgreSQL table
    
    Returns:
        tuple[bool, str]: (success status, message)
    """
    try:
        # Read Parquet file
        df = pd.read_parquet(parquet_path)
        
        # Get column names and types
        columns = df.columns.tolist()
        dtypes = df.dtypes
        
        # Create connection
        conn = get_postgres_connection()
        cur = conn.cursor()
        
        # Check if table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            );
        """, (table_name,))
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            # Create table with appropriate column types
            column_definitions = []
            for col, dtype in zip(columns, dtypes):
                if dtype == 'object':  # For text columns
                    col_type = 'TEXT'
                elif dtype == 'int64':
                    col_type = 'BIGINT'
                elif dtype == 'float64':
                    col_type = 'DOUBLE PRECISION'
                elif dtype == 'bool':
                    col_type = 'BOOLEAN'
                else:
                    col_type = 'TEXT'  # Default to TEXT for unknown types
                
                column_definitions.append(f'"{col}" {col_type}')
            
            create_table_sql = f"""
                CREATE TABLE {table_name} (
                    {', '.join(column_definitions)}
                );
            """
            cur.execute(create_table_sql)
            conn.commit()
        
        # Prepare data for insertion
        # Convert numpy arrays to lists for PostgreSQL
        for col in df.columns:
            if isinstance(df[col].iloc[0], np.ndarray):
                df[col] = df[col].apply(lambda x: x.tolist())
        
        # Insert data
        columns_str = ', '.join(f'"{col}"' for col in columns)
        values = [tuple(x) for x in df.values]
        
        insert_sql = f"""
            INSERT INTO {table_name} ({columns_str})
            VALUES %s;
        """
        
        execute_values(cur, insert_sql, values)
        conn.commit()
        
        # Close connections
        cur.close()
        conn.close()
        
        success_msg = f"Successfully {'created and populated' if not table_exists else 'appended data to'} table '{table_name}'"
        logging.info(success_msg)
        return True, success_msg
        
    except Exception as e:
        error_msg = f"Error converting Parquet to PostgreSQL: {str(e)}"
        logging.error(error_msg)
        if 'conn' in locals():
            conn.close()
        return False, error_msg 