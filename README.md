# parquet_mcp_server

A powerful MCP (Model Control Protocol) server that provides tools for manipulating and analyzing Parquet files. This server is designed to work with Claude Desktop and offers three main functionalities:

1. **Text Embedding Generation**: Convert text columns in Parquet files into vector embeddings using Ollama models
2. **Parquet File Analysis**: Extract detailed information about Parquet files including schema, row count, and file size
3. **DuckDB Integration**: Convert Parquet files to DuckDB databases for efficient querying and analysis

This server is particularly useful for:
- Data scientists working with large Parquet datasets
- Applications requiring vector embeddings for text data
- Projects needing to analyze or convert Parquet files
- Workflows that benefit from DuckDB's fast querying capabilities

## Installation

### Clone this repository

```bash
git clone ...
cd parquet_mcp_server
```

### Create and activate virtual environment

```bash
uv venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On macOS/Linux
```

### Install the package

```bash
uv pip install -e .
```

### Environment

Create a `.env` file with the following variables:

```bash
EMBEDDING_URL=  # URL for the embedding service
OLLAMA_URL=    # URL for Ollama server
EMBEDDING_MODEL=nomic-embed-text  # Model to use for generating embeddings
```

## Usage with Claude Desktop

Add this to your Claude Desktop configuration file (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "parquet-mcp-server": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/${USER}/workspace/parquet_mcp_server/src/parquet_mcp_server",
        "run",
        "main.py"
      ]
    }
  }
}
```

## Available Tools

The server provides three main tools:

1. **Embed Parquet**: Adds embeddings to a specific column in a Parquet file
   - Required parameters:
     - `input_path`: Path to input Parquet file
     - `output_path`: Path to save the output
     - `column_name`: Column containing text to embed
     - `embedding_column`: Name for the new embedding column
     - `batch_size`: Number of texts to process in each batch (for better performance)

2. **Parquet Information**: Get details about a Parquet file
   - Required parameters:
     - `file_path`: Path to the Parquet file to analyze

3. **Convert to DuckDB**: Convert a Parquet file to a DuckDB database
   - Required parameters:
     - `parquet_path`: Path to the input Parquet file
   - Optional parameters:
     - `output_dir`: Directory to save the DuckDB database (defaults to same directory as input file)

## Example Prompts

Here are some example prompts you can use with the agent:

### For Embedding:
```
"Please embed the column 'text' in the parquet file '/path/to/input.parquet' and save the output to '/path/to/output.parquet'. Use 'embeddings' as the final column name and a batch size of 2"
```

### For Parquet Information:
```
"Please give me some information about the parquet file '/path/to/input.parquet'"
```

### For DuckDB Conversion:
```
"Please convert the parquet file '/path/to/input.parquet' to DuckDB format and save it in '/path/to/output/directory'"
```

## Testing the MCP Server

The project includes a comprehensive test suite in the `src/tests` directory. You can run all tests using:

```bash
python src/tests/run_tests.py
```

Or run individual tests:

```bash
# Test embedding functionality
python src/tests/test_embedding.py

# Test parquet information tool
python src/tests/test_parquet_info.py

# Test DuckDB conversion
python src/tests/test_duckdb_conversion.py
```

You can also test the server using the client directly:

```python
from parquet_mcp_server.client import convert_to_duckdb, embed_parquet, get_parquet_info

# Test DuckDB conversion
result = convert_to_duckdb(
    parquet_path="input.parquet",
    output_dir="db_output"
)

# Test embedding
result = embed_parquet(
    input_path="input.parquet",
    output_path="output.parquet",
    column_name="text",
    embedding_column="embeddings",
    batch_size=2
)

# Test parquet information
result = get_parquet_info("input.parquet")
```

### Troubleshooting

1. If you get SSL verification errors, make sure the SSL settings in your `.env` file are correct
2. If embeddings are not generated, check:
   - The Ollama server is running and accessible
   - The model specified is available on your Ollama server
   - The text column exists in your input Parquet file
3. If DuckDB conversion fails, check:
   - The input Parquet file exists and is readable
   - You have write permissions in the output directory
   - The Parquet file is not corrupted

### API Response Format

The embeddings are returned in the following format:

```json
{
    "object": "list",
    "data": [{
        "object": "embedding",
        "embedding": [0.123, 0.456, ...],
        "index": 0
    }],
    "model": "llama2",
    "usage": {
        "prompt_tokens": 4,
        "total_tokens": 4
    }
}
```

Each embedding vector is stored in the Parquet file as a NumPy array in the specified embedding column.

The DuckDB conversion tool returns a success message with the path to the created database file or an error message if the conversion fails.
