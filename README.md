# parquet_mcp_server

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

The server provides two main tools:

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

## Testing the MCP Server

You can test the server using the provided test client:

```bash
python src/test_mcp_embedding.py '{
    "input_path": "sample.parquet",
    "output_path": "output.parquet",
    "column_name": "text",
    "embedding_column": "embeddings",
    "batch_size": 10
}'
```

### Troubleshooting

1. If you get SSL verification errors, make sure the SSL settings in your `.env` file are correct
2. If embeddings are not generated, check:
   - The Ollama server is running and accessible
   - The model specified is available on your Ollama server
   - The text column exists in your input Parquet file

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
