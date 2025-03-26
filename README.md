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

you need to define this .env file:

```bash
EMBEDDING_URL=
OLLAMA_URL=
```

EMBEDDING_URL is the embedding url of ollama and the ollama url is used in the agent for testing.

## Usage with an Agent

In (`client.py`) there is an anget in langchain which you can test the tools with sample prompt.

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

---

That's it! Now you can run the server using Claude Desktop.

## Testing the MCP Server

You can test the MCP server in two ways: using the test client directly or using the provided agent.

### Using the Test Client

1. First, create a sample Parquet file with some text data:

```python
import pandas as pd

# Create a sample DataFrame
df = pd.DataFrame({
    'text': ['This is a test', 'Another test sentence']
})

# Save it as a Parquet file
df.to_parquet('sample.parquet')
```

2. Run the test client with your parameters:

```bash
python src/test_mcp_embedding.py '{
    "input_path": "sample.parquet",
    "output_path": "output.parquet",
    "column_name": "text",
    "embedding_column": "embeddings"
}'
```

3. Verify the output:

```python
import pandas as pd

# Read the output file
df = pd.read_parquet('output.parquet')

# Check the contents
print('Columns:', df.columns.tolist())
print('Shape:', df.shape)
print('Sample embedding size:', len(df['embeddings'].iloc[0]))
```

The output should show that:
- The file has both 'text' and 'embeddings' columns
- Each embedding is a 768-dimensional vector
- The original text data is preserved

### Using the Agent

The agent provides a more natural language interface to the embedding functionality. You can use it by running:

```bash
python src/parquet_mcp_server/client.py
```

Example prompts for the agent:
- "Please embed the column 'text' in the parquet file 'input.parquet' and save the output to 'output.parquet'"
- "Generate embeddings for the 'description' column in 'data.parquet' and save them as 'vectors' in 'result.parquet'"

### Environment Setup

Make sure your `.env` file is properly configured:

```bash
EMBEDDING_URL=https://your-ollama-server/v1/embeddings
OLLAMA_URL=https://your-ollama-server
PYTHONWARNINGS=ignore:Unverified HTTPS request
REQUESTS_CA_BUNDLE=""
CURL_CA_BUNDLE=""
SSL_CERT_VERIFY=false
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
