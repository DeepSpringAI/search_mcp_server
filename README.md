# parquet_mcp_server
[![smithery badge](https://smithery.ai/badge/@DeepSpringAI/parquet_mcp_server)](https://smithery.ai/server/@DeepSpringAI/parquet_mcp_server)

A powerful MCP (Model Control Protocol) server that provides tools for performing web searches and finding similar content. This server is designed to work with Claude Desktop and offers two main functionalities:

1. **Web Search**: Perform a web search and scrape results
2. **Similarity Search**: Extract relevant information from previous searches

This server is particularly useful for:
- Applications requiring web search capabilities
- Projects needing to find similar content based on search queries

## Installation

### Installing via Smithery

To install Parquet MCP Server for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@DeepSpringAI/parquet_mcp_server):

```bash
npx -y @smithery/cli install @DeepSpringAI/parquet_mcp_server --client claude
```

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
EMBEDDING_URL=http://sample-url.com/api/embed  # URL for the embedding service
OLLAMA_URL=http://sample-url.com/  # URL for Ollama server
EMBEDDING_MODEL=sample-model  # Model to use for generating embeddings
SEARCHAPI_API_KEY=your_searchapi_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key
VOYAGE_API_KEY=your_voyage_api_key
AZURE_OPENAI_ENDPOINT=http://sample-url.com/azure_openai
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
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

1. **Search Web**: Perform a web search and scrape results
   - Required parameters:
     - `queries`: List of search queries
   - Optional parameters:
     - `page_number`: Page number for the search results (defaults to 1)

2. **Extract Info from Search**: Extract relevant information from previous searches
   - Required parameters:
     - `queries`: List of search queries to merge

## Example Prompts

Here are some example prompts you can use with the agent:

### For Web Search:
```
"Please perform a web search for 'macbook' and 'laptop' and scrape the results from page 1"
```

### For Extracting Info from Search:
```
"Please extract relevant information from the previous searches for 'macbook'"
```

## Testing the MCP Server

The project includes a comprehensive test suite in the `src/tests` directory. You can run all tests using:

```bash
python src/tests/run_tests.py
```

Or run individual tests:

```bash
# Test Web Search
python src/tests/test_search_web.py

# Test Extract Info from Search
python src/tests/test_extract_info_from_search.py
```

You can also test the server using the client directly:

```python
from parquet_mcp_server.client import (
    perform_search_and_scrape,  # New web search function
    find_similar_chunks  # New extract info function
)

# Perform a web search
perform_search_and_scrape(["macbook", "laptop"], page_number=1)

# Extract information from the search results
find_similar_chunks(["macbook"])
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
4. If PostgreSQL conversion fails, check:
   - The PostgreSQL connection settings in your `.env` file are correct
   - The PostgreSQL server is running and accessible
   - You have the necessary permissions to create/modify tables
   - The pgvector extension is installed in your database

## PostgreSQL Function for Vector Similarity Search

To perform vector similarity searches in PostgreSQL, you can use the following function:

```sql
-- Create the function for vector similarity search
CREATE OR REPLACE FUNCTION match_web_search(
  query_embedding vector(1024),  -- Adjusted vector size
  match_threshold float,
  match_count int  -- User-defined limit for number of results
)
RETURNS TABLE (
  id bigint,
  metadata jsonb,
  text TEXT,  -- Added text column to the result
  date TIMESTAMP,  -- Using the date column instead of created_at
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    web_search.id,
    web_search.metadata,
    web_search.text,  -- Returning the full text of the chunk
    web_search.date,  -- Returning the date timestamp
    1 - (web_search.embedding <=> query_embedding) as similarity
  FROM web_search
  WHERE 1 - (web_search.embedding <=> query_embedding) > match_threshold
  ORDER BY web_search.date DESC,  -- Sort by date in descending order (newest first)
           web_search.embedding <=> query_embedding  -- Sort by similarity
  LIMIT match_count;  -- Limit the results to the match_count specified by the user
END;
$$;
```

This function allows you to perform similarity searches on vector embeddings stored in a PostgreSQL database, returning results that meet a specified similarity threshold and limiting the number of results based on user input. The results are sorted by date and similarity.



## Postgres table creation
```
CREATE TABLE web_search (
    id SERIAL PRIMARY KEY,
    text TEXT,
    metadata JSONB,
    embedding VECTOR(1024),

    -- This will be auto-updated
    date TIMESTAMP DEFAULT NOW()
);
```