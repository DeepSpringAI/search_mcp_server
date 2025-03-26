
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
                "run",
                "main.py"
            ]
        }
    }
}
```

---

That's it! Now you can run the server using Claude Desktop.