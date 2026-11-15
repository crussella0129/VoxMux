# VoxMux

The Local AI Multiplexer: a desktop chat app for running local models through llama.cpp (GGUF files) or PyTorch (safetensors folders), without a browser.

Want to build it yourself, line by line? See the step-by-step tutorial at [VoxMux_Tutorial](https://github.com/crussella0129/VoxMux_Tutorial). This repository is the finished app.

## Features

- **Two backends:** llama.cpp for a `.gguf` file, or PyTorch (Hugging Face transformers) for a folder of `.safetensors` weights.
- **Streaming replies,** shown as the model writes them.
- **System prompt and temperature** controls.
- **Thinking mode,** with your own thinking instructions. Reasoning is marked off from the answer.
- **The largest context your graphics card can hold.** The app asks for the model's full trained context and halves it until the model loads.
- **Compaction.** When a conversation outgrows the context, older messages are summarised by the model rather than dropped.
- **Tool calling.** The model gets an exact calculator and a random-number tool.
- **An MCP client.** Any [Model Context Protocol](https://modelcontextprotocol.io) server listed in `mcp_servers.json` adds its tools. A small demo server, with a clock and a notes file, is included.

## Requirements

- Windows
- An NVIDIA graphics card with a current driver
- Python 3.13

## Install

```bash
python install.py
```

This installs the CUDA builds of PyTorch and llama-cpp-python, transformers and tzdata into the Python you ran it with, then copies three CUDA runtime files that llama.cpp needs. It ends by checking that both libraries can see the graphics card. You should see `cuda: True llama gpu: True`. It is safe to run again.

Do not use `pip install -r requirements.txt` on Windows. It would fetch the CPU-only PyTorch and try to compile llama-cpp-python from source.

## Run

```bash
python app.py
```

1. Pick **llama.cpp** or **PyTorch**.
2. Click **Browse** to choose a `.gguf` file or a model folder.
3. Click **Load**, and wait for the status line to say **Ready**.
4. Type a message and press Enter.

Tested with Qwen3.5 models (0.8B, 2B and 4B) on an RTX 2080 Ti.

## MCP servers

`mcp_servers.json` uses the same `mcpServers` layout as other MCP clients:

```json
{
  "mcpServers": {
    "demo": {"command": "python", "args": ["demo_mcp_server.py"]}
  }
}
```

Servers start when the first model loads and stop when the window closes. Each server's error output goes to `mcp-<name>.log`. Only the included demo server has been tested.

## Files

| File | What it is |
| --- | --- |
| `app.py` | The window, the worker threads and the tool-calling loop. |
| `backends.py` | The llama.cpp and PyTorch backends, plus context trimming and compaction. |
| `tools.py` | The built-in tools, tool-call parsing, and the toolbox. |
| `mcp_client.py` | Starts MCP servers and talks JSON-RPC to them. |
| `demo_mcp_server.py` | The demo MCP server. |
| `install.py` | Installs the dependencies. |

## License

MIT
