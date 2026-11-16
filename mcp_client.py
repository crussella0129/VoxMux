"""mcp_client.py: a small Model Context Protocol (MCP) client."""

import json
import os
import pathlib
import queue
import shutil
import subprocess
import sys
import threading

PROTOCOL_VERSION = "2025-06-18"
TIMEOUT_SECONDS = 60


class MCPServer:
    """One running MCP server and the tools it offers."""

    def __init__(self, name, command, args, folder, env=None):
        """Start the server, introduce ourselves, and list its tools."""
        self.name = name
        if command == "python":
            program = sys.executable
        else:
            program = shutil.which(command) or command
        environment = dict(os.environ)
        if env:
            environment.update(env)
        log_path = pathlib.Path(folder) / f"mcp-{name}.log"
        self.log = open(log_path, "w", encoding="utf-8")
        self.process = subprocess.Popen(
            [program] + list(args),
            cwd=folder,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            encoding="utf-8",
        )
        self.lines = queue.Queue()
        self.next_id = 1
        reader = threading.Thread(target=self.read_lines, daemon=True)
        reader.start()

        try:
            self.request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "voxmux", "version": "1.0"},
                },
            )
            self.notify("notifications/initialized")
            self.schemas = self.list_tools()
        except Exception:
            self.close()
            raise

    def read_lines(self):
        """Background thread: copy each line the server writes into the queue."""
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def send(self, message):
        """Write one message to the server as a single line of JSON."""
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def notify(self, method, params=None):
        """Send a notification: a message that gets no answer."""
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)

    def request(self, method, params):
        """Send a request and wait for the response with the same id."""
        request_id = self.next_id
        self.next_id = self.next_id + 1
        self.send(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )

        while True:
            try:
                line = self.lines.get(timeout=TIMEOUT_SECONDS)
            except queue.Empty:
                raise TimeoutError(f"MCP server {self.name} did not answer {method}")
            if line is None:
                raise ConnectionError(f"MCP server {self.name} stopped")
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if not isinstance(message, dict):
                continue
            if "method" in message:
                if "id" in message:
                    self.answer_server_request(message)
                continue
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(message["error"].get("message", "unknown error"))
                return message.get("result", {})

    def answer_server_request(self, message):
        """Answer a request the server sent us."""
        if message["method"] == "ping":
            self.send({"jsonrpc": "2.0", "id": message["id"], "result": {}})
        else:
            error = {"code": -32601, "message": "Method not found"}
            self.send({"jsonrpc": "2.0", "id": message["id"], "error": error})

    def list_tools(self):
        """Ask for every tool and turn each into a chat-template schema."""
        schemas = []
        cursor = None
        while True:
            params = {}
            if cursor:
                params["cursor"] = cursor
            result = self.request("tools/list", params)
            for item in result.get("tools", []):
                schemas.append(
                    {
                        "type": "function",
                        "function": {
                            "name": item["name"],
                            "description": item.get("description", ""),
                            "parameters": item.get("inputSchema", {"type": "object"}),
                        },
                    }
                )
            cursor = result.get("nextCursor")
            if not cursor:
                return schemas

    def call_tool(self, name, arguments):
        """Run one of this server's tools and return its result as text."""
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        parts = []
        for part in result.get("content", []):
            if part.get("type") == "text":
                parts.append(part["text"])
            else:
                parts.append(f"[{part.get('type')} content not shown]")
        text = "\n".join(parts)
        if result.get("isError"):
            return "Error: " + text
        return text

    def close(self):
        """Stop the server politely, then close its log file."""
        try:
            self.process.stdin.close()
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()
        self.log.close()


def start_servers(config_path):
    """Start every server listed in the config file."""
    path = pathlib.Path(config_path)
    servers = []
    problems = []
    if not path.exists():
        return servers, problems
    config = json.loads(path.read_text(encoding="utf-8"))
    for name, settings in config.get("mcpServers", {}).items():
        try:
            server = MCPServer(
                name,
                settings["command"],
                settings.get("args", []),
                path.parent,
                settings.get("env"),
            )
            servers.append(server)
        except Exception as error:
            problems.append(f"{name}: {error}")
    return servers, problems
