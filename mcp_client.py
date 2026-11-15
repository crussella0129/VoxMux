"""A minimal MCP client: JSON-RPC 2.0 over a server's stdin and stdout."""

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
    def __init__(self, name, command, args, folder, env=None):
        self.name = name
        # "python" means the interpreter running VoxMux, not whatever is on PATH.
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
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def send(self, message):
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def notify(self, method, params=None):
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)

    def request(self, method, params):
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
        if message["method"] == "ping":
            self.send({"jsonrpc": "2.0", "id": message["id"], "result": {}})
        else:
            error = {"code": -32601, "message": "Method not found"}
            self.send({"jsonrpc": "2.0", "id": message["id"], "error": error})

    def list_tools(self):
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
        # Closing stdin asks the server to exit; kill it only if it does not.
        try:
            self.process.stdin.close()
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()
        self.log.close()


def start_servers(config_path):
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
