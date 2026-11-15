"""A small MCP server with a clock and a notes file, for testing the client.

stdout is the protocol channel, so nothing else may be printed to it.
"""

import datetime
import json
import pathlib
import sys
import zoneinfo

NOTES_FILE = pathlib.Path(__file__).with_name("notes.json")
SERVER_INFO = {"name": "voxmux-demo", "version": "1.0"}

TOOLS = [
    {
        "name": "get_current_time",
        "description": "Get the current date and time in a time zone, such as Europe/London or Asia/Tokyo.",
        "inputSchema": {
            "type": "object",
            "properties": {"timezone": {"type": "string"}},
            "required": ["timezone"],
        },
    },
    {
        "name": "save_note",
        "description": "Save a short note so it can be read back later, even after the app is closed.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "list_notes",
        "description": "List every note saved so far.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def get_current_time(timezone):
    now = datetime.datetime.now(zoneinfo.ZoneInfo(timezone))
    return now.strftime(f"%A %d %B %Y, %H:%M in {timezone}")


def load_notes():
    if NOTES_FILE.exists():
        return json.loads(NOTES_FILE.read_text(encoding="utf-8"))
    return []


def save_note(text):
    notes = load_notes()
    notes.append(text)
    NOTES_FILE.write_text(json.dumps(notes, indent=2), encoding="utf-8")
    return f"Saved. There are now {len(notes)} notes."


def list_notes():
    notes = load_notes()
    if not notes:
        return "There are no notes yet."
    lines = []
    for number, note in enumerate(notes, start=1):
        lines.append(f"{number}. {note}")
    return "\n".join(lines)


HANDLERS = {
    "get_current_time": get_current_time,
    "save_note": save_note,
    "list_notes": list_notes,
}


def handle(message):
    method = message.get("method")
    if "id" not in message:
        return None

    if method == "initialize":
        version = message.get("params", {}).get("protocolVersion", "2025-06-18")
        result = {
            "protocolVersion": version,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = message.get("params", {})
        handler = HANDLERS.get(params.get("name"))
        try:
            if handler is None:
                raise ValueError(f"unknown tool {params.get('name')}")
            text = handler(**params.get("arguments", {}))
            result = {"content": [{"type": "text", "text": text}], "isError": False}
        except Exception as error:
            result = {
                "content": [{"type": "text", "text": str(error)}],
                "isError": True,
            }
    else:
        error = {"code": -32601, "message": f"Method not found: {method}"}
        return {"jsonrpc": "2.0", "id": message["id"], "error": error}

    return {"jsonrpc": "2.0", "id": message["id"], "result": result}


def main():
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except ValueError:
            continue
        response = handle(message)
        if response is not None:
            print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
