"""app.py: the VoxMux window, a tiny desktop chat for local models."""

import json
import pathlib
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, scrolledtext

from backends import load_backend
from mcp_client import start_servers
from tools import Toolbox, parse_tool_calls, text_before_call

POLL_INTERVAL_MS = 50
DEFAULT_TEMPERATURE = 0.7
DEFAULT_THINKING_INSTRUCTIONS = (
    "Think through the question step by step before you answer."
)

MCP_CONFIG = pathlib.Path(__file__).with_name("mcp_servers.json")
MAX_TOOL_ROUNDS = 5

SETTINGS_FILE = pathlib.Path(__file__).with_name("settings.json")
DENIED = "Error: the user did not allow this tool call."

CHATS_FOLDER = pathlib.Path(__file__).with_name("chats")


class ChatApp:
    """The whole window: its widgets, its state, and its behaviour."""

    def __init__(self, root):
        """Set up the window, state, and queue check."""
        self.root = root
        self.root.title("VoxMux")
        self.root.geometry("800x600")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.backend = None
        self.model_path = ""
        self.messages = []
        self.busy = False
        self.results = queue.Queue()
        self.thinking_instructions = DEFAULT_THINKING_INSTRUCTIONS
        self.toolbox = None
        self.stop_requested = threading.Event()
        self.answers = queue.Queue()
        self.allowed_tools = []
        self.chat_file = None
        self.offer_tools = False

        self.build_widgets()
        self.load_settings()
        self.poll_results()

    def build_widgets(self):
        """Create every widget and place it in the window."""
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=8)

        self.backend_kind = tk.StringVar(value="llama_cpp")
        tk.Radiobutton(
            top,
            text="llama.cpp (GGUF)",
            variable=self.backend_kind,
            value="llama_cpp",
        ).pack(side="left")
        tk.Radiobutton(
            top,
            text="PyTorch (safetensors)",
            variable=self.backend_kind,
            value="pytorch",
        ).pack(side="left")

        tk.Button(top, text="Browse...", command=self.browse).pack(side="left", padx=8)
        self.load_button = tk.Button(top, text="Load", command=self.load_model)
        self.load_button.pack(side="left")
        self.new_chat_button = tk.Button(top, text="New chat", command=self.new_chat)
        self.new_chat_button.pack(side="right")
        self.open_chat_button = tk.Button(
            top, text="Open chat...", command=self.open_chat
        )
        self.open_chat_button.pack(side="right", padx=8)

        settings = tk.Frame(self.root)
        settings.pack(fill="x", padx=8)

        tk.Label(settings, text="Temperature").pack(side="left")
        self.temperature = tk.DoubleVar(value=DEFAULT_TEMPERATURE)
        tk.Scale(
            settings,
            variable=self.temperature,
            from_=0,
            to=2,
            resolution=0.1,
            orient="horizontal",
            length=160,
        ).pack(side="left", padx=(4, 16))

        self.thinking = tk.BooleanVar(value=False)
        tk.Checkbutton(settings, text="Thinking mode", variable=self.thinking).pack(
            side="left"
        )
        tk.Button(
            settings,
            text="Thinking instructions...",
            command=self.open_thinking_window,
        ).pack(side="left", padx=8)

        self.use_tools = tk.BooleanVar(value=True)
        tk.Checkbutton(settings, text="Use tools", variable=self.use_tools).pack(
            side="left", padx=(8, 0)
        )

        tk.Label(self.root, text="System prompt:", anchor="w").pack(
            fill="x", padx=8, pady=(8, 0)
        )
        self.system_box = tk.Text(self.root, height=3, wrap="word")
        self.system_box.pack(fill="x", padx=8)

        self.status = tk.Label(self.root, text="No model loaded.", anchor="w")
        self.status.pack(fill="x", padx=8, pady=(8, 0))

        bottom = tk.Frame(self.root)
        bottom.pack(side="bottom", fill="x", padx=8, pady=8)

        self.entry = tk.Entry(bottom)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", self.send)

        self.send_button = tk.Button(bottom, text="Send", command=self.send)
        self.send_button.pack(side="left", padx=(8, 0))
        self.stop_button = tk.Button(
            bottom, text="Stop", command=self.stop, state="disabled"
        )
        self.stop_button.pack(side="left", padx=(8, 0))

        self.chat_log = scrolledtext.ScrolledText(
            self.root, wrap="word", state="disabled"
        )
        self.chat_log.pack(fill="both", expand=True, padx=8, pady=8)

    def browse(self):
        """Browse button: let the user pick a model, remember the path."""
        if self.backend_kind.get() == "llama_cpp":
            path = filedialog.askopenfilename(filetypes=[("GGUF model", "*.gguf")])
        else:
            path = filedialog.askdirectory()

        if path:
            self.model_path = path
            self.status.config(text=f"Selected: {path}")

    def open_thinking_window(self):
        """Thinking instructions button: open a small editing window."""
        window = tk.Toplevel(self.root)
        window.title("Thinking mode instructions")

        tk.Label(
            window,
            text="Added to the system prompt whenever Thinking mode is on:",
            anchor="w",
        ).pack(fill="x", padx=8, pady=(8, 0))
        box = tk.Text(window, width=60, height=8, wrap="word")
        box.pack(fill="both", expand=True, padx=8, pady=8)
        box.insert("1.0", self.thinking_instructions)

        tk.Button(
            window,
            text="Save",
            command=lambda: self.save_thinking_instructions(window, box),
        ).pack(pady=(0, 8))

    def save_thinking_instructions(self, window, box):
        """Save button: keep the edited instructions, close the window."""
        self.thinking_instructions = box.get("1.0", "end").strip()
        window.destroy()

    def load_model(self):
        """Load button: start loading the picked model."""
        if not self.model_path:
            self.status.config(text="Pick a model with Browse first.")
            return

        self.set_busy(True)
        if self.backend is not None:
            self.backend.unload()
            self.backend = None
        self.status.config(text="Loading model...")
        self.start_worker(self.load_worker, self.backend_kind.get(), self.model_path)

    def new_chat(self):
        """New chat button: start a fresh conversation."""
        self.messages = []
        self.chat_file = None
        self.chat_log.config(state="normal")
        self.chat_log.delete("1.0", "end")
        self.chat_log.config(state="disabled")

    def open_chat(self):
        """Open chat button: pick a saved conversation and show it."""
        path = filedialog.askopenfilename(
            initialdir=CHATS_FOLDER, filetypes=[("Saved chat", "*.json")]
        )
        if path:
            self.read_chat(path)

    def read_chat(self, path):
        """Bring one saved conversation back into the window."""
        try:
            saved = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
            messages = saved["messages"]
            log = saved["log"]
        except (OSError, ValueError, KeyError, TypeError):
            self.status.config(text=f"Could not open the chat in {path}")
            return
        self.new_chat()
        self.messages = messages
        self.append_chat(log)
        self.chat_file = pathlib.Path(path)

    def save_chat(self):
        """Write this conversation to its file in the chats folder."""
        try:
            if self.chat_file is None:
                CHATS_FOLDER.mkdir(exist_ok=True)
                name = time.strftime("%Y-%m-%d %H.%M.%S")
                self.chat_file = CHATS_FOLDER / f"{name}.json"
                number = 2
                while self.chat_file.exists():
                    self.chat_file = CHATS_FOLDER / f"{name} ({number}).json"
                    number = number + 1
            saved = {
                "messages": self.messages,
                "log": self.chat_log.get("1.0", "end-1c"),
            }
            text = json.dumps(saved, indent=2, ensure_ascii=False)
            self.chat_file.write_text(text, encoding="utf-8")
        except OSError as error:
            self.status.config(text=f"Could not save this chat: {error}")

    def send(self, event=None):
        """Send button or Enter: add the message, ask."""
        text = self.entry.get().strip()
        if not text or self.backend is None or self.busy:
            return

        system_prompt = self.system_box.get("1.0", "end").strip()
        temperature = self.temperature.get()
        thinking = self.thinking.get()
        use_tools = self.use_tools.get() and self.offer_tools
        if thinking and self.thinking_instructions:
            system_prompt = system_prompt + "\n\n" + self.thinking_instructions
            system_prompt = system_prompt.strip()

        self.entry.delete(0, "end")
        self.messages.append({"role": "user", "content": text})
        self.append_chat(f"You: {text}\n\nModel: ")

        self.set_busy(True)
        self.stop_requested.clear()
        self.stop_button.config(state="normal")
        self.status.config(text="Writing a reply...")
        self.start_worker(
            self.reply_worker, system_prompt, temperature, thinking, use_tools
        )

    def stop(self):
        """Stop button: ask the worker to cut the reply short."""
        self.stop_requested.set()
        self.stop_button.config(state="disabled")
        self.status.config(text="Stopping...")

    def start_worker(self, target, *args):
        """Run target(*args) on a new background thread."""
        thread = threading.Thread(target=target, args=args, daemon=True)
        thread.start()

    def load_worker(self, kind, model_path):
        """Background thread: load a model and tools."""
        try:
            backend = load_backend(kind, model_path)
            toolbox = self.toolbox
            problems = []
            if toolbox is None:
                servers, problems = start_servers(MCP_CONFIG)
                toolbox = Toolbox(servers)
            self.results.put(("loaded", (backend, toolbox, problems)))
        except Exception as error:
            self.results.put(("error", str(error)))

    def reply_worker(self, system_prompt, temperature, thinking, use_tools):
        """Background thread: reply, using tools."""
        try:
            tools = None
            if use_tools:
                tools = self.toolbox.schemas
            for round_number in range(MAX_TOOL_ROUNDS):
                text, calls = self.stream_reply(
                    system_prompt, temperature, thinking, tools
                )
                stopped = self.stop_requested.is_set()
                if stopped or not calls:
                    if stopped:
                        self.results.put(("piece", " [stopped]"))
                    self.results.put(("reply", text))
                    return

                tool_calls = []
                for call in calls:
                    tool_calls.append({"type": "function", "function": call})
                self.messages.append(
                    {"role": "assistant", "content": text, "tool_calls": tool_calls}
                )
                for call in calls:
                    if self.may_run(call):
                        self.results.put(("status", f"Running tool {call['name']}..."))
                        result = self.toolbox.run(call["name"], call["arguments"])
                    else:
                        result = DENIED
                    self.results.put(
                        (
                            "piece",
                            f"\n[{call['name']} {call['arguments']} -> {result}]\n",
                        )
                    )
                    self.messages.append(
                        {"role": "tool", "name": call["name"], "content": result}
                    )
                self.results.put(("status", "Writing a reply..."))
            note = "(Stopped: too many tool calls in a row.)"
            self.results.put(("piece", note))
            self.results.put(("reply", note))
        except Exception as error:
            self.results.put(("failed", str(error)))

    def stream_reply(self, system_prompt, temperature, thinking, tools):
        """Stream one reply; return its text and calls."""
        reply = ""
        shown = ""
        pieces = self.backend.chat(
            self.messages, system_prompt, temperature, thinking, tools
        )
        for piece in pieces:
            if self.stop_requested.is_set():
                break
            reply = reply + piece
            thoughts, end_mark, answer = reply.rpartition("</think>")
            if tools:
                answer = text_before_call(answer)
            visible = thoughts + end_mark + answer
            visible = visible.replace("</think>", "\n[end of thinking]").strip()
            start = len(shown)
            new_text = visible[start:]
            if new_text:
                self.results.put(("piece", new_text))
                shown = visible
        pieces.close()

        answer = reply.split("</think>")[-1].strip()
        if not tools:
            return answer, []
        text, calls = parse_tool_calls(answer)
        if not calls:
            whole = reply.replace("</think>", "\n[end of thinking]").strip()
            held_back = whole[len(shown) :]
            if held_back:
                self.results.put(("piece", held_back))
        return text, calls

    def may_run(self, call):
        """Background thread: is this tool call allowed to run?"""
        name = call["name"]
        owner = self.toolbox.owners.get(name)
        if owner is None or callable(owner) or name in self.allowed_tools:
            return True
        self.results.put(("status", f"Waiting for your answer about {name}..."))
        self.results.put(("ask", call))
        return self.answers.get() != "deny"

    def poll_results(self):
        """Main thread, on a timer: apply results."""
        while not self.results.empty():
            kind, value = self.results.get()

            if kind == "piece":
                self.append_chat(value)
            elif kind == "status":
                self.status.config(text=value)
            elif kind == "ask":
                self.open_approval_window(value)
            elif kind == "reply":
                self.messages.append({"role": "assistant", "content": value})
                self.append_chat("\n\n")
                self.status.config(text=f"Ready: {self.model_path}")
                self.save_chat()
                self.set_busy(False)
            elif kind == "loaded":
                backend, toolbox, problems = value
                self.backend = backend
                self.toolbox = toolbox
                self.offer_tools = backend.offers_tools(toolbox.schemas)
                backend.on_compact = lambda: self.results.put(
                    ("status", "Summarising older messages to make room...")
                )
                text = (
                    f"Ready: {self.model_path} ({backend.context_size} tokens of "
                    f"context, {len(toolbox.schemas)} tools)"
                )
                if not self.offer_tools:
                    text = text + " This model's template has no place for tools."
                if problems:
                    text = text + " MCP problems: " + "; ".join(problems)
                self.status.config(text=text)
                self.set_busy(False)
            elif kind == "failed":
                while self.messages and self.messages.pop()["role"] != "user":
                    pass
                self.append_chat(" [failed]\n\n")
                self.status.config(text=f"Error: {value}")
                self.set_busy(False)
            elif kind == "error":
                self.status.config(text=f"Error: {value}")
                self.set_busy(False)

        self.root.after(POLL_INTERVAL_MS, self.poll_results)

    def open_approval_window(self, call):
        """Main thread: ask whether one MCP tool call may run."""
        name = call["name"]
        window = tk.Toplevel(self.root)
        window.title("Allow this tool?")
        window.protocol(
            "WM_DELETE_WINDOW", lambda: self.answer_tool(window, name, "deny")
        )

        tk.Label(
            window,
            text=f"The model wants to run {name} with:\n{call['arguments']}",
            justify="left",
            wraplength=420,
        ).pack(padx=8, pady=8)

        buttons = tk.Frame(window)
        buttons.pack(padx=8, pady=(0, 8))
        tk.Button(
            buttons,
            text="Allow once",
            command=lambda: self.answer_tool(window, name, "once"),
        ).pack(side="left")
        tk.Button(
            buttons,
            text="Always allow",
            command=lambda: self.answer_tool(window, name, "always"),
        ).pack(side="left", padx=8)
        tk.Button(
            buttons,
            text="Deny",
            command=lambda: self.answer_tool(window, name, "deny"),
        ).pack(side="left")

    def answer_tool(self, window, name, answer):
        """A button in the approval window: pass the answer on."""
        if answer == "always":
            self.allowed_tools.append(name)
        self.answers.put(answer)
        window.destroy()

    def append_chat(self, text):
        """Add text to the end of the chat log, exactly as given."""
        self.chat_log.config(state="normal")
        self.chat_log.insert("end", text)
        self.chat_log.config(state="disabled")
        self.chat_log.see("end")

    def set_busy(self, busy):
        """Grey out the buttons while work is running."""
        self.busy = busy
        if busy:
            state = "disabled"
        else:
            state = "normal"
        self.load_button.config(state=state)
        self.send_button.config(state=state)
        self.new_chat_button.config(state=state)
        self.open_chat_button.config(state=state)
        if not busy:
            self.stop_button.config(state="disabled")

    def load_settings(self):
        """Read settings.json back into the window."""
        if not SETTINGS_FILE.exists():
            return
        try:
            saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            self.backend_kind.set(saved["backend_kind"])
            self.model_path = saved["model_path"]
            if self.model_path:
                self.status.config(text=f"Selected: {self.model_path}")
            self.temperature.set(saved["temperature"])
            self.thinking.set(saved["thinking"])
            self.thinking_instructions = saved["thinking_instructions"]
            self.use_tools.set(saved["use_tools"])
            self.system_box.insert("1.0", saved["system_prompt"])
            self.allowed_tools = saved["allowed_tools"]
            if saved["chat_file"]:
                self.read_chat(saved["chat_file"])
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def save_settings(self):
        """Write the current settings to settings.json."""
        chat_file = ""
        if self.chat_file is not None:
            chat_file = str(self.chat_file)
        saved = {
            "backend_kind": self.backend_kind.get(),
            "model_path": self.model_path,
            "temperature": self.temperature.get(),
            "thinking": self.thinking.get(),
            "thinking_instructions": self.thinking_instructions,
            "use_tools": self.use_tools.get(),
            "system_prompt": self.system_box.get("1.0", "end").strip(),
            "allowed_tools": self.allowed_tools,
            "chat_file": chat_file,
        }
        SETTINGS_FILE.write_text(json.dumps(saved, indent=2), encoding="utf-8")

    def close(self):
        """Window closed: save, stop servers, close."""
        try:
            self.save_settings()
        except OSError:
            pass
        if self.toolbox is not None:
            for server in self.toolbox.servers:
                server.close()
        self.root.destroy()


def main():
    """Create the window, attach the app to it, and run until closed."""
    root = tk.Tk()
    ChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
