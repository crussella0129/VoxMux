"""VoxMux: a desktop chat window for local models.

Models run on worker threads, which hand results to the window through a
queue. Only the main thread touches the widgets.
"""

import pathlib
import queue
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext

from backends import load_backend
from mcp_client import start_servers
from tools import Toolbox, parse_tool_calls

POLL_INTERVAL_MS = 50
DEFAULT_TEMPERATURE = 0.7
DEFAULT_THINKING_INSTRUCTIONS = (
    "Think through the question step by step before you answer."
)

MCP_CONFIG = pathlib.Path(__file__).with_name("mcp_servers.json")
MAX_TOOL_ROUNDS = 5


class ChatApp:
    def __init__(self, root):
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

        self.build_widgets()
        self.poll_results()

    def build_widgets(self):
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

        self.chat_log = scrolledtext.ScrolledText(
            self.root, wrap="word", state="disabled"
        )
        self.chat_log.pack(fill="both", expand=True, padx=8, pady=8)

    def browse(self):
        if self.backend_kind.get() == "llama_cpp":
            path = filedialog.askopenfilename(filetypes=[("GGUF model", "*.gguf")])
        else:
            path = filedialog.askdirectory()

        if path:
            self.model_path = path
            self.status.config(text=f"Selected: {path}")

    def open_thinking_window(self):
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
        self.thinking_instructions = box.get("1.0", "end").strip()
        window.destroy()

    def load_model(self):
        if not self.model_path:
            self.status.config(text="Pick a model with Browse first.")
            return

        self.set_busy(True)
        self.status.config(text="Loading model...")
        self.start_worker(self.load_worker, self.backend_kind.get(), self.model_path)

    def send(self, event=None):
        text = self.entry.get().strip()
        if not text or self.backend is None or self.busy:
            return

        system_prompt = self.system_box.get("1.0", "end").strip()
        temperature = self.temperature.get()
        thinking = self.thinking.get()
        use_tools = self.use_tools.get()
        if thinking and self.thinking_instructions:
            system_prompt = system_prompt + "\n\n" + self.thinking_instructions
            system_prompt = system_prompt.strip()

        self.entry.delete(0, "end")
        self.messages.append({"role": "user", "content": text})
        self.append_chat(f"You: {text}\n\nModel: ")

        self.set_busy(True)
        self.status.config(text="Writing a reply...")
        self.start_worker(
            self.reply_worker, system_prompt, temperature, thinking, use_tools
        )

    def start_worker(self, target, *args):
        thread = threading.Thread(target=target, args=args, daemon=True)
        thread.start()

    def load_worker(self, kind, model_path):
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
        try:
            tools = None
            if use_tools:
                tools = self.toolbox.schemas
            for round_number in range(MAX_TOOL_ROUNDS):
                answer = self.stream_reply(system_prompt, temperature, thinking, tools)
                text, calls = parse_tool_calls(answer)
                if not calls or not use_tools:
                    self.results.put(("reply", text))
                    return

                tool_calls = []
                for call in calls:
                    tool_calls.append({"type": "function", "function": call})
                self.messages.append(
                    {"role": "assistant", "content": text, "tool_calls": tool_calls}
                )
                for call in calls:
                    self.results.put(("status", f"Running tool {call['name']}..."))
                    result = self.toolbox.run(call["name"], call["arguments"])
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
            self.results.put(("error", str(error)))

    def stream_reply(self, system_prompt, temperature, thinking, tools):
        reply = ""
        shown = ""
        for piece in self.backend.chat(
            self.messages, system_prompt, temperature, thinking, tools
        ):
            reply = reply + piece
            # Tool-call markup is for the app, not the reader: hide it.
            visible = reply.split("<tool_call>")[0]
            visible = visible.replace("</think>", "\n[end of thinking]").strip()
            start = len(shown)
            new_text = visible[start:]
            if new_text:
                self.results.put(("piece", new_text))
                shown = visible
        return reply.split("</think>")[-1].strip()

    def poll_results(self):
        while not self.results.empty():
            kind, value = self.results.get()

            if kind == "piece":
                self.append_chat(value)
            elif kind == "status":
                self.status.config(text=value)
            elif kind == "reply":
                self.messages.append({"role": "assistant", "content": value})
                self.append_chat("\n\n")
                self.status.config(text=f"Ready: {self.model_path}")
                self.set_busy(False)
            elif kind == "loaded":
                backend, toolbox, problems = value
                self.backend = backend
                self.toolbox = toolbox
                self.messages = []
                backend.on_compact = lambda: self.results.put(
                    ("status", "Summarising older messages to make room...")
                )
                text = (
                    f"Ready: {self.model_path} ({backend.context_size} tokens of "
                    f"context, {len(toolbox.schemas)} tools)"
                )
                if problems:
                    text = text + " MCP problems: " + "; ".join(problems)
                self.status.config(text=text)
                self.set_busy(False)
            elif kind == "error":
                self.status.config(text=f"Error: {value}")
                self.set_busy(False)

        self.root.after(POLL_INTERVAL_MS, self.poll_results)

    def append_chat(self, text):
        self.chat_log.config(state="normal")
        self.chat_log.insert("end", text)
        self.chat_log.config(state="disabled")
        self.chat_log.see("end")

    def set_busy(self, busy):
        self.busy = busy
        if busy:
            state = "disabled"
        else:
            state = "normal"
        self.load_button.config(state=state)
        self.send_button.config(state=state)

    def close(self):
        if self.toolbox is not None:
            for server in self.toolbox.servers:
                server.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    ChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
