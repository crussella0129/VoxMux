"""backends.py: the two ways VoxMux can run a model."""

import threading

MAX_NEW_TOKENS = 4096
SMALLEST_CONTEXT = 4096

SUMMARY_INSTRUCTIONS = (
    "Summarise the conversation below so it can be continued later. Keep "
    "every name, number, fact, decision and open question. Write plain "
    "sentences, no more than about 200 words."
)


def find_user(messages, start):
    """Return the index of the first user message at or after start."""
    for index in range(start, len(messages)):
        if messages[index]["role"] == "user":
            return index
    return None


class Backend:
    """What every backend shares: fit, then stream."""

    on_compact = None

    def chat(self, messages, system_prompt, temperature, thinking, tools=None):
        """Fit the conversation, then stream a reply."""
        prompt = self.fit_to_context(messages, system_prompt, thinking, tools)
        yield from self.stream(prompt, temperature)

    def fit_to_context(self, messages, system_prompt, thinking, tools):
        """Build the prompt, making room until it fits."""
        reserve = min(MAX_NEW_TOKENS, self.context_size // 2)
        room = self.context_size - reserve
        while True:
            full = messages
            if system_prompt:
                full = [{"role": "system", "content": system_prompt}] + messages
            prompt = self.build_prompt(full, thinking, tools)
            if self.count_tokens(prompt) <= room:
                return prompt
            before = len(messages)
            self.compact(messages, room)
            if len(messages) < before:
                continue
            second_user = find_user(messages, 1)
            if second_user is None:
                return prompt
            del messages[:second_user]

    def compact(self, messages, room):
        """Replace the older half of messages with a short summary."""
        cut = find_user(messages, max(len(messages) // 2, 3))
        if cut is None:
            return
        older = messages[:cut]

        lines = []
        for message in older:
            if message.get("content"):
                lines.append(f"{message['role']}: {message['content']}")
        request = [
            {"role": "system", "content": SUMMARY_INSTRUCTIONS},
            {"role": "user", "content": "\n\n".join(lines)},
        ]
        prompt = self.build_prompt(request, False, None)
        if self.count_tokens(prompt) > room:
            return

        if self.on_compact:
            self.on_compact()
        summary = "".join(self.stream(prompt, 0)).split("</think>")[-1].strip()
        messages[:cut] = [
            {
                "role": "user",
                "content": "Summary of our conversation so far:\n" + summary,
            },
            {"role": "assistant", "content": "Thanks, I have that summary."},
        ]


class LlamaCppBackend(Backend):
    """Runs a single .gguf file through llama.cpp."""

    def __init__(self, model_path):
        """Load the .gguf file with as much context as will fit."""
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        info = Llama(model_path=model_path, vocab_only=True, verbose=False)
        architecture = info.metadata["general.architecture"]
        size = int(info.metadata[f"{architecture}.context_length"])

        while True:
            try:
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=size,
                    n_gpu_layers=-1,
                    verbose=False,
                )
                break
            except ValueError as error:
                if "llama_context" not in str(error) or size // 2 < SMALLEST_CONTEXT:
                    raise
                size = size // 2
        self.context_size = size

        end_token = info.detokenize([info.token_eos()], special=True)
        start_token = info.detokenize([info.token_bos()], special=True)
        self.formatter = Jinja2ChatFormatter(
            template=info.metadata["tokenizer.chat_template"],
            eos_token=end_token.decode("utf-8", errors="ignore"),
            bos_token=start_token.decode("utf-8", errors="ignore"),
        )

    def build_prompt(self, messages, thinking, tools=None):
        """Turn messages into one prompt string."""
        response = self.formatter(
            messages=messages, tools=tools, enable_thinking=thinking
        )
        return response.prompt

    def count_tokens(self, prompt):
        """Count how many tokens a prompt string takes up."""
        tokens = self.llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        return len(tokens)

    def stream(self, prompt, temperature):
        """Yield the text the model writes after prompt, piece by piece."""
        chunks = self.llm.create_completion(
            prompt,
            max_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            stream=True,
        )
        for chunk in chunks:
            text = chunk["choices"][0]["text"]
            if text:
                yield text


class TransformersBackend(Backend):
    """Runs a folder of safetensors weights through PyTorch."""

    def __init__(self, model_path):
        """Load the tokenizer and the model from the folder model_path."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype="auto")
        self.model.to(self.device)

        text_config = self.model.config.get_text_config()
        self.context_size = text_config.max_position_embeddings

    def build_prompt(self, messages, thinking, tools=None):
        """Turn messages into one prompt string."""
        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=thinking,
        )

    def count_tokens(self, prompt):
        """Count how many tokens a prompt string takes up."""
        tokens = self.tokenizer(prompt, add_special_tokens=False)
        return len(tokens["input_ids"])

    def stream(self, prompt, temperature):
        """Yield the text the model writes after prompt."""
        from transformers import StoppingCriteriaList, TextIteratorStreamer

        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        inputs = inputs.to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        stop = threading.Event()
        failures = []

        def should_stop(input_ids, scores, **extra):
            return stop.is_set()

        settings = dict(inputs)
        settings["max_new_tokens"] = MAX_NEW_TOKENS
        settings["streamer"] = streamer
        settings["stopping_criteria"] = StoppingCriteriaList([should_stop])
        if temperature > 0:
            settings["do_sample"] = True
            settings["temperature"] = temperature
        else:
            settings["do_sample"] = False

        def run():
            try:
                self.model.generate(**settings)
            except Exception as error:
                failures.append(error)
                streamer.end()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        try:
            for text in streamer:
                if text:
                    yield text
        finally:
            stop.set()
            thread.join()
        if failures:
            raise failures[0]


def load_backend(kind, model_path):
    """Build and return the backend named by kind."""
    if kind == "llama_cpp":
        return LlamaCppBackend(model_path)
    if kind == "pytorch":
        return TransformersBackend(model_path)
    raise ValueError(f"Unknown backend: {kind}")
