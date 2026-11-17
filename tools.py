"""tools.py: the tools a model may call, and reading its calls back."""

import ast
import inspect
import json
import operator
import random
import re

TOOLS = {}
TYPE_NAMES = {str: "string", int: "integer", float: "number", bool: "boolean"}

CALL_PATTERN = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)
FUNCTION_PATTERN = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.DOTALL)
PARAMETER_PATTERN = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
LARGEST_POWER = 1000
LARGEST_RESULT_BITS = 100_000


def tool(function):
    """Decorator: register function as a tool the model may call."""
    TOOLS[function.__name__] = function
    return function


def evaluate(node):
    """Work out the value of one piece of a parsed arithmetic expression."""
    if isinstance(node, ast.Expression):
        return evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left = evaluate(node.left)
        right = evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            result_bits = abs(int(left)).bit_length() * abs(right)
            if abs(right) > LARGEST_POWER or result_bits > LARGEST_RESULT_BITS:
                raise ValueError("that power is too large")
        return OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](evaluate(node.operand))
    raise ValueError("only numbers, brackets and + - * / // % ** are allowed")


@tool
def calculate(expression: str) -> str:
    """Work out an arithmetic expression exactly, for example (17 * 23) + 4 / 2. Use it for any sum you are not certain of."""
    return str(evaluate(ast.parse(expression, mode="eval")))


@tool
def random_number(low: int, high: int) -> str:
    """Pick a random whole number between low and high, including both ends. Use it for dice rolls and random choices."""
    return str(random.randint(low, high))


def schema_for(function):
    """Describe a function the way chat templates expect a tool described."""
    properties = {}
    required = []
    for name, parameter in inspect.signature(function).parameters.items():
        properties[name] = {"type": TYPE_NAMES.get(parameter.annotation, "string")}
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": function.__name__,
            "description": inspect.getdoc(function),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def text_before_call(text):
    """Return the part of a reply that comes before any tool call."""
    if text.lstrip()[:1] in ("{", "["):
        return ""
    return text.split("<tool_call>")[0]


def json_calls(text):
    """Read every tool call that is written as JSON somewhere in text."""
    decoder = json.JSONDecoder()
    calls = []
    position = 0
    while position < len(text):
        if text[position] not in "{[":
            position = position + 1
            continue
        try:
            value, position = decoder.raw_decode(text, position)
        except ValueError:
            position = position + 1
            continue
        if isinstance(value, dict):
            value = [value]
        for item in value:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("function")
            arguments = item.get("arguments", item.get("parameters"))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    continue
            if isinstance(name, str) and isinstance(arguments, dict):
                calls.append({"name": name, "arguments": arguments})
    return calls


def parse_tool_calls(text):
    """Split a reply into its text and its tool calls."""
    before = text_before_call(text)
    rest = text[len(before) :]
    calls = []
    for body in CALL_PATTERN.findall(rest):
        match = FUNCTION_PATTERN.search(body)
        if match:
            arguments = {}
            for key, value in PARAMETER_PATTERN.findall(match.group(2)):
                arguments[key] = value.strip("\n")
            calls.append({"name": match.group(1), "arguments": arguments})
    if not calls:
        calls = json_calls(rest)
    if not calls:
        return text.strip(), []
    return before.strip(), calls


def convert(value, type_name):
    """Turn a value the model wrote into the type a tool asked for."""
    if not isinstance(value, str):
        return value
    if type_name == "integer":
        return int(value)
    if type_name == "number":
        return float(value)
    if type_name == "boolean":
        return value.strip().lower() == "true"
    if type_name in ("array", "object"):
        return json.loads(value)
    return value


class Toolbox:
    """Every tool the model may call, built-in or from an MCP server."""

    def __init__(self, servers):
        """Gather the built-in tools, then each server's tools."""
        self.servers = servers
        self.schemas = []
        self.owners = {}
        self.properties = {}
        for function in TOOLS.values():
            self.add(schema_for(function), function)
        for server in servers:
            for schema in server.schemas:
                self.add(schema, server)

    def add(self, schema, owner):
        """Add one tool, unless a tool with that name is already in."""
        name = schema["function"]["name"]
        if name in self.owners:
            return
        self.schemas.append(schema)
        self.owners[name] = owner
        parameters = schema["function"].get("parameters", {})
        self.properties[name] = parameters.get("properties", {})

    def run(self, name, arguments):
        """Run the tool called name and return its result as text."""
        if name not in self.owners:
            return f"Error: there is no tool called {name}."
        try:
            wanted = self.properties[name]
            converted = {}
            for key, value in arguments.items():
                type_name = wanted.get(key, {}).get("type")
                converted[key] = convert(value, type_name)
            owner = self.owners[name]
            if callable(owner):
                return str(owner(**converted))
            return owner.call_tool(name, converted)
        except Exception as error:
            return f"Error: {error}"
