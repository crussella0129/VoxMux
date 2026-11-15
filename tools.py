"""Built-in tools, tool-call parsing, and the toolbox that runs local and MCP tools."""

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
    # A tool's docstring is the description the model reads, so keep it.
    TOOLS[function.__name__] = function
    return function


def evaluate(node):
    # Walks the syntax tree instead of calling eval(), so only arithmetic runs.
    if isinstance(node, ast.Expression):
        return evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left = evaluate(node.left)
        right = evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            # Refuse powers whose result would take minutes to compute.
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


def parse_tool_calls(text):
    before = text.split("<tool_call>")[0].strip()
    calls = []
    for body in CALL_PATTERN.findall(text):
        body = body.strip()
        # Qwen writes <function=...> XML; most other families write JSON.
        if body.startswith("{"):
            try:
                data = json.loads(body)
                arguments = data.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
            except (ValueError, AttributeError):
                continue
            calls.append({"name": data.get("name", ""), "arguments": arguments})
            continue
        match = FUNCTION_PATTERN.search(body)
        if match:
            arguments = {}
            for key, value in PARAMETER_PATTERN.findall(match.group(2)):
                arguments[key] = value.strip("\n")
            calls.append({"name": match.group(1), "arguments": arguments})
    return before, calls


def convert(value, type_name):
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
    def __init__(self, servers):
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
        name = schema["function"]["name"]
        if name in self.owners:
            return
        self.schemas.append(schema)
        self.owners[name] = owner
        parameters = schema["function"].get("parameters", {})
        self.properties[name] = parameters.get("properties", {})

    def run(self, name, arguments):
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
