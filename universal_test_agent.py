# universal_test_agent_v2.py
# -*- coding: utf-8 -*-
"""
Universal Python Unit Test Generator AI Agent — v2
==================================================
Optimized, project-wide, context-aware unit test generator that supports:
1) Any project structure (packages, Django/Flask, DS/ML, microservices, CLI tools, libs)
2) All function types (async, generators, methods, properties, etc.)
3) Complex dependencies (imports, external libs, circular, dynamic)
4) Advanced Python features (type hints, dataclasses, ABCs, descriptors, etc.)
5) Real-world patterns (DB, I/O, HTTP, logging, threading/async)

Key Capabilities
----------------
- Recursive project scanning with exclusion patterns
- Advanced AST parsing for functions, classes, async, generators, properties
- Batch LLM calls to minimize cost (pluggable model providers: Gemini, OpenAI, Ollama)
- Smart caching based on function signature & behavior
- Rate limiting with exponential backoff
- Individual pytest validation for each test function
- Self-heal loop: model fixes failing tests
- Mirrors your package structure under tests/ automatically
- Generates mocks/fixtures guidance in prompt for external/network/DB calls

Usage
-----
pip install google-generativeai openai requests pytest python-dotenv

# Set your provider env vars (example: Gemini)
export GEMINI_API_KEY=...

# Single file
python universal_test_agent_v2.py --source path/to/file.py --provider gemini

# Whole project (recursively)
python universal_test_agent_v2.py --source . --project --tests-dir tests --provider gemini --heal-rounds 1

# OpenAI
python universal_test_agent_v2.py --source . --project --provider openai --model gpt-4o-mini

# Ollama (local)
ollama pull llama3
python universal_test_agent_v2.py --source . --project --provider ollama --model llama3

"""

import os
import re
import ast
import sys
import io
import json
import time
import math
import pickle
import shutil
import tempfile
import logging
import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    List, Dict, Any, Optional, Tuple, Set, Protocol, Iterable
)

from dotenv import load_dotenv

# ---------------- Bootstrap ----------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("universal_test_agent_v2")


# ---------------- Provider Abstraction ----------------
class ModelProvider(Protocol):
    def generate(self, prompt: str, *, temperature: float = 0.1, max_tokens: int = 4000) -> str: ...
    def name(self) -> str: ...


class GeminiProvider:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        import google.generativeai as genai
        self.genai = genai
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")
        self.genai.configure(api_key=self.api_key)
        self.model_name = model or os.getenv("MODEL_NAME", "gemini-1.5-flash")
        self._model = self.genai.GenerativeModel(self.model_name)

    def generate(self, prompt: str, *, temperature: float = 0.1, max_tokens: int = 4000) -> str:
        config = self.genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=0.8,
            top_k=40,
        )
        res = self._model.generate_content(prompt, generation_config=config)
        return (res.text or "").strip()

    def name(self) -> str:
        return f"gemini:{self.model_name}"


class OpenAIProvider:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        import openai
        self.openai = openai
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.openai.api_key = self.api_key
        self.model_name = model or os.getenv("MODEL_NAME", "gpt-4o-mini")

    def generate(self, prompt: str, *, temperature: float = 0.1, max_tokens: int = 4000) -> str:
        resp = self.openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message["content"].strip()

    def name(self) -> str:
        return f"openai:{self.model_name}"


class OllamaProvider:
    def __init__(self, model: Optional[str] = None, host: Optional[str] = None):
        import requests
        self.requests = requests
        self.model_name = model or os.getenv("MODEL_NAME", "llama3")
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def generate(self, prompt: str, *, temperature: float = 0.1, max_tokens: int = 4000) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        r = self.requests.post(url, json=payload, timeout=600)
        r.raise_for_status()
        text_chunks = []
        for line in r.text.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if "response" in obj:
                    text_chunks.append(obj["response"])
            except Exception:
                pass
        return "".join(text_chunks).strip()

    def name(self) -> str:
        return f"ollama:{self.model_name}"


def build_provider(provider: str, model: Optional[str]) -> ModelProvider:
    p = provider.lower()
    if p == "gemini":
        return GeminiProvider(model=model)
    if p == "openai":
        return OpenAIProvider(model=model)
    if p == "ollama":
        return OllamaProvider(model=model)
    raise ValueError(f"Unsupported provider: {provider}")


# ---------------- Rate Limit Manager ----------------
class RateLimitManager:
    def __init__(self, rpm: int = 8, rph: int = 200):
        self.rpm = rpm
        self.rph = rph
        self.minute = []
        self.hour = []
        self.delay = 1.0
        self.max_delay = 60.0

    def can_go(self) -> bool:
        now = time.time()
        self.minute = [t for t in self.minute if now - t < 60]
        self.hour = [t for t in self.hour if now - t < 3600]
        return len(self.minute) < self.rpm and len(self.hour) < self.rph

    def wait(self):
        while not self.can_go():
            log.info("⏳ Hitting rate limit; sleeping %.1fs", self.delay)
            time.sleep(self.delay)
            self.delay = min(self.delay * 1.5, self.max_delay)
        self.delay = 1.0

    def tick(self):
        now = time.time()
        self.minute.append(now)
        self.hour.append(now)

    def backoff(self):
        self.delay = min(self.delay * 2, self.max_delay)
        log.warning("Rate limit error, increased delay to %.1fs", self.delay)


# ---------------- Data Models ----------------
from dataclasses import dataclass, field

@dataclass
class FunctionInfo:
    name: str
    args: List[str]
    defaults: Dict[str, Any]
    kwonlyargs: List[str]
    return_annotation: Optional[str]
    docstring: Optional[str]
    source_code: str
    line_number: int
    decorators: List[str]
    is_method: bool
    is_async: bool = False
    is_generator: bool = False
    is_property: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False
    class_name: Optional[str] = None
    complexity_score: int = 0
    exception_types: List[str] = field(default_factory=list)
    actual_exceptions: List[str] = field(default_factory=list)
    param_types: Dict[str, str] = field(default_factory=dict)

    def signature_hash(self) -> str:
        sig = f"{self.qualified_name()}({','.join(self.args+self.kwonlyargs)})->{self.return_annotation}"
        sig += f"|{self.complexity_score}|{','.join(sorted(self.actual_exceptions))}|{self.is_async}|{self.is_generator}"
        return hashlib.md5(sig.encode()).hexdigest()

    def qualified_name(self) -> str:
        if self.class_name:
            return f"{self.class_name}.{self.name}"
        return self.name


@dataclass
class ClassInfo:
    name: str
    methods: List[FunctionInfo]
    attributes: List[str]
    docstring: Optional[str]
    inheritance: List[str]
    source_code: str
    line_number: int
    is_abstract: bool = False
    properties: List[str] = field(default_factory=list)
    constructor_args: List[str] = field(default_factory=list)


@dataclass
class ImportInfo:
    module: str
    alias: Optional[str]
    from_import: bool
    specific_imports: List[str]
    is_external: bool = True
    is_standard_lib: bool = False


@dataclass
class FileContext:
    file_path: str
    module: str
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    imports: List[ImportInfo]
    global_variables: Dict[str, Any]
    external_dependencies: Set[str] = field(default_factory=set)
    test_complexity: str = "medium"
    actual_behavior: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestCase:
    name: str
    code: str
    target_function: str  # qualified name
    test_type: str
    is_valid: bool = False
    error_message: str = ""


# ---------------- Cache Manager ----------------
class TestCache:
    def __init__(self, path: str = ".universal_test_cache_v2.pkl"):
        self.path = path
        self._cache: Dict[str, Dict[str, List[TestCase]]] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    self._cache = pickle.load(f)
                log.info("📦 Loaded %d cached signatures", len(self._cache))
            except Exception as e:
                log.warning("Cache load failed: %s", e)

    def save(self):
        try:
            with open(self.path, "wb") as f:
                pickle.dump(self._cache, f)
        except Exception as e:
            log.warning("Cache save failed: %s", e)

    def get(self, fn: FunctionInfo, test_type: str) -> Optional[List[TestCase]]:
        return self._cache.get(fn.signature_hash(), {}).get(test_type)

    def put(self, fn: FunctionInfo, test_type: str, tests: List[TestCase]):
        key = fn.signature_hash()
        self._cache.setdefault(key, {})[test_type] = tests
        if len(self._cache) % 25 == 0:
            self.save()


# ---------------- AST Parser ----------------
class ProjectParser:
    def __init__(self):
        self.standard_libs = self._get_std_libs()

    def parse_project(self, root: str, exclude: Optional[List[str]] = None) -> List[FileContext]:
        if exclude is None:
            exclude = ['__pycache__', '.git', '.venv', 'venv', 'env', 'tests', 'build', 'dist', '.mypy_cache']
        py_files = self._discover_python_files(root, exclude)
        contexts = []
        for f in py_files:
            try:
                contexts.append(self.parse_file(f, project_root=root))
            except Exception as e:
                log.warning("Parse failed for %s: %s", f, e)
        return contexts

    def parse_file(self, file_path: str, project_root: Optional[str] = None) -> FileContext:
        file_path_abs = os.path.abspath(file_path)
        with open(file_path_abs, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)
        module_name = self._compute_module_name(file_path_abs, project_root)

        ctx = FileContext(
            file_path=file_path_abs,
            module=module_name,
            functions=[],
            classes=[],
            imports=[],
            global_variables={},
            external_dependencies=set(),
        )

        self._extract_imports(tree, ctx)
        self._extract_globals(tree, ctx)
        self._extract_functions_and_classes(tree, ctx, source)
        self._analyze_behavior(ctx, tree)
        self._analyze_complexity(ctx)

        return ctx

    # ---------- Helpers ----------
    def _discover_python_files(self, root: str, exclude: List[str]) -> List[str]:
        out = []
        for r, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if not any(p in d for p in exclude)]
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                if fn.startswith("test_") or fn.endswith("_test.py"):
                    continue
                out.append(os.path.join(r, fn))
        return out

    def _compute_module_name(self, file_path: str, project_root: Optional[str]) -> str:
        if not project_root:
            return Path(file_path).stem
        try:
            rel = os.path.relpath(file_path, project_root)
            parts = Path(rel).with_suffix("").parts
            parts = [p for p in parts if p not in (".",)]
            return ".".join(parts)
        except Exception:
            return Path(file_path).stem

    def _get_std_libs(self) -> Set[str]:
        return {
            'os', 'sys', 'json', 'math', 'datetime', 'time', 'random', 're',
            'collections', 'itertools', 'functools', 'typing', 'pathlib',
            'logging', 'unittest', 'sqlite3', 'urllib', 'http', 'email',
            'xml', 'csv', 'io', 'tempfile', 'shutil', 'subprocess', 'threading',
            'multiprocessing', 'asyncio', 'socket', 'ssl', 'hashlib', 'hmac',
            'base64', 'pickle', 'copy', 'weakref', 'gc', 'inspect', 'dataclasses',
            'contextlib', 'abc', 'enum'
        }

    def _extract_imports(self, tree: ast.AST, ctx: FileContext):
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base = alias.name.split('.')[0]
                    is_std = base in self.standard_libs
                    ctx.imports.append(ImportInfo(
                        module=alias.name,
                        alias=alias.asname,
                        from_import=False,
                        specific_imports=[],
                        is_external=not is_std,
                        is_standard_lib=is_std
                    ))
                    if not is_std:
                        ctx.external_dependencies.add(base)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                base = mod.split('.')[0] if mod else ""
                is_std = base in self.standard_libs
                ctx.imports.append(ImportInfo(
                    module=mod,
                    alias=None,
                    from_import=True,
                    specific_imports=[n.name for n in node.names],
                    is_external=not is_std,
                    is_standard_lib=is_std
                ))
                if not is_std and base:
                    ctx.external_dependencies.add(base)

    def _extract_globals(self, tree: ast.AST, ctx: FileContext):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                try:
                    if isinstance(node.targets[0], ast.Name):
                        name = node.targets[0].id
                        if isinstance(node.value, ast.Constant):
                            ctx.global_variables[name] = {"value": node.value.value, "type": type(node.value.value).__name__}
                        else:
                            ctx.global_variables[name] = {"value": "complex", "type": type(node.value).__name__}
                except Exception:
                    pass

    def _extract_functions_and_classes(self, tree: ast.AST, ctx: FileContext, source: str):
        lines = source.splitlines()
        class_methods: Dict[str, List[ast.AST]] = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_methods[node.name] = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(node in v for v in class_methods.values()):
                    continue
                fi = self._make_function_info(node, lines, False, None)
                ctx.functions.append(fi)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods: List[FunctionInfo] = []
                attributes: List[str] = []
                properties: List[str] = []
                constructor_args: List[str] = []
                is_abs = False

                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        finfo = self._make_function_info(item, lines, True, node.name)
                        methods.append(finfo)
                        if finfo.name == "__init__":
                            constructor_args = [a for a in finfo.args if a != "self"]
                        if finfo.is_property:
                            properties.append(finfo.name)
                        if any(isinstance(d, ast.Name) and d.id == "abstractmethod" for d in item.decorator_list):
                            is_abs = True
                    elif isinstance(item, ast.Assign):
                        for t in item.targets:
                            if isinstance(t, ast.Name):
                                attributes.append(t.id)

                inheritance = []
                for base in node.bases:
                    try:
                        inheritance.append(ast.unparse(base))
                    except Exception:
                        inheritance.append(str(base))

                src_code = "\n".join(lines[node.lineno - 1: node.end_lineno or node.lineno])
                ctx.classes.append(ClassInfo(
                    name=node.name,
                    methods=methods,
                    attributes=attributes,
                    docstring=ast.get_docstring(node),
                    inheritance=inheritance,
                    source_code=src_code,
                    line_number=node.lineno,
                    is_abstract=is_abs,
                    properties=properties,
                    constructor_args=constructor_args
                ))

    def _make_function_info(self, node: ast.AST, lines: List[str], is_method: bool, class_name: Optional[str]) -> FunctionInfo:
        is_async = isinstance(node, ast.AsyncFunctionDef)
        name = node.name
        args_obj = node.args

        args = [a.arg for a in args_obj.args]
        kwonlyargs = [a.arg for a in args_obj.kwonlyargs]
        defaults = {}
        for arg, default in zip(args[-len(args_obj.defaults):], args_obj.defaults):
            defaults[arg] = self._const_to_str(default)

        return_annotation = None
        if getattr(node, "returns", None) is not None:
            try:
                return_annotation = ast.unparse(node.returns)
            except Exception:
                return_annotation = str(node.returns)

        decorators = []
        is_property = False
        is_classmethod = False
        is_staticmethod = False
        for d in getattr(node, "decorator_list", []):
            try:
                dtext = ast.unparse(d)
            except Exception:
                dtext = str(d)
            decorators.append(dtext)
            if dtext.endswith(".setter") or dtext == "property":
                is_property = True
            if dtext == "classmethod":
                is_classmethod = True
            if dtext == "staticmethod":
                is_staticmethod = True

        complexity = self._cyclomatic(node)
        exception_types = self._extract_exception_types(node)
        actual_exceptions = exception_types[:]
        param_types = self._extract_param_types(node)

        is_generator = any(isinstance(n, ast.Yield) or isinstance(n, ast.YieldFrom) for n in ast.walk(node))

        end_line = getattr(node, "end_lineno", None) or (node.lineno + 20)
        source_code = "\n".join(lines[node.lineno - 1: end_line])

        return FunctionInfo(
            name=name,
            args=args,
            defaults=defaults,
            kwonlyargs=kwonlyargs,
            return_annotation=return_annotation,
            docstring=ast.get_docstring(node),
            source_code=source_code,
            line_number=node.lineno,
            decorators=decorators,
            is_method=is_method,
            is_async=is_async,
            is_generator=is_generator,
            is_property=is_property,
            is_classmethod=is_classmethod,
            is_staticmethod=is_staticmethod,
            class_name=class_name,
            complexity_score=complexity,
            exception_types=exception_types,
            actual_exceptions=actual_exceptions,
            param_types=param_types
        )

    def _const_to_str(self, node: ast.AST) -> str:
        try:
            if isinstance(node, ast.Constant):
                return repr(node.value)
            return ast.unparse(node)
        except Exception:
            return str(node)

    def _extract_param_types(self, node: ast.AST) -> Dict[str, str]:
        out: Dict[str, str] = {}
        args_obj = node.args
        for a in list(args_obj.args) + list(args_obj.kwonlyargs):
            if a.annotation:
                try:
                    out[a.arg] = ast.unparse(a.annotation)
                except Exception:
                    out[a.arg] = str(a.annotation)
        return out

    def _cyclomatic(self, node: ast.AST) -> int:
        c = 1
        for n in ast.walk(node):
            if isinstance(n, (ast.If, ast.For, ast.While, ast.AsyncFor, ast.ExceptHandler)):
                c += 1
            elif isinstance(n, ast.BoolOp):
                c += len(n.values) - 1
        return c

    def _extract_exception_types(self, node: ast.AST) -> List[str]:
        out = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Raise) and n.exc:
                if isinstance(n.exc, ast.Call) and isinstance(n.exc.func, ast.Name):
                    out.add(n.exc.func.id)
                elif isinstance(n.exc, ast.Name):
                    out.add(n.exc.id)
        return list(out)

    def _analyze_behavior(self, ctx: FileContext, tree: ast.AST):
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ctx.actual_behavior[n.name] = {
                    "has_async": isinstance(n, ast.AsyncFunctionDef),
                    "has_yield": any(isinstance(x, (ast.Yield, ast.YieldFrom)) for x in ast.walk(n)),
                    "raises_exceptions": self._extract_exception_types(n),
                    "uses_isinstance": any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "isinstance" for x in ast.walk(n)),
                }

    def _analyze_complexity(self, ctx: FileContext):
        total_fns = len(ctx.functions) + sum(len(c.methods) for c in ctx.classes)
        avg_complexity = 0.0
        if ctx.functions:
            avg_complexity = sum(f.complexity_score for f in ctx.functions) / len(ctx.functions)
        if total_fns > 40 or avg_complexity > 10 or len(ctx.external_dependencies) > 20:
            ctx.test_complexity = "high"
        elif total_fns > 20 or avg_complexity > 5 or len(ctx.external_dependencies) > 10:
            ctx.test_complexity = "medium"
        else:
            ctx.test_complexity = "low"


# ---------------- Pytest Validator ----------------
class PytestValidator:
    def __init__(self, source_file: str, module_name: str):
        self.source_file = source_file
        self.module_name = module_name

    def validate(self, test_case: TestCase) -> Tuple[bool, str]:
        tmp = tempfile.mkdtemp()
        try:
            src_dst_dir = os.path.join(tmp, os.path.dirname(self.source_file).strip("/\\"))
            os.makedirs(src_dst_dir, exist_ok=True)
            dst = os.path.join(tmp, self.source_file.lstrip("/\\"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(self.source_file, dst)

            test_file = os.path.join(tmp, f"test_{Path(self.source_file).stem}.py")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(self._imports() + "\n" + test_case.code + "\n")

            old = os.getcwd()
            os.chdir(tmp)
            cmd = [sys.executable, "-m", "pytest", f"{Path(test_file).name}::{test_case.name}", "-q", "--no-header", "-x"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
            os.chdir(old)

            if res.returncode == 0:
                return True, ""
            return False, res.stdout + "\n" + res.stderr
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:
            return False, str(e)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _imports(self) -> str:
        return f"import pytest\nfrom {self.module_name} import *\n"


# ---------------- Prompt Builder ----------------
class PromptBuilder:
    @staticmethod
    def build_batch_prompt(functions: List[FunctionInfo], ctx: FileContext) -> str:
        infos = []
        for f in functions:
            kind = "async function" if f.is_async else "function"
            if f.is_method:
                kind = "class method" if f.is_classmethod else ("static method" if f.is_staticmethod else "instance method")
            if f.is_property:
                kind = "property method"
            if f.is_generator:
                kind += " (generator)"

            infos.append(f"""
TARGET: {f.qualified_name()} ({kind})
MODULE: {ctx.module}
ARGS: {', '.join(f.args + f.kwonlyargs)}
DEFAULTS: {f.defaults}
RETURN: {f.return_annotation}
DECORATORS: {f.decorators}
EXCEPTIONS: {f.actual_exceptions}
DOCSTRING:
{f.docstring or 'N/A'}
SOURCE:
```python
{f.source_code}
```
""")

        return f"""
You are an expert Python test engineer. Generate **pytest** unit tests for **ALL** the following functions/methods in one response.

Rules:
- For each function/method:
  * At least 2 basic tests
  * 1 edge-case test
  * 1 exception test ONLY if the implementation raises exceptions
  * For async code, use `pytest.mark.asyncio`
  * For floating points, use `pytest.approx`
  * For external side effects (DB/HTTP/files/env), **mock them** using `unittest.mock` or pytest fixtures
  * For properties, test getter/setter behavior
  * For generators, consume yielded values and assert expected sequences
  * Never fabricate imports or functions that don't exist

- Output **ONLY Python code**. Start with ```python and end with ```.
- Structure test function names like: test_<qualifiedname>_<case>_<#>

Project context:
- Module: {ctx.module}
- External deps: {sorted(ctx.external_dependencies)}
- Complexity: {ctx.test_complexity}

Functions to cover:
{''.join(infos)}
"""


# ---------------- Batch Test Generator ----------------
class BatchTestGenerator:
    def __init__(self, provider: ModelProvider, cache: TestCache, rate: RateLimitManager):
        self.provider = provider
        self.cache = cache
        self.rate = rate
        self.api_calls = 0

    def generate_tests_batch(self, functions: List[FunctionInfo], ctx: FileContext, batch_size: int = 4) -> List[TestCase]:
        all_tests: List[TestCase] = []
        uncached: List[FunctionInfo] = []

        for fn in functions:
            cb = self.cache.get(fn, "batch")
            if cb:
                log.info("📦 Cache hit: %s", fn.qualified_name())
                all_tests.extend(cb)
            else:
                uncached.append(fn)

        if not uncached:
            return all_tests

        for i in range(0, len(uncached), batch_size):
            group = uncached[i:i + batch_size]
            prompt = PromptBuilder.build_batch_prompt(group, ctx)

            self.rate.wait()
            text = self.provider.generate(prompt, temperature=0.05, max_tokens=8000)
            self.rate.tick()
            self.api_calls += 1

            parsed = self._parse_batch_tests(text, group)
            all_tests.extend(parsed)

            for fn in group:
                related = [t for t in parsed if t.target_function == fn.qualified_name()]
                if related:
                    self.cache.put(fn, "batch", related)

            if i + batch_size < len(uncached):
                time.sleep(1.5)

        return all_tests

    def _parse_batch_tests(self, text: str, functions: List[FunctionInfo]) -> List[TestCase]:
        code_block = re.findall(r"```python(.*?)```", text, re.DOTALL)
        if code_block:
            text = code_block[0]
        text = re.sub(r"```+", "", text).strip()

        test_defs = re.findall(r"(def\s+test_[\w\d_]+\s*\(.*?\):(?:.*\n)*?(?=^def\s+test_|\Z))", text, re.DOTALL | re.MULTILINE)
        tests: List[TestCase] = []
        for block in test_defs:
            first_line = block.splitlines()[0]
            m = re.match(r"\s*def\s+(test_[\w\d_]+)", first_line)
            if not m:
                continue
            test_name = m.group(1)
            target = self._infer_target_from_name(test_name, functions)
            test_type = self._infer_test_type(test_name)
            tests.append(TestCase(
                name=test_name,
                code=block.strip(),
                target_function=target,
                test_type=test_type
            ))
        log.info("Parsed %d tests from batch", len(tests))
        return tests

    def _infer_target_from_name(self, test_name: str, functions: List[FunctionInfo]) -> str:
        clean = re.sub(r"^test_", "", test_name)
        clean = re.sub(r"_(basic|edge|exception|async)_?\d*$", "", clean)
        for f in functions:
            if f.name.lower() in clean.lower() or f.qualified_name().replace(".", "_").lower() in clean.lower():
                return f.qualified_name()
        return clean

    def _infer_test_type(self, test_name: str) -> str:
        name = test_name.lower()
        if "exception" in name:
            return "exception"
        if "edge" in name:
            return "edge_case"
        if "async" in name:
            return "async"
        return "basic"


# ---------------- Self-heal Agent ----------------
class SelfHealAgent:
    def __init__(self, provider: ModelProvider, rate: RateLimitManager):
        self.provider = provider
        self.rate = rate
        self.heal_calls = 0

    def fix(self, test_case: TestCase, error: str, func: FunctionInfo, ctx: FileContext) -> Optional[TestCase]:
        prompt = f"""
The following pytest unit test failed. Please FIX it so it correctly reflects the function's implementation.

Module: {ctx.module}
Function: {func.qualified_name()}

Function source:
```python
{func.source_code}
```

Failing test:
```python
{test_case.code}
```

Pytest error (keep it in mind):
```
{error}
```

Return ONLY the corrected test function starting with:
def {test_case.name}(
"""
        self.rate.wait()
        out = self.provider.generate(prompt, temperature=0.0, max_tokens=2000)
        self.rate.tick()
        self.heal_calls += 1

        m = re.search(rf"(def\s+{re.escape(test_case.name)}\s*\(.*?\):(?:.*\n)*?)\Z", out.strip(), re.DOTALL)
        if not m:
            m = re.search(r"(def\s+test_[\w\d_]+\s*\(.*?\):(?:.*\n)*?)\Z", out.strip(), re.DOTALL)
        if not m:
            return None

        fixed_block = m.group(1).rstrip() + "\n"
        return TestCase(
            name=test_case.name,
            code=fixed_block,
            target_function=test_case.target_function,
            test_type=test_case.test_type
        )


# ---------------- Orchestrator ----------------
class UniversalTestAgentV2:
    def __init__(self, provider: ModelProvider, rpm: int = 8, rph: int = 200, cache_file: str = ".universal_test_cache_v2.pkl"):
        self.provider = provider
        self.rate = RateLimitManager(rpm, rph)
        self.cache = TestCache(cache_file)
        self.parser = ProjectParser()
        self.generator = BatchTestGenerator(self.provider, self.cache, self.rate)
        self.self_healer = SelfHealAgent(self.provider, self.rate)

        self.valid_tests: List[TestCase] = []
        self.failed_tests: List[TestCase] = []

    def generate_for_file(self, src: str, out_dir: Optional[str] = None, heal_rounds: int = 1) -> Dict[str, Any]:
        if out_dir is None:
            out_dir = "tests"
        os.makedirs(out_dir, exist_ok=True)

        ctx = self.parser.parse_file(src, project_root=os.path.dirname(src))
        all_functions = ctx.functions[:]
        for c in ctx.classes:
            all_functions.extend(c.methods)

        if not all_functions:
            log.warning("No functions/methods found in %s", src)
            return {}

        tests = self.generator.generate_tests_batch(all_functions, ctx)
        validator = PytestValidator(ctx.file_path, ctx.module)

        for t in tests:
            ok, err = validator.validate(t)
            if ok:
                t.is_valid = True
                self.valid_tests.append(t)
            else:
                t.error_message = err
                self.failed_tests.append(t)

        self._self_heal(ctx, validator, heal_rounds, all_functions)

        out_file = self._write_tests_file(out_dir, ctx, self.valid_tests)
        self.cache.save()

        return self._summary(ctx, out_file)

    def generate_for_project(self, root: str, tests_dir: str = "tests", heal_rounds: int = 1) -> Dict[str, Any]:
        ctxs = self.parser.parse_project(root)
        os.makedirs(tests_dir, exist_ok=True)

        totals = dict(gen=0, valid=0, failed=0, files=0, healed=0)
        for ctx in ctxs:
            all_functions = ctx.functions[:]
            for c in ctx.classes:
                all_functions.extend(c.methods)

            if not all_functions:
                continue

            totals["files"] += 1

            tests = self.generator.generate_tests_batch(all_functions, ctx)
            totals["gen"] += len(tests)

            validator = PytestValidator(ctx.file_path, ctx.module)
            file_valid: List[TestCase] = []
            file_failed: List[TestCase] = []

            for t in tests:
                ok, err = validator.validate(t)
                if ok:
                    t.is_valid = True
                    file_valid.append(t)
                else:
                    t.error_message = err
                    file_failed.append(t)

            healed = self._self_heal(ctx, validator, heal_rounds, all_functions, file_failed, file_valid)
            totals["healed"] += healed
            totals["valid"] += len(file_valid)
            totals["failed"] += len(file_failed)

            _ = self._write_tests_file(tests_dir, ctx, file_valid)

        self.cache.save()
        return dict(
            project_root=root,
            files_processed=totals["files"],
            total_api_calls=self.generator.api_calls,
            total_generated=totals["gen"],
            total_valid=totals["valid"],
            total_failed=totals["failed"],
            healed=totals["healed"],
        )

    def _self_heal(self, ctx: FileContext, validator: PytestValidator, heal_rounds: int,
                   all_functions: List[FunctionInfo],
                   failed_list: Optional[List[TestCase]] = None,
                   valid_list: Optional[List[TestCase]] = None) -> int:
        if failed_list is None:
            failed_list = self.failed_tests
        if valid_list is None:
            valid_list = self.valid_tests

        healed_count = 0
        for r in range(heal_rounds):
            if not failed_list:
                break
            still_failed: List[TestCase] = []
            for t in failed_list:
                fn = self._find_fn(t.target_function, all_functions)
                if not fn:
                    still_failed.append(t)
                    continue
                fixed = self.self_healer.fix(t, t.error_message, fn, ctx)
                if not fixed:
                    still_failed.append(t)
                    continue
                ok, err = validator.validate(fixed)
                if ok:
                    fixed.is_valid = True
                    valid_list.append(fixed)
                    healed_count += 1
                else:
                    fixed.error_message = err
                    still_failed.append(fixed)
            failed_list[:] = still_failed
        return healed_count

    def _find_fn(self, qname: str, fns: List[FunctionInfo]) -> Optional[FunctionInfo]:
        for f in fns:
            if f.qualified_name() == qname or f.name == qname:
                return f
        return None

    def _write_tests_file(self, tests_dir: str, ctx: FileContext, tests: List[TestCase]) -> str:
        rel_module_path = ctx.module.replace(".", "/")
        test_path_dir = os.path.join(tests_dir, os.path.dirname(rel_module_path))
        os.makedirs(test_path_dir, exist_ok=True)
        test_file = os.path.join(test_path_dir, f"test_{Path(ctx.file_path).stem}.py")

        header = f'''# -*- coding: utf-8 -*-
"""
Auto-generated tests for module: {ctx.module}
Provider: {self.provider.name()}
File: {ctx.file_path}
Valid tests: {len(tests)}
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
import pytest
from {ctx.module} import *
'''

        with open(test_file, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            for t in tests:
                f.write(t.code + "\n\n")

        log.info("📝 Wrote %d valid tests -> %s", len(tests), test_file)
        return test_file

    def _summary(self, ctx: FileContext, out_file: str) -> Dict[str, Any]:
        total = len(self.valid_tests) + len(self.failed_tests)
        success_rate = (len(self.valid_tests) / total * 100) if total else 0.0
        return dict(
            module=ctx.module,
            file=ctx.file_path,
            output=out_file,
            provider=self.provider.name(),
            total_generated=total,
            valid=len(self.valid_tests),
            failed=len(self.failed_tests),
            success_rate=f"{success_rate:.1f}%",
            api_calls=self.generator.api_calls,
            heal_calls=self.self_healer.heal_calls,
        )


# ---------------- CLI ----------------
def main():
    import argparse
    p = argparse.ArgumentParser(description="Universal AI Unit Test Generator v2")
    p.add_argument("--source", required=True, help="Python file or project directory")
    p.add_argument("--project", action="store_true", help="Treat source as project root; scan recursively")
    p.add_argument("--tests-dir", default="tests", help="Directory to place generated tests")
    p.add_argument("--provider", default=os.getenv("PROVIDER", "gemini"), choices=["gemini", "openai", "ollama"])
    p.add_argument("--model", default=os.getenv("MODEL_NAME", None), help="Override model name")
    p.add_argument("--rpm", type=int, default=8, help="Requests per minute")
    p.add_argument("--rph", type=int, default=200, help="Requests per hour")
    p.add_argument("--heal-rounds", type=int, default=1, help="Self-heal rounds per file")
    args = p.parse_args()

    provider = build_provider(args.provider, args.model)
    agent = UniversalTestAgentV2(provider, rpm=args.rpm, rph=args.rph)

    if args.project or os.path.isdir(args.source):
        result = agent.generate_for_project(args.source, tests_dir=args.tests_dir, heal_rounds=args.heal_rounds)
        log.info("\nProject Summary:\n%s", json.dumps(result, indent=2))
    else:
        result = agent.generate_for_file(args.source, out_dir=args.tests_dir, heal_rounds=args.heal_rounds)
        log.info("\nFile Summary:\n%s", json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
