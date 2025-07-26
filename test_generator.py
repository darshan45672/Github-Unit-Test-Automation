import ast
import os
import sys
import inspect
import importlib.util
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import google.generativeai as genai
from dataclasses import dataclass, field
import json
import re
from dotenv import load_dotenv
import logging
from concurrent.futures import ThreadPoolExecutor
import time
import subprocess
import tempfile
import shutil
import hashlib
from functools import lru_cache
import threading
from queue import Queue
import pickle

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FunctionInfo:
    name: str
    args: List[str]
    return_annotation: Optional[str]
    docstring: Optional[str]
    source_code: str
    line_number: int
    decorators: List[str]
    is_method: bool
    class_name: Optional[str] = None
    complexity_score: int = 0
    called_functions: List[str] = field(default_factory=list)
    exception_types: List[str] = field(default_factory=list)
    param_types: Dict[str, str] = field(default_factory=dict)
    actual_exceptions: List[str] = field(default_factory=list)
    
    def get_signature_hash(self) -> str:
        """Get a hash of the function signature for caching"""
        signature = f"{self.name}({','.join(self.args)})->{self.return_annotation}"
        signature += f"|{self.complexity_score}|{','.join(self.actual_exceptions)}"
        return hashlib.md5(signature.encode()).hexdigest()

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
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    imports: List[ImportInfo]
    global_variables: Dict[str, Any]
    dependencies: List[str]
    external_dependencies: Set[str] = field(default_factory=set)
    test_complexity: str = "medium"
    actual_behavior: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TestCase:
    """Represents a single test case"""
    name: str
    code: str
    target_function: str
    test_type: str  # 'basic', 'exception', 'edge_case'
    is_valid: bool = False
    error_message: str = ""

@dataclass
class BatchTestRequest:
    """Represents a batch request for multiple test cases"""
    functions: List[FunctionInfo]
    context: FileContext
    request_type: str  # 'basic', 'exception', 'edge_case'
    timestamp: float = field(default_factory=time.time)

class RateLimitManager:
    """Manages API rate limits with exponential backoff"""
    
    def __init__(self, requests_per_minute: int = 10, requests_per_hour: int = 300):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.minute_requests = []
        self.hour_requests = []
        self.lock = threading.Lock()
        self.current_delay = 1.0  # Start with 1 second delay
        self.max_delay = 60.0     # Max 60 seconds delay
        
    def can_make_request(self) -> bool:
        """Check if we can make a request without hitting rate limits"""
        with self.lock:
            now = time.time()
            
            # Clean old requests
            self.minute_requests = [t for t in self.minute_requests if now - t < 60]
            self.hour_requests = [t for t in self.hour_requests if now - t < 3600]
            
            # Check limits
            return (len(self.minute_requests) < self.requests_per_minute and 
                   len(self.hour_requests) < self.requests_per_hour)
    
    def wait_if_needed(self):
        """Wait if we're hitting rate limits"""
        while not self.can_make_request():
            logger.info(f"⏳ Rate limit reached, waiting {self.current_delay:.1f}s...")
            time.sleep(self.current_delay)
            self.current_delay = min(self.current_delay * 1.5, self.max_delay)
        
        # Reset delay on successful check
        self.current_delay = 1.0
    
    def record_request(self):
        """Record that we made a request"""
        with self.lock:
            now = time.time()
            self.minute_requests.append(now)
            self.hour_requests.append(now)
    
    def handle_rate_limit_error(self):
        """Handle when we get a rate limit error from the API"""
        with self.lock:
            self.current_delay = min(self.current_delay * 2, self.max_delay)
            logger.warning(f"🚫 API rate limit hit, increasing delay to {self.current_delay:.1f}s")

class TestCacheManager:
    """Caches generated tests to avoid redundant API calls"""
    
    def __init__(self, cache_file: str = "test_cache.pkl"):
        self.cache_file = cache_file
        self.cache = {}
        self.load_cache()
    
    def load_cache(self):
        """Load cache from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"📦 Loaded {len(self.cache)} cached test patterns")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.cache = {}
    
    def save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get_cached_tests(self, func_info: FunctionInfo, test_type: str) -> Optional[List[TestCase]]:
        """Get cached tests for a function"""
        cache_key = f"{func_info.get_signature_hash()}_{test_type}"
        return self.cache.get(cache_key)
    
    def cache_tests(self, func_info: FunctionInfo, test_type: str, tests: List[TestCase]):
        """Cache tests for a function"""
        cache_key = f"{func_info.get_signature_hash()}_{test_type}"
        self.cache[cache_key] = tests
        
        # Auto-save every 10 cache entries
        if len(self.cache) % 10 == 0:
            self.save_cache()

class BatchTestGenerator:
    """Generates tests in batches to minimize API calls"""
    
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be provided")
        
        genai.configure(api_key=api_key)
        
        # Rate limiting configuration - conservative settings
        self.rate_limiter = RateLimitManager(
            requests_per_minute=8,  # Conservative: 8 requests per minute
            requests_per_hour=200   # Conservative: 200 requests per hour
        )
        
        # Cache manager
        self.cache_manager = TestCacheManager()
        
        # Generation config - optimized for efficiency
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            top_p=0.8,
            top_k=40,
            max_output_tokens=3000,  # Larger output for batch processing
        )
        
        self.model = self._initialize_model()
        self.batch_queue = Queue()
        self.api_call_count = 0
        
    def _initialize_model(self):
        """Initialize model with error handling"""
        model_priority = [
            'gemini-1.5-flash',      # Fastest and cheapest
            'gemini-1.5-pro', 
            'gemini-1.5-flash-latest',
        ]
        
        for model_name in model_priority:
            try:
                model = genai.GenerativeModel(model_name)
                # Test with rate limiting
                self.rate_limiter.wait_if_needed()
                test_response = model.generate_content(
                    "Hello", 
                    generation_config=self.generation_config
                )
                self.rate_limiter.record_request()
                if test_response.text:
                    logger.info(f"✓ Using model: {model_name}")
                    return model
            except Exception as e:
                logger.warning(f"✗ Model {model_name} failed: {e}")
                self.rate_limiter.handle_rate_limit_error()
                continue
        
        raise Exception("No compatible Gemini model found")
    
    def generate_tests_batch(self, functions: List[FunctionInfo], context: FileContext) -> List[TestCase]:
        """Generate tests for multiple functions in a single API call"""
        if not functions:
            return []
        
        # Check cache first
        all_tests = []
        uncached_functions = []
        
        for func_info in functions:
            cached_basic = self.cache_manager.get_cached_tests(func_info, 'basic')
            cached_exception = self.cache_manager.get_cached_tests(func_info, 'exception')
            cached_edge = self.cache_manager.get_cached_tests(func_info, 'edge_case')
            
            if cached_basic and cached_exception and cached_edge:
                logger.info(f"📦 Using cached tests for {func_info.name}")
                all_tests.extend(cached_basic + cached_exception + cached_edge)
            else:
                uncached_functions.append(func_info)
        
        if not uncached_functions:
            return all_tests
        
        # Generate tests for uncached functions in batches
        batch_size = min(3, len(uncached_functions))  # Process max 3 functions per API call
        
        for i in range(0, len(uncached_functions), batch_size):
            batch_functions = uncached_functions[i:i + batch_size]
            logger.info(f"🔄 Generating tests for batch of {len(batch_functions)} functions")
            
            try:
                batch_tests = self._generate_batch_tests(batch_functions, context)
                all_tests.extend(batch_tests)
                
                # Cache the results
                self._cache_batch_results(batch_functions, batch_tests)
                
                # Add delay between batches to respect rate limits
                if i + batch_size < len(uncached_functions):
                    time.sleep(2)  # 2 second delay between batches
                    
            except Exception as e:
                logger.error(f"❌ Batch generation failed: {e}")
                # Fall back to individual generation with longer delays
                for func_info in batch_functions:
                    try:
                        time.sleep(5)  # Longer delay for fallback
                        individual_tests = self._generate_individual_tests(func_info, context)
                        all_tests.extend(individual_tests)
                    except Exception as e2:
                        logger.error(f"❌ Individual generation failed for {func_info.name}: {e2}")
        
        return all_tests
    
    def _generate_batch_tests(self, functions: List[FunctionInfo], context: FileContext) -> List[TestCase]:
        """Generate tests for a batch of functions in one API call"""
        prompt = self._create_batch_prompt(functions, context)
        
        # Rate limiting
        self.rate_limiter.wait_if_needed()
        
        try:
            result = self.model.generate_content(prompt, generation_config=self.generation_config)
            self.rate_limiter.record_request()
            self.api_call_count += 1
            
            if result.text:
                logger.info(f"📊 API call #{self.api_call_count} completed successfully")
                return self._parse_batch_tests(result.text, functions)
            else:
                logger.warning("Empty response from API")
                return []
                
        except Exception as e:
            self.rate_limiter.handle_rate_limit_error()
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                logger.error(f"🚫 Rate limit or quota exceeded: {e}")
                time.sleep(30)  # Wait 30 seconds on rate limit
                raise
            else:
                logger.error(f"API call failed: {e}")
                raise
    
    def _create_batch_prompt(self, functions: List[FunctionInfo], context: FileContext) -> str:
        """Create a single prompt for multiple functions"""
        functions_info = []
        
        for func_info in functions:
            behavior = context.actual_behavior.get(func_info.name, {})
            functions_info.append(f"""
FUNCTION: {func_info.name}
ARGUMENTS: {', '.join(func_info.args)}
EXCEPTIONS: {func_info.actual_exceptions}
SOURCE CODE:
```python
{func_info.source_code}
```
BEHAVIOR: Type checks={behavior.get('has_type_checks', False)}, Division={behavior.get('performs_division', False)}
""")
        
        return f"""
Generate comprehensive test cases for ALL the following Python functions in ONE response:

{chr(10).join(functions_info)}

REQUIREMENTS:
1. For EACH function, generate:
   - 2 basic functionality tests (test_functionname_basic_1, test_functionname_basic_2)
   - 1 exception test ONLY if exceptions are actually raised (test_functionname_exception)
   - 1 edge case test (test_functionname_edge)

2. Use realistic input values and correct expected outputs
3. For floating point results, use pytest.approx() with appropriate tolerance
4. Only test exceptions that are explicitly raised in the source code
5. Each test should be a complete function starting with 'def test_'

FORMAT - Generate tests in this exact structure:
```python
# Tests for function1
def test_function1_basic_1():
    # implementation

def test_function1_basic_2():
    # implementation

def test_function1_exception():  # Only if function raises exceptions
    # implementation

def test_function1_edge():
    # implementation

# Tests for function2  
def test_function2_basic_1():
    # implementation
# ... continue for all functions
```

Generate ALL test functions for ALL {len(functions)} functions in this single response.
"""
    
    def _parse_batch_tests(self, generated_text: str, functions: List[FunctionInfo]) -> List[TestCase]:
        """Parse batch-generated tests"""
        test_cases = []
        
        # Clean up the generated text
        generated_text = re.sub(r'```python\n?', '', generated_text)
        generated_text = re.sub(r'```\n?', '', generated_text)
        
        # Extract all test functions
        test_functions = re.findall(r'def (test_\w+.*?)(?=def test_|\Z)', generated_text, re.DOTALL)
        
        for test_match in test_functions:
            lines = test_match.split('\n')
            if lines:
                first_line = lines[0].strip()
                test_name_match = re.match(r'(test_\w+)', first_line)
                if test_name_match:
                    test_name = test_name_match.group(1)
                    
                    # Determine target function and test type
                    target_function = self._extract_target_function(test_name, functions)
                    test_type = self._determine_test_type(test_name)
                    
                    # Reconstruct the complete test function
                    test_code = f"def {test_match.strip()}\n"
                    
                    test_cases.append(TestCase(
                        name=test_name,
                        code=test_code,
                        target_function=target_function,
                        test_type=test_type
                    ))
        
        logger.info(f"📋 Parsed {len(test_cases)} tests from batch response")
        return test_cases
    
    def _extract_target_function(self, test_name: str, functions: List[FunctionInfo]) -> str:
        """Extract the target function name from test name"""
        # Remove test_ prefix and common suffixes
        clean_name = re.sub(r'^test_', '', test_name)
        clean_name = re.sub(r'_(basic|exception|edge)_?\d*$', '', clean_name)
        
        # Find matching function
        for func_info in functions:
            if func_info.name.lower() == clean_name.lower():
                return func_info.name
        
        # Fallback - return the cleaned name
        return clean_name
    
    def _determine_test_type(self, test_name: str) -> str:
        """Determine test type from test name"""
        if 'exception' in test_name.lower():
            return 'exception'
        elif 'edge' in test_name.lower():
            return 'edge_case'
        else:
            return 'basic'
    
    def _cache_batch_results(self, functions: List[FunctionInfo], tests: List[TestCase]):
        """Cache batch results by function and test type"""
        # Group tests by function and type
        for func_info in functions:
            func_tests = {
                'basic': [],
                'exception': [],
                'edge_case': []
            }
            
            for test in tests:
                if test.target_function == func_info.name:
                    func_tests[test.test_type].append(test)
            
            # Cache each type separately
            for test_type, type_tests in func_tests.items():
                if type_tests:
                    self.cache_manager.cache_tests(func_info, test_type, type_tests)
    
    def _generate_individual_tests(self, func_info: FunctionInfo, context: FileContext) -> List[TestCase]:
        """Fallback: Generate tests for individual function"""
        logger.info(f"🔄 Fallback: Generating individual tests for {func_info.name}")
        
        # Simple individual generation with longer delays
        prompt = f"""
Generate test cases for this single Python function:

FUNCTION: {func_info.name}
ARGUMENTS: {', '.join(func_info.args)}
SOURCE CODE:
```python
{func_info.source_code}
```

Generate 2-3 simple test functions:
def test_{func_info.name}_basic():
    # test basic functionality

def test_{func_info.name}_edge():
    # test edge case

Only generate tests that will actually pass based on the function implementation.
"""
        
        self.rate_limiter.wait_if_needed()
        
        try:
            result = self.model.generate_content(prompt, generation_config=self.generation_config)
            self.rate_limiter.record_request()
            self.api_call_count += 1
            
            if result.text:
                return self._parse_batch_tests(result.text, [func_info])
            
        except Exception as e:
            logger.error(f"Individual generation failed: {e}")
            self.rate_limiter.handle_rate_limit_error()
        
        return []
    
    def get_api_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        return {
            'total_api_calls': self.api_call_count,
            'cache_hits': len(self.cache_manager.cache),
            'estimated_cost_savings': f"{len(self.cache_manager.cache) * 0.02:.2f} USD",
            'current_delay': self.rate_limiter.current_delay,
            'requests_this_minute': len(self.rate_limiter.minute_requests),
            'requests_this_hour': len(self.rate_limiter.hour_requests)
        }

class IndividualTestValidator:
    """Validates individual test cases one by one"""
    
    def __init__(self, source_file_path: str):
        self.source_file_path = source_file_path
        self.source_module_name = Path(source_file_path).stem
        self.temp_dir = None
        self.base_imports = self._generate_base_imports()
        
    def _generate_base_imports(self) -> str:
        """Generate base imports needed for testing"""
        return f"""# -*- coding: utf-8 -*-
import pytest
import json
import math
import os
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from {self.source_module_name} import *

"""
    
    def validate_single_test(self, test_case: TestCase) -> Tuple[bool, str]:
        """
        Validate a single test case by running it in isolation
        Returns: (is_valid, error_message)
        """
        logger.info(f"🔍 Validating test: {test_case.name}")
        
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        try:
            # Copy source file to temp directory
            source_dest = os.path.join(self.temp_dir, os.path.basename(self.source_file_path))
            shutil.copy2(self.source_file_path, source_dest)
            
            # Create complete test file with just this one test
            full_test_code = self.base_imports + test_case.code
            
            # Write test file to temp directory
            test_file_path = os.path.join(self.temp_dir, f"test_{self.source_module_name}.py")
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(full_test_code)
            
            # Run the specific test
            is_valid, error = self._run_single_test(test_file_path, test_case.name)
            
            if is_valid:
                logger.info(f"✅ Test {test_case.name} passed!")
                test_case.is_valid = True
                return True, ""
            else:
                logger.warning(f"❌ Test {test_case.name} failed: {error}")
                test_case.error_message = error
                return False, error
                
        except Exception as e:
            error_msg = f"Validation error: {str(e)}"
            logger.error(f"Error validating {test_case.name}: {error_msg}")
            test_case.error_message = error_msg
            return False, error_msg
        finally:
            # Cleanup
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
    
    def _run_single_test(self, test_file_path: str, test_name: str) -> Tuple[bool, str]:
        """Run a specific test function and capture results"""
        try:
            # Change to temp directory
            old_cwd = os.getcwd()
            os.chdir(self.temp_dir)
            
            # Run pytest on the specific test
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 
                f"{os.path.basename(test_file_path)}::{test_name}",
                '-v', '--tb=short', '--no-header', '-x'
            ], capture_output=True, text=True, timeout=10)
            
            os.chdir(old_cwd)
            
            if result.returncode == 0:
                return True, ""
            else:
                # Extract error information
                error_output = result.stdout + result.stderr
                return False, self._extract_error_message(error_output)
            
        except subprocess.TimeoutExpired:
            return False, "Test execution timed out"
        except Exception as e:
            return False, f"Test execution failed: {str(e)}"
    
    def _extract_error_message(self, output: str) -> str:
        """Extract meaningful error message from pytest output"""
        lines = output.split('\n')
        error_lines = []
        
        for line in lines:
            if 'AssertionError' in line or 'TypeError' in line or 'ValueError' in line:
                error_lines.append(line.strip())
            elif line.startswith('E ') and len(error_lines) < 3:
                error_lines.append(line.strip())
            elif 'FAILED' in line:
                error_lines.append(line.strip())
        
        return ' | '.join(error_lines[:3]) if error_lines else "Unknown error"

class EnhancedPythonFileParser:
    """Enhanced parser with better context extraction and actual behavior analysis"""
    
    def __init__(self):
        self.visited_files = set()
        self.standard_libs = self._get_standard_libraries()
    
    def _get_standard_libraries(self) -> Set[str]:
        """Get list of Python standard library modules"""
        return {
            'os', 'sys', 'json', 'math', 'datetime', 'time', 'random', 're', 
            'collections', 'itertools', 'functools', 'typing', 'pathlib',
            'logging', 'unittest', 'sqlite3', 'urllib', 'http', 'email',
            'xml', 'csv', 'io', 'tempfile', 'shutil', 'subprocess', 'threading',
            'multiprocessing', 'asyncio', 'socket', 'ssl', 'hashlib', 'hmac',
            'base64', 'pickle', 'copy', 'weakref', 'gc', 'inspect'
        }
    
    def parse_file(self, file_path: str) -> FileContext:
        """Parse a Python file with enhanced context extraction"""
        file_path = os.path.abspath(file_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in {file_path}: {e}")
        
        context = FileContext(
            file_path=file_path,
            functions=[],
            classes=[],
            imports=[],
            global_variables={},
            dependencies=[]
        )
        
        self._extract_imports(tree, context)
        self._extract_functions(tree, context, source_code)
        self._extract_classes(tree, context, source_code)
        self._extract_global_variables(tree, context)
        self._analyze_actual_behavior(context, source_code)
        self._analyze_complexity(context)
        
        return context
    
    def _analyze_actual_behavior(self, context: FileContext, source_code: str):
        """Analyze the actual behavior of functions by examining their implementation"""
        tree = ast.parse(source_code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                behavior = self._analyze_function_behavior(node)
                context.actual_behavior[func_name] = behavior
    
    def _analyze_function_behavior(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze what a function actually does based on its AST"""
        behavior = {
            'raises_exceptions': [],
            'handles_none': False,
            'performs_division': False,
            'uses_external_calls': [],
            'has_type_checks': False,
            'return_type': 'unknown',
            'validates_inputs': False
        }
        
        # Analyze function body
        for node in ast.walk(func_node):
            # Check for explicit raises
            if isinstance(node, ast.Raise):
                if isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
                    behavior['raises_exceptions'].append(node.exc.func.id)
            
            # Check for type validation
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'isinstance':
                    behavior['has_type_checks'] = True
                elif isinstance(node.func, ast.Attribute):
                    behavior['uses_external_calls'].append(f"{node.func.attr}")
            
            # Check for division operations
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                behavior['performs_division'] = True
            
            # Check for None comparisons
            elif isinstance(node, ast.Compare):
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Constant) and comparator.value is None:
                        behavior['handles_none'] = True
            
            # Check for input validation patterns
            elif isinstance(node, ast.If):
                behavior['validates_inputs'] = True
        
        return behavior
    
    def _extract_imports(self, tree: ast.AST, context: FileContext):
        """Extract import statements with enhanced categorization - FIXED VERSION"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # Handle 'import module' statements
                for alias in node.names:
                    is_standard = alias.name.split('.')[0] in self.standard_libs
                    context.imports.append(ImportInfo(
                        module=alias.name,  # Use alias.name for Import nodes
                        alias=alias.asname,
                        from_import=False,  # This is not a from import
                        specific_imports=[],  # No specific imports for 'import module'
                        is_external=not is_standard,
                        is_standard_lib=is_standard
                    ))
                    if not is_standard:
                        context.external_dependencies.add(alias.name.split('.')[0])
            
            elif isinstance(node, ast.ImportFrom):
                # Handle 'from module import ...' statements
                if node.module:  # node.module can be None for relative imports
                    specific_imports = [alias.name for alias in node.names]
                    is_standard = node.module.split('.')[0] in self.standard_libs
                    context.imports.append(ImportInfo(
                        module=node.module,
                        alias=None,
                        from_import=True,
                        specific_imports=specific_imports,
                        is_external=not is_standard,
                        is_standard_lib=is_standard
                    ))
                    if not is_standard:
                        context.external_dependencies.add(node.module.split('.')[0])
    
    def _extract_functions(self, tree: ast.AST, context: FileContext, source_code: str):
        """Extract function definitions with enhanced analysis"""
        lines = source_code.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip methods (they'll be handled in class extraction)
                if any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree) 
                       if hasattr(parent, 'body') and node in parent.body):
                    continue
                
                func_info = self._create_enhanced_function_info(node, lines, False)
                context.functions.append(func_info)
    
    def _extract_classes(self, tree: ast.AST, context: FileContext, source_code: str):
        """Extract class definitions with enhanced analysis"""
        lines = source_code.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                attributes = []
                properties = []
                is_abstract = False
                constructor_args = []
                
                # Check for abstract methods and constructor
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_info = self._create_enhanced_function_info(item, lines, True, node.name)
                        methods.append(method_info)
                        
                        # Check for constructor
                        if item.name == '__init__':
                            constructor_args = [arg.arg for arg in item.args.args if arg.arg != 'self']
                        
                        # Check for abstract methods or properties
                        if any(d.id == 'abstractmethod' if isinstance(d, ast.Name) else False 
                               for d in item.decorator_list):
                            is_abstract = True
                        
                        if any(d.id == 'property' if isinstance(d, ast.Name) else False 
                               for d in item.decorator_list):
                            properties.append(item.name)
                    
                    elif isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                attributes.append(target.id)
                
                inheritance = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        inheritance.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        inheritance.append(f"{base.value.id}.{base.attr}")
                
                class_info = ClassInfo(
                    name=node.name,
                    methods=methods,
                    attributes=attributes,
                    docstring=ast.get_docstring(node),
                    inheritance=inheritance,
                    source_code='\n'.join(lines[node.lineno-1:node.end_lineno]) if node.end_lineno else '',
                    line_number=node.lineno,
                    is_abstract=is_abstract,
                    properties=properties,
                    constructor_args=constructor_args
                )
                context.classes.append(class_info)
    
    def _create_enhanced_function_info(self, node: ast.FunctionDef, lines: List[str], 
                                     is_method: bool, class_name: Optional[str] = None) -> FunctionInfo:
        """Create enhanced FunctionInfo with complexity analysis"""
        args = [arg.arg for arg in node.args.args]
        return_annotation = None
        
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return_annotation = node.returns.id
            elif isinstance(node.returns, ast.Constant):
                return_annotation = str(node.returns.value)
            else:
                return_annotation = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)
        
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            else:
                decorators.append(ast.unparse(decorator) if hasattr(ast, 'unparse') else str(decorator))
        
        # Analyze function complexity and calls
        complexity_score = self._calculate_complexity(node)
        called_functions = self._extract_function_calls(node)
        exception_types = self._extract_exception_types(node)
        param_types = self._extract_parameter_types(node)
        actual_exceptions = self._extract_actual_exceptions(node)
        
        end_line = node.end_lineno if node.end_lineno else min(node.lineno + 20, len(lines))
        source_code = '\n'.join(lines[node.lineno-1:end_line])
        
        return FunctionInfo(
            name=node.name,
            args=args,
            return_annotation=return_annotation,
            docstring=ast.get_docstring(node),
            source_code=source_code,
            line_number=node.lineno,
            decorators=decorators,
            is_method=is_method,
            class_name=class_name,
            complexity_score=complexity_score,
            called_functions=called_functions,
            exception_types=exception_types,
            param_types=param_types,
            actual_exceptions=actual_exceptions
        )
    
    def _extract_parameter_types(self, node: ast.FunctionDef) -> Dict[str, str]:
        """Extract parameter type annotations"""
        param_types = {}
        for arg in node.args.args:
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    param_types[arg.arg] = arg.annotation.id
                else:
                    param_types[arg.arg] = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation)
        return param_types
    
    def _extract_actual_exceptions(self, node: ast.FunctionDef) -> List[str]:
        """Extract exceptions that are actually raised in the code"""
        exceptions = []
        for child in ast.walk(node):
            if isinstance(child, ast.Raise) and child.exc:
                if isinstance(child.exc, ast.Call) and isinstance(child.exc.func, ast.Name):
                    exceptions.append(child.exc.func.id)
                elif isinstance(child.exc, ast.Name):
                    exceptions.append(child.exc.id)
        return list(set(exceptions))
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _extract_function_calls(self, node: ast.FunctionDef) -> List[str]:
        """Extract function calls within a function"""
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(f"{child.func.value.id if isinstance(child.func.value, ast.Name) else 'obj'}.{child.func.attr}")
        return list(set(calls))  # Remove duplicates
    
    def _extract_exception_types(self, node: ast.FunctionDef) -> List[str]:
        """Extract exception types that can be raised"""
        exceptions = []
        for child in ast.walk(node):
            if isinstance(child, ast.Raise) and child.exc:
                if isinstance(child.exc, ast.Call) and isinstance(child.exc.func, ast.Name):
                    exceptions.append(child.exc.func.id)
                elif isinstance(child.exc, ast.Name):
                    exceptions.append(child.exc.id)
        return list(set(exceptions))
    
    def _extract_global_variables(self, tree: ast.AST, context: FileContext):
        """Extract global variable assignments with type inference"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.targets[0], ast.Name):
                    var_name = node.targets[0].id
                    if isinstance(node.value, ast.Constant):
                        context.global_variables[var_name] = {
                            'value': node.value.value,
                            'type': type(node.value.value).__name__
                        }
                    elif isinstance(node.value, ast.List):
                        context.global_variables[var_name] = {
                            'value': 'list',
                            'type': 'list'
                        }
                    elif isinstance(node.value, ast.Dict):
                        context.global_variables[var_name] = {
                            'value': 'dict',
                            'type': 'dict'
                        }
                    else:
                        context.global_variables[var_name] = {
                            'value': 'complex_value',
                            'type': 'unknown'
                        }
    
    def _analyze_complexity(self, context: FileContext):
        """Analyze overall file complexity for test generation strategy"""
        total_functions = len(context.functions) + sum(len(cls.methods) for cls in context.classes)
        avg_complexity = sum(f.complexity_score for f in context.functions) / max(1, len(context.functions)) if context.functions else 0
        external_deps = len(context.external_dependencies)
        
        # More reasonable thresholds
        if total_functions > 30 or avg_complexity > 8 or external_deps > 15:
            context.test_complexity = "high"
        elif total_functions > 15 or avg_complexity > 5 or external_deps > 8:
            context.test_complexity = "medium"
        else:
            context.test_complexity = "low"


class OptimizedIncrementalTestGenerator:
    """Main class that generates tests with minimal API calls and intelligent caching"""
    
    def __init__(self, gemini_api_key: str = None):
        self.parser = EnhancedPythonFileParser()
        self.batch_generator = BatchTestGenerator(gemini_api_key)
        self.validator = None
        self.valid_tests = []
        self.failed_tests = []
        
    def generate_incremental_tests(self, source_file: str, output_file: str = None) -> Dict[str, Any]:
        """Generate tests with optimized batch processing and caching"""
        if output_file is None:
            output_file = f"test_{Path(source_file).name}"
        
        logger.info(f"🚀 Starting OPTIMIZED incremental test generation for: {source_file}")
        
        # Initialize validator
        self.validator = IndividualTestValidator(source_file)
        
        # Parse the source file
        context = self.parser.parse_file(source_file)
        
        # Collect all functions for batch processing
        all_functions = []
        all_functions.extend(context.functions)
        
        # Add class methods to functions list
        for class_info in context.classes:
            for method in class_info.methods:
                if not method.name.startswith('__') or method.name == '__init__':
                    all_functions.append(method)
        
        logger.info(f"📋 Found {len(all_functions)} functions/methods total")
        logger.info(f"🎯 Will process in batches to minimize API calls")
        
        # Generate tests in optimized batches
        total_generated = 0
        total_valid = 0
        
        if all_functions:
            # Process functions in batches
            test_cases = self.batch_generator.generate_tests_batch(all_functions, context)
            total_generated = len(test_cases)
            
            logger.info(f"📊 Generated {total_generated} test cases from batch processing")
            
            # Validate each test case
            for i, test_case in enumerate(test_cases, 1):
                logger.info(f"🔍 Validating test {i}/{total_generated}: {test_case.name}")
                
                # Find the corresponding function info
                target_func = self._find_function_info(test_case.target_function, all_functions)
                
                if target_func and self._validate_test_case(test_case, target_func, context):
                    self.valid_tests.append(test_case)
                    total_valid += 1
                    logger.info(f"  ✅ Valid ({total_valid}/{i})")
                else:
                    self.failed_tests.append(test_case)
                    logger.info(f"  ❌ Failed: {test_case.error_message}")
        
        # Write all valid tests to file
        final_test_file = self._write_final_test_file(output_file, source_file, context)
        
        # Get API usage stats
        api_stats = self.batch_generator.get_api_usage_stats()
        
        # Generate summary
        summary = {
            'source_file': source_file,
            'output_file': output_file,
            'total_generated': total_generated,
            'total_valid': total_valid,
            'success_rate': f"{(total_valid/total_generated*100):.1f}%" if total_generated > 0 else "0%",
            'valid_tests': len(self.valid_tests),
            'failed_tests': len(self.failed_tests),
            'test_file_created': final_test_file,
            'api_usage': api_stats
        }
        
        logger.info(f"✅ OPTIMIZED generation complete!")
        logger.info(f"📊 Summary: {total_valid}/{total_generated} tests valid ({summary['success_rate']})")
        logger.info(f"🚀 API calls made: {api_stats['total_api_calls']}")
        logger.info(f"💾 Cache hits: {api_stats['cache_hits']}")
        logger.info(f"💰 Estimated savings: {api_stats['estimated_cost_savings']}")
        
        return summary
    
    def _find_function_info(self, target_name: str, all_functions: List[FunctionInfo]) -> Optional[FunctionInfo]:
        """Find function info by name"""
        for func_info in all_functions:
            if func_info.name == target_name:
                return func_info
        return None
    
    def _validate_test_case(self, test_case: TestCase, func_info: FunctionInfo, context: FileContext) -> bool:
        """Validate a single test case"""
        is_valid, error = self.validator.validate_single_test(test_case)
        
        if is_valid:
            test_case.is_valid = True
            return True
        else:
            test_case.is_valid = False
            test_case.error_message = error
            return False
    
    def _write_final_test_file(self, output_file: str, source_file: str, context: FileContext) -> str:
        """Write all valid tests to the final test file"""
        logger.info(f"📝 Writing {len(self.valid_tests)} valid tests to: {output_file}")
        
        api_stats = self.batch_generator.get_api_usage_stats()
        
        # Generate file header
        header = f'''# -*- coding: utf-8 -*-
"""
Unit tests for {Path(source_file).name}
Generated by OPTIMIZED AI Test Generator with Batch Processing & Caching

Generation Metadata:
- Source file: {source_file}
- Total valid tests: {len(self.valid_tests)}
- Failed tests: {len(self.failed_tests)}
- Success rate: {(len(self.valid_tests)/(len(self.valid_tests)+len(self.failed_tests))*100):.1f}%
- API calls made: {api_stats['total_api_calls']}
- Cache hits: {api_stats['cache_hits']}
- Estimated cost savings: {api_stats['estimated_cost_savings']}
- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
- All tests have been individually validated ✅
"""

import pytest
import json
import math
import os
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from {Path(source_file).stem} import *

'''
        
        # Combine all valid test codes
        all_test_code = header
        
        # Group tests by target function for better organization
        tests_by_function = {}
        for test in self.valid_tests:
            func_name = test.target_function
            if func_name not in tests_by_function:
                tests_by_function[func_name] = []
            tests_by_function[func_name].append(test)
        
        # Write tests grouped by function
        for func_name, tests in tests_by_function.items():
            all_test_code += f"\n# Tests for {func_name}\n"
            for test in tests:
                all_test_code += test.code + "\n"
        
        # Add summary comment at the end
        all_test_code += f'''
# OPTIMIZATION SUMMARY:
# - Generated {len(self.valid_tests)} valid tests with only {api_stats['total_api_calls']} API calls
# - Used intelligent batching and caching to minimize costs
# - Cache provided {api_stats['cache_hits']} reusable test patterns
# - Estimated cost savings: {api_stats['estimated_cost_savings']}
# - All tests individually validated to ensure they pass ✅
'''
        
        # Write to file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(all_test_code)
            logger.info(f"✅ Test file successfully written: {output_file}")
            
            # Save cache after successful generation
            self.batch_generator.cache_manager.save_cache()
            
            return output_file
        except Exception as e:
            logger.error(f"❌ Failed to write test file: {e}")
            raise
    
    def get_generation_report(self) -> str:
        """Generate a detailed report with optimization metrics"""
        if not self.valid_tests and not self.failed_tests:
            return "No tests were generated."
        
        total_tests = len(self.valid_tests) + len(self.failed_tests)
        success_rate = (len(self.valid_tests) / total_tests * 100) if total_tests > 0 else 0
        api_stats = self.batch_generator.get_api_usage_stats()
        
        report = f"""
# OPTIMIZED Test Generation Report

## Performance Summary
- **Total API calls**: {api_stats['total_api_calls']} (MINIMIZED! 🚀)
- **Cache hits**: {api_stats['cache_hits']}
- **Estimated cost savings**: {api_stats['estimated_cost_savings']}
- **Rate limit status**: {api_stats['requests_this_minute']}/8 per minute, {api_stats['requests_this_hour']}/200 per hour

## Test Results
- **Total tests generated**: {total_tests}
- **Valid tests**: {len(self.valid_tests)}
- **Failed tests**: {len(self.failed_tests)}
- **Success rate**: {success_rate:.1f}%

## Valid Tests ({len(self.valid_tests)})
"""
        for test in self.valid_tests:
            report += f"- ✅ {test.name} ({test.test_type})\n"
        
        if self.failed_tests:
            report += f"\n## Failed Tests ({len(self.failed_tests)})\n"
            for test in self.failed_tests:
                report += f"- ❌ {test.name} ({test.test_type}): {test.error_message[:100]}...\n"
        
        report += f"""
## Optimization Benefits
- 🚀 **Batch Processing**: Generated multiple tests per API call
- 💾 **Smart Caching**: Reused patterns for similar functions  
- ⏱️ **Rate Limiting**: Prevented API quota exhaustion
- 💰 **Cost Effective**: Estimated {api_stats['estimated_cost_savings']} savings
- 🎯 **High Success Rate**: {success_rate:.1f}% of generated tests are valid

## Next Steps
- Run: `pytest {Path(self.valid_tests[0].target_function).name if self.valid_tests else 'test_file.py'} -v`
- All {len(self.valid_tests)} valid tests should pass ✅
- Cache will speed up future test generation
"""
        
        return report


# Utility functions
def discover_python_files(directory: str, exclude_patterns: List[str] = None) -> List[str]:
    """Discover Python files in a directory"""
    if exclude_patterns is None:
        exclude_patterns = ['__pycache__', '.git', '.venv', 'venv', 'env', 'test_*.py', '*_test.py']
    
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
        
        for file in files:
            if file.endswith('.py') and not any(pattern in file for pattern in exclude_patterns):
                python_files.append(os.path.join(root, file))
    
    return python_files


def main():
    """OPTIMIZED main function with batch processing and caching"""
    print("🧪 OPTIMIZED AI Unit Test Generator v4.0")
    print("🚀 Batch Processing + Smart Caching + Rate Limiting")
    print("=" * 70)
    
    try:
        # Initialize the optimized test generator
        generator = OptimizedIncrementalTestGenerator()
        
        # Example usage - you can modify this for your needs
        source_file = "main.py"
        
        if not os.path.exists(source_file):
            print(f"❌ Source file not found: {source_file}")
            
            # Try to discover Python files
            discovered_files = discover_python_files(".", exclude_patterns=['test_*.py'])
            if discovered_files:
                print(f"\n📁 Found Python files in current directory:")
                for i, file in enumerate(discovered_files[:5], 1):  # Show first 5
                    print(f"   {i}. {file}")
                
                if len(discovered_files) > 5:
                    print(f"   ... and {len(discovered_files) - 5} more")
                
                # Use the first discovered file for demo
                source_file = discovered_files[0]
                print(f"\n🎯 Using: {source_file}")
            else:
                print("No Python files found in current directory")
                return
        
        print(f"\n🚀 Starting OPTIMIZED test generation for: {source_file}")
        print("🎯 Using batch processing and smart caching!")
        print("-" * 50)
        
        # Generate tests with optimization
        result = generator.generate_incremental_tests(source_file)
        
        print("-" * 50)
        print("🎉 OPTIMIZED test generation completed!")
        print(f"📂 Generated file: {result['output_file']}")
        print(f"📊 Success rate: {result['success_rate']}")
        print(f"✅ Valid tests: {result['valid_tests']}")
        print(f"🚀 API calls: {result['api_usage']['total_api_calls']}")
        print(f"💰 Cost savings: {result['api_usage']['estimated_cost_savings']}")
        
        # Show detailed report
        print("\n📋 Detailed Report:")
        print(generator.get_generation_report())
        
        print("\n💡 Next steps:")
        print("   1. All valid tests have been individually verified ✅")
        print("   2. Install dependencies: pip install pytest pytest-mock")
        print(f"   3. Run tests: pytest {result['output_file']} -v")
        print("   4. All tests should pass without issues! 🎯")
        
        print("\n🛡️ OPTIMIZATION BENEFITS:")
        print("   • Batch processing: Multiple tests per API call")
        print("   • Smart caching: Reuses patterns for similar functions")
        print("   • Rate limiting: Prevents quota exhaustion")
        print("   • Cost effective: Minimizes API usage")
        print("   • Individual validation: Only valid tests included")
        
        # Offer batch processing
        if len(discover_python_files(".")) > 1:
            print(f"\n🚀 Want to generate optimized tests for all Python files?")
            print("   Run: python test_generator.py --batch-optimized")
        
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def batch_optimized_generation():
    """Batch process multiple files with MAXIMUM optimization"""
    try:
        generator = OptimizedIncrementalTestGenerator()
        python_files = discover_python_files(".")
        
        if not python_files:
            print("No Python files found for batch processing")
            return
        
        print(f"🔄 OPTIMIZED batch processing {len(python_files)} files...")
        print("🚀 Using batch processing, caching, and rate limiting!")
        
        results = []
        total_generated = 0
        total_valid = 0
        total_api_calls = 0
        
        os.makedirs("tests", exist_ok=True)
        
        for source_file in python_files:
            try:
                print(f"\n📁 Processing: {source_file}")
                output_file = os.path.join("tests", f"test_{Path(source_file).name}")
                result = generator.generate_incremental_tests(source_file, output_file)
                
                results.append(result)
                total_generated += result['total_generated']
                total_valid += result['total_valid']
                total_api_calls += result['api_usage']['total_api_calls']
                
                print(f"  ✅ {result['valid_tests']} valid tests, {result['api_usage']['total_api_calls']} API calls")
                
                # Brief pause between files to be extra careful with rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                results.append({
                    'source_file': source_file,
                    'success': False,
                    'error': str(e)
                })
        
        # Generate batch summary
        successful = sum(1 for r in results if r.get('valid_tests', 0) > 0)
        overall_success_rate = (total_valid / total_generated * 100) if total_generated > 0 else 0
        
        print(f"\n🎉 OPTIMIZED batch processing complete!")
        print(f"📊 Overall success: {successful}/{len(python_files)} files processed")
        print(f"🎯 Test success rate: {overall_success_rate:.1f}% ({total_valid}/{total_generated})")
        print(f"🚀 Total API calls: {total_api_calls} (MINIMIZED!)")
        print(f"🛡️ All {total_valid} valid tests have been individually verified!")
        print(f"💰 Massive cost savings achieved through optimization!")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")


if __name__ == "__main__":
    # Add CLI argument handling
    if len(sys.argv) > 1 and sys.argv[1] == '--batch-optimized':
        batch_optimized_generation()
    else:
        main()