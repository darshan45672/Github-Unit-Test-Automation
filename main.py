import math
import json
import requests
from typing import List, Dict, Optional
from datetime import datetime

class Calculator:
    """A simple calculator class with various mathematical operations"""
    
    def __init__(self, precision: int = 2):
        """Initialize calculator with specified precision"""
        self.precision = precision
        self.history = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers"""
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError("Both arguments must be numbers")
        
        result = round(a + b, self.precision)
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def divide(self, a: float, b: float) -> float:
        """Divide two numbers"""
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError("Both arguments must be numbers")
        
        if b == 0:
            raise ValueError("Cannot divide by zero")
        
        result = round(a / b, self.precision)
        self.history.append(f"{a} / {b} = {result}")
        return result
    
    def power(self, base: float, exponent: float) -> float:
        """Calculate base raised to the power of exponent"""
        if not isinstance(base, (int, float)) or not isinstance(exponent, (int, float)):
            raise TypeError("Both arguments must be numbers")
        
        result = round(math.pow(base, exponent), self.precision)
        self.history.append(f"{base} ^ {exponent} = {result}")
        return result
    
    def get_history(self) -> List[str]:
        """Get calculation history"""
        return self.history.copy()
    
    def clear_history(self):
        """Clear calculation history"""
        self.history.clear()

def factorial(n: int) -> int:
    """Calculate factorial of a number"""
    if not isinstance(n, int):
        raise TypeError("Argument must be an integer")
    
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    
    if n == 0 or n == 1:
        return 1
    
    result = 1
    for i in range(2, n + 1):
        result *= i
    
    return result

def is_prime(n: int) -> bool:
    """Check if a number is prime"""
    if not isinstance(n, int):
        raise TypeError("Argument must be an integer")
    
    if n < 2:
        return False
    
    if n == 2:
        return True
    
    if n % 2 == 0:
        return False
    
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    
    return True

def fetch_user_data(user_id: int) -> Dict:
    """Fetch user data from an API"""
    if not isinstance(user_id, int) or user_id <= 0:
        raise ValueError("User ID must be a positive integer")
    
    try:
        response = requests.get(f"https://jsonplaceholder.typicode.com/users/{user_id}")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch user data: {e}")

def process_numbers(numbers: List[float], operation: str = "sum") -> float:
    """Process a list of numbers with the specified operation"""
    if not isinstance(numbers, list):
        raise TypeError("Numbers must be provided as a list")
    
    if not numbers:
        raise ValueError("List cannot be empty")
    
    if not all(isinstance(n, (int, float)) for n in numbers):
        raise TypeError("All elements must be numbers")
    
    if operation == "sum":
        return sum(numbers)
    elif operation == "average":
        return sum(numbers) / len(numbers)
    elif operation == "max":
        return max(numbers)
    elif operation == "min":
        return min(numbers)
    else:
        raise ValueError(f"Unsupported operation: {operation}")

class FileProcessor:
    """Process and analyze files"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = base_path
        self.processed_files = []
    
    def read_json_file(self, filename: str) -> Dict:
        """Read and parse a JSON file"""
        if not filename.endswith('.json'):
            raise ValueError("File must have .json extension")
        
        try:
            with open(f"{self.base_path}/{filename}", 'r') as f:
                data = json.load(f)
                self.processed_files.append(filename)
                return data
        except FileNotFoundError:
            raise FileNotFoundError(f"File {filename} not found")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    
    def write_json_file(self, filename: str, data: Dict) -> bool:
        """Write data to a JSON file"""
        if not filename.endswith('.json'):
            raise ValueError("File must have .json extension")
        
        try:
            with open(f"{self.base_path}/{filename}", 'w') as f:
                json.dump(data, f, indent=2)
                self.processed_files.append(filename)
                return True
        except Exception as e:
            raise RuntimeError(f"Failed to write file: {e}")
    
    def get_processed_files(self) -> List[str]:
        """Get list of processed files"""
        return self.processed_files.copy()

# Global constants
PI = 3.14159
MAX_ITERATIONS = 1000

def generate_report(data: Dict, format: str = "json") -> str:
    """Generate a report from data"""
    if not isinstance(data, dict):
        raise TypeError("Data must be a dictionary")
    
    timestamp = datetime.now().isoformat()
    report = {
        "timestamp": timestamp,
        "data_summary": {
            "total_keys": len(data),
            "has_data": bool(data)
        },
        "data": data
    }
    
    if format == "json":
        return json.dumps(report, indent=2)
    elif format == "summary":
        return f"Report generated at {timestamp} with {len(data)} data points"
    else:
        raise ValueError(f"Unsupported format: {format}")