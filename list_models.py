from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# List available models
try:
    models = genai.list_models()
    print("Available Gemini models:")
    print("-" * 50)
    
    for model in models:
        print(f"Name: {model.name}")
        if hasattr(model, 'display_name'):
            print(f"Display Name: {model.display_name}")
        if hasattr(model, 'description'):
            print(f"Description: {model.description}")
        print("-" * 30)
        
except Exception as e:
    print(f"Error listing models: {e}")
    print("Make sure your GEMINI_API_KEY is set correctly in your .env file")