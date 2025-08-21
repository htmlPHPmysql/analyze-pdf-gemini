from dotenv import load_dotenv
import google.generativeai as genai
import os # Import the os module

def main():
    """
    The main function that runs the Streamlit application.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the API key from the environment variable
    api_key = os.getenv("GOOGLE_API_KEY")

    # Configure the Google API with the retrieved key
    # This is the line that was missing or commented out
    genai.configure(api_key=api_key)

    print("List of models:")
    for model in genai.list_models():
        # Check if the model supports the 'generateContent' method
        if 'generateContent' in model.supported_generation_methods:
            print(f"Model Name: {model.name}")
            print(f"  Supported Methods: {model.supported_generation_methods}")
            print(f"  Description: {model.description}")
            print("-" * 20)

if __name__ == '__main__':
    main()