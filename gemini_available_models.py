import google.generativeai as genai

# Configure your API key
genai.configure(api_key="AIzaSyAxmQYYvVA1Ue_pMIlfxZDrLMUZJzRixrs")

print("List of models:")
for model in genai.list_models():
    # Check if the model supports the 'generateContent' method
    if 'generateContent' in model.supported_generation_methods:
        print(f"Model Name: {model.name}")
        print(f"  Supported Methods: {model.supported_generation_methods}")
        print(f"  Description: {model.description}")
        print("-" * 20)