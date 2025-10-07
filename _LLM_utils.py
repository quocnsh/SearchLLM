from _9_constant import *
from openai import AzureOpenAI
from dotenv import load_dotenv  # Import load_dotenv to load environment variables from .env file  

import os

MAX_TOKENS = 16384 

load_dotenv()  
AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT')  
API_KEY = os.getenv('API_KEY')  
API_VERSION = os.getenv('API_VERSION')  
  
# Instantiate the AzureOpenAI client with endpoint, API key, and version  
azure_client = AzureOpenAI(  
    azure_endpoint=AZURE_ENDPOINT,  
    api_key=API_KEY,  
    api_version=API_VERSION  
)  

API_ERROR = "API_ERROR"
IGNORE_BY_API_ERROR = "IGNORE_BY_API_ERROR"


def paraphrase_by_azure_openAI(input_text, deplopment_name):  
    """Return a paraphrased version of input_text using Azure OpenAI."""  
    prompt = f"Paraphrasing for the text: ```{input_text}``` Only output the paraphrased text without explanation."  
    if input_text == API_ERROR:  
        return API_ERROR  
    try:  
        response = azure_client.chat.completions.create(  
            model=deplopment_name,  # model = "deployment_name".  
            messages=[  
                {"role": "system", "content": "You are a helpful assistant."},  
                {"role": "user", "content": prompt},  
            ],  
            max_tokens=MAX_TOKENS,  
            temperature=0,  
        )  
        return response.choices[0].message.content  
    except Exception:  
        return API_ERROR  
  
  
if __name__ == "__main__":  
    a = 2