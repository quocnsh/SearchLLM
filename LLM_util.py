from openai import AzureOpenAI  # Import the AzureOpenAI class from openai package  
from dotenv import load_dotenv  # Import load_dotenv to load environment variables from .env file  
import os  # Import os to access environment variables  
  
# Set the maximum number of tokens for the language model  
MAX_TOKENS = 16384  # For GPT-4o: 100,000, for gpt-4o-mini: 32,768  

# Constant representing the identifier for the GPT-4o Mini model  
GPT_4O_MINI = "gpt-4o-mini"  
  
# Constant representing the OpenAI provider name  
OPENAI = "openai"  

# Load environment variables from a .env file  
load_dotenv()  
  
# Retrieve Azure endpoint, API key, and API version from environment variables  
AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT')  
API_KEY = os.getenv('API_KEY')  
API_VERSION = os.getenv('API_VERSION')  
  
# Instantiate the AzureOpenAI client with endpoint, API key, and version  
azure_client = AzureOpenAI(  
    azure_endpoint=AZURE_ENDPOINT,  
    api_key=API_KEY,  
    api_version=API_VERSION  
)  
  
# Constant representing a generic API error message  
API_ERROR = "API_ERROR"  
  
if __name__ == "__main__":  
    # Main entry point (no code to execute when run as script)  
    pass  
  
  
def paraphrase_by_azure_openAI(input_text, deployment_name):  
    """  
    Paraphrases the provided input text using Azure OpenAI's chat completion API.  
  
    Parameters:  
        input_text (str): The text to paraphrase.  
        deployment_name (str): The name of the deployment/model to use.  
  
    Returns:  
        str: The paraphrased text if successful, or API_ERROR if an error occurs.  
    """  
    # Construct the prompt with the input text to request a paraphrase  
    prompt = (  
        f"Paraphrasing for the text: ```{input_text}``` "  
        "Only output the paraphrased text without explanation."  
    )  
  
    # Return API_ERROR if the input is the error constant  
    if input_text == API_ERROR:  
        return API_ERROR  
  
    try:  
        # Send a completion request to the Azure OpenAI service  
        response = azure_client.chat.completions.create(  
            model=deployment_name,  # Specify the deployment/model to use  
            messages=[  
                {"role": "system", "content": "You are a helpful assistant."},  
                {"role": "user", "content": prompt},  
            ],  
            max_tokens=MAX_TOKENS,  # Set the maximum number of tokens for the output  
            temperature=0,  # Use deterministic output (no randomness)  
        )  
        # Return the content of the first choice message from the response  
        return response.choices[0].message.content  
    except Exception:  
        # Return API_ERROR in case of exceptions (e.g., network or API issues)  
        return API_ERROR  