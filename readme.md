# SearchLLM: Detecting LLM Generated Text by Measuring the Similarity with Regeneration of the Candidate Source via Search Engine

## Requirements

1. **API Keys**   
   - Obtain your Azure OpenAI API key.
   - Add the following entries to your `.env` file:
     - `AZURE_ENDPOINT`
     - `API_KEY`     
     - `API_VERSION`     

2. **Data**
   - Prepare your dataset and save it as `5_data/data.csv`.
   - Each line in the file should contain a text and a label, where the label is either `human` or `machine`.

     

2. **Dependencies**
   - Install the required Python packages:
     ```
     pip install -r requirements.txt     
     ```

## Usage

To run the script, execute the following command:

```bash
python SearchLLM.py
```
