
import numpy as np  # Import NumPy for numerical operations  
import torch  # Import PyTorch for tensor calculations and deep learning  
  
from nltk.tokenize import sent_tokenize  # Import the sentence tokenizer from NLTK  
  
from sentence_transformers import SentenceTransformer, util    
# Import the SentenceTransformer class and utility functions for sentence embeddings  
  
from wikipedia_util import (  
    get_wiki_content,  # Function to fetch Wikipedia article content  
    wiki_search_whole_phrase,  # Function to search Wikipedia for whole phrases  
    NUMBER_OF_RESULTS  # Constant: number of results to retrieve from Wikipedia  
)  
  
from LLM_util import OPENAI    
# Import the OPENAI API or utilities for large language models (LLMs)  
  
from LLM_util import paraphrase_by_azure_openAI    
# Import the function to paraphrase text using Azure's OpenAI service  
  
# Set the device to GPU if available, else fall back to CPU  
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
  
# Global variable for a paraphrase model, to be initialized later  
PARAPHASE_MODEL = None  

if __name__ == "__main__":
    pass


def estimate_filtered_similarity(  
    text,   
    best_url,   
    matching_data,   
    filtered_threshold=0.8,   
    remain_ratio_threshold=0.5  
):  
    """  
    Estimates the average similarity score for sentences that pass a specific threshold,  
    given a set of sentence similarities, and returns 0 if the ratio of remaining   
    sentences below the threshold is not met.  
  
    Parameters:  
        text (str): The input text to compare (not used in function currently).  
        best_url (str or None): The best-matching URL (may be empty or None).  
        matching_data (list): List of tuples containing similarity information where  
            each tuple is expected to have the similarity score at index 2.  
        filtered_threshold (float, optional): Minimum similarity score to be considered. Default is 0.8.  
        remain_ratio_threshold (float, optional): Minimum ratio of sentences passing  
            the similarity threshold to return an average. Default is 0.5.  
  
    Returns:  
        float: The average similarity of filtered sentences if the ratio of such sentences   
        is above the specified threshold, else 0.  
    """  
    average = 0  # Initialize the average similarity score  
    remain_ratio = 0  # Initialize the ratio of sentences that pass the threshold  
    num_sentence = 0  # Initialize the total number of sentences  
  
    # Check if best_url is provided and non-empty  
    if best_url is not None and best_url != "":  
        filtered_sim = []  # List to hold similarity scores that pass the threshold  
        num_sentence = len(matching_data)  # Count the number of sentence matches  
  
        # Iterate through all sentence matches in matching_data  
        for match in matching_data:  
            sim = float(match[2])  # Get the similarity score from the tuple  
            if sim >= filtered_threshold:  
                filtered_sim.append(sim)  # Add score if above the threshold  
  
        # Compute the ratio of sentences that remain after filtering  
        if num_sentence > 0:  
            remain_ratio = len(filtered_sim) / num_sentence  
  
        # Calculate the average similarity of the filtered sentences  
        if len(filtered_sim) > 0:  
            average = np.average(filtered_sim)  
  
    # Return the average if the remaining ratio meets the threshold, else return 0  
    if remain_ratio >= remain_ratio_threshold:  
        return average  
    return 0  


def find_best_similarity_by_relative_search(  
    input_text,  
    is_check_bbc=False,  
    is_wikipedia=False,  
    number_of_wiki_query=NUMBER_OF_RESULTS,  
    min_threshold=0.963  
):  
    """  
    Finds and returns the best matching URL and its similarity to the input text using relative search techniques,  
    either limited to Wikipedia search or general search, optionally restricted to BBC URLs.  
  
    Parameters:  
        input_text (str): The text to find the best similarity for.  
        is_check_bbc (bool): If True, only considers BBC URLs.  
        is_wikipedia (bool): If True, searches using Wikipedia only.  
        number_of_wiki_query (int): Number of Wikipedia search results to consider.  
        min_threshold (float): Minimum similarity threshold to early return result.  
  
    Returns:  
        best_url (str or None): URL with the best similarity score found.  
        best_avg_similarity (float): Best average similarity score found.  
        best_data (any): Additional data from similarity computation.  
  
    """  
    # Disable BBC check if searching on Wikipedia only  
    if is_wikipedia:  
        is_check_bbc = False  
  
    # Set to track URLs that have already been checked  
    checked_urls = set()  
    # Initialize result variables  
    best_url = None  
    best_avg_similarity = -1  
    best_data = None  
  
    # If using Wikipedia search, tokenize input_text into sentences for searching  
    if is_wikipedia:  
        candidate_phrases = sent_tokenize(input_text)  
    else:  
        candidate_phrases = [input_text]  
  
    # Iterate through each candidate phrase to search for matches  
    for phrase in candidate_phrases:  
        if is_wikipedia:  
            # Perform Wikipedia search for the phrase  
            search_results = wiki_search_whole_phrase(phrase, number_of_wiki_query)  
        else:  
            # Perform general search (not explicitly shown in the original code)  
            search_results = None  # Needs implementation if not Wikipedia  
  
        if search_results is not None:  
            # Extract URLs from search results  
            if is_wikipedia:  
                urls = search_results  
            else:  
                urls = [item['link'] for item in search_results.get("items", [])]  
  
            # Iterate through each found URL  
            for url in urls:  
                # If BBC filtering is enabled, skip URLs not from BBC  
                if is_check_bbc:  
                    if "https://www.bbc." not in url:  
                        continue  
                # Skip already checked URLs  
                if url in checked_urls:  
                    continue  
  
                # Add the current URL to the set of checked URLs  
                checked_urls.add(url)  
                print(f"checking the url = {url}")  
  
                # Calculate average similarity and obtain data for this URL  
                avg_similarity, data = measure_similarity_with_url(input_text, url)  
  
                # If similarity is better than previous best, update best values  
                if avg_similarity > best_avg_similarity:  
                    best_avg_similarity = avg_similarity  
                    best_data = data  
                    best_url = url  
  
                    # Further filter similarity if necessary before early return  
                    filter_similarity = estimate_filtered_similarity(input_text, url, data)  
                    if filter_similarity >= min_threshold:  
                        return best_url, best_avg_similarity, best_data  
  
    # Return best found result after exhausting all candidates  
    return best_url, best_avg_similarity, best_data  

def split_to_sentences(input_text):  
    """  
    Splits the input text into sentences.  
  
    This function first splits the input text into paragraphs using the newline  
    character. Then, it tokenizes each paragraph into individual sentences.  
  
    Parameters:  
        input_text (str): The text to be split into sentences.  
  
    Returns:  
        list: A list of sentences extracted from the input text.  
    """  
    # Split the input text into paragraphs by newline character  
    paragraphs = input_text.split("\n")  
    # Initialize a list to store the resulting sentences  
    result = []  
    # Iterate over each paragraph in the list  
    for paragraph in paragraphs:  
        # Remove leading and trailing whitespace from each paragraph  
        paragraph = paragraph.strip()  
        # Continue only if the paragraph is not empty  
        if paragraph != "":  
            # Tokenize the paragraph into sentences  
            sentences = sent_tokenize(paragraph)  
            # Add the list of sentences to the result  
            result.extend(sentences)  
    # Return the final list of sentences  
    return result  

def extract_text(url):  
    """  
    Determines the file type of the given URL and extracts text accordingly.  
      
    Parameters:  
        url (str): The URL from which text needs to be extracted.  
      
    Returns:  
        str: The extracted text from the URL, or an empty string if extraction is not possible.  
    """  
    try:  
        # Check if the URL is from Wikipedia  
        if ".wikipedia." in url:  
            # Extract and return content from the Wikipedia page  
            return get_wiki_content(url)  
        else:  
            # Return empty string if the URL is not from Wikipedia  
            return ""  
    except Exception:  
        # Return empty string in case of any exceptions during extraction  
        return ""  


def calculate_sim_for_regeneration(  
    input_text,  
    url,  
    matching_index,  
    input_index,  
    source_from_url_index,  
    LLM_model,  
    verbose=False,  
    provider=OPENAI  
):  
    """  
    Calculates the average similarity score between the input sentences and  
    their regenerated/ paraphrased versions taken from a specific URL.  
    Utilizes sentence embeddings for similarity calculation and a language model  
    for paraphrasing.  
  
    Parameters:  
        input_text (str): The input text to compare against.  
        url (str): The URL from which to extract the content for regeneration.  
        matching_index (list of int): Indices of sentences in the URL text to use.  
        input_index (list of int): Indices in the input text (for similarity comparison).  
        source_from_url_index (list of int): Indices to map source sentences from URL.  
        LLM_model (str): Language model name to use for paraphrasing.  
        verbose (bool, optional): If True, prints debug information. Default is False.  
        provider (str, optional): Provider for LLM. Default is OPENAI.  
  
    Returns:  
        tuple:  
            regeneration_sim (float or None): The average similarity score between  
            input and regenerated sentences. None if computation fails.  
            regeneration_text (str): The paraphrased text from the URL sentences.  
    """  
    regeneration_sim = None  # Initialize similarity value  
    regeneration_text = ""   # Initialize regenerated text  
  
    print("start matching")  
    if input_text is None:  # Return early if input is None  
        return regeneration_sim, regeneration_text  
  
    input_sentences = split_to_sentences(input_text)  # Split input into sentences  
  
    if verbose:  
        print(f"input_sentences = {input_sentences}")  
  
    page_text = extract_text(url)  # Extract text content from the URL  
    url_page_sentences = split_to_sentences(page_text)  # Split URL text into sentences  
  
    filter_sentences = []  
    for index in matching_index:  
        filter_sentences.append(url_page_sentences[index])  # Select matched sentences  
  
    candidate_text = " ".join(filter_sentences)  # Concatenate selected sentences  
  
    if provider == OPENAI:  
        # Paraphrase candidate text using the LLM  
        regeneration_text = paraphrase_by_azure_openAI(candidate_text, LLM_model)  
  
    if regeneration_text is None:  
        print("return due to regeneration_text == None")  
        return regeneration_sim, regeneration_text  
  
    page_sentences = split_to_sentences(regeneration_text)  # Split regenerated text  
  
    if verbose:  
        print(f"url_page_sentences = {url_page_sentences}")  
        print(f"filter_sentences = {filter_sentences}")  
        print(f"regeneration_sentences = {page_sentences}")  
        print(f"matching_index = {matching_index}")  
        print(f"input_index = {input_index}")  
        print(f"source_from_url_index = {source_from_url_index}")  
  
    if len(input_sentences) == 0 or len(page_sentences) == 0:  
        # Return if no sentences found  
        print("return due to len(input_sentences) == 0 or len(page_sentences) == 0")  
        return regeneration_sim, regeneration_text  
  
    if len(page_sentences) != len(matching_index):  
        # The number of sentences does not match; can't compare reliably  
        if verbose:  
            print("return due to len(page_sentences) != len(matching_index)")  
        return regeneration_sim, regeneration_text  
  
    global PARAPHASE_MODEL  
    if PARAPHASE_MODEL is None:  
        # Load sentence transformer model for similarity computation  
        PARAPHASE_MODEL = SentenceTransformer('paraphrase-MiniLM-L6-v2')  
        PARAPHASE_MODEL.to(DEVICE)  
  
    # Encode input sentences into embeddings  
    embeddings1 = PARAPHASE_MODEL.encode(  
        input_sentences, convert_to_tensor=True, device=DEVICE  
    )  
    # Encode regenerated sentences into embeddings  
    embeddings2 = PARAPHASE_MODEL.encode(  
        page_sentences, convert_to_tensor=True, device=DEVICE  
    )  
  
    # Calculate cosine similarity matrix for input and regenerated sentences  
    similarity_matrix = util.cos_sim(embeddings1, embeddings2).cpu().numpy()  
  
    all_sim = []  
    for index in input_index:  
        sim = similarity_matrix[index][index]  # Diagonal: compare aligned sentences  
        all_sim.append(sim)  
  
    if verbose:  
        print(f"all_sim with regeneration = {all_sim}")  
  
    if len(all_sim) > 0:  
        # Calculate average similarity if there are values  
        regeneration_sim = np.average(all_sim)  
        return regeneration_sim, regeneration_text  
    else:  
        return regeneration_sim, regeneration_text  


def measure_similarity_with_url_return_matching_index(input_text, url):  
    """  
    Measures sentence-level similarity between an input text and the content of a web page from a given URL.  
    Returns the average similarity score, detailed alignments, and matching indices of similar sentences.  
  
    Parameters:  
        input_text (str): The text whose sentences will be compared to the page content.  
        url (str): The URL of the web page whose content will be used for comparison.  
  
    Returns:  
        avg_similarity (float): The average of the highest similarity scores found for each sentence in input_text.  
        alignment (list): List of [input_sentence, matched_page_sentence, similarity_score] for each sentence in input_text.  
        matching_index (list): List of indices indicating which sentence in the page_text is the best match to each input_sentence.  
    """  
  
    avg_similarity = -1  # Default to -1 in case of early exit or errors  
    alignment = []       # Will store the alignments of matched sentences  
    similarities = []    # List to collect individual similarity scores  
    matching_index = []  # List to collect indices of matched sentences in the page  
  
    print("start matching")  # Indicate process start  
  
    if input_text is None:  
        # If no input text, return defaults  
        return avg_similarity, alignment, matching_index  
  
    # Split input_text into sentences  
    input_sentences = split_to_sentences(input_text)  
  
    # Extract web page text from URL  
    page_text = extract_text(url)  
  
    if page_text is None:  
        # If extraction failed, return defaults  
        return avg_similarity, alignment, matching_index  
  
    # Split extracted text into sentences  
    page_sentences = split_to_sentences(page_text)  
  
    # Check for empty sentence lists  
    if len(input_sentences) == 0 or len(page_sentences) == 0:  
        return avg_similarity, alignment, matching_index  
  
    # Ensure the SentenceTransformer model is loaded and on correct device  
    global PARAPHASE_MODEL  
    if PARAPHASE_MODEL is None:  
        PARAPHASE_MODEL = SentenceTransformer('paraphrase-MiniLM-L6-v2')  
        PARAPHASE_MODEL.to(DEVICE)  
  
    # Encode sentences into embeddings  
    embeddings1 = PARAPHASE_MODEL.encode(input_sentences, convert_to_tensor=True, device=DEVICE)  
    embeddings2 = PARAPHASE_MODEL.encode(page_sentences, convert_to_tensor=True, device=DEVICE)  
  
    # Compute cosine similarity between each pair of sentences  
    similarity_matrix = util.cos_sim(embeddings1, embeddings2).cpu().numpy()  
  
    # For each sentence in input_text, find the most similar sentence in the page  
    for i, sentence1 in enumerate(input_sentences):  
        # Find the page sentence with the highest similarity  
        max_sim_index = np.argmax(similarity_matrix[i])  
        max_similarity = similarity_matrix[i][max_sim_index]  
  
        # Record the similarity score and matching index  
        similarities.append(max_similarity)  
        matching_index.append(max_sim_index)  
  
        # Store the aligned sentence pair and their similarity  
        item = [sentence1, page_sentences[max_sim_index], max_similarity]  
        alignment.append(item)  
  
    # Compute average similarity if any matches found  
    if len(similarities) > 0:  
        avg_similarity = np.average(similarities)  
  
    # Return the results  
    return avg_similarity, alignment, matching_index  


def measure_similarity_with_url(input_text, url):  
    """  
    Measures the average sentence similarity between a given input text and the text extracted from a given URL.  
      
    This function splits both the input text and the web page text into sentences, encodes sentences as embeddings,  
    computes cosine similarity between each pair of sentences, aligns each input sentence to its most similar sentence  
    from the web page, and returns the average similarity score as well as the detailed alignments.  
      
    Parameters:  
        input_text (str): The input text to compare.  
        url (str): The URL from which to extract web page text.  
          
    Returns:  
        avg_similarity (float): The average of maximal sentence similarities; -1 if unavailable.  
        data (list): A list of alignments, each as [input_sentence, matched_page_sentence, similarity_score].  
    """  
    # Default similarity score (used as error/empty state sentinel)  
    avg_similarity = -1  
    # Holds sentence-by-sentence alignments (input sentence, matched page sentence, similarity)  
    data = []  
    # List of similarity scores for each input sentence  
    similarities = []  
  
    print("start matching")  
  
    # If there is no input text, return default values  
    if input_text is None:  
        return avg_similarity, data  
  
    # Split input text into sentences  
    input_sentences = split_to_sentences(input_text)  
    # Extract the web page's text  
    page_text = extract_text(url)  
  
    # If page could not be extracted, return default values  
    if page_text is None:  
        return avg_similarity, data  
  
    # Split extracted web page text into sentences  
    page_sentences = split_to_sentences(page_text)  
  
    # If either text has zero sentences, return default values  
    if len(input_sentences) == 0 or len(page_sentences) == 0:  
        return avg_similarity, data  
  
    global PARAPHASE_MODEL  
  
    # Initialize the SentenceTransformer model if not already loaded  
    if PARAPHASE_MODEL is None:  
        PARAPHASE_MODEL = SentenceTransformer('paraphrase-MiniLM-L6-v2')  
        PARAPHASE_MODEL.to(DEVICE)  
  
    # Number of input sentences (may not be used further)  
    total_sentence = len(input_sentences)  
  
    # Encode input sentences into embeddings (using the BERT-like model)  
    embeddings1 = PARAPHASE_MODEL.encode(input_sentences, convert_to_tensor=True, device=DEVICE)  
    # Encode page sentences into embeddings  
    embeddings2 = PARAPHASE_MODEL.encode(page_sentences, convert_to_tensor=True, device=DEVICE)  
  
    # Compute cosine similarity between all pairs (input_sentence_i, page_sentence_j)  
    similarity_matrix = util.cos_sim(embeddings1, embeddings2).cpu().numpy()  
  
    # List to hold aligned pairs  
    alignment = []  
    # Counter (not used anywhere, legacy variable)  
    count = 0  
  
    # For each input sentence, find the most similar page sentence  
    for i, sentence1 in enumerate(input_sentences):  
        # Identify the index of the most similar page sentence for the current input sentence  
        max_sim_index = np.argmax(similarity_matrix[i])  
        # Obtain the similarity score of this best match  
        max_similarity = similarity_matrix[i][max_sim_index]  
  
        # Append this similarity score to the list  
        similarities.append(max_similarity)  
  
        # Prepare aligned item: [input sentence, matched page sentence, similarity score]  
        item = [sentence1, page_sentences[max_sim_index], max_similarity]  
        alignment.append(item)  
  
    # If similarities were found, calculate the average similarity score  
    if len(similarities) > 0:  
        avg_similarity = np.average(similarities)  
  
    # Return average similarity score and the list of alignments  
    return avg_similarity, alignment  