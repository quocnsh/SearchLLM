import wikipedia  
import time  
  
NUMBER_OF_RESULTS = 1  # Default number of Wikipedia search results  
  
def safe_wikipedia_search(query, results, retries=5, delay=5):  
    """  
    Safely searches Wikipedia handling exceptions and retrying if needed.  
  
    Parameters:  
        query (str): The search query.  
        results (int): Number of results to return.  
        retries (int): Number of retry attempts on failure.  
        delay (int): Delay in seconds between retries.  
  
    Returns:  
        list: List of Wikipedia page titles matching the search query.  
    """  
    for attempt in range(retries):  
        try:  
            # Perform the Wikipedia search  
            return wikipedia.search(query, results=results)  
        except wikipedia.exceptions.WikipediaException as e:  
            # Print the exception and wait before retrying  
            print(f"Attempt {attempt + 1} failed: {e}")  
            time.sleep(delay)  
        except Exception:  
            # Handle any other exceptions  
            return []  
    # If all attempts fail, return an empty list  
    return []  
  
def search_output_urls(sentence, number_of_results):  
    """  
    Searches Wikipedia for a sentence and outputs a list of page URLs.  
  
    Parameters:  
        sentence (str): The input query or sentence.  
        number_of_results (int): Number of results to return.  
  
    Returns:  
        list: List of Wikipedia page URLs matching the query.  
    """  
    # Limit the query length to 300 characters  
    if len(sentence) > 300:  
        sentence = sentence[:300]  
  
    # Get search result titles  
    search_results = safe_wikipedia_search(sentence, results=number_of_results)  
    result = []  
    for title in search_results:  
        print(f"title = {title}")  
        try:  
            # Fetch the page object for this title  
            page = wikipedia.page(title, auto_suggest=False)  
            # Append the page URL to the result list  
            result.append(page.url)  
        except wikipedia.exceptions.DisambiguationError:  
            # If disambiguation error occurs, skip this title  
            continue  
        except wikipedia.exceptions.PageError:  
            # If page does not exist, skip this title  
            continue  
    return result  
  
def wiki_search_whole_phrase(input_text, number_of_wiki_query=NUMBER_OF_RESULTS):  
    """  
    Searches Wikipedia for the whole input phrase and returns a list of URLs.  
  
    Parameters:  
        input_text (str): The phrase to search on Wikipedia.  
        number_of_wiki_query (int): Number of search results to return.  
  
    Returns:  
        list: List of Wikipedia page URLs found for the input phrase.  
    """  
    # Call search_output_urls to get Wikipedia URLs  
    urls = search_output_urls(input_text, number_of_wiki_query)  
    return urls  
  
def get_wiki_content(url):  
    """  
    Given a Wikipedia page URL, extracts and returns the page content.  
  
    Parameters:  
        url (str): The Wikipedia page URL.  
  
    Returns:  
        str: The textual content of the Wikipedia page, or empty string if error occurs.  
    """  
    # Extract the title from the URL  
    title = url.split("/")[-1]  # E.g., "Python_(programming_language)"  
    # Replace underscores with spaces for the title  
    title = title.replace("_", " ")  
  
    result = ""  
    try:  
        # Fetch the Wikipedia page  
        page = wikipedia.page(title, auto_suggest=False)  
        result = page.content  
    except wikipedia.exceptions.DisambiguationError:  
        # Handle ambiguous title error  
        print(f"Error parsing page: {url}")  
    except wikipedia.exceptions.PageError:  
        # Handle page not found error  
        print(f"Error parsing page: {url}")  
  
    return result  
  
if __name__ == "__main__":  
    # The main section can be used for testing or as an entry point  
    pass  