import wikipedia
import torch
import numpy as np
import spacy

from nltk.tokenize import sent_tokenize

from _9_constant import *
from sentence_transformers import SentenceTransformer, util
from _LLM_utils import paraphrase_by_azure_openAI
import requests
from bs4 import BeautifulSoup

IGNORE_NER_LIST = ["DATE", "TIME", "CARDINAL", ]
nlp = spacy.load("en_core_web_trf")
# MODEL = None
NUMBER_OF_RESULTS = 1
MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import pandas as pd


def safe_wikipedia_search(query, results, retries=5, delay=5):  
    """Search Wikipedia with retries and delay on failure."""  
    for attempt in range(retries):  
        try:  
            return wikipedia.search(query, results=results)  
        except wikipedia.exceptions.WikipediaException as e:  
            print(f"Attempt {attempt + 1} failed: {e}")  
            time.sleep(delay)  
        except Exception:  
            return []  
    return []  
  
  
def search_output_urls(sentence, number_of_results):  
    """  
    Search for a sentence and return a list of Wikipedia page URLs.  
    """  
    if len(sentence) > 300:  
        sentence = sentence[:300]  # Truncate the query to 300 characters  
    search_results = safe_wikipedia_search(sentence, results=number_of_results)  
    result = []  
    for title in search_results:  
        print(f"title = {title}")  
        try:  
            page = wikipedia.page(title, auto_suggest=False)  
            result.append(page.url)  
        except wikipedia.exceptions.DisambiguationError:  
            continue  
        except wikipedia.exceptions.PageError:  
            continue  
    return result

def wiki_search_whole_phrase(input_text, number_of_wiki_query=NUMBER_OF_RESULTS):  
    """Search Wikipedia for the whole input phrase and return result URLs."""  
    urls = search_output_urls(input_text, number_of_wiki_query)  
    return urls  
  
  
def get_wiki_content(url):  
    """Get the content of a Wikipedia page given its URL."""  
    # Extract the title from the URL  
    title = url.split("/")[-1]  
    # Replace underscores with spaces  
    title = title.replace("_", " ")  
    result = ""  
    try:  
        page = wikipedia.page(title, auto_suggest=False)  
        result = page.content  
    except wikipedia.exceptions.DisambiguationError:  
        print(f"Error parsing: {url}")  
    except wikipedia.exceptions.PageError:  
        print(f"Error parsing: {url}")  
    except Exception:  
        print(f"Error parsing: {url}")  
    return result  
  
  
def measure_similarity_two_text(text_1, text_2):  
    """Measure similarity between two texts using a transformer model."""  
    global MODEL  
    if MODEL is None:  
        MODEL = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")  
        MODEL.to(DEVICE)  # Move the model to the appropriate device  
    embeddings = MODEL.encode([text_1, text_2], convert_to_tensor=True)  
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])  
    return similarity.item()  
  
  
def extract_NER(input_text, ignore_ner_list=IGNORE_NER_LIST):  
    """Extract named entities from text, excluding those in ignore_ner_list."""  
    doc = nlp(input_text)  
    result = []  
    for ent in doc.ents:  
        if ent.label_ not in ignore_ner_list:  
            result.append((ent.text, ent.label_))  
    return result  
  
  
def extract_first_NER(input_text, ignore_ner_list=IGNORE_NER_LIST):  
    """Extract the first named entity from the input text, ignoring specified entities."""  
    ners = extract_NER(input_text, ignore_ner_list)  
    result = None  
    if len(ners) == 0:  
        return result  
    else:  
        return ners[0][0]  
  
  
def search_by_NER_and_first_k_sentence(  
    input_text,  
    k,  
    minimum_similarity_for_mapping,  
    min_similarity_for_found=0.8  
):  
    """  
    Search Wikipedia using the first named entity and first k sentences.  
    Return the best URL and its similarity score and additional information.  
    """  
    input_sentences = sent_tokenize(input_text)  
    if len(input_sentences) <= 0:  
        return None, (-2, None)  
    ner = extract_first_NER(input_sentences[0])  
    queries = []  
    if ner is not None:  
        queries.append(ner)  
    queries.extend(input_sentences[:k])  
    best_url = None  
    best_similarity = -2  
    best_additional_information = None  
    processed_urls = []  
    for query in queries:  
        urls = wiki_search_whole_phrase(query)  
        for url in urls:  
            if url not in processed_urls:  
                processed_urls.append(url)  
                similarity, additional_information = measure_similarity_by_mapping_sentence(  
                    input_text, url, minimum_similarity_for_mapping  
                )  
                if similarity > best_similarity:  
                    best_similarity = similarity  
                    best_url = url  
                    best_additional_information = additional_information  
                if similarity >= min_similarity_for_found:  
                    return best_url, (best_similarity, best_additional_information)  
    return best_url, (best_similarity, best_additional_information)  

def measure_similarity_mapping(sentences_1, sentences_2, chose_sen_1, chose_sen_2, verbose=False):  
    """  
    Map selected sentences from two lists and measure their similarity.  
    """  
    similarity = None  # error default  
    n1 = len(sentences_1)  
    n2 = len(sentences_2)  
    text_1 = []  
    text_2 = []  
  
    if verbose:  
        print(f"chose_sen_1 = {chose_sen_1}")  
  
    for index in chose_sen_1:  
        if index >= n1:  
            return similarity  
        text_1.append(sentences_1[index])  
  
    for index in chose_sen_2:  
        if index >= n2:  
            return similarity  
        text_2.append(sentences_2[index])  
  
    combine_text_1 = " ".join(text_1)  
    combine_text_2 = " ".join(text_2)  
    similarity = measure_similarity_two_text(combine_text_1, combine_text_2)  
    return similarity  
  
  
def evaluate_candidate_matching(sentences_1, sentences_2, index_1, index_2):  
    """  
    Evaluate the quality of the current matching pair:  
    = sim(selected pair) + sim(next pair) (if available)  
    """  
    combine_sim = None  
    current_sim = measure_similarity_mapping(sentences_1, sentences_2, index_1, index_2)  
    if current_sim is None:  
        return combine_sim, combine_sim  
    next_index_1 = [index_1[-1] + 1]  
    next_index_2 = [index_2[-1] + 1]  
    next_sim = measure_similarity_mapping(sentences_1, sentences_2, next_index_1, next_index_2)  
    if next_sim is not None:  
        combine_sim = current_sim + next_sim  
    else:  
        combine_sim = current_sim  
    return current_sim, combine_sim  
  
  
def mapping_sentence(sentences_1, sentences_2, min_similarity=0, verbose=False):  
    """  
    Heuristic mapping:  
        - Map the first sentence of sentences_1 to the first sentence of sentences_2.  
        - Try to concatenate the second sentence of sentences_1 and check if the similarity of the next pair increases.  
        - Do the same for the second sentence of sentences_2.  
        - Choose the next pair with the highest score.  
    Output:  
        average similarity  
        matched sentences  
            ([sen1, sen2], [sent3])  
            or  
            ([sen1], [sen2, sent3])  
            ([sen1], [sen2])  
    """  
    n1 = len(sentences_1)  
    n2 = len(sentences_2)  
    avg_similarity = -1  
    all_similarity = []  
    mapping = []  
    index1 = -1  
    index2 = -1  
  
    while index1 < n1 - 1:  
        best_combine_sim = -1  
        best_current_sim = -1  
        best_mapping = []  
        all_candidate = [  
            [[index1 + 1], [index2 + 1]],  
            [[index1 + 1, index1 + 2], [index2 + 1]],  
            [[index1 + 1], [index2 + 1, index2 + 2]],  
        ]  
        for candidate in all_candidate:  
            current_sim, current_combine_sim = evaluate_candidate_matching(  
                sentences_1, sentences_2, candidate[0], candidate[1]  
            )  
            if current_combine_sim is not None and current_combine_sim > best_combine_sim:  
                best_mapping = candidate  
                best_current_sim = current_sim  
                best_combine_sim = current_combine_sim  
        if best_combine_sim == -1:  
            return None, [], []  
        if best_current_sim > min_similarity:  
            mapping.append(best_mapping)  
            index1 = best_mapping[0][-1]  
            index2 = best_mapping[1][-1]  
            all_similarity.append(best_current_sim)  
        else:  
            return None, [], []  
  
    if verbose:  
        print(f"all sim = {all_similarity}")  
        for sim, (idx_1, idx_2) in zip(all_similarity, mapping):  
            print(f"******sim = {sim}*********")  
            print("first sentence")  
            for idx in idx_1:  
                print(f"index = {idx} text = {sentences_1[idx]}")  
            print("second sentence")  
            for idx in idx_2:  
                print(f"index = {idx} text = {sentences_2[idx]}")  
  
    if len(all_similarity) > 0:  
        avg_similarity = np.average(all_similarity)  
    return avg_similarity, all_similarity, mapping        


def extract_text_from_html(url):  
    """  
    Extract text from an HTML page at the given URL.  
    """  
    response = requests.get(url)  
    soup = BeautifulSoup(response.text, "html.parser")  
    return soup.get_text(separator="\n")  
  
  
def extract_text(url):  
    """  
    Determine the file type and extract text accordingly from the given URL.  
    """  
    try:  
        if ".wikipedia." in url:  
            return get_wiki_content(url)  
        print(f"extract url = {url}")  
        file_extension = url.split('.')[-1].lower()  
        if file_extension in ["html", "htm"]:  
            return extract_text_from_html(url)  
        elif file_extension == "pdf":  
            return ""  
        elif file_extension in ["doc", "docx"]:  
            return ""  
        else:  
            print(f"Unsupported file type: {file_extension}")  
            return extract_text_from_html(url)  
    except Exception:  
        return ""  
  
  
def measure_similarity_by_mapping_sentence(  
    input_text,  
    url,  
    minimum_similarity_for_mapping,  
    verbose=True  
):  
    """  
    Measure similarity between the input text and the content at the given URL  
    by mapping sentences and calculating similarity.  
    """  
    similarity = -1  
    input_sentences = sent_tokenize(input_text)  
    wiki_text = extract_text(url)  
    wiki_sentences = sent_tokenize(wiki_text)  
  
    if verbose:  
        print(f"len(wiki_text)= {len(wiki_text)}")  
        print(f"len(wiki_sentences)= {len(wiki_sentences)}")  
  
    if len(input_sentences) == 0 or len(wiki_sentences) == 0:  
        return -1, ([], [])  
  
    first_sentence = input_sentences[0]  
    global MODEL  
    if MODEL is None:  
        MODEL = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")  
        MODEL.to(DEVICE)  # Move the model to the appropriate device  
  
    query_embedding = MODEL.encode(  
        [first_sentence], convert_to_tensor=True, device=DEVICE, batch_size=64  
    )  
  
    if ".wikipedia." in url:  
        sentence_embeddings = MODEL.encode(  
            wiki_sentences[:5], convert_to_tensor=True, device=DEVICE, batch_size=64  
        )  
    else:  
        sentence_embeddings = MODEL.encode(  
            wiki_sentences, convert_to_tensor=True, device=DEVICE, batch_size=64  
        )  
  
    similarities = util.cos_sim(query_embedding, sentence_embeddings).squeeze(0)  
    sorted_indices = similarities.argsort(descending=True)  
    best_index = sorted_indices[0]  
  
    if verbose:  
        print(f"similarity = {sorted_indices}")  
        print(f"similarities = {similarities}")  
        print(f"best_index = {best_index}")  
        print(f"similarities[best_index] = {similarities[best_index]}")  
  
    if similarities[best_index] < minimum_similarity_for_mapping:  
        return -1, ([], [])  
  
    filter_wiki_sentences = wiki_sentences[  
        best_index:best_index + (len(input_sentences) * 2 + 1)  
    ]  
    min_similarity = 0  
  
    if verbose:  
        print(f"len(input_sentences) = {len(input_sentences)}")  
        print(f"len(filter_wiki_sentences) = {len(filter_wiki_sentences)}")  
  
    if len(filter_wiki_sentences) < (len(input_sentences) / 2) - 1:  
        return -1, ([], [])  
  
    avg_similarity, all_similarity, mapping = mapping_sentence_improve_speed(  
        input_sentences, filter_wiki_sentences, min_similarity  
    )  
  
    if verbose:  
        print(f"similarity = {similarity}")  
        print(f"mapping = {mapping}")  
  
    text_mapping = []  
    if avg_similarity is not None:  
        similarity = avg_similarity  
        for idx_1, idx_2 in mapping:  
            pair = [[], []]  
            for idx in idx_1:  
                pair[0].append(input_sentences[idx])  
            for idx in idx_2:  
                pair[1].append(filter_wiki_sentences[idx])  
            text_mapping.append(pair)  
  
    return similarity, (all_similarity, text_mapping)

def calculate_similarity_with_regeneration(input_text, text_mapping, model_name, custom_prompt=None, verbose=False):  
    """  
    Calculate similarity between input sentences and regenerated sentences using a specified model.  
    """  
    all_similarity = []  
    mapping = []  
    input_sentences = sent_tokenize(input_text)  
    if len(input_sentences) <= 0:  
        return -1, ([], [])  
    wiki_sentences = []  
    for _, wiki in text_mapping:  
        wiki_sentences.extend(wiki)  
    if len(wiki) <= 0:  
        return -1, ([], [])  
    wiki_text = " ".join(wiki_sentences)  
    provider = PROVIDER_MAPPING[model_name]  
    regeneration = paraphrase_by_azure_openAI(wiki_text, model_name)  
    try:  
        regeneration_sentences = sent_tokenize(regeneration)  
    except Exception:  
        return -1, ([], [])  
    if len(regeneration_sentences) <= 0:  
        return -1, ([], [])  
    avg_similarity, all_similarity, mapping = mapping_sentence(input_sentences, regeneration_sentences)  
    if verbose:  
        print(f"all_similarity = {all_similarity}")  
        print(f"avg_similarity = {avg_similarity}")  
        print(f"mapping = {mapping}")  
    text_mapping = []  
    if avg_similarity is not None:  
        for idx_1, idx_2 in mapping:  
            pair = [[], []]  
            for idx in idx_1:  
                pair[0].append(input_sentences[idx])  
            for idx in idx_2:  
                pair[1].append(regeneration_sentences[idx])  
            text_mapping.append(pair)  
        return avg_similarity, (all_similarity, text_mapping)  
  
  
def create_mapping_similarity(sentences_1, sentences_2, verbose=False):  
    """  
    Create a mapping of sentence pairs and compute their similarity scores.  
    """  
    result = dict()  
    n1 = len(sentences_1)  
    n2 = len(sentences_2)  
    text_1 = []  
    text_2 = []  
    index_pair = []  
    for index_1 in range(n1):  
        for index_2 in range(n2):  
            candidate_idx_1 = [(index_1,), (index_1, index_1 + 1), (index_1,)]  
            candidate_idx_2 = [(index_2,), (index_2,), (index_2, index_2 + 1)]  
            for candidate_1, candidate_2 in zip(candidate_idx_1, candidate_idx_2):  
                if candidate_1[-1] >= n1:  
                    continue  
                if candidate_2[-1] >= n2:  
                    continue  
                index_pair.append([candidate_1, candidate_2])  
    global MODEL  
    if MODEL is None:  
        MODEL = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")  
        MODEL.to(DEVICE)  # Move the model to the appropriate device  
    pair_text_1 = []  
    for index_1 in range(n1 - 1):  
        pair_text_1.append(sentences_1[index_1] + " " + sentences_1[index_1 + 1])  
    pair_text_2 = []  
    for index_2 in range(n2 - 1):  
        pair_text_2.append(sentences_2[index_2] + " " + sentences_2[index_2 + 1])  
    sentence_embedding_1 = MODEL.encode(sentences_1, convert_to_tensor=True, device=DEVICE, batch_size=64)  
    sentence_embedding_2 = MODEL.encode(sentences_2, convert_to_tensor=True, device=DEVICE, batch_size=64)  
    pair_sentence_embedding_1 = MODEL.encode(pair_text_1, convert_to_tensor=True, device=DEVICE, batch_size=64)  
    pair_sentence_embedding_2 = MODEL.encode(pair_text_2, convert_to_tensor=True, device=DEVICE, batch_size=64)  
    for pair_idx in index_pair:  
        index_1 = pair_idx[0]  
        index_2 = pair_idx[1]  
        if len(index_1) == 1:  
            embedding_1 = sentence_embedding_1[index_1[0]]  
        else:  
            embedding_1 = pair_sentence_embedding_1[index_1[0]]  
        if len(index_2) == 1:  
            embedding_2 = sentence_embedding_2[index_2[0]]  
        else:  
            embedding_2 = pair_sentence_embedding_2[index_2[0]]  
        similarity = util.pytorch_cos_sim(embedding_1, embedding_2)  
        result[tuple(pair_idx)] = similarity.item()  
    if verbose:  
        print(result)  
    return result



def measure_similarity_mapping_improve_speed(  
    sentences_1, sentences_2, chose_sen_1, chose_sen_2, mapping_sim, verbose=False  
):  
    """  
    Measure the similarity between selected sentences using a mapping.  
    """  
    if verbose:  
        a = 2  # Placeholder variable for demonstration  
  
    sen_1 = tuple(chose_sen_1)  
    sen_2 = tuple(chose_sen_2)  
  
    if len(chose_sen_1) == 1:  
        sen_1 = (chose_sen_1[0],)  
  
    if len(chose_sen_2) == 1:  
        sen_2 = (chose_sen_2[0],)  
  
    index = (sen_1, sen_2)  
  
    if verbose:  
        print(index)  
  
    if index in mapping_sim:  
        if verbose:  
            a = 2  # Placeholder variable for demonstration  
            print(f"mapping_sim[index] = {mapping_sim[index]}")  
        return mapping_sim[index]  
    else:  
        return None  
  
  
def evaluate_candidate_matching_improve_speed(  
    sentences_1, sentences_2, index_1, index_2, mapping_sim  
):  
    """  
    Evaluate the quality of the current matching pair.  
    The score is the similarity of the selected pair plus the similarity of the next pair (if exists).  
    """  
    combine_sim = None  
    current_sim = measure_similarity_mapping_improve_speed(  
        sentences_1, sentences_2, index_1, index_2, mapping_sim  
    )  
    if current_sim is None:  
        return combine_sim, combine_sim  
  
    next_index_1 = [index_1[-1] + 1]  
    next_index_2 = [index_2[-1] + 1]  
    next_sim = measure_similarity_mapping_improve_speed(  
        sentences_1, sentences_2, next_index_1, next_index_2, mapping_sim  
    )  
    if next_sim is not None:  
        combine_sim = current_sim + next_sim  
    else:  
        combine_sim = current_sim  
    return current_sim, combine_sim  
  
  
def mapping_sentence_improve_speed(  
    sentences_1, sentences_2, min_similarity=0, verbose=False  
):  
    """  
    Heuristic mapping:  
        - Map the first sentence of sentences_1 with the first sentence of sentences_2  
        - Try to concatenate the second sentence of sentences_1, and check if the similarity of the next pair increases.  
        - Do the same for the second sentence of sentences_2  
        - Find the pair with the highest next similarity  
  
    Output:  
        - average similarity  
        - match sentence:  
            ([sen1, sen2], [sent3])  
            or  
            ([sen1], [sen2, sent3])  
            ([sen1], [sen2])  
    """  
    mapping_sim = create_mapping_similarity(sentences_1, sentences_2)  
    n1 = len(sentences_1)  
    n2 = len(sentences_2)  
  
    avg_similarity = -1  
    all_similarity = []  
    mapping = []  
    index1 = -1  
    index2 = -1  
  
    while index1 < n1 - 1:  
        best_combine_sim = -1  
        best_current_sim = -1  
        best_mapping = []  
        all_candidate = [  
            [[index1 + 1], [index2 + 1]],  
            [[index1 + 1, index1 + 2], [index2 + 1]],  
            [[index1 + 1], [index2 + 1, index2 + 2]],  
        ]  
        for candidate in all_candidate:  
            current_sim, current_combine_sim = evaluate_candidate_matching_improve_speed(  
                sentences_1, sentences_2, candidate[0], candidate[1], mapping_sim  
            )  
            if current_combine_sim is not None and current_combine_sim > best_combine_sim:  
                best_mapping = candidate  
                best_current_sim = current_sim  
                best_combine_sim = current_combine_sim  
        if best_combine_sim == -1:  
            return None, [], []  
  
        if best_current_sim > min_similarity:  
            mapping.append(best_mapping)  
            index1 = best_mapping[0][-1]  
            index2 = best_mapping[1][-1]  
            all_similarity.append(best_current_sim)  
        else:  
            return None, [], []  
  
    if verbose:  
        print(f"all sim = {all_similarity}")  
        for sim, (idx_1, idx_2) in zip(all_similarity, mapping):  
            print(f"******sim = {sim}*********")  
            print("first sentence")  
            for idx in idx_1:  
                print(f"index = {idx} text = {sentences_1[idx]}")  
            print("second sentence")  
            for idx in idx_2:  
                print(f"index = {idx} text = {sentences_2[idx]}")  
  
    if len(all_similarity) > 0:  
        avg_similarity = np.average(all_similarity)  
    return avg_similarity, all_similarity, mapping  
  
  
if __name__ == "__main__":  
    a = 2  # Placeholder variable for demonstration