# Define the output file path for storing results  
OUTPUT_FILE = "5_data/result_testing.csv"  
  
# Import required utilities and models from custom modules  
from LLM_util import OPENAI, GPT_4O_MINI  
from search_engine_util import (  
    find_best_similarity_by_relative_search,  
    measure_similarity_with_url_return_matching_index,  
    calculate_sim_for_regeneration,  
)  
  
# Import accuracy and AUC metric calculations from sklearn  
from sklearn.metrics import auc, roc_curve, accuracy_score  
  
# Import numpy for numerical computations  
import numpy as np  
  
# Import os for file and directory manipulations  
import os  
  
# Import pandas for structured data manipulation  
import pandas as pd  
  
# Import csv for reading and writing csv files  
import csv  
  
# Import the Hugging Face transformers pipeline for model inference  
from transformers import pipeline  
  
# Constant for representing API errors  
API_ERROR = "API_ERROR"  
  
# Human label constant, usually used as a ground truth label  
HUMAN_LABEL = 0  
  
# Machine label constant, usually used for machine-generated content  
MACHINE_LABEL = 1  
  
# String representation for human type  
HUMAN = "human"  
  
# String representation for machine type  
MACHINE = "machine"  

def write_to_file(filename, text):
    """
    Writes the given text to a file and prints it to the console.


    Parameters:  
        filename (str): The path to the file where the text will be written.  
        text (str): The text to be written and printed.  

    Returns:  
        None  
    """  
    # Print the text to the console  
    print(text)  
    # Open the file in append mode with UTF-8 encoding and write the text  
    with open(filename, "a+", encoding="utf-8") as f:  
        f.write(text)


def read_csv_data(input_file):  
    """  
    Reads data from a CSV file and returns its contents as a NumPy array.  
  
    Parameters:  
        input_file (str): The path to the CSV file to be read.  
  
    Returns:  
        numpy.ndarray: The contents of the CSV file as a NumPy array of strings.  
    """  
    # Read the CSV file into a pandas DataFrame with all columns as string type,  
    # do not convert default NA values, and use comma as the separator  
    my_file = pd.read_csv(  
        input_file,  
        dtype='string',  
        keep_default_na=False,  
        sep=','  
    ).values  
  
    # Return the data as a NumPy array  
    return my_file  

# def write_to_csv(filename, row):
#     with open(filename, 'a+', encoding='UTF8', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(row)

def write_to_csv(filename, row):  
    """  
    Appends a single row to a CSV file.  
  
    Parameters:  
        filename (str): The path of the CSV file to write to.  
        row (list): The row data to write into the CSV file.  
  
    Returns:  
        None  
    """  
    # Open the file in append mode with UTF-8 encoding and proper newline handling  
    with open(filename, 'a+', encoding='UTF8', newline='') as f:  
        # Create a CSV writer object  
        writer = csv.writer(f)  
        # Write the given row to the CSV file  
        writer.writerow(row)  

# def detect_by_huggingface_with_search_engine_support(model, human, machine, human_label, machine_label, human_similarity, machine_similarity, human_threshold, machine_threshold, is_hard = True, max_length=512):
#     if not is_hard:
#         return detect_by_huggingface_with_search_engine_support_by_soft(model, human, machine, human_label, machine_label, human_similarity, machine_similarity, human_threshold, machine_threshold, max_length) 

def detect_by_huggingface_with_search_engine_support(  
    model,   
    human,   
    machine,   
    human_label,   
    machine_label,   
    human_similarity,   
    machine_similarity,   
    human_threshold,   
    machine_threshold,   
    is_hard=True,   
    max_length=512  
):  
    """  
    Detects whether the input text is written by a human or a machine, with support for   
    search engine based similarity, using a HuggingFace model.  
  
    This function allows switching between a "hard" and "soft" detection method, depending   
    on the `is_hard` flag. If `is_hard` is False, the function delegates to another   
    detection function tailored for soft detection strategy.  
  
    Parameters:  
        model:   
            The HuggingFace model to use for detection.  
        human:   
            The input text believed to be written by a human.  
        machine:   
            The input text believed to be written by a machine (e.g., AI-generated text).  
        human_label:   
            The label associated with human-written text.  
        machine_label:   
            The label associated with machine-written text.  
        human_similarity:   
            The similarity score for the human text (from a search engine or similarity model).  
        machine_similarity:   
            The similarity score for the machine text (from a search engine or similarity model).  
        human_threshold:   
            The threshold above which text is considered human-written.  
        machine_threshold:   
            The threshold above which text is considered machine-written.  
        is_hard (bool, optional):   
            If True, apply hard detection logic (default: True).  
            If False, apply soft detection logic using an alternative function.  
        max_length (int, optional):  
            The maximum length of the input text to consider (default: 512).  
  
    Returns:  
        The result of the detection, possibly from a soft detection function.  
    """  
    # If not using a hard detection strategy, delegate to the soft detection function  
    if not is_hard:  
        return detect_by_huggingface_with_search_engine_support_by_soft(  
            model,   
            human,   
            machine,   
            human_label,   
            machine_label,   
            human_similarity,   
            machine_similarity,   
            human_threshold,   
            machine_threshold,   
            max_length  
        )  



def detect_by_huggingface_with_search_engine_support_by_soft(  
        model,  
        human,  
        machine,  
        human_label,  
        machine_label,  
        human_similarity,  
        machine_similarity,  
        human_threshold,  
        machine_threshold,  
        max_length=512  
    ):  
    """  
    Uses a Hugging Face text-classification model with soft label adjustment based on   
    search engine similarity to evaluate human and machine-generated texts.  
  
    Parameters:  
        model (str):                 The name or path of the pretrained Hugging Face model.  
        human (list of str):         A list of human-written texts.  
        machine (list of str):       A list of machine-generated texts.  
        human_label (str):           The label assigned by the model indicating 'human'.  
        machine_label (str):         The label assigned by the model indicating 'machine'.  
        human_similarity (list):     Similarity scores (e.g. cosine) for human texts.  
        machine_similarity (list):   Similarity scores for machine texts.  
        human_threshold (float):     Similarity threshold to consider a text as 'human'.  
        machine_threshold (float):   Similarity threshold to consider a text as 'machine'.  
        max_length (int):            Maximum sequence length for the model inputs.  
  
    Returns:  
        float: The accuracy if no texts or the ROC AUC computed by 'calculate_roc_by_RADAR_code'.  
    """  
  
    # Sentinel hard prediction for 'human'  
    HUMAN_prediction = -10000000  
  
    # Sentinel hard prediction for 'machine'  
    MACHINE_prediction = 10000000  
  
    # Print info about the model usage  
    print(f"model = {model}_with_search_engine_support\n")  
  
    # Setup HuggingFace classification pipeline  
    pipe = pipeline(  
        "text-classification",  
        model=model,  
        tokenizer=model,  
        max_length=max_length,  
        truncation=True,  
        device_map="auto"  
    )  
  
    label = []        # Store true labels  
    predict = []      # Store predicted probabilities or sentinel values  
    hard_predict = [] # Store hard predictions  
  
    # Run prediction for human texts  
    result = pipe(human)  
    for sub_result, human_sim in zip(result, human_similarity):  
        score = float(sub_result['score'])  # Model's confidence score  
  
        if human_sim >= human_threshold:  
            # Predict as 'human' with sentinel if similarity above human threshold  
            predict.append(HUMAN_prediction)  
            hard_predict.append(HUMAN)  
        elif human_sim >= machine_threshold:  
            # Predict as 'machine' with sentinel if similarity above machine threshold  
            predict.append(MACHINE_prediction)  
            hard_predict.append(MACHINE)  
        else:  
            # Use model's prediction depending on label  
            if sub_result['label'] == human_label:  
                hard_predict.append(HUMAN)  
                predict.append(1.0 - score)  
            else:  
                hard_predict.append(MACHINE)  
                predict.append(score)  
        # True label is 'human'  
        label.append(HUMAN)  
  
    # Run prediction for machine texts  
    result = pipe(machine)  
    for sub_result, machine_sim in zip(result, machine_similarity):  
        score = float(sub_result['score'])  # Model's confidence score  
  
        if machine_sim >= human_threshold:  
            # Predict as 'human' if similarity above human threshold  
            predict.append(HUMAN_prediction)  
            hard_predict.append(HUMAN)  
        elif machine_sim >= machine_threshold:  
            # Predict as 'machine' if similarity above machine threshold  
            predict.append(MACHINE_prediction)  
            hard_predict.append(MACHINE)  
        else:  
            # Use model's prediction depending on label  
            if sub_result['label'] == human_label:  
                hard_predict.append(HUMAN)  
                predict.append(1.0 - score)  
            else:  
                hard_predict.append(MACHINE)  
                predict.append(score)  
        # True label is 'machine'  
        label.append(MACHINE)  
  
    # If there are no texts, return accuracy; otherwise, compute ROC AUC  
    if len(human) == 0 or len(machine) == 0:  
        acc = accuracy_score(label, hard_predict)  
        print(f"acc = {acc}\n")  
        return acc  
    else:  
        roc_auc_by_RADAR = calculate_roc_by_RADAR_code(label, predict)  
        return roc_auc_by_RADAR  


def detect_by_huggingface(model, human, machine, human_label, machine_label, is_hard=True, max_length=512):  
    """  
    Detects the type of content (human-generated or machine-generated) using a HuggingFace model.  
      
    This function selects between a 'hard' and a 'soft' detection method based on the is_hard flag.  
    For 'hard' detection, this function assumes further logic will be implemented.  
    For 'soft' detection (is_hard=False), it delegates the detection task to detect_by_huggingface_by_soft().  
      
    Parameters:  
        model: The HuggingFace model used for detection.  
        human: The human-generated texts or samples.  
        machine: The machine-generated texts or samples.  
        human_label: The label corresponding to human-generated content.  
        machine_label: The label corresponding to machine-generated content.  
        is_hard (bool, optional): If True, use the 'hard' detection method. If False, use the 'soft' method. Default is True.  
        max_length (int, optional): The maximum sequence length for input. Default is 512.  
  
    Returns:  
        The detection result, either from 'hard' or 'soft' detection logic.  
    """  
    # If not using the 'hard' detection method, use the 'soft' method  
    if not is_hard:  
        return detect_by_huggingface_by_soft(  
            model, human, machine, human_label, machine_label, max_length  
        )  
    # For the 'hard' detection method, further logic can be implemented here  

def detect_by_huggingface_by_soft(  
    model,  
    human,  
    machine,  
    human_label,  
    machine_label,  
    max_length=512  
):  
    """  
    Evaluates a HuggingFace text classification model to distinguish between human and machine-generated texts.  
  
    Parameters:  
        model (str): The HuggingFace model identifier or path.  
        human (list of str): List of human-generated texts for classification.  
        machine (list of str): List of machine-generated texts for classification.  
        human_label (str): The expected label for human-generated text.  
        machine_label (str): The expected label for machine-generated text.  
        max_length (int, optional): Maximum length of text sequences for the model input. Default is 512.  
  
    Returns:  
        float: ROC AUC score (or accuracy if only one of human or machine is provided).  
    """  
  
    # Print out the used model  
    print(f"model = {model}\n")  
  
    # Create a HuggingFace pipeline for text classification using the provided model  
    pipe = pipeline(  
        "text-classification",  
        model=model,  
        tokenizer=model,  
        max_length=max_length,  
        truncation=True,  
        device_map="auto"  
    )  
  
    # Initialize lists to store labels, predictions, and scores  
    label = []  
    predict = []  
    human_score = []  
    hard_predict = []  
  
    # Run the model on human-generated texts  
    result = pipe(human)  
    for sub_result in result:  
        # Extract the confidence score of the prediction  
        score = float(sub_result['score'])  
        # Check if the predicted label matches the human label  
        if sub_result['label'] == human_label:  
            hard_predict.append(HUMAN)                # Add HUMAN to hard predictions  
            predict.append(1.0 - score)               # Append inverted probability for human label  
            human_score.append(score)                 # Record score for human  
        else:  
            hard_predict.append(MACHINE)              # Add MACHINE to hard predictions  
            predict.append(score)                     # Append probability for machine label  
            human_score.append(1.0 - score)           # Record inverted score for human  
        label.append(HUMAN)                           # Append HUMAN to ground truth labels  
  
    # Run the model on machine-generated texts  
    result = pipe(machine)  
    machine_score = []  
    for sub_result in result:  
        score = float(sub_result['score'])  
        if sub_result['label'] == human_label:  
            hard_predict.append(HUMAN)  
            predict.append(1.0 - score)  
            machine_score.append(1.0 - score)  
        else:  
            hard_predict.append(MACHINE)  
            predict.append(score)  
            machine_score.append(score)  
        label.append(MACHINE)  
  
    # If either human or machine input is empty, calculate and return the accuracy  
    if len(human) == 0 or len(machine) == 0:  
        acc = accuracy_score(label, hard_predict)  
        print(f"acc = {acc}\n")  
        return acc  
      
    # Otherwise, calculate and return the ROC AUC score using the provided calculate_roc_by_RADAR_code  
    else:  
        roc_auc_by_RADAR = calculate_roc_by_RADAR_code(label, predict)  
        print(f"roc_auc_by_RADAR = {roc_auc_by_RADAR}\n")  
        return roc_auc_by_RADAR  


def calculate_roc_by_RADAR_code(label, pred):  
    """  
    Calculates the ROC metric (Receiver Operating Characteristic) for inputs labeled as HUMAN and non-HUMAN.  
      
    This function separates predicted values into human and machine lists based on their labels,  
    then calculates ROC-related metrics using the 'get_roc_metrics' function.  
      
    Parameters:  
        label (list): List of labels corresponding to each prediction. HUMAN and others.  
        pred (list): List of prediction scores corresponding to each label.  
      
    Returns:  
        The last value from the ROC metrics computed by get_roc_metrics.  
    """  
    # Initialize a list to hold prediction scores with HUMAN labels  
    human = []  
    # Initialize a list to hold prediction scores with non-HUMAN labels (machines)  
    machine = []  
    # Iterate over the labels and prediction scores in parallel  
    for la, pr in zip(label, pred):  
        # If the label indicates human, add the prediction to the human list  
        if la == HUMAN:  
            human.append(pr)  
        # Otherwise, add the prediction to the machine list  
        else:  
            machine.append(pr)  
    # Compute ROC metrics using the separated lists  
    result = get_roc_metrics(human, machine)  
    # Return the desired metric (the last element of the result)  
    return result[-1]  

def get_roc_metrics(human_preds, ai_preds):  
    """  
    Computes ROC curve metrics (FPR, TPR, AUC) for two prediction lists.  
  
    Parameters:  
        human_preds (list of float): Prediction scores for the 'human' class (negative samples).  
        ai_preds (list of float): Prediction scores for the 'AI' class (positive samples).  
  
    Returns:  
        tuple:   
            - fpr (list of float): False Positive Rate values for the ROC curve.  
            - tpr (list of float): True Positive Rate values for the ROC curve.  
            - roc_auc (float): Area Under the ROC Curve.  
  
    This function combines the provided prediction scores into a single list,  
    constructs the true label array, computes the ROC curve,  
    and finally calculates the area under the curve.  
    """  
  
    # Create ground-truth labels: 0 for humans, 1 for AI  
    labels = [0] * len(human_preds) + [1] * len(ai_preds)  
  
    # Concatenate both predictions (order: humans first, then AI)  
    scores = human_preds + ai_preds  
  
    # Calculate False Positive Rate (fpr), True Positive Rate (tpr), and thresholds  
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)  
  
    # Compute the Area Under the ROC Curve (AUC)  
    roc_auc = auc(fpr, tpr)  
  
    # Convert fpr and tpr to lists, ensure roc_auc is a float, and return  
    return fpr.tolist(), tpr.tolist(), float(roc_auc)  


def calculate_roc_by_radar_code(label, pred):  
    """  
    Calculates the ROC (Receiver Operating Characteristic) metric based on label and prediction lists  
    following the RADAR code classification.  
  
    Parameters:  
        label (list): List of ground truth labels, each label is expected to be either HUMAN or another type (e.g., MACHINE).  
        pred (list): List of model predictions corresponding to the label list.  
  
    Returns:  
        float: The last element of the result returned by get_roc_metrics,   
               which likely represents the ROC AUC score or a similar metric.  
    """  
  
    # Initialize list to collect predictions where the label is HUMAN  
    human = []  
    # Initialize list to collect predictions where the label is not HUMAN (assumed MACHINE)  
    machine = []  
  
    # Iterate over label and prediction pairs  
    for la, pr in zip(label, pred):  
        # If label is HUMAN, append prediction to human list  
        if la == HUMAN:  
            human.append(pr)  
        # Otherwise (assumed MACHINE), append prediction to machine list  
        else:  
            machine.append(pr)  
      
    # Call the get_roc_metrics function with the two lists to obtain ROC-related metrics  
    result = get_roc_metrics(human, machine)  
    # Return the last element of the result, possibly the ROC AUC score  
    return result[-1]  


import os  
  
def create_folder_for_file(file_name):  
    """  
    Creates a folder (directory) for the given file path if it does not already exist.  
  
    Parameters:  
        file_name (str): The path to the file for which the containing folder should be created.  
  
    Returns:  
        None  
    """  
    # Extract the directory path from the file path  
    path_name = os.path.dirname(file_name)  
    # Print the directory path to the console for debugging  
    print(f"path_name = {path_name}")  
    # Check if the directory exists  
    if not os.path.exists(path_name):  
        # Create the directory (including intermediate folders if necessary)  
        os.makedirs(path_name)  


def search_engine_predict_with_unparallel_data(  
    input_file,  
    output_file,  
    is_check_bbc=True,  
    number_sample=-1,  
    is_wikipedia=True  
):  
    """  
    Processes a dataset for search engine prediction tasks with unparallel data.  
  
    Parameters:  
        input_file (str): Path to the input CSV file. The CSV must have two columns: 'text' and 'label'.  
        output_file (str): Path to the output CSV file where the results will be saved.  
        is_check_bbc (bool, optional): Flag to indicate whether to include BBC sources during the search. Defaults to True.  
        number_sample (int, optional): The number of samples to process. If -1, process all samples. Defaults to -1.  
        is_wikipedia (bool, optional): Flag to indicate whether to include Wikipedia sources. Defaults to True.  
  
    The output CSV file will have the following columns:  
        text: The input text.  
        label: HUMAN/MACHINE ground truth label.  
        best_url: The URL with the highest similarity.  
        best_avg_similarity: The similarity score for the best_url.  
        best_data: The content of the best-matched data.  
  
    Returns:  
        None  
    """  
  
    # Read input data from CSV file  
    data = read_csv_data(input_file)  
  
    # If number_sample is -1, process all data  
    if number_sample == -1:  
        number_sample = len(data)  
  
    # Ensure the output file's folder exists  
    create_folder_for_file(output_file)  
  
    # If output file does not exist, create it and write the header  
    if not os.path.exists(output_file):  
        header = ["text", "label", "best_url", "best_avg_similarity", "best_data"]  
        write_to_csv(output_file, header)  
  
    # Read any already-processed output data for resuming capability  
    output_data = read_csv_data(output_file)  
    number_of_process_samples = len(output_data)  
  
    # Select unprocessed samples to work on  
    data = data[number_of_process_samples:number_sample]  
  
    # Process each data item  
    for item in data:  
        input_text = item[0]  # Extract the input text  
        label = item[1]       # Extract the corresponding label  
  
        # Handle API error cases in the input text  
        if API_ERROR in input_text:  
            best_url = ""              # No result for API error  
            best_avg_similarity = -1   # Indicate similarity not available  
            best_data = ""             # No data available  
        else:  
            # Find the best similarity using a relative search method  
            best_url, best_avg_similarity, best_data = find_best_similarity_by_relative_search(  
                input_text,  
                is_check_bbc,  
                is_wikipedia  
            )  
  
        # Construct the row with processed information  
        row = [input_text, label, best_url, best_avg_similarity, best_data]  
  
        # Write result row to the output CSV file  
        write_to_csv(output_file, row=row)  

def evaluate_baseline_with_search_engine_support(  
    samples,   
    human_threshold,   
    machine_threshold,   
    is_hard=True  
):  
    """  
    Evaluates the performance of baseline models (with and without search engine support)   
    on a dataset of labeled text samples.  
  
    Parameters:  
        samples (list): Each element is a tuple of (text, label, best_avg_similarity).  
        human_threshold (float): Threshold for human-written text classification.  
        machine_threshold (float): Threshold for machine-generated text classification.  
        is_hard (bool, optional): If True, use strict/hard decision in classification. Default is True.  
  
    Returns:  
        None: Results are written to OUTPUT_FILE as a string.  
    """  
  
    # Lists to store human and machine texts separately  
    human = []  
    machine = []  
    # Lists to store similarity scores for human and machine texts  
    human_similarity = []  
    machine_similarity = []  
  
    # # Constants for label comparison  
    # HUMAN = 1  
    # MACHINE = 0  
  
    # Loop over all samples in the dataset and distribute them based on their label  
    for sample in samples:  
        text = sample[0]  # The text content  
        label = sample[1]  # Label to indicate human or machine-generated  
        best_avg_similarity = sample[2]  # Precomputed average similarity score  
  
        if label == HUMAN:  
            human.append(text)  
            human_similarity.append(best_avg_similarity)  
        else:  
            machine.append(text)  
            machine_similarity.append(best_avg_similarity)  
  
    # Initialize a dictionary to store classification results from different models  
    result = dict()  
  
    # Evaluate using the "yaful/MAGE" model  
    model = "yaful/MAGE"  
    human_label = 1  
    machine_label = 0  
    # Detect without search engine support  
    result[model] = detect_by_huggingface(  
        model, human, machine, human_label, machine_label, is_hard  
    )  
    # Detect with search engine support and similarity thresholds  
    result[model + "_search_engine_support"] = detect_by_huggingface_with_search_engine_support(  
        model, human, machine, human_label, machine_label,  
        human_similarity, machine_similarity,   
        human_threshold, machine_threshold,   
        is_hard  
    )  
 
    # Evaluate using the "TrustSafeAI/RADAR-Vicuna-7B" model  
    model = "TrustSafeAI/RADAR-Vicuna-7B"
    human_label = "LABEL_1"
    machine_label = "LABEL_0"

    # Detect without search engine support  
    result[model] = detect_by_huggingface(  
        model, human, machine, human_label, machine_label, is_hard  
    )  
    # Detect with search engine support and similarity thresholds  
    result[model + "_search_engine_support"] = detect_by_huggingface_with_search_engine_support(  
        model, human, machine, human_label, machine_label,  
        human_similarity, machine_similarity,   
        human_threshold, machine_threshold,   
        is_hard  
    ) 

    # Write the results to the output file  
    write_to_file(OUTPUT_FILE, str(result))  

def longest_increasing_subsequence(arr):  
    """  
    Finds the indices of the Longest Increasing Subsequence (LIS) in a given list.  
  
    Parameters:  
        arr (List[int]): The input list of integers.  
  
    Returns:  
        List[int]: The list of indices that form the LIS in the input list.  
    """  
    # Return empty list if input is empty  
    if not arr:  
        return []  
  
    # Store the length of the input array  
    n = len(arr)  
  
    # Initialize the dp array with all elements set to 1 (each element is a subsequence)  
    dp = [1] * n  
  
    # Initialize the trace array for reconstructing the LIS  
    trace = [-1] * n  
  
    # Compute dp values and trace the predecessors  
    for i in range(n):  
        for j in range(i):  
            # If a valid increasing pair is found, update dp and trace  
            if arr[j] < arr[i] and dp[j] + 1 > dp[i]:  
                dp[i] = dp[j] + 1  
                trace[i] = j  
  
    # Find the length of the longest increasing subsequence  
    max_length = max(dp)  
  
    # Find the index where LIS ends  
    index = dp.index(max_length)  
  
    # Reconstruct the sequence of indices for the LIS  
    lis_indices = []  
    while index != -1:  
        lis_indices.append(index)  
        index = trace[index]  
  
    # Return the indices in the correct order  
    return lis_indices[::-1]  

def check_matching_index(matching_index, error_matching_ratio, max_matching_len):  
    """  
    Checks if a list of matching indexes is ordered and the gaps between   
    consecutive indexes are within a specified limit, allowing for a certain   
    error ratio.  
  
    Parameters:  
        matching_index (list of int): The list of indexes to check.  
        error_matching_ratio (float): The maximum allowed ratio of errors.  
        max_matching_len (int): The maximum allowed difference between consecutive indexes.  
  
    Returns:  
        bool: True if the proportion of correctly ordered and close-enough indexes   
              meets the threshold, False otherwise.  
    """  
    correct = 0  # Initialize counter for correct matches  
  
    for index in range(len(matching_index)):  
        if index == 0:  
            # The first element is always considered correct  
            correct += 1  
        else:  
            current = matching_index[index]         # Current element  
            previous = matching_index[index - 1]    # Previous element  
            # Check if current index is >= previous and difference is within limit  
            if current >= previous and current - previous <= max_matching_len:  
                correct += 1   # Count as correct if condition satisfied  
  
    if len(matching_index) <= 0:  
        # If input list is empty, return False  
        return False  
  
    correct_ratio = correct / len(matching_index)  # Calculate ratio of correct matches  
  
    # Return True if the error ratio is within allowable bounds  
    if 1 - correct_ratio <= error_matching_ratio:  
        return True  
  
    return False  # Otherwise, return False  

def estimate_sample_similarity_by_SearchLLM(  
    text,  
    best_url,  
    filtered_threshold,  
    remain_ratio_threshold,  
    verbose=False,  
    model_name=GPT_4O_MINI,  
    provider=OPENAI  
):  
    """  
    Estimates the similarity between a sample text and a reference found at best_url  
    using search and language modeling. If the similarity is too low, attempts   
    to regenerate the text and measure similarity again.  
  
    Parameters:  
        text (str): The sample text to be compared.  
        best_url (str): The URL containing the reference text for comparison.  
        filtered_threshold (float): Minimum similarity score to consider a match as valid.  
        remain_ratio_threshold (float): Minimum ratio of matched sentences to accept sim score.  
        verbose (bool, optional): If True, prints debug information. Default is False.  
        model_name (str, optional): The name of the model used for regeneration, default to GPT_4O_MINI.  
        provider (str, optional): The provider for the language model, default to OPENAI.  
  
    Returns:  
        tuple: (regenerated_text (str or None), original_sim (float), regeneration_sim (float))  
        - regenerated_text: Regenerated text if similarity is too low; otherwise, None.  
        - original_sim: Similarity score before regeneration.  
        - regeneration_sim: Similarity score after regeneration (only if performed).  
    """  
  
    # Set thresholds  
    error_matching_ratio = 0.1  
    max_matching_len = 3  
  
    regenerated_text = None      # Variable to store regenerated text if needed  
    original_sim = 0             # Original similarity score  
    regeneration_sim = 0         # Similarity score after regeneration, if performed  
    average = 0                  # Average similarity of filtered matches  
    remain_ratio = 0             # Ratio of content that remains after filtering  
    num_sentence = 0             # Total number of compared sentence pairs  
  
    # Check if a valid URL is provided for comparison  
    if best_url is not None and best_url != "":  
        # Calculate similarities and matching data with the reference content from the url  
        avg_similarity, matching_data, matching_index = measure_similarity_with_url_return_matching_index(text, best_url)  
  
        # Validate matching indices with error thresholds  
        if not check_matching_index(matching_index, error_matching_ratio, max_matching_len):  
            if verbose:  
                print("check_matching_index is false")  
            # Early exit if not valid  
            return regenerated_text, original_sim, regeneration_sim  
  
        if verbose:  
            print(f"matching_index = {matching_index}")  
  
        filtered_sim = []           # List to store similarity scores after filtering  
        num_sentence = len(matching_data)   # Number of sentence pairs  
  
        filtered_match = []         # List of matches with similarity above threshold  
        filter_matching_index = []  # Matching indices above threshold  
        filter_original_index = []  # Indices of the original sentences that matched  
  
        # Iterate through matched sentence pairs and their similarity scores  
        for ori_index, (match, match_index) in enumerate(zip(matching_data, matching_index)):  
            sentence_1 = match[0]   # Sentence from sample text  
            sentence_2 = match[1]   # Sentence from url reference text  
            sim = float(match[2])   # Similarity score  
  
            # Keep only matches with similarity above the given threshold  
            if sim >= filtered_threshold:  
                filtered_match.append(match)  
                filter_matching_index.append(match_index)  
                filter_original_index.append(ori_index)  
                if verbose:  
                    print(f"sentence_1 = {sentence_1}")  
                    print(f"sentence_2 = {sentence_2}")  
                    print(f"sim = {sim}")  
  
        # Select longest increasing subsequence to keep matched sentence order  
        selected_index = longest_increasing_subsequence(filter_matching_index)  
        if verbose:  
            print(f"filter_matching_index = {filter_matching_index}")  
            print(f"selected_index = {selected_index}")  
  
        # Gather similarity scores from filtered, ordered matches  
        for index in selected_index:  
            sim = float(filtered_match[index][2])  
            filtered_sim.append(sim)  
  
        # Calculate the ratio of matched sentences to original  
        if num_sentence > 0:  
            remain_ratio = len(filtered_sim) / num_sentence  
  
        # If any filtered similarities, calculate their average  
        if len(filtered_sim) > 0:  
            average = np.average(filtered_sim)  
  
    # If the ratio of matched sentences passes the threshold:  
    if remain_ratio >= remain_ratio_threshold:  
        original_sim = average  
        # If high enough similarity, no regeneration needed  
        if average >= 0.97:  
            return regenerated_text, original_sim, regeneration_sim  
        else:  
            input_index = []          # Index of input sentences to regenerate  
            source_from_url_index = []# Indices of reference sentences to check against  
            for index in selected_index:  
                input_index.append(filter_original_index[index])  
                source_from_url_index.append(filter_matching_index[index])  
  
            # Attempt regeneration using the given model and provider  
            regeneration_sim, regenerated_text = calculate_sim_for_regeneration(  
                text,  
                best_url,  
                matching_index,  
                input_index,  
                source_from_url_index,  
                model_name,  
                verbose,  
                provider  
            )  
  
            if verbose:  
                print(f"filtered_sim = {filtered_sim}")  
                print(f"original_sim = {original_sim}")  
                print(f"regeneration_sim = {regeneration_sim}")  
  
            # If regeneration fails, set similarity to 0  
            if regeneration_sim is None:  
                regeneration_sim = 0  
                return regenerated_text, original_sim, regeneration_sim  
            else:  
                return regenerated_text, original_sim, regeneration_sim  
  
    # If not enough content remains or no similarity can be calculated  
    return regenerated_text, original_sim, regeneration_sim  

def estimate_similarity_by_SearchLLM(  
    search_engine_csv,  
    output_file,  
    filtered_threshold,  
    human_threshold,  
    machine_threshold,  
    remain_ratio_threshold,  
    number_sample=-1,  
    verbose=False,  
    model_name=GPT_4O_MINI,  
    provider=OPENAI  
):  
    """  
    Estimates text similarity using a search engine and LLM (Language Learning Model) regeneration.  
  
    This function processes a CSV file containing search engine results, and for each sample,   
    estimates the similarity between the original text and text regenerated using a LLM,   
    particularly for results from Wikipedia. The results are stored or appended to an output CSV file,   
    enabling checkpointing and progress continuation if interrupted.  
  
    Parameters:  
        search_engine_csv (str): Path to the input CSV file containing search engine data.  
        output_file (str): Path to the output CSV file in which to store results.  
        filtered_threshold (float): The similarity threshold for filtering candidates.  
        human_threshold (float): The similarity threshold used for human evaluation.  
        machine_threshold (float): The similarity threshold used for machine evaluation.  
        remain_ratio_threshold (float): The threshold for filtering candidates based on remain ratio.  
        number_sample (int, optional): Number of samples to process; set to -1 to process all.  
        verbose (bool, optional): Whether to print verbose progress information.  
        model_name (str, optional): Name of the LLM model to use for regeneration/comparison.  
        provider (str, optional): The service provider for the LLM model.  
  
    Returns:  
        None  
    """  
  
    # Read the input data from the search engine CSV file  
    data = read_csv_data(search_engine_csv)  
  
    # If number_sample is -1, process all samples; else, use the specified number  
    if number_sample == -1:  
        number_sample = len(data)  
  
    # Ensure the output folder exists, create if it doesn't  
    create_folder_for_file(output_file)  
  
    # If the output file doesn't already exist, create it with the appropriate header  
    if not os.path.exists(output_file):  
        header = [  
            "text",  
            "label",  
            "best_url",  
            "original_sim",  
            "regeneration_sim",  
            "regenerated_text",  
            "best_avg_similarity",  
            "best_data",  
        ]  
        write_to_csv(output_file, header)  
  
    # Read the existing output data for progress tracking (for checkpointing/resuming)  
    output_data = read_csv_data(output_file)  
    number_of_process_samples = len(output_data)  
  
    # Select the data not yet processed, up to number_sample samples  
    data = data[number_of_process_samples:number_sample]  
  
    # Process each item in the selected data  
    for item in data:  
        text = item[0]            # Original text to estimate similarity with  
        label = item[1]           # Corresponding label for the text  
        best_url = item[2]        # The best-matching URL from search engine results  
        best_avg_similarity = item[3]  # Precomputed best average similarity  
        best_data = item[4]       # Associated data for the best match  
  
        regenerated_text = None   # Placeholder for text regenerated by LLM  
        original_sim = 0          # Placeholder for original similarity value  
        regeneration_sim = 0      # Placeholder for similarity after regeneration  
  
        # Only process Wikipedia URLs for regeneration/similarity estimation  
        if "wikipedia.org" in best_url:  
            regenerated_text, original_sim, regeneration_sim = estimate_sample_similarity_by_SearchLLM(  
                text,  
                best_url,  
                filtered_threshold,  
                remain_ratio_threshold,  
                verbose,  
                model_name,  
                provider  
            )  
  
        # Combine results into a row for CSV writing  
        row = [  
            text,  
            label,  
            best_url,  
            original_sim,  
            regeneration_sim,  
            regenerated_text,  
            best_avg_similarity,  
            best_data,  
        ]  
  
        # Write the result row to the output CSV file  
        write_to_csv(output_file, row)  

def baseline_with_search_engine_support_filter_from_pre_estimate(  
    search_engine_csv, human_threshold, machine_threshold, min_diff_regeneration  
):  
    """  
    Processes search engine results, filters and evaluates samples based on provided similarity thresholds.  
  
    This function reads a CSV of search engine outputs.  
    It determines which samples pass given similarity thresholds for 'human' and 'machine' acceptability.  
    It appends qualifying samples to a list, which is then evaluated.  
  
    Parameters:  
        search_engine_csv (str): Path to the CSV file containing search engine results.  
        human_threshold (float): Similarity threshold to accept as 'human-quality' output.  
        machine_threshold (float): Similarity threshold to accept as 'machine-quality' output.  
        min_diff_regeneration (float): Minimum required difference between regeneration and original similarity to pass as 'machine-quality'.  
  
    Returns:  
        None  
    """  
    # Log the start of the baseline with search engine support evaluation  
    write_to_file(OUTPUT_FILE, "\n\nBASELINE WITH SEARCH ENGINE SUPPORT\n")  
  
    # Log the name of the CSV file being processed  
    write_to_file(OUTPUT_FILE, f"csv_file = {search_engine_csv}\n")  
  
    # Read in CSV data using a helper function  
    data = read_csv_data(search_engine_csv)  
  
    # Prepare a list to hold filtered samples: (text, label, best_avg_similarity)  
    samples = []  
  
    # Iterate through each row in the CSV data  
    for item in data:  
        text = item[0]  # The text sample  
        label = item[1]  # The label/classification, e.g. human or machine  
        original_sim = float(item[3])  # Similarity score before regeneration  
        regeneration_sim = float(item[4])  # Similarity score after regeneration  
  
        # Determine final similarity based on thresholds  
        if original_sim >= human_threshold:  
            # If original similarity passes human threshold, accept it  
            final_similarity = original_sim  
        elif original_sim >= machine_threshold:  
            # If only machine threshold is passed, check regeneration improvement  
            if regeneration_sim - original_sim >= min_diff_regeneration:  
                # If improvement is sufficient, accept; else reject  
                final_similarity = original_sim  
            else:  
                final_similarity = 0  
        else:  
            # If neither threshold is met, reject  
            final_similarity = 0  
  
        # Create a triple (text, label, final_similarity)  
        triple = (text, label, final_similarity)  
  
        # Add the filtered sample to the list  
        samples.append(triple)  
  
    # Flag to indicate evaluation difficulty, set as False for baseline evaluation  
    is_hard = False  
  
    # Log the difficulty flag  
    write_to_file(OUTPUT_FILE, f"is_hard = {is_hard}\n")  
  
    # Perform baseline evaluation with search engine support  
    evaluate_baseline_with_search_engine_support(  
        samples, human_threshold, machine_threshold, is_hard  
    )  

if __name__ == "__main__":  
    """  
    The main entry point of the script.  
    Executes three main steps:  
    1. Generate search engine predictions for input data.  
    2. Estimate similarity using a SearchLLM model.  
    3. Apply baseline filtering based on these estimates.  
    """  
  
    # ===== Step 1: Generate search engine predictions =====  
  
    # Specify the path to the input data file  
    input_file = "5_data/data.csv"  
    # Specify the path to save the search engine results  
    output_file = "5_data/search_engine.csv"  
    # Number of samples to collect per entry  
    num_sample = -1  
    # Whether to check for BBC results  
    is_check_bbc = False  
    # Whether to enable Wikipedia as a source  
    is_wikipedia = True  
  
    # Run search engine prediction with the provided configuration  
    search_engine_predict_with_unparallel_data(  
        input_file, output_file, is_check_bbc, num_sample, is_wikipedia  
    )  
  
    # ===== Step 2: Estimate similarity using SearchLLM =====  
  
    # Path to search engine predictions file  
    search_engine_csv = "5_data/search_engine.csv"  
    # Path to output similarity estimates  
    output_file = "5_data/searchLLM.csv"  
    # Threshold for filtering similarity  
    human_threshold = 0.97  
    machine_threshold = 0.8  
    filtered_threshold = 0.8  
    # Minimum ratio of samples to retain after filtering.  
    remain_ratio_threshold = 0.5  
    # Number of samples to process, -1 means all  
    number_sample = -1  
    # Name of the model to use for similarity estimation  
    model_name = GPT_4O_MINI
    # Provider of the model  
    provider = OPENAI
    # Toggle for verbose output  
    verbose = True  
  
    # Run similarity estimation using a language model  
    estimate_similarity_by_SearchLLM(  
        search_engine_csv, output_file, filtered_threshold,  
        human_threshold, machine_threshold, remain_ratio_threshold,  
        number_sample, verbose, model_name, provider  
    )  
  
    # ===== Step 3: Baseline filter using pre-estimate results =====  
  
    # Path to similarity estimation results for baseline filtering  
    search_engine_csv = "5_data/searchLLM.csv"  
    # Similarity threshold for filtering as human-level  
    human_threshold = 0.97  
    # Similarity threshold for filtering as machine-level  
    machine_threshold = 0.9  
    # Minimum difference required for regeneration  
    min_diff_regeneration = 0  
  
    # Apply baseline filter using the estimated similarity results  
    baseline_with_search_engine_support_filter_from_pre_estimate(  
        search_engine_csv, human_threshold, machine_threshold, min_diff_regeneration  
    )  