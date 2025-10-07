import torch
import ast 
OUTPUT_FILE = "20_result/result_ablation.txt"
from _9_constant import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc,roc_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import ttest_rel
from _wikipedia_util import search_by_NER_and_first_k_sentence, calculate_similarity_with_regeneration
from datetime import datetime
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import accuracy_score



MIN_RATIO = 0.5
MIN_SIM_PARAPHRASE = 0.86
MIN_SIM_MACHINE = 0.86
MIN_SIM_HUMAN = 0.99
MIN_REGENERATE_DIFF = 0.00923
MIN_NUM_WORD = 30


TILDE_HUMAN = "~human"

FOUND = "found"
NOT_FOUND = "not_found"
import numpy as np
from sklearn.metrics import accuracy_score
import os
import pandas as pd 
import csv
from transformers import pipeline

from datetime import datetime


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAGE_MODEL = "yaful/MAGE"  
RADAR_MODEL = "TrustSafeAI/RADAR-Vicuna-7B"
MODELS = {MAGE_MODEL: None, RADAR_MODEL: None}

import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

HUMAN_LABEL = 0
MACHINE_LABEL = 1

HUMAN = "human"
MACHINE = "machine"
UNKNOWN = "unknown"
TRAIN = "train"
VALIDAION = "validation"
TEST = "test"



def write_to_file(filename, text):  
    """Print text and append it to a file."""  
    print(text)  
    with open(filename, "a+", encoding="utf-8") as f:  
        f.write(text)


def read_csv_data(input_file):  
    """Read CSV file and return its values as a numpy array of strings."""  
    data = pd.read_csv(input_file, dtype='string', keep_default_na=False, sep=',').values  
    return data  
  
  
def write_to_csv(filename, row):  
    """Append a row to a CSV file."""  
    with open(filename, 'a+', encoding='UTF8', newline='') as f:  
        writer = csv.writer(f)  
        writer.writerow(row)  
  
  
def create_folder_for_file(file_name):  
    """Create the folder for the given file if it does not exist."""  
    path_name = os.path.dirname(file_name)  
    print(f"path_name = {path_name}")  
    if not os.path.exists(path_name):  
        os.makedirs(path_name)

def convert_to_object(data):  
    """  
    Convert the input string to a Python object using ast.literal_eval.  
  
    Example:  
        input: '["a", "b", "c"]' (str)  
        output: ["a", "b", "c"] (list)  
    """  
    return ast.literal_eval(data)  
  
  
def calculate_similarity_by_SeachLLM_greedy_mapping_with_regeneration(  
    input_file,  
    output_file,  
    model_name,  
    number_of_first_sentence,  
    minimum_similarity_for_mapping,  
    minimum_similarity_for_regeneration,  
    min_rate,  
    number_sample=-1,  
    custom_prompts=None  
):  
    """  
    Calculate similarity using SearchLLM greedy mapping with regeneration.  
  
    Args:  
        input_file: Input CSV file containing 'text' and 'label'.  
        output_file: Output CSV file for results.  
        model_name: Model name for similarity calculation.  
        number_of_first_sentence: Number of first sentences to consider.  
        minimum_similarity_for_mapping: Minimum similarity threshold for mapping.  
        minimum_similarity_for_regeneration: Minimum similarity for regeneration.  
        min_rate: Minimum rate for regeneration filter.  
        number_sample: Number of samples to process (-1 for all).  
        custom_prompts: List of custom prompts for regeneration.  
  
    Output file columns:  
        text, label, similarity, url, all_similarity, mapping, regeneration_similarity, regeneration_mapping  
    """  
    start = datetime.now()  
    write_to_file(OUTPUT_FILE, "******************************\n")  
    write_to_file(OUTPUT_FILE, "calculate_similarity_by_SeachLLM_greedy_mapping_with_regeneration\n")  
    write_to_file(OUTPUT_FILE, f"input_file - {input_file}\n")  
    write_to_file(OUTPUT_FILE, f"output_file - {output_file}\n")  
    data = read_csv_data(input_file)  
    create_folder_for_file(output_file)  
    if not os.path.exists(output_file):  
        header = [  
            "text",  
            "label",  
            "similarity",  
            "url",  
            "all_similarity",  
            "mapping",  
            "regeneration_similarity",  
            "regeneration_mapping"  
        ]  
        write_to_csv(output_file, header)  
    processed_data = read_csv_data(output_file)  
    number_of_processed_item = len(processed_data)  
    if number_sample == -1:  
        number_sample = len(data)  
    data = data[number_of_processed_item:number_sample]  
    write_to_file(OUTPUT_FILE, f"Number of sample - {len(data)}\n")  
    for index, item in enumerate(data):  
        text = item[0]  
        label = item[1]  
        all_similarity = []  
        mapping = []  
        similarity, (url, additional_info) = search_by_NER_and_first_k_sentence(  
            text,  
            number_of_first_sentence,  
            minimum_similarity_for_mapping  
        )  
        if additional_info is not None:  
            all_similarity = additional_info[0]  
            mapping = additional_info[1]  
        regeneration_similarity = []  
        regeneration_mapping = []  
        if mapping is not None and len(mapping) > 0:  
            org_filter_sim = []  
            for sim in all_similarity:  
                if sim > minimum_similarity_for_regeneration:  
                    org_filter_sim.append(sim)  
            if len(org_filter_sim) / len(all_similarity) >= min_rate:  
                if custom_prompts is None:  
                    ave_generation_sim, (regeneration_similarity, regeneration_mapping) = calculate_similarity_with_regeneration(  
                        text, mapping, model_name  
                    )  
                else:  
                    best_average = -2  
                    for custom_prompt in custom_prompts:  
                        current_ave_generation_sim, (current_regeneration_similarity, current_regeneration_mapping) = calculate_similarity_with_regeneration(  
                            text, mapping, model_name, custom_prompt  
                        )  
                        if (  
                            current_ave_generation_sim is not None and  
                            current_ave_generation_sim > best_average  
                        ):  
                            best_average = current_ave_generation_sim  
                            regeneration_similarity = current_regeneration_similarity  
                            regeneration_mapping = current_regeneration_mapping  
        row = [  
            text,  
            label,  
            similarity,  
            url,  
            all_similarity,  
            mapping,  
            regeneration_similarity,  
            regeneration_mapping  
        ]  
        write_to_csv(output_file, row)  
        print(f"processed {index + 1}/{len(data)}")  
    end = datetime.now()  
    write_to_file(OUTPUT_FILE, f"Running time: {end - start}")  

def measure_difference_similarity(  
    original_sim,  
    original_map,  
    regeneration_sim,  
    regeneration_map,  
    min_paraphrase_diff,  
    verbose=False  
):  
    """  
    Measure the difference in similarity between original and regenerated sentences.  
  
    Args:  
        original_sim: Similarity scores between input and Wikipedia text.  
        original_map: Mapping between input and Wikipedia sentences.  
        regeneration_sim: Similarity scores between input and regeneration.  
        regeneration_map: Mapping between input and regeneration sentences.  
        min_paraphrase_diff: Minimum paraphrase difference threshold.  
        verbose: Whether to print debug information.  
  
    Returns:  
        Tuple containing the average difference and the lists of similarities, or (None, (None, None)) if conditions are not met.  
    """  
    ori_sim_by_sent = []  
    for org_sim, org_map in zip(original_sim, original_map):  
        num_sent = len(org_map[0])  
        for _ in range(num_sent):  
            ori_sim_by_sent.append(org_sim)  
  
    re_sim_by_sent = []  
    for re_sim, re_map in zip(regeneration_sim, regeneration_map):  
        num_re_sent = len(re_map[0])  
        for _ in range(num_re_sent):  
            re_sim_by_sent.append(re_sim)  
  
    if verbose:  
        print(f"ori_sim_by_sent = {ori_sim_by_sent}")  
        print(f"re_sim_by_sent = {re_sim_by_sent}")  
  
    if len(ori_sim_by_sent) != len(re_sim_by_sent):  
        return None, (None, None)  
  
    filter_index = []  
    for index, (org_sim_sent, re_sim_sent) in enumerate(zip(ori_sim_by_sent, re_sim_by_sent)):  
        if org_sim_sent > min_paraphrase_diff and re_sim_sent > min_paraphrase_diff:  
            filter_index.append(index)  
  
    if len(filter_index) == 0:  
        return None, (None, None)  
  
    diff = []  
    for index in filter_index:  
        diff.append(re_sim_by_sent[index] - ori_sim_by_sent[index])  
  
    if verbose:  
        print(f"diff = {diff}")  
  
    return np.average(diff), (ori_sim_by_sent, re_sim_by_sent)  


def predict_by_SearchLLM_with_regeneration_and_alignment_and_confidence(  
    original_similarity,  
    original_mapping,  
    regeneration_similarity,  
    regeneration_mapping,  
    min_ratio,  
    min_sim_paraphrase,  
    min_sim_machine,  
    min_sim_human,  
    min_regenerate_diff,  
    verbose=False  
):  
    """  
    Predict the source (HUMAN, MACHINE, or UNKNOWN) based on similarity metrics and regeneration.  
    Returns a tuple of prediction and confidence value.  
    """  
    avg_sim = 0  
    predict = UNKNOWN  
    filter_sim = []  
  
    # Filter similarities above the paraphrase threshold  
    for sim in original_similarity:  
        if sim > min_sim_paraphrase:  
            filter_sim.append(sim)  
  
    N = len(original_similarity)  
    pass_number = len(filter_sim)  
  
    if N > 0 and (pass_number / N) >= min_ratio:  
        if len(filter_sim) == 0:  
            avg_sim = 0  
        else:  
            avg_sim = np.average(filter_sim)  
        if avg_sim >= min_sim_machine:  
            if avg_sim >= min_sim_human:  
                predict = HUMAN  
            else:  
                sim_diff, _ = measure_difference_similarity(  
                    original_similarity,  
                    original_mapping,  
                    regeneration_similarity,  
                    regeneration_mapping,  
                    min_sim_paraphrase  
                )  
                if sim_diff is not None and sim_diff > min_regenerate_diff:  
                    predict = MACHINE  
  
    if predict == HUMAN:  
        return predict, 1 - avg_sim  
    elif predict == MACHINE:  
        return predict, avg_sim  
    else:  
        return predict, 0

def detect_by_huggingface_with_search_engine_support_by_soft_and_confidence(  
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
    Perform text classification using HuggingFace pipeline with search engine support.  
    Soft prediction and confidence thresholds are used to determine final labels.  
    Returns metrics for baseline and full evaluation.  
    """  
    start = datetime.now()  
    HUMAN_prediction = -10000000  
    MACHINE_prediction = 10000000  
    print(f"model = {model}_with_search_engine_support\n")  
  
    global MODELS  
    if MODELS[model] is None:  
        MODELS[model] = pipeline(  
            "text-classification",  
            model=model,  
            tokenizer=model,  
            max_length=512,  
            truncation=True  
        )  
  
    label = []  
    predict = []  
    hard_predict = []  
    label_for_searchLLM = []  
    predict_by_SearchLLM = []  
    baseline_predict = []  
    label_for_baseline_only = []  
    predict_for_baseline_only = []  
  
    # Process human samples  
    result = MODELS[model](human)  
    for sub_result, (human_sim, confidence) in zip(result, human_similarity):  
        score = float(sub_result['score'])  
        if human_sim >= human_threshold:  
            predict.append(HUMAN_prediction - confidence)  
            hard_predict.append(HUMAN)  
            label_for_searchLLM.append(HUMAN)  
            predict_by_SearchLLM.append(HUMAN)  
        elif human_sim >= machine_threshold:  
            predict.append(MACHINE_prediction + confidence)  
            hard_predict.append(MACHINE)  
            label_for_searchLLM.append(HUMAN)  
            predict_by_SearchLLM.append(MACHINE)  
        else:  
            label_for_baseline_only.append(HUMAN)  
            if sub_result['label'] == human_label:  
                hard_predict.append(HUMAN)  
                predict.append(1.0 - score)  
                predict_for_baseline_only.append(1.0 - score)  
            else:  
                hard_predict.append(MACHINE)  
                predict.append(score)  
                predict_for_baseline_only.append(score)  
        label.append(HUMAN)  
        if sub_result['label'] == human_label:  
            baseline_predict.append(1.0 - score)  
        else:  
            baseline_predict.append(score)  
  
    # Process machine samples  
    result = MODELS[model](machine)  
    for sub_result, (machine_sim, confidence) in zip(result, machine_similarity):  
        score = float(sub_result['score'])  
        if machine_sim >= human_threshold:  
            label_for_searchLLM.append(MACHINE)  
            predict_by_SearchLLM.append(HUMAN)  
            predict.append(HUMAN_prediction - confidence)  
            hard_predict.append(HUMAN)  
        elif machine_sim >= machine_threshold:  
            predict.append(MACHINE_prediction + confidence)  
            hard_predict.append(MACHINE)  
            label_for_searchLLM.append(MACHINE)  
            predict_by_SearchLLM.append(MACHINE)  
        else:  
            label_for_baseline_only.append(MACHINE)  
            if sub_result['label'] == human_label:  
                hard_predict.append(HUMAN)  
                predict.append(1.0 - score)  
                predict_for_baseline_only.append(1.0 - score)  
            else:  
                hard_predict.append(MACHINE)  
                predict.append(score)  
                predict_for_baseline_only.append(score)  
        label.append(MACHINE)  
        if sub_result['label'] == human_label:  
            baseline_predict.append(1.0 - score)  
        else:  
            baseline_predict.append(score)  
  
    # Evaluate SearchLLM  
    accuracy = accuracy_score(label_for_searchLLM, predict_by_SearchLLM)  
    matrix = confusion_matrix(label_for_searchLLM, predict_by_SearchLLM, labels=[HUMAN, MACHINE])  
  
    if len(human) == 0 or len(machine) == 0:  
        acc = accuracy_score(label, hard_predict)  
        print(f"acc = {acc}\n")  
        return acc  
    else:  
        baseline_accuracy = full_evaluation(label, baseline_predict)  
        baseline_accuracy["roc_auc_by_RADAR"] = calculate_roc_by_RADAR_code(label, baseline_predict)  
  
        full_metrics = full_evaluation(label, predict)  
        full_metrics["roc_auc_by_RADAR"] = calculate_roc_by_RADAR_code(label, predict)  
        print(f"full_metrics = {full_metrics}\n")  
        end = datetime.now()  
        print(f"Running time: {end - start}")  
        full_metrics["Running time"] = end - start  
        full_metrics["searchLLM matrix"] = matrix.tolist()  
        baseline_only_matrix = matrix_based_on_threshold(  
            label_for_baseline_only,  
            predict_for_baseline_only,  
            full_metrics["Best Threshold (by F1)"]  
        )  
        full_metrics["baseline_only_matrix"] = baseline_only_matrix.tolist()  
        full_metrics["p_value"] = compare_models_scaled(  
            label,  
            baseline_predict,  
            predict,  
            scaling='minmax'  
        )  
        return baseline_accuracy, full_metrics

def full_evaluation(y_true_text, y_scores):  
    """  
    Evaluate classifier performance with F1, precision, recall, confusion matrix, ROC AUC, and partial AUCs.  
  
    Args:  
        y_true_text (list): List or array of true binary labels (0 or 1), in text format.  
        y_scores (list): List or array of real-valued scores (higher = more likely class 1).  
  
    Returns:  
        dict: Dictionary containing evaluation metrics.  
    """  
    y_true = []  
    for y in y_true_text:  
        if y == HUMAN:  
            y_true.append(HUMAN_LABEL)  
        else:  
            y_true.append(MACHINE_LABEL)  
    y_true = np.array(y_true)  
    y_scores = np.array(y_scores)  
  
    # Compute ROC curve and thresholds  
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)  
  
    # Compute F1 score for each threshold  
    f1_scores = []  
    for thresh in thresholds:  
        y_pred = (y_scores >= thresh).astype(int)  
        f1 = f1_score(y_true, y_pred)  
        f1_scores.append(f1)  
  
    # Find best threshold by F1  
    best_idx = np.argmax(f1_scores)  
    best_threshold = thresholds[best_idx]  
  
    # Final prediction using best threshold  
    y_pred_optimal = (y_scores >= best_threshold).astype(int)  
  
    # Metrics  
    best_f1 = f1_score(y_true, y_pred_optimal)  
    precision = precision_score(y_true, y_pred_optimal)  
    recall = recall_score(y_true, y_pred_optimal)  
    conf_matrix = confusion_matrix(y_true, y_pred_optimal)  
  
    # Full ROC AUC  
    roc_auc = auc(fpr, tpr)  
  
    # Partial AUC at FPR ≤ 0.1  
    idx_10 = np.where(fpr <= 0.10)[0]  
    auc_10 = auc(fpr[idx_10], tpr[idx_10]) / 0.10 if len(idx_10) > 1 else 0.0  
  
    # Partial AUC at FPR ≤ 0.01  
    idx_01 = np.where(fpr <= 0.01)[0]  
    auc_01 = auc(fpr[idx_01], tpr[idx_01]) / 0.01 if len(idx_01) > 1 else 0.0  
  
    return {  
        "Best Threshold (by F1)": best_threshold,  
        "F1 Score": best_f1,  
        "Precision": precision,  
        "Recall": recall,  
        "Confusion Matrix": conf_matrix.tolist(),  # as nested list  
        "ROC AUC": roc_auc,  
        "ROC AUC@FPR<=10%": auc_10,  
        "ROC AUC@FPR<=1%": auc_01  
    }  

def detect_by_huggingface_with_search_engine_support_and_confidence(  
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
    Detect using HuggingFace model with search engine support and confidence.  
    """  
    return detect_by_huggingface_with_search_engine_support_by_soft_and_confidence(  
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
  
  
def evaluate_baseline_with_search_engine_support_and_confidence(  
    samples,  
    human_threshold,  
    machine_threshold,  
    is_hard=True  
):  
    """  
    Evaluate baseline with search engine support and confidence.  
  
    Args:  
        samples: List of samples. Each item contains (text, label, best_avg_similarity).  
        human_threshold: Threshold for human classification.  
        machine_threshold: Threshold for machine classification.  
        is_hard: Whether to use hard thresholding.  
  
    Writes results to OUTPUT_FILE.  
    """  
    human = []  
    machine = []  
    human_similarity = []  
    machine_similarity = []  
  
    for sample in samples:  
        text = sample[0]  
        label = sample[1]  
        best_avg_similarity = sample[2]  
  
        if label == HUMAN:  
            human.append(text)  
            human_similarity.append(best_avg_similarity)  
        else:  
            machine.append(text)  
            machine_similarity.append(best_avg_similarity)  
  
    result = dict()  
    model = "yaful/MAGE"  
    human_label = 1  
    machine_label = 0  
  
    result[model], result[model + "_search_engine_support"] = (  
        detect_by_huggingface_with_search_engine_support_and_confidence(  
            model,  
            human,  
            machine,  
            human_label,  
            machine_label,  
            human_similarity,  
            machine_similarity,  
            human_threshold,  
            machine_threshold,  
            is_hard  
        )  
    )  
  
    model = "TrustSafeAI/RADAR-Vicuna-7B"  
    human_label = "LABEL_1"  
    machine_label = "LABEL_0"  
  
    result[model], result[model + "_search_engine_support"] = (  
        detect_by_huggingface_with_search_engine_support_and_confidence(  
            model,  
            human,  
            machine,  
            human_label,  
            machine_label,  
            human_similarity,  
            machine_similarity,  
            human_threshold,  
            machine_threshold,  
            is_hard  
        )  
    ) 

    write_to_file(OUTPUT_FILE, str(result))


def baseline_with_SearchLLM_with_regeneration_and_confidence(  
    input_file,  
    min_ratio,  
    min_sim_paraphrase,  
    min_sim_machine,  
    min_sim_human,  
    min_regenerate_diff,  
    verbose=False  
):  
    """  
    Process input data with SearchLLM using regeneration and confidence thresholds.  
  
    Args:  
        input_file (str): Input CSV file path.  
        min_ratio (float): Minimum ratio for alignment.  
        min_sim_paraphrase (float): Minimum similarity for paraphrase.  
        min_sim_machine (float): Minimum similarity for machine.  
        min_sim_human (float): Minimum similarity for human.  
        min_regenerate_diff (float): Minimum regeneration difference.  
        verbose (bool, optional): Verbosity flag. Defaults to False.  
  
    Returns:  
        None  
    """  
    human_threshold = 0.8  
    machine_threshold = 0.5  
  
    write_to_file(OUTPUT_FILE, "\n\nBASELINE WITH SEARCH ENGINE SUPPORT AND CONFIDENCE\n")  
    write_to_file(OUTPUT_FILE, f"csv_file = {input_file}\n")  
    data = read_csv_data(input_file)  
  
    samples = []  # text, label, best_avg_similarity  
  
    for item in data:  
        text = item[0]  
        label = item[1]  
        original_similarity = convert_to_object(item[4])  
        original_mapping = convert_to_object(item[5])  
        regeneration_similarity = convert_to_object(item[6])  
        regeneration_mapping = convert_to_object(item[7])  
  
        searchLLM_predict, confidence = predict_by_SearchLLM_with_regeneration_and_alignment_and_confidence(  
            original_similarity,  
            original_mapping,  
            regeneration_similarity,  
            regeneration_mapping,  
            min_ratio,  
            min_sim_paraphrase,  
            min_sim_machine,  
            min_sim_human,  
            min_regenerate_diff  
        )  
  
        if searchLLM_predict == HUMAN:  
            final_similarity = (human_threshold, confidence)  
        elif searchLLM_predict == MACHINE:  
            final_similarity = (machine_threshold, confidence)  
        else:  
            final_similarity = (0, 0)  # undecided  
  
        triple = (text, label, final_similarity)  
        samples.append(triple)  
  
    is_hard = False  
    write_to_file(OUTPUT_FILE, f"is_hard = {is_hard}\n")  
    evaluate_baseline_with_search_engine_support_and_confidence(  
        samples, human_threshold, machine_threshold, is_hard  
    )

def testing():  
    """  
    Set up parameters and files, then run similarity calculations and baseline evaluation.  
    """  
    a = 2  
    global OUTPUT_FILE  
    OUTPUT_FILE = "5_data/result.txt"  
  
    # Set parameters for similarity calculation  
    number_of_first_sentences = 1  
    number_sample = -1  
    minimum_similarity_for_mapping = 0.7  
    minimum_similarity_for_regeneration = 0.7  
    min_rate = 0.3  
    model_name = GPT_4O_MINI  
    input_file = "5_data/data.csv"  
    output_file = "5_data/similarity.csv"  
  
    calculate_similarity_by_SeachLLM_greedy_mapping_with_regeneration(  
        input_file,  
        output_file,  
        model_name,  
        number_of_first_sentences,  
        minimum_similarity_for_mapping,  
        minimum_similarity_for_regeneration,  
        min_rate,  
        number_sample  
    )  
  
    input_file = output_file  
    baseline_with_SearchLLM_with_regeneration_and_confidence(  
        input_file,  
        MIN_RATIO,  
        MIN_SIM_PARAPHRASE,  
        MIN_SIM_MACHINE,  
        MIN_SIM_HUMAN,  
        MIN_REGENERATE_DIFF  
    )  
  
  
def get_roc_metrics(human_preds, ai_preds):  
    """  
    Compute ROC curve and AUC for predictions from human and AI.  
    human_preds: probabilities for human-generated text.  
    ai_preds: probabilities for AI-generated text.  
    """  
    fpr, tpr, _ = roc_curve(  
        [0] * len(human_preds) + [1] * len(ai_preds),  
        human_preds + ai_preds,  
        pos_label=1  
    )  
    roc_auc = auc(fpr, tpr)  
    return fpr.tolist(), tpr.tolist(), float(roc_auc)  
  
  
def calculate_roc_by_RADAR_code(label, pred):  
    """  
    Separate predictions into human and machine, then compute ROC AUC.  
    """  
    human = []  
    machine = []  
    for la, pr in zip(label, pred):  
        if la == HUMAN:  
            human.append(pr)  
        else:  
            machine.append(pr)  
    result = get_roc_metrics(human, machine)  
    return result[-1]  
  
  
def matrix_based_on_threshold(y_true_text, predict_for_baseline_only, threshold):  
    """  
    Compute confusion matrix based on thresholded predictions.  
    """  
    y_true = []  
    for y in y_true_text:  
        if y == HUMAN:  
            y_true.append(HUMAN_LABEL)  
        else:  
            y_true.append(MACHINE_LABEL)  
    y_true = np.array(y_true)  
    y_pred_optimal = (predict_for_baseline_only >= threshold).astype(int)  
    conf_matrix = confusion_matrix(  
        y_true,  
        y_pred_optimal,  
        labels=[HUMAN_LABEL, MACHINE_LABEL]  
    )  
    return conf_matrix

def compare_models_scaled(gold_labels, scores_model1, scores_model2, scaling='minmax'):  
    """  
    Compare two models based on adjusted scores and normalization to the same scale.  
    - For label 'human': higher score is better.  
    - For label 'machine': lower score is better (invert sign).  
    - Normalize scores to the same scale (min-max or z-score).  
    - Perform Paired t-test.  
      
    Parameters:  
        gold_labels: list of actual labels ['human', 'machine', ...]  
        scores_model1: list of scores from model 1  
        scores_model2: list of scores from model 2  
        scaling: 'minmax' or 'zscore' (default is 'minmax')  
      
    Returns:  
        p-value  
        Conclusion about which model is better if p-value < 0.05  
    """  
    # Step 1: Adjust scores so that higher is always better  
    adjusted1 = []  
    adjusted2 = []  
    for label, s1, s2 in zip(gold_labels, scores_model1, scores_model2):  
        if label == 'human':  
            adjusted1.append(s1)  
            adjusted2.append(s2)  
        elif label == 'machine':  
            adjusted1.append(-s1)  
            adjusted2.append(-s2)  
        else:  
            raise ValueError(f"Invalid label: {label}")  
  
    # Step 2: Normalize scores to the same scale  
    adjusted1 = np.array(adjusted1).reshape(-1, 1)  
    adjusted2 = np.array(adjusted2).reshape(-1, 1)  
    combined = np.hstack([adjusted1, adjusted2])  
  
    if scaling == 'minmax':  
        scaler = MinMaxScaler()  
    elif scaling == 'zscore':  
        scaler = StandardScaler()  
    else:  
        raise ValueError("Parameter 'scaling' must be 'minmax' or 'zscore'.")  
  
    scaled = scaler.fit_transform(combined)  
    scores1_scaled = scaled[:, 0]  
    scores2_scaled = scaled[:, 1]  
  
    # Step 3: Perform Paired t-test  
    statistic, p_value = ttest_rel(scores1_scaled, scores2_scaled)  
    return p_value  
  
  
if __name__ == "__main__":  
    a = 2  
    testing()