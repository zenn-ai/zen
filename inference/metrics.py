import warnings
import pandas as pd

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

from rouge import Rouge
from bert_score import score


# nltk.download('punkt')
# nltk.download('wordnet')
warnings.filterwarnings('ignore')


def calculate_bleu(references, candidates):
    """
    Calculate BLEU score for two lists of strings.
    :param references: List of reference strings (ground truth).
    :param candidates: List of candidate strings (generated text).
    :return: BLEU score.
    """
    # Tokenize the sentences
    tokenized_references = [word_tokenize(ref.lower()) for ref in references]
    tokenized_candidates = [word_tokenize(candidate.lower()) for candidate in candidates]
    # Calculate BLEU score
    scores = [sentence_bleu([ref], cand) for ref, cand in zip(tokenized_references, tokenized_candidates)]
    score = round(sum(scores) * 100 / len(scores), 2)
    return score


def rouge_dict_to_dataframe(rouge_dict):
    """
    Convert a ROUGE scores dictionary to a pandas DataFrame.
    :param rouge_dict: Dictionary containing ROUGE scores.
    :return: DataFrame with the ROUGE scores.
    """
    # Initialize an empty dictionary to hold the data
    data = {}
    # Loop through each ROUGE type and extract the values
    for rouge_type, scores in rouge_dict.items():
        for score_type, value in scores.items():
            # Create a column name like 'rouge-1_r', 'rouge-1_p', etc.
            col_name = f'{rouge_type}_{score_type}'
            data[col_name] = [value*100]
    # Create and return the DataFrame
    return pd.DataFrame(data).round(2)


def calculate_rouge(references, candidates):
    """
    Calculate ROUGE score for two lists of strings.
    :param references: List of reference strings (ground truth).
    :param candidates: List of candidate strings (generated text).
    :return: Dictionary of ROUGE scores.
    """
    rouge = Rouge()
    scores = rouge.get_scores(candidates, references, avg=True)
    return scores


def calculate_meteor(references, candidates):
    """
    Calculate METEOR score for lists of reference and candidate strings.
    :param references: List of reference strings.
    :param candidates: List of candidate strings.
    :return: Average METEOR score.
    """
    # Tokenize the candidate sentences
    tokenized_references = [word_tokenize(ref.lower()) for ref in references]
    tokenized_candidates = [word_tokenize(cand.lower()) for cand in candidates]
    # Calculate METEOR score for each pair of reference and tokenized candidate
    scores = [meteor_score([ref], cand) for ref, cand in zip(tokenized_references, tokenized_candidates)]
    score = round(sum(scores) * 100 / len(scores), 2)
    return score


def calculate_bert_score(references, candidates):
    """
    Calculate BERTScore for lists of reference and candidate strings.
    :param references: List of reference strings.
    :param candidates: List of candidate strings.
    :return: Average BERTScore.
    """
    P, R, F1 = score(candidates, references, lang="en")
    P = round(P.mean().item() * 100, 2)
    R = round(R.mean().item() * 100, 2)
    F1 = round(F1.mean().item() * 100, 2)
    return {"precision": P, "recall": R, "f1": F1}


def calculate_all_metrics(references, candidates, label):
    """
    Calculate BLEU, ROUGE, METEOR, and BERTScore for given references and candidates.
    :param references: List of reference strings.
    :param candidates: List of candidate strings.
    :return: DataFrame with each metric's score in separate columns.
    """
    # Calculate each metric
    bleu_score = calculate_bleu(references, candidates)
    meteor_score = calculate_meteor(references, candidates)
    bert_scores = calculate_bert_score(references, candidates)
    rouge_scores = calculate_rouge(references, candidates)
    # Prepare data for DataFrame
    data = {
        'BLEU': [bleu_score],
        'METEOR': [meteor_score],
        'BERTScore_Precision': [bert_scores['precision']],
        'BERTScore_Recall': [bert_scores['recall']],
        'BERTScore_F1': [bert_scores['f1']]
    }
    name_dict = {
        "r": "Recall",
        "p": "Precision",
        "f": "F1"
    }
    # Add ROUGE scores to the data dictionary
    for rouge_key, rouge_values in rouge_scores.items():
        for score_type, value in rouge_values.items():
            data[f'{rouge_key.upper()}_{name_dict[score_type]}'] = [round(value * 100, 2)]
    # Create DataFrame
    data = pd.DataFrame(data)
    data["Model"] = label
    return data
