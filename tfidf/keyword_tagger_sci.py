"""
This module provides various functions to extract and manipulate keyword data 
from textual documents using TF-IDF (Term Frequency-Inverse Document Frequency) methodology. 
Running this script provides the scores when testing against three datasets:
inspec, duc2001, and nus
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
import numpy as np

# Load training, development, and test datasets from their respective sources
train_dataset = load_dataset('midas/inspec', 'raw')
dev_dataset = load_dataset('midas/duc2001', 'raw')
test_dataset = load_dataset('midas/nus', 'raw')

# Prepare documents from each dataset
train_documents = [' '.join(item['document']) for item in train_dataset['test']]
dev_documents = [' '.join(item['document']) for item in dev_dataset['test']]
test_documents = [' '.join(item['document']) for item in test_dataset['test']]

#documents = [" ".join(item["document"]) for item in dataset["test"]]

# Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

# Fit and transform the documents
train_tfidf_matrix = tfidf_vectorizer.fit_transform(train_documents)
dev_tfidf_matrix = tfidf_vectorizer.transform(dev_documents)
test_tfidf_matrix = tfidf_vectorizer.transform(test_documents)
# Feature names will give you the terms (including n-grams)
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())


def get_dynamic_threshold_terms_scores(tfidf_matrix, feature_names, quantile=0.6):
    """
    Retrieves terms from a TF-IDF matrix whose scores meet a dynamically determined threshold 
        based on a specified quantile for each document.

    Args:
        tfidf_matrix (array-like): A matrix where each row represents a document and each column 
            a term's TF-IDF score.
        feature_names (list): A list of all feature names/terms corresponding to the columns of 
            the tfidf_matrix.
        quantile (float, optional): Quantile used to determine the threshold for including terms. 
            Defaults to 0.6.

    Returns:
        list of lists: Each inner list contains tuples of terms and their scores that exceed the 
            dynamically determined threshold, sorted by score in descending order for each document.
    """
    terms_scores = []
    for row in tfidf_matrix:
        # Get threshold based on quantile, also ignoring values that are too low
        threshold = max(
            np.quantile(row.toarray().flatten(), quantile), 0.1
        )  # Ensure a minimum threshold of 0.1

        scores = row.toarray().flatten()
        valid_indices = np.where(scores >= threshold)[0]

        valid_feature_names = feature_names[valid_indices]
        valid_scores = scores[valid_indices]

        feature_scores = list(zip(valid_feature_names, valid_scores))
        feature_scores_sorted = sorted(feature_scores, key=lambda x: x[1], reverse=True)

        terms_scores.append(feature_scores_sorted)

    return terms_scores

def get_top_keyphrases_sci_dynamic(tfidfmatrix):
    """
    Extracts and concatenates the top keyphrases for each document in a given TF-IDF matrix,
    using a dynamic threshold based on quantile to determine term relevance.

    Returns:
        list: A list of strings, where each string represents the concatenated top 
            keyphrases for a document.
    """
    top_terms_scores = get_dynamic_threshold_terms_scores(tfidfmatrix, feature_names)
    top_keyphrases = []
    for doc_index, terms_scores in enumerate(top_terms_scores, start=1):
        doc_keyphrases = ""
        for term, score in terms_scores:
            if doc_keyphrases != "":
                doc_keyphrases += " "
            doc_keyphrases += term
        top_keyphrases.append(doc_keyphrases)
    return top_keyphrases

def get_output_file(output_file_path, dynamic=False):
    """
    Writes the top terms and their scores from a TF-IDF matrix to a specified file,
    allowing for either a fixed or dynamic threshold for term selection 
        based on the 'dynamic' parameter.

    Args:
        output_file_path (str): The path to the file where the output will be written.
        dynamic (bool, optional): Determines the threshold strategy for term selection.
            If True, uses a dynamic quantile-based threshold. If False, uses a fixed threshold.
            Defaults to False.

    Returns:
        None: The function writes to a file and does not return any value.
    """
    if dynamic:
        top_terms_scores = get_dynamic_threshold_terms_scores(
            train_tfidf_matrix, feature_names
        )

    with open(output_file_path, "w", encoding="utf-8") as file:
        for doc_index, terms_scores in enumerate(top_terms_scores, start=1):
            file.write(f"Document {doc_index} top terms:\n")
            for term, score in terms_scores:
                if score > 0.1:  # Only write terms with a score above 0.1
                    line = f"{term}: {score:.12f}\n"
                    file.write(line)
            file.write("\n")

#####################SCORING#######################################################################################
##################################################################################################################

train_answer_keyphrases = [
    " ".join(item["extractive_keyphrases"]) for item in train_dataset["test"]
]
dev_answer_keyphrases = [
    " ".join(item["extractive_keyphrases"]) for item in dev_dataset["test"]
]
test_answer_keyphrases = [
    " ".join(item["extractive_keyphrases"]) for item in test_dataset["test"]
]

train_tfidf = get_top_keyphrases_sci_dynamic(train_tfidf_matrix)
dev_tfidf = get_top_keyphrases_sci_dynamic(dev_tfidf_matrix)
test_tfidf = get_top_keyphrases_sci_dynamic(test_tfidf_matrix)

train_tfidf_scores = get_dynamic_threshold_terms_scores(train_tfidf_matrix, feature_names)
dev_tfidf_scores = get_dynamic_threshold_terms_scores(dev_tfidf_matrix, feature_names)
test_tfidf_scores = get_dynamic_threshold_terms_scores(test_tfidf_matrix, feature_names)

def evaluate(system_keyphrases, answer_keyphrases):
    """
    Calculate Precision, Recall, and F1-Score for a set of predicted keyphrases 
    against a set of ground truth keyphrases.

    This function assesses the precision of a keyphrase prediction system by comparing 
    each predicted keyphrase against a set of actual keyphrases and computing the 
    precision across all documents. The function iteratively calculates the precision 
    for individual documents by determining the ratio of relevant keyphrases correctly 
    predicted and averages these values to provide the MAP.

    Parameters:
        system_keyphrases (list of str): A list containing the 
                                        predicted keyphrases for each document.
                                        
        answer_keyphrases (list of str): A list of actual keyphrases 
                                        for each document used as the ground truth.

    Returns:
        float: The mean average precision score as a float.
    """
    total_precision = 0
    total_recall=0
    total_f1_score = 0
    valid_docs = 0 #  docs where we actually calculate precision

    # Iterate over each document's keyphrases
    for system_key, answer_key in zip(system_keyphrases, answer_keyphrases):
        # Split keyphrases into words
        system_words = set(system_key.lower().split())
        answer_words = set(answer_key.lower().split())

        if not system_words:
            continue  # Skip documents with no system keyphrases to avoid division by zero

        # Calculate precision and recall for this document
        correct_count = 0
        for word in system_words:
            if word in answer_words:
                correct_count += 1
        document_recall = correct_count / len(answer_words) if answer_words else 0
        document_precision = correct_count / len(system_words) if system_words else 0

        # Calculate f1 score
        if document_precision + document_recall != 0:
            document_f1_score = 2 * (document_recall*document_precision) / (document_precision + document_recall)
        else:
            document_f1_score=0

        # Aggregate
        total_precision += document_precision
        total_recall+=document_recall
        total_f1_score+=document_f1_score
        valid_docs += 1

    # Calculate mean scores for other measures
    mean_precision = (total_precision / valid_docs) if valid_docs else 0
    mean_recall = (total_recall / valid_docs) if valid_docs else 0
    mean_f1_score = (total_f1_score / valid_docs) if valid_docs else 0

    return {
        "precision": mean_precision,
        "recall": mean_recall,
        "f1_score": mean_f1_score,
    }
def dcg_at_k(relevance_scores, k, method=1):
    """Calculate discounted cumulative gain (DCG) at rank k.

    Args:
        relevance_scores (list of float): The list of relevance scores.
        k (int): The number of results to consider.
        method (int): The method to compute DCG, 0 or 1.

    Returns:
        float: The DCG score.

    Raises:
        ValueError: If the method is not 0 or 1.
    """
    relevance_scores = np.asfarray(relevance_scores)[:k]
    if relevance_scores.size:
        if method == 0:
            return relevance_scores[0] + np.sum(relevance_scores[1:] / np.log2(np.arange(2, relevance_scores.size + 1)))
        elif method == 1:
            return np.sum(relevance_scores / np.log2(np.arange(2, relevance_scores.size + 2)))
    return 0.0

def ndcg_at_k(relevance_scores, k, method=1):
    """Calculate normalized discounted cumulative gain (NDCG) at rank k.

    Args:
        relevance_scores (list of float): The list of relevance scores.
        k (int): The number of results to consider.
        method (int): The method to compute DCG, 0 or 1.

    Returns:
        float: The NDCG score.
    """
    dcg_max = dcg_at_k(sorted(relevance_scores, reverse=True), k, method)
    if not dcg_max:
        return 0.0
    return dcg_at_k(relevance_scores, k, method) / dcg_max

def mean_reciprocal_rank(ranking_lists):
    """Calculate the mean reciprocal rank (MRR).

    Args:
        ranking_lists (list of list of int): Each inner list is a set of binary values (0 or 1)
            indicating the absence or presence of relevant items.

    Returns:
        float: The MRR score.
    """
    first_relevant = (np.asarray(rankings).nonzero()[0] for rankings in ranking_lists)
    return np.mean([1.0 / (ranking[0] + 1) if ranking.size else 0 for ranking in first_relevant])

def calculate_relevance_scores(true_keywords, predicted_keywords):
    """Calculates relevance scores where 1 indicates relevance and 0 indicates irrelevance.
   
    Args:
        true_keywords (list of str): The list of true keywords.
        predicted_keywords (list of tuples): List of predicted keywords with their scores.
   
    Returns:
        list of int: Relevance scores (1 or 0) for each predicted keyword.
    """
    return [1 if keyword in true_keywords else 0 for keyword, _ in predicted_keywords]

def evaluate_keyword_extraction(true_data, predictions):
    """Evaluates the keyword extraction algorithm using NDCG and MRR scoring metrics.
   
    Args:
        true_data (list of list of str): List of lists containing true keywords for each document.
        predictions (list of list of tuples): List of lists, each containing tuples of keywords and their confidence scores.
   
    Returns:
        tuple of (float, float): Mean NDCG score and Mean MRR score.
    """
    ndcg_scores = []
    mrr_scores = []

    for true_keywords, predicted_keywords_with_scores in zip(true_data, predictions):
        predicted_keywords_with_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by confidence score descending
        predicted_keywords = [kw for kw, _ in predicted_keywords_with_scores]
        relevance_scores = calculate_relevance_scores(true_keywords, predicted_keywords_with_scores)
        # Compute NDCG
        ndcg_score = ndcg_at_k(relevance_scores, k=len(relevance_scores))
        #print(ndcg_score)
        ndcg_scores.append(ndcg_score)
       
        # Compute MRR
        rs = [[1 if keyword in true_keywords else 0 for keyword in predicted_keywords]]
        mrr_score = mean_reciprocal_rank(rs)
        #print(mrr_score)
        mrr_scores.append(mrr_score)
    
    mean_ndcg = np.mean(ndcg_scores)
    mean_mrr = np.mean(mrr_scores)
    return mean_ndcg, mean_mrr

train_results = evaluate(train_tfidf, train_answer_keyphrases)
train_mean_ndcg, train_mean_mrr = evaluate_keyword_extraction(train_answer_keyphrases, train_tfidf_scores)
print("INSPEC")
print("Precision: {:.2%}".format(train_results["precision"]))
print("Recall: {:.2%}".format(train_results["recall"]))
print("F1 Score: {:.2%}".format(train_results["f1_score"]))
print(f"Mean NDCG: {train_mean_ndcg:.3f}")
print(f"Mean MRR: {train_mean_mrr:.3f} \n")


dev_results = evaluate(dev_tfidf, dev_answer_keyphrases)
dev_mean_ndcg, dev_mean_mrr = evaluate_keyword_extraction(dev_answer_keyphrases, dev_tfidf_scores)
print("DUC2001")
print("Precision: {:.2%}".format(dev_results["precision"]))
print("Recall: {:.2%}".format(dev_results["recall"]))
print("F1 Score: {:.2%}".format(dev_results["f1_score"]))
print(f"Mean NDCG: {dev_mean_ndcg:.3f}")
print(f"Mean MRR: {dev_mean_mrr:.3f}\n")

test_results = evaluate(test_tfidf, test_answer_keyphrases)
test_mean_ndcg, test_mean_mrr = evaluate_keyword_extraction(test_answer_keyphrases, test_tfidf_scores)
print("NUS")
print("Precision: {:.2%}".format(test_results["precision"]))
print("Recall: {:.2%}".format(test_results["recall"]))
print("F1 Score: {:.2%}".format(test_results["f1_score"]))
print(f"Mean NDCG: {test_mean_ndcg:.3f}")
print(f"Mean MRR: {test_mean_mrr:.3f}\n")