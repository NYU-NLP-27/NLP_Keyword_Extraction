# TF-IDF Keyword Extractor

The TF-IDF Keyword Extractor is a Python program designed to efficiently identify and extract key terms from a corpus of text documents using the Term Frequency-Inverse Document Frequency (TF-IDF) method. This tool allows for both fixed and dynamic thresholding of terms based on their relevance, providing flexibility in how keywords are selected and utilized. Running `python3 keyword_tagger_sci.py` provides the scores when testing the model against three datasets: inspec, duc2001, and nus.

## Features

- **TF-IDF Matrix Calculation**: Build a matrix representation of documents and terms weighted by their importance.
- **Keyword Extraction**: Extract top terms based on fixed or dynamic thresholds.
- **Keyphrase Concatenation**: Combine top terms into keyphrases for better contextual understanding.
- **Flexible Thresholding**: Choose between a static threshold or a dynamic quantile-based threshold to suit different analytical needs.
- **Output to Text File**: Save the extracted keywords and their scores to a text file for further analysis or reporting.
- **Scoring**: Evaluate the extraction performance using precision, recall, and F1-score, as well as advanced metrics like NDCG and MRR.

## Installation

Ensure you have Python 3.6 or later installed, and then set up a virtual environment:

```bash
pip install pipenv
pipenv install
pipenv shell
```

Alternatively, install the required packages as follows:

```bash
pip install scikit-learn
pip install datasets
pip install numpy
```

## Functions
- **get_top_terms_scores(tfidf_matrix, feature_names, n=10)**: Extracts the top n terms with the highest TF-IDF scores.
- **get_terms_above_threshold(tfidf_matrix, feature_names, threshold=0.1)**: Filters terms by a fixed score threshold.
- **get_dynamic_threshold_terms_scores(tfidf_matrix, feature_names, quantile=0.6)**: Uses a quantile-based dynamic threshold to filter terms.
- **get_top_keyphrases_sci()**: Concatenates top terms into keyphrases using a fixed threshold.
- **get_top_keyphrases_sci_dynamic()**: Concatenates top terms into keyphrases using a dynamic threshold based on quantiles.
- **get_output_file(output_file_path, dynamic=False)**: Writes the top terms and their scores to a specified file, with an option to use dynamic thresholding.
- **evaluate**: Computes precision, recall, and F1-score for extracted keyphrases.
- **dcg_at_k, ndcg_at_k, mean_reciprocal_rank**: Functions for calculating ranking-based metrics.

The module uses the `load_dataset` function from the `datasets` library to 
fetch datasets for keyword extraction. 
Various datasets can be tested by switching the dataset source in the script.

## Example Usage

```bash
# Load your dataset
    dataset = load_dataset("midas/nus", "raw")  # test corpus

    # Prepare documents
    documents = [" ".join(item["document"]) for item in dataset["test"]]

    # Compute TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

    # Extract keywords
    keywords = get_top_terms_scores(tfidf_matrix, feature_names)

    # Evaluate extraction performance
    results = evaluate(predicted_keyphrases, actual_keyphrases)
```
## Evaluation Metrics
- **Precision**: Proportion of predicted keyphrases that are relevant.
- **Recall**: Proportion of relevant keyphrases that are correctly predicted.
- **F1 Score** : Harmonic mean of precision and recall.
- **NDCG**: Normalized Discounted Cumulative Gain, a measure of ranking quality.
- **MRR**: Mean Reciprocal Rank, a statistic for evaluating the ranking of relevant documents.

## Datasets
- **Training Corpus**: [Inspec](https://huggingface.co/datasets/midas/inspec) database comprised of 2000 short documents from science journal abstracts.
- **Development Corpus**:  [DUC2001](https://huggingface.co/datasets/midas/duc2001) is comprised of 308 mid-length news articles organized in 30 topics.
- **Test Corpus**: [Nus](https://huggingface.co/datasets/midas/nus) which contains 211 long scientific conference papers (4~12 pages).

## Current Results

To get results run:
```bash
python3 keyword_tagger_score.py
```
To get:

```bash
INSPEC
Precision: 46.27%
Recall: 55.52%
F1 Score: 42.59%
Mean NDCG: 0.788
Mean MRR: 0.783 

DUC2001
Precision: 19.52%
Recall: 21.07%
F1 Score: 19.19%
Mean NDCG: 0.611
Mean MRR: 0.536

NUS
Precision: 26.70%
Recall: 37.20%
F1 Score: 27.85%
Mean NDCG: 0.711
Mean MRR: 0.702
```

## Authors
- [Leeya Howley](https://github.com/rlh9398)
- [Corina Luca](https://github.com/CorinaLucaFocsan)
