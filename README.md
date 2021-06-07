# App Reviews

## Getting Started
Begin with installing the required dependencies and then the spaCy models. 
Please note that optional dependencies are not in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

For speed, or when first time running the code. This is also the default spaCy model we use. 
```
python -m spacy en_core_web_sm download 
```

## Inputs
The data directory consists of two main directories: `raw` and `tagged`. 

## 1.0 Interactive Text Exploration App

Run the app from `src` directory with:
```bash
streamlit run app.py
```

This app covers our text exploration. We do the following to understand the text better:

- Vocabulary / Word Counts 
- N-grams
- Nouns

**Topic Extraction**
- (Unsupervised) Key Word Extraction with YAKE

Each of the above has a Seaborn or Altair Plot of the first 10-20 results
- [x] Visualization: Streamlit + Altair Demo
- [x] Topics, grouped by aggregate word frequency of top k words

## Notebooks

For each app review, we have more than one label possible. This means our dataset is multi-label.

### 2.1 Label Finding

We attack the problem of using Topic Modeling and Linguisic terms to find relevant topics and select meaningful labels. We also use `PyDantic` to define our Topic class to manage all the terms related to a specific topic in one object. 
We assume that we have no labels for the dataset in order to reflect more common situations.

### 3.1 TF-IDF-NB-LR

We build on the work from [Wang and Manning](https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf).

The [J. Howard implementation from Kaggle](https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline) proposes two changes: 
1. Using Logistic Regression instead of Support Vector Machine as the 2nd stacked model
2. Using TF-IDF instead of TF for feature extraction

We keep the model equations (Naive Bayes and LR) unchanged but instead binarize each label against the review. This allows us to use the existing implementation with relative ease.

### 3.2 Model Evaluation

We train and evaluate a separate model for each label, where each label now only has binary values against review. This makes evaluation relatively straightforward.

We calculate the following metrics for each model, trained on a separate label:
1. Precision
2. Recall
3. F1 Score

We do so at multiple thresholds to get a sense of the model performance itself.