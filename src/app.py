"""
To run this app, after you've the data, execute the following:

pip install -r requirements.txt
cd src
streamlit run app.py
"""

from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import spacy
import streamlit as st
from textacy import Corpus, extract
from textacy.extract.basics import entities, ngrams, noun_chunks

from pathlib import Path
from typing import List

DEBUG = True

# Define Text
st.title("Text Exploration App")
st.write(
    "Depending on your hard disk and CPU, this can take a few minutes to run. Be patient!"
)
# File Paths
data_dir = Path("../data/")
assert data_dir.exists()
clients = {"Uber": data_dir / "Uber_us_app_store_reviews.json"}
file_path = clients["Uber"]
assert file_path.exists()

st.sidebar.title("Options")
word_count = st.sidebar.checkbox("Word Counts")
ngrams_count = st.sidebar.checkbox("n-grams")
show_ncs = st.sidebar.checkbox("Noun", value=True)
show_ents = st.sidebar.checkbox("Entities")


@st.cache
def load_data(file_path: Path) -> List[str]:
    """Load data as a list of reviews"""
    with file_path.open("r") as f:
        df = pd.read_json(f)
    reviews = df.review.to_list()

    return reviews


def make_corpus(reviews: List[str]):
    corpus = Corpus("en_core_web_sm", data=reviews)
    return corpus


# Reduce Corpus Load
reviews = load_data(file_path)
count = st.number_input(
    label="Number of Reviews you want to analyze", value=50, max_value=len(reviews)
)

if DEBUG:
    reviews = reviews[:count]

corpus = make_corpus(reviews)
st.write(
    f"Total Reviews: {corpus.n_docs},\nNumber of Sentences: {corpus.n_sents},\nNumber of words/tokens: {corpus.n_tokens}"
)


def get_word_counts(
    corpus=corpus, by="lemma_", filter_stops=True, filter_nums=True, filter_punct=True
):
    word_counts = corpus.word_counts(
        by="lemma_", filter_stops=True, filter_nums=True, filter_punct=True
    )
    most_common = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:25]
    most_common_df = pd.DataFrame(most_common, columns=["Words", "Count"])
    return most_common_df


if word_count:
    st.write("## Word Counts")

    most_common_df = get_word_counts(corpus)
    if DEBUG:
        st.dataframe(most_common_df[:10])

    fig, ax = plt.subplots()
    sns.scatterplot(data=most_common_df[:10], x="count", y="words")
    st.pyplot(fig)


def spacy_span_hash(input):
    return str(input)


@st.cache(hash_funcs={spacy.tokens.span.Span: spacy_span_hash})
def get_count(list_of_doc_terms: List[List], lower=True) -> pd.DataFrame:
    if lower:
        terms_list = [
            str(val).lower() for sublist in list_of_doc_terms for val in sublist
        ]
    else:
        terms_list = [str(val) for sublist in list_of_doc_terms for val in sublist]
    terms_freq = dict(Counter(terms_list))
    df = pd.DataFrame.from_dict(terms_freq, orient="index")
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Phrase", 0: "Count"}, inplace=True)
    # st.write(df.columns)
    df.sort_values(by=["Count", "Phrase"], inplace=True, ascending=False)
    return df


def get_linguistic_counts(corpus=corpus, by="noun_chunks", **kwargs):
    if by == "noun_chunks":
        list_of_doc_terms = [
            list(noun_chunks(doc, drop_determiners=True)) for doc in corpus
        ]
    elif by == "ents":
        list_of_doc_terms = [
            list(entities(doc, drop_determiners=True, exclude_types="NUMERIC"))
            for doc in corpus
        ]
    return get_count(list_of_doc_terms)


if show_ncs:
    st.write("## Noun Phrases")
    st.write(get_linguistic_counts(corpus))

if show_ents:
    st.write("## Entities")
    st.write(get_linguistic_counts(corpus, by="ents"))
    # terms_df = pd.DataFrame(dict(Counter(terms)))
    # st.write(dict(Counter(terms)))
