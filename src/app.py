"""
To run this app, after you've the data, execute the following:

pip install -r requirements.txt
cd src
streamlit run app.py
"""

from logging import disable
from numpy import load
import pandas as pd
import spacy
import streamlit as st
from streamlit.elements.text_widgets import TextWidgetsMixin
import seaborn as sns
import textacy
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List

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


@st.cache
def load_data(file_path: Path) -> List[str]:
    """Load data as a list of reviews"""
    with file_path.open("r") as f:
        df = pd.read_json(f)
    reviews = df.review.to_list()

    return reviews


def make_corpus(reviews: List[str]):
    corpus = textacy.Corpus("en_core_web_sm", data=reviews)
    return corpus


# Reduce Corpus Load
reviews = load_data(file_path)
count = st.number_input(
    label="Number of Reviews you want to analyze", value=50, max_value=len(reviews)
)
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
    most_common_df = pd.DataFrame(most_common, columns=["words", "count"])
    return most_common_df


most_common_df = get_word_counts(corpus)
st.dataframe(most_common_df[:10])

fig, ax = plt.subplots()
sns.scatterplot(data=most_common_df[:10], x="count", y="words")
st.pyplot(fig)
