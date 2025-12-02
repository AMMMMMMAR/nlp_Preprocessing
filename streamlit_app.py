"""Streamlit UI that visualizes each NLP preprocessing step."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import run_pipeline


st.set_page_config(page_title="NLP Preprocessing Visualizer", layout="wide")

st.title("NLP Preprocessing Visualizer")
st.write(
    "NLP preprocessing means cleaning raw text so that models can understand it. "
    "This page lets you run the classic steps and shows short notes for each stage."
)
st.write(
    "Paste any English text in the box below and press **Run NLP Pipeline** to see how "
    "the text changes from the original words to lemmas."
)

placeholder_text = (
    "Natural Language Processing (NLP) helps computers understand people. "
    "Try typing your own paragraph to explore the steps!"
)

user_text = st.text_area(
    "Write or paste your text",
    value="",
    height=220,
    placeholder=placeholder_text,
)

run_clicked = st.button("Run NLP Pipeline", type="primary")


def format_list(items: list[str]) -> str:
    """Format tokens as a multi-line string for code blocks."""
    return "\n".join(items) if items else "(no tokens)"


if run_clicked:
    if not user_text.strip():
        st.warning("Please type some text first.")
    else:
        results = run_pipeline(user_text)
        token_count = len(results["tokens"])
        clean_token_count = len(results["tokens_no_sw"])

        st.subheader("Original Text")
        st.caption("The exact text you typed before any cleaning happens.")
        st.code(user_text.strip(), language="text")

        st.subheader("Cleaned Text")
        st.caption(
            "We lowercase the text and remove links, numbers, punctuation, and extra spaces "
            "so that the remaining words are neat."
        )
        st.code(results["cleaned"], language="text")

        st.subheader("Tokens")
        st.caption("Tokenization splits the cleaned text into simple word pieces.")
        col1, col2 = st.columns(2)
        col1.metric("Number of tokens", token_count)
        col2.metric("Unique tokens", len(set(results["tokens"])))
        st.code(format_list(results["tokens"]), language="text")

        st.subheader("Tokens without Stopwords")
        st.caption(
            "Stopwords are very common helper words (like 'the' or 'is'). We remove them to focus on "
            "the words that carry the main meaning."
        )
        col3, col4 = st.columns(2)
        col3.metric("After removing stopwords", clean_token_count)
        col4.metric("Words removed", token_count - clean_token_count)
        st.code(format_list(results["tokens_no_sw"]), language="text")

        st.subheader("Stemmed Tokens")
        st.caption(
            "Stemming chops words down to a rough root. It is quick but can look a little strange."
        )
        st.code(format_list(results["stemmed"]), language="text")

        st.subheader("Lemmatized Tokens")
        st.caption(
            "Lemmatization finds the proper dictionary form of each word, so it looks cleaner."
        )
        st.code(format_list(results["lemmas"]), language="text")

        comparison_df = pd.DataFrame(
            {
                "Original word": results["tokens_no_sw"],
                "Stemmed": results["stemmed"],
                "Lemmatized": results["lemmas"],
            }
        )

        st.markdown("#### Compare each token")
        st.caption(
            "Use this table to see how a word changes after stemming and lemmatization."
        )
        st.dataframe(comparison_df, use_container_width=True)
else:
    st.info("Press **Run NLP Pipeline** to see each preprocessing step.")
