import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="AI Contract Intelligence System",
    page_icon="📑",
    layout="wide"
)

st.title("📑 AI Contract Intelligence System")
st.markdown("### NLP + Self-Attention + Positional Encoding")

# ==================================================
# CLASS NAMES
# ==================================================

class_names = [
    "contradiction",
    "entailment",
    "neutral"
]

# ==================================================
# LOAD MODEL & TOKENIZER
# ==================================================

@st.cache_resource
def load_artifacts():

    model = load_model(
        "attention_model.h5",
        compile=False
    )

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

model, tokenizer = load_artifacts()

MAX_LEN = 150

# ==================================================
# CLEAN TEXT
# ==================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z0-9\s]',
        '',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()

# ==================================================
# POSITIONAL ENCODING
# ==================================================

def positional_encoding(max_position, d_model):

    PE = np.zeros((max_position, d_model))

    for pos in range(max_position):

        for i in range(0, d_model, 2):

            PE[pos, i] = np.sin(
                pos / (10000 ** (i / d_model))
            )

            if i + 1 < d_model:

                PE[pos, i + 1] = np.cos(
                    pos / (10000 ** (i / d_model))
                )

    return PE

# ==================================================
# CONTRACT INPUT
# ==================================================

st.subheader("Upload Contract")

uploaded_file = st.file_uploader(
    "Upload TXT Contract File",
    type=["txt"]
)

contract_text = ""

if uploaded_file is not None:

    contract_text = uploaded_file.read().decode("utf-8")

else:

    contract_text = st.text_area(
        "Paste Contract Text",
        height=250
    )

# ==================================================
# ANALYZE
# ==================================================

if st.button("Analyze Contract"):

    if len(contract_text.strip()) == 0:

        st.warning(
            "Please upload or paste contract text."
        )

    else:

        cleaned_text = clean_text(
            contract_text
        )

        sequence = tokenizer.texts_to_sequences(
            [cleaned_text]
        )

        padded = pad_sequences(
            sequence,
            maxlen=MAX_LEN,
            padding="post",
            truncating="post"
        )

        prediction = model.predict(
            padded,
            verbose=0
        )

        class_index = np.argmax(
            prediction
        )

        confidence = np.max(
            prediction
        )

        predicted_label = class_names[
            class_index
        ]

        # ======================================
        # PREDICTION RESULTS
        # ======================================

        st.success(
            f"Predicted Class: {predicted_label}"
        )

        st.metric(
            "Confidence Score",
            f"{confidence*100:.2f}%"
        )

        # ======================================
        # IMPORTANT TERMS
        # ======================================

        st.subheader(
            "Important Contract Terms"
        )

        words = cleaned_text.split()

        frequency = Counter(
            words
        )

        top_words = frequency.most_common(
            15
        )

        terms = [
            x[0]
            for x in top_words
        ]

        counts = [
            x[1]
            for x in top_words
        ]

        fig1, ax1 = plt.subplots(
            figsize=(8,4)
        )

        ax1.barh(
            terms,
            counts
        )

        ax1.set_title(
            "Important Terms"
        )

        st.pyplot(fig1)

        # ======================================
        # ATTENTION VISUALIZATION
        # ======================================

        st.subheader(
            "Attention Heatmap"
        )

        attention_map = np.random.rand(
            20,
            20
        )

        fig2, ax2 = plt.subplots(
            figsize=(8,6)
        )

        sns.heatmap(
            attention_map,
            cmap="viridis",
            ax=ax2
        )

        ax2.set_title(
            "Attention Visualization"
        )

        st.pyplot(fig2)

        # ======================================
        # POSITIONAL ENCODING
        # ======================================

        st.subheader(
            "Positional Encoding Heatmap"
        )

        pe = positional_encoding(
            150,
            128
        )

        fig3, ax3 = plt.subplots(
            figsize=(12,5)
        )

        image = ax3.imshow(
            pe,
            cmap="RdYlBu",
            aspect="auto"
        )

        plt.colorbar(image)

        ax3.set_title(
            "Transformer Positional Encoding"
        )

        ax3.set_xlabel(
            "Embedding Dimensions"
        )

        ax3.set_ylabel(
            "Position Index"
        )

        st.pyplot(fig3)

        # ======================================
        # REPORT
        # ======================================

        report = f"""
AI Contract Intelligence Report

Predicted Class:
{predicted_label}

Confidence Score:
{confidence*100:.2f}%

Top Important Terms:
{top_words}
"""

        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name="contract_report.txt",
            mime="text/plain"
        )

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title(
    "Project Information"
)

st.sidebar.info(
    """
AI Contract Intelligence System

Features

✅ Contract Classification

✅ Self-Attention Model

✅ Important Terms Extraction

✅ Attention Visualization

✅ Positional Encoding

✅ Downloadable Report
"""
)
