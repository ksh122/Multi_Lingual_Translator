import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from data import df


def preprocess(df):
# Conducting Exploratory Data Analysis (EDA)
# Sentence length distribution
    df["English_Length"] = df["English"].apply(lambda x: len(x.split()))
    df["Hindi_Length"] = df["Hindi"].apply(lambda x: len(x.split()))
    df["Gujarati_Length"] = df["Gujarati"].apply(lambda x: len(x.split()))
    df["Marathi_Length"] = df["Marathi"].apply(lambda x: len(x.split()))


    # Streamlit UI
    st.title("Sentence Length Distribution Across Languages")
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot sentence length distributions
    # plt.figure(figsize=(10, 5))
    sns.histplot(df["English_Length"], bins=10, kde=True, label="English", color="blue")
    sns.histplot(df["Hindi_Length"], bins=10, kde=True, label="Hindi", color="red", alpha=0.6)
    sns.histplot(df["Gujarati_Length"], bins=10, kde=True, label="Gujarati", color="green", alpha=0.6)
    sns.histplot(df["Marathi_Length"], bins=10, kde=True, label="Marathi", color="purple", alpha=0.6)
    plt.legend()
    plt.title("Sentence Length Distribution Across Languages")
    plt.xlabel("Sentence Length")
    plt.ylabel("Frequency")
    # plt.show()
    st.pyplot(fig)

    # Function to calculate vocabulary richness (unique words)
    def vocabulary_richness(texts):
        words = [word for text in texts for word in text.split()]
        return len(set(words)), len(words)

    # Compute vocabulary richness for each language
    vocab_stats = {
        "Language": ["English", "Hindi", "Gujarati", "Marathi"],
        "Unique_Words": [],
        "Total_Words": [],
        "Richness": []
    }

    for lang in ["English", "Hindi", "Gujarati", "Marathi"]:
        unique_words, total_words = vocabulary_richness(df[lang])
        vocab_stats["Unique_Words"].append(unique_words)
        vocab_stats["Total_Words"].append(total_words)
        vocab_stats["Richness"].append(unique_words / total_words)

    # Convert to DataFrame for better visualization
    vocab_df = pd.DataFrame(vocab_stats)
    st.header("Vocab Stats")
    st.write(vocab_df)