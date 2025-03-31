from model import translate
from model import evaluate_translations
from preprocess import preprocess
from data import df
import streamlit as st

preprocess(df)

st.title("Educational Translator App")
# Test Translation
sample_text = st.text_input("Enter an English sentence to translate")
# sample_text = "My name is Kshitiz."

translate_button = st.button("Translate")

if translate_button:
    hindi_translation = translate(sample_text, src_lang="eng_Latn", tgt_lang="hin_Deva")
    gujarati_translation = translate(sample_text, src_lang="eng_Latn", tgt_lang="guj_Gujr")
    marathi_translation = translate(sample_text, src_lang="eng_Latn", tgt_lang="mar_Deva")

    st.write("Hindi Translation:", hindi_translation)
    st.write("Gujarati Translation:", gujarati_translation)
    st.write("Marathi Translation:", marathi_translation)


evaluate_error_button = st.button("Evaluate Errors")

if evaluate_error_button:
    # Collect translations for evaluation
    translations = [translate(text, "eng_Latn", "hin_Deva") for text in df["English"].tolist()]
    references = df["Hindi"].tolist()

    # Compute evaluation metrics
    eval_scores = evaluate_translations(references, translations)
    st.write("Evaluation Scores:", eval_scores)

# # Error Analysis
# for eng, ref, trans in zip(df["English"], references, translations):
#     print(f"\nEnglish: {eng}")
#     print(f"Reference: {ref}")
#     print(f"Generated: {trans}")
#     print("Errors: ", "[Identify mistranslations, loss of meaning, grammar issues]")