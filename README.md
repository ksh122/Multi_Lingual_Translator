# Multilingual Translation using Transformers

## Overview
This project focuses on translating English text into Hindi, Gujarati, and Marathi using Transformer models. It involves data preparation, model training, evaluation, and visualization using Streamlit.

## Features
- Preprocessing and exploration of multilingual datasets.
- Training a Transformer-based model for translation.
- Evaluation using BLEU, METEOR, and chrF metrics.
- Visualization of sentence length distributions.
- Streamlit web app for interactive translation.

## Installation
### Clone the Repository
```bash
git clone https://github.com/ksh122/Multi_Lingual_Translator.git
```
### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Streamlit App
```bash
streamlit run app.py
```

## Evaluation Metrics
- **BLEU**: Measures translation precision.
- **METEOR**: Accounts for synonymy and stemming.
- **chrF**: Character-level evaluation for multilingual text.

## Future Improvements
- Fine-tune for better fluency and accuracy.
- Incorporate domain-specific knowledge.
- Extend to additional languages.

## Author
Kshitiz Agrahari
