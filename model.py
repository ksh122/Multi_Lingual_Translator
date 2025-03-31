from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sacrebleu import corpus_bleu, corpus_chrf
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import torch
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from data import df
import streamlit as st

# Load model and tokenizer
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_function(examples):
    inputs = tokenizer(examples["English"], padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(examples["Hindi"], padding="max_length", truncation=True, max_length=128)
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": targets["input_ids"]}

# Convert dataset to Hugging Face format
dataset = Dataset.from_pandas(df)
tokenized_datasets = dataset.map(preprocess_function, batched=True)

def collate_fn(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch])
    attention_mask = torch.tensor([item["attention_mask"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# DataLoader setup
dataloader = DataLoader(tokenized_datasets, batch_size=8, shuffle=True, collate_fn=collate_fn)

# Training setup
device = "cuda" if torch.cuda.is_available() else "cpu"
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
st.write("Training in progress for 3 epochs.")
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    st.write(f"Epoch {epoch+1} completed. Loss: {loss.item()}")

# Translation function
def translate(text, src_lang="eng_Latn", tgt_lang="hin_Deva"):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang))
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# Evaluation function
def evaluate_translations(references, translations):
    bleu = corpus_bleu(translations, [references]).score
    chrf = corpus_chrf(translations, [references]).score
    meteor = sum(meteor_score([word_tokenize(ref)], word_tokenize(trans)) for ref, trans in zip(references, translations)) / len(references)
    return {"BLEU": bleu, "chrF": chrf, "METEOR": meteor}