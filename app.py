import streamlit as st
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from transformers import BertTokenizer, BertModel
import json
import numpy as np
import torch
import re
from PyPDF2 import PdfReader
import openai
from openai import OpenAI

# Set OpenAI API key (use environment variable for security)
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
# Load models from Hugging Face Hub
@st.cache_resource
def load_sentence_transformer_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource
def load_bert_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

embedder = load_sentence_transformer_model()
tokenizer, bert_model = load_bert_model_and_tokenizer()

def sent_embedding(sent):
    tokens = tokenizer.encode_plus(sent, max_length=128, truncation=True,
                                    padding='max_length', return_tensors='pt')
    outputs = bert_model(**tokens)
    embedding = outputs.pooler_output.detach().numpy()
    return embedding

def compute_cosine_similarity(text1, text2):
    sentences1 = [sentence.strip() for sentence in text1.split('.') if sentence.strip()]
    sentences2 = [sentence.strip() for sentence in text2.split('.') if sentence.strip()]

    embeddings1 = embedder.encode(sentences1)
    embeddings2 = embedder.encode(sentences2)

    matched_sentences2 = set()
    similar_sentences = []
    different_sentences = []

    for i, embedding1 in enumerate(embeddings1):
        match_found = False
        for j, embedding2 in enumerate(embeddings2):
            similarity_score = cosine_similarity([embedding1], [embedding2])[0][0]
            if similarity_score > 0.7 and j not in matched_sentences2:
                matched_sentences2.add(j)
                similar_sentences.append({"similar-sentence": sentences1[i]})
                match_found = True
                break
        if not match_found:
            different_sentences.append({"different-sentence": sentences1[i]})

    unmatched_sentences2 = [sentences2[j] for j in range(len(sentences2)) if j not in matched_sentences2]
    for sentence in unmatched_sentences2:
        different_sentences.append({"different-sentence": sentence})

    overall_embedding1 = embedder.encode([text1])[0]
    overall_embedding2 = embedder.encode([text2])[0]
    overall_similarity_score = cosine_similarity([overall_embedding1], [overall_embedding2])[0][0] * 100

    if not similar_sentences:
        overall_similarity_score = 0.0

    return round(overall_similarity_score, 2), similar_sentences, different_sentences

def compute_euclidean_similarity(text1, text2):
    sentences1 = [sentence.strip() for sentence in text1.split('.') if sentence.strip()]
    sentences2 = [sentence.strip() for sentence in text2.split('.') if sentence.strip()]

    embeddings1 = embedder.encode(sentences1)
    embeddings2 = embedder.encode(sentences2)

    overall_similarity_scores = []
    for embedding1 in embeddings1:
        for embedding2 in embeddings2:
            distance = euclidean_distances([embedding1], [embedding2])[0][0]
            similarity_score = 1 / (1 + distance)
            overall_similarity_scores.append(similarity_score)

    if overall_similarity_scores:
        overall_similarity_score = round((sum(overall_similarity_scores) / len(overall_similarity_scores)) * 100, 2)
    else:
        overall_similarity_score = 0.0

    matched_sentences2 = set()
    similar_sentences = []
    different_sentences = []

    for i, embedding1 in enumerate(embeddings1):
        min_distance = float('inf')
        best_pair = None
        best_j = None
        for j, embedding2 in enumerate(embeddings2):
            distance = euclidean_distances([embedding1], [embedding2])[0][0]
            if distance < min_distance and j not in matched_sentences2:
                min_distance = distance
                best_pair = sentences2[j].strip()
                best_j = j
        if best_pair and min_distance < 0.5:
            matched_sentences2.add(best_j)
            similar_sentences.append({"similar-sentence": sentences1[i]})
        else:
            different_sentences.append({"different-sentence": sentences1[i]})

    unmatched_sentences2 = [sentences2[j] for j in range(len(sentences2)) if j not in matched_sentences2]
    for sentence in unmatched_sentences2:
        different_sentences.append({"different-sentence": sentence})

    return overall_similarity_score, similar_sentences, different_sentences

def compute_similarity_with_bert(text1, text2):
    sentences1 = [sentence.strip() for sentence in text1.split('.') if sentence.strip()]
    sentences2 = [sentence.strip() for sentence in text2.split('.') if sentence.strip()]

    embeddings1 = [sent_embedding(sent) for sent in sentences1]
    embeddings2 = [sent_embedding(sent) for sent in sentences2]

    matched_sentences2 = set()
    similar_sentences = []
    different_sentences = []

    overall_similarity_scores = []
    for i, embedding1 in enumerate(embeddings1):
        match_found = False
        for j, embedding2 in enumerate(embeddings2):
            similarity_score = cosine_similarity(embedding1, embedding2)
            overall_similarity_scores.append(similarity_score.item())
            if similarity_score > 0.8 and j not in matched_sentences2:
                matched_sentences2.add(j)
                similar_sentences.append({"similar-sentence": sentences1[i]})
                match_found = True
                break
        if not match_found:
            different_sentences.append({"different-sentence": sentences1[i]})

    unmatched_sentences2 = [sentences2[j] for j in range(len(sentences2)) if j not in matched_sentences2]
    for sentence in unmatched_sentences2:
        different_sentences.append({"different-sentence": sentence})

    overall_similarity_score = (sum(overall_similarity_scores) / len(overall_similarity_scores)) * 100

    return round(overall_similarity_score, 2), similar_sentences, different_sentences



# Utility function to read text from PDF
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to classify differences using GPT-4
def classify_differences(differences, categories):
    classification_results = []
    prompt_template = (
        "You are an expert classifier. I will give you a sentence and a list of categories. "
        "Assign the sentence to the most appropriate category or categories.\n\n"
        "Sentence: {sentence}\n"
        "Categories: {categories}\n"
        "Answer: "
    )
    for diff in differences:
        prompt = prompt_template.format(
            sentence=diff["different-sentence"],
            categories=", ".join(categories)
        )
        
        response = client.chat.completions.create(
            model = "gpt-4o",
            messages = [
                {"role": "user", "content": prompt}
            ]
        )
        classified_category = response.choices[0].message.content
        classification_results.append(
            {"sentence": diff["different-sentence"], "category": classified_category}
        )
    return classification_results

# Streamlit app

import streamlit as st

def classify_differences(different_sentences, categories):
    """Placeholder function to classify differences based on categories."""
    # Replace with actual classification logic
    classifications = []
    for i, sentence in enumerate(different_sentences):
        classifications.append({"sentence": sentence, "category": categories[i % len(categories)]})
    return classifications

st.title("Document Similarity Comparison")
st.write("Upload two documents to compare their similarity and classify differences.")

uploaded_files = st.file_uploader("Upload two documents", type=["txt", "pdf"], accept_multiple_files=True)
algo = st.selectbox("Select a similarity algorithm", ["Cosine", "Euclidean", "BERT Cosine"])

# Predefined categories
categories_input = 'legal rights', 'obligations', 'liabilities', 'penalties', 'compliance requirements'

# categories = ["Grammar", "Formatting", "Content", "Spelling"]
st.write("### Categories for Classification")
st.write(categories_input)

if st.button("Compare Documents"):
    if uploaded_files and len(uploaded_files) == 2:
        file1, file2 = uploaded_files


        # Read the first document
        if file1.type == "application/pdf":
            text1 = read_pdf(file1)
        else:
            text1 = file1.read().decode('utf-8')


        # Read the second document
        if file2.type == "application/pdf":
            text2 = read_pdf(file2)
        else:
            text2 = file2.read().decode('utf-8')

        if algo == "Cosine":
            score, similar_sentences, different_sentences = compute_cosine_similarity(text1, text2)
        elif algo == "Euclidean":
            score, similar_sentences, different_sentences = compute_euclidean_similarity(text1, text2)
        elif algo == "BERT Cosine":
            score, similar_sentences, different_sentences = compute_similarity_with_bert(text1, text2)

        st.write(f"Overall Similarity Score: {score}%")

        # st.write("### Similar Sentences")
        # for sim in similar_sentences:
        #     st.write(sim["similar-sentence"])

        # st.write("### Different Sentences")
        # for diff in different_sentences:
        #     st.write(diff["different-sentence"])
        
        # Ensure categories_input is a string
        if isinstance(categories_input, tuple):
            categories_input = ",".join(categories_input)  # Or categories_input[0] if only the first element matters

        # Now split and process
        categories = [cat.strip() for cat in categories_input.split(",")]

        if categories_input:
            # categories = [cat.strip() for cat in categories_input.split(",")]
            st.write("### Classifying Differences")
            classifications = classify_differences(different_sentences, categories)
            for classification in classifications:
                st.write(f"Sentence: {classification['sentence']}")
                st.write(f"Category: {classification['category']}")
                st.write("---")
    else:
        st.error("Please upload both documents before comparing.")


# st.title("Document Similarity Comparison")
# st.write("Upload two documents to compare their similarity using different algorithms.")

# uploaded_file1 = st.file_uploader("Upload the first document", type=["txt", "pdf"])
# uploaded_file2 = st.file_uploader("Upload the second document", type=["txt", "pdf"])

# algo = st.selectbox("Select a similarity algorithm", ["Cosine", "Euclidean", "BERT Cosine"])

# categories_input = st.text_input(
#     "Enter categories (comma-separated) for classifying differences",
#     placeholder="e.g., Grammar, Formatting, Content, Spelling"
# )

# if st.button("Compare Documents"):
#     if uploaded_file1 and uploaded_file2:
#         if uploaded_file1.type == "application/pdf":
#             text1 = read_pdf(uploaded_file1)
#         else:
#             text1 = uploaded_file1.read().decode('utf-8')

#         if uploaded_file2.type == "application/pdf":
#             text2 = read_pdf(uploaded_file2)
#         else:
#             text2 = uploaded_file2.read().decode('utf-8')

#         if algo == "Cosine":
#             score, similar_sentences, different_sentences = compute_cosine_similarity(text1, text2)
#         elif algo == "Euclidean":
#             score, similar_sentences, different_sentences = compute_euclidean_similarity(text1, text2)
#         elif algo == "BERT Cosine":
#             score, similar_sentences, different_sentences = compute_similarity_with_bert(text1, text2)

#         st.write(f"Overall Similarity Score: {score}%")

#         st.write("### Similar Sentences")
#         for sim in similar_sentences:
#             st.write(sim["similar-sentence"])

#         st.write("### Different Sentences")
#         for diff in different_sentences:
#             st.write(diff["different-sentence"])

#         if categories_input:
#             categories = [cat.strip() for cat in categories_input.split(",")]
#             st.write("### Classifying Differences")
#             classifications = classify_differences(different_sentences, categories)
#             for classification in classifications:
#                 st.write(f"Sentence: {classification['sentence']}")
#                 st.write(f"Category: {classification['category']}")
#                 st.write("---")
#     else:
#         st.error("Please upload both documents before comparing.")
