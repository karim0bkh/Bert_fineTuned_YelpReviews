import streamlit as st 
import numpy as np 
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@st.cache(allow_output_mutation=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("karimbkh/BERT_fineTuned_Sentiment_Classification_Yelp")
    model = AutoModelForSequenceClassification.from_pretrained("karimbkh/BERT_fineTuned_Sentiment_Classification_Yelp")
    return tokenizer, model

tokenizer, model = load_model()

st.title("Sentiment Analysis")
st.header("Enter a sentence to get the sentiment")
text = st.text_input("Text")
if text:
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    probs = probs.detach().numpy()
    probs = np.squeeze(probs)
    sentiment = np.argmax(probs)
    if sentiment == 0:
        sentiment_ = "Negative"
    else:
        sentiment_ = "Positive"
    st.write(f"Sentiment: {sentiment_}")
    st.write(f"Confidence: {probs[sentiment]}")



st.header("How to Use in Your Own Code")
st.subheader("Install the Transformers Library:")
st.code("pip install transformers")

st.subheader("Load the Model in Python:")
st.code("""
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("karimbkh/BERT_fineTuned_Sentiment_Classification_Yelp")
model = AutoModelForSequenceClassification.from_pretrained("karimbkh/BERT_fineTuned_Sentiment_Classification_Yelp")
""")

st.subheader("Perform Sentiment Analysis:")
st.code("""
text = "Your input text here"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
probs = outputs[0].softmax(1)
sentiment = np.argmax(probs)
confidence = probs[sentiment]

print(f"Sentiment: {sentiment}")
print(f"Confidence: {confidence}")
""")



st.sidebar.title("Model Information (By Karim Ben Khaled)")
st.sidebar.header("Model Data and Scores")

st.sidebar.subheader("Model Architecture:")
st.sidebar.write("The sentiment analysis model is based on BERT architecture.")

st.sidebar.subheader("Training Data:")
st.sidebar.write("The model is fine-tuned on the Yelp reviews dataset.")

st.sidebar.subheader("Performance Scores:")
st.sidebar.write(f"Precision: 0: 96.1%, 1: 98.2%")
st.sidebar.write(f"Recall: 0: 98.3%, 1: 96.0%")
st.sidebar.write(f"F1 Score: 0: 97.2%, 1: 97.1%")

st.sidebar.header("Hugging Face Link")
st.sidebar.write("For more details, visit the [Hugging Face model page](https://huggingface.co/karimbkh/BERT_fineTuned_Sentiment_Classification_Yelp).")
