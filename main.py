import streamlit as st
import numpy
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

def preprocess_text(text):
    words =  text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

model = load_model('simple_rnn_imdb.h5')
    
def predict_sentiment(review):
    preprocess_input = preprocess_text(review)
    prediction = model.predict(preprocess_input)
    sentiment ="Positive" if prediction[0][0] > 0.5 else 'Negative'
    return sentiment , prediction[0][0]

st.title("Sentiment Analyzer")
Text = st.text_area("Enter your review here:")

if st.button('Classify'):
    sentiment,score = predict_sentiment(Text)
    st.write(f"Review : {Text}")
    st.write(f"Sentiment : {sentiment}")
    st.write(f"Prediction : {score:2f}")
else:
    st.write("Enter a movie review ")
