# Streamlit application for SMS/Email Spam Classification
# This script deploys a web interface for the refined Spam Classifier model.
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize PorterStemmer for stemming (reducing words to their root form)
ps=PorterStemmer()

def transform_text(text):
    '''
    Preprocesses input text by lowercasing, tokenizing, removing stopwords/punctuation, and stemming.
    '''
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Tokenize
    text = nltk.word_tokenize(text)
    
    # 3. Keep only alphanumeric tokens
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # 4. Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # 5. Apply Stemming
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

# Load the pre-trained TF-IDF Vectorizer and the saved Machine Learning model
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

# Create the main user interface with a title and text input area
st.title("SMS/Email Spam Classifier")


input_sms=st.text_area("Enter the message")

# Trigger prediction when the button is clicked
if st.button('Predict'):
    # 1. Preprocess: Apply the same cleaning steps as in training
    transformed_sms=transform_text(input_sms)

    # 2. Vectorize: Convert text to numerical features using the loaded TF-IDF
    vector_input=tfidf.transform([transformed_sms])

    # 3. Predict: Use the classifier to determine if it is Spam (1) or Ham (0)
    result=model.predict(vector_input)


    # 4. Display Result: Show the classification output to the user
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")

