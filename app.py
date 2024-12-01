import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load Porter Stemmer
ps = PorterStemmer()

# Function to preprocess and transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load models
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Streamlit app for spam classification
st.title("SMS Spam Classifier")

# Decorative UI: Header and Subheader
st.header("Detect Spam Messages")
st.subheader("Enter the message below to determine if it's spam or not:")

# Text area for user input
input_sms = st.text_area("Enter the message")

# Predict button
if st.button('Predict'):
    # Preprocess the input text
    transformed_sms = transform_text(input_sms)
    # Vectorize the preprocessed text
    vector_input = tfidf.transform([transformed_sms])
    # Predict
    result = model.predict(vector_input)[0]
    # Probability of the prediction
    probability = model.predict_proba(vector_input)[0][1]

    # Display prediction with icons and additional styling
    if result == 1:
        st.markdown('<i class="fas fa-times-circle" style="color:red;"></i> <h2 style="color:red;">Spam</h2>', unsafe_allow_html=True)
    else:
        st.markdown('<i class="fas fa-check-circle" style="color:green;"></i> <h2 style="color:green;">Not Spam</h2>', unsafe_allow_html=True)

    # Display probability with increased text size
    st.markdown(f'<h2 style="font-size:24px;">Confidence Score (Spam Probability): {round(probability * 100, 2)}%</h2>', unsafe_allow_html=True)

# Custom CSS styling
st.markdown(
    """
    <style>
    .stTextInput>div>div>textarea {
        border-radius: 10px;
        border: 2px solid #3498db;
        padding: 10px;
        font-size: 16px;
        background-color: #f8f9fa;
    }
    .stButton>button {
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 18px;
        background-color: #4CAF50;
        color: white;
        border: none;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTitle {
        color: white;
        background-color: #343a40;
        padding: 10px 20px;
        border-radius: 10px;
    }
    .stHeader>div {
        color: white;
        background-color: #343a40;
        padding: 10px 20px;
        border-radius: 10px;
    }
    .stMarkdown {
        color: #343a40;
    }
    </style>
    """,
    unsafe_allow_html=True
)





