import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# Titre de l'application
st.title("🧠 Analyse de Sentiment Amazon (NLP)")

st.markdown("Saisis un **avis client** et découvre s'il est **positif**, **négatif** ou **neutre**.")

# Zone de saisie de texte
user_input = st.text_area("✍️ Écris ton avis ici :", height=150)

# Charger le modèle et tokenizer
@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Fonction d'analyse de sentiment
def predict_sentiment(text):
    # Nettoyage basique
    text = text.strip().lower()

    # Tokenisation
    encoded_input = tokenizer(text, return_tensors='pt')
    
    # Prédiction
    with torch.no_grad():
        output = model(**encoded_input)
    
    scores = output.logits[0].numpy()
    scores = softmax(scores)

    # Labels du modèle
    labels = ['Negative', 'Neutral', 'Positive']
    sentiment = labels[scores.argmax()]
    confidence = scores.max()

    return sentiment, confidence

# Bouton pour lancer l'analyse
if st.button("🔍 Analyser le sentiment") and user_input.strip() != "":
    with st.spinner("Analyse en cours..."):
        sentiment, confidence = predict_sentiment(user_input)
        st.success(f"✅ Sentiment détecté : **{sentiment}**")
        st.metric(label="Score de confiance", value=f"{confidence:.2%}")
