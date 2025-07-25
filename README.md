
# Analyse de Sentiment avec NLP (Streamlit App)

Ce projet est une **application web interactive** développée avec **Streamlit** qui permet d'analyser automatiquement le **sentiment d’un avis client** (positif, négatif ou neutre) à l’aide d’un **modèle NLP pré-entraîné**.

---

## Fonctionnalités

- Saisie manuelle d’un avis client.
- Analyse des sentiments à l’aide du modèle `cardiffnlp/twitter-roberta-base-sentiment` (HuggingFace).
- Retour du **sentiment détecté** avec un **score de confiance**.
- Interface web simple et rapide grâce à Streamlit.

---

## Lancer l’application en local

### 1. Cloner le dépôt

```bash
git https://github.com/TINBYTE/NLP-Streamlit.git
cd NLP-Streamlit
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

> Si `requirements.txt` n'existe pas encore, tu peux utiliser :
```bash
pip install streamlit transformers torch scipy
```

### 3. Lancer l’application

```bash
streamlit run app.py
```

---

## Modèle utilisé

Modèle : [`cardiffnlp/twitter-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)  
Ce modèle est basé sur **RoBERTa**, pré-entraîné pour détecter les sentiments dans les textes courts (tweets, avis...).
