# Natural Language Processing (NLP) Essentials

**Author:** Marouan Taleb  
## Project Overview

This project serves as a comprehensive practical implementation of Natural Language Processing (NLP) fundamentals. It is designed as a structured learning resource and technical reference that walks through the entire NLP pipeline, making complex concepts accessible and actionable.

The work covers every critical stage of NLP, starting with essential text preprocessing techniques and progressing to advanced topics like topic modeling and transformer-based sentiment analysis. Each notebook bridges theory with real-world Python implementation.

## Project Structure

The project is organized into modular notebooks, each focusing on a specific aspect of the NLP workflow:

### 1. Text Preprocessing
- **`lowercase.ipynb`**: Text normalization techniques.
- **`stopwords.ipynb`**: Handling and removing common stopwords.
- **`regular_expressions.ipynb`**: Pattern matching and text cleaning using Regex.
- **`tokenizing_text.ipynb`**: Breaking text into meaningful units (tokens).
- **`stemming.ipynb`**: Reducing words to their root form (Stemming).
- **`lemmatization.ipynb`**: Advanced word reduction using vocabulary and morphological analysis (Lemmatization).
- **`preprocessing_practical.ipynb`**: A practical exercise combining all preprocessing steps.

### 2. Feature Extraction
- **`n_grams.ipynb`**: Capturing context with N-gram models.
- **`bag_of_words.ipynb`**: Converting text to numerical vectors using BoW.
- **`tfidf.ipynb`**: Importance weighting with Term Frequency-Inverse Document Frequency.

### 3. Advanced NLP Tasks
- **`pos_tagging.ipynb`**: Part-of-Speech tagging using spaCy.
- **`ner.ipynb`**: Named Entity Recognition for information extraction.
- **`information_extraction_practical.ipynb`**: Hands-on information extraction tasks.

### 4. Topic Modeling
- **`lda_topic_modeling.ipynb`**: Latent Dirichlet Allocation for discovering abstract topics.
- **`lsa_topic_modeling.ipynb`**: Latent Semantic Analysis for dimensionality reduction and topic finding.

### 5. Sentiment Analysis
- **`rule_based_sentiment.ipynb`**: Sentiment analysis using TextBlob and VADER.
- **`pretrained_transformer_models.ipynb`**: Leveraging state-of-the-art HuggingFace Transformers.
- **`sentiment_analysis_practical.ipynb`**: Real-world sentiment analysis application.

### 6. Classification & Pipeline
- **`custom_classifier.ipynb`**: Building custom text classifiers using Scikit-learn.
- **`final_pipeline_practical.ipynb`**: A complete end-to-end NLP pipeline integrating multiple components.

## Technologies Used

- **Python 3.11**
- **NLTK**: Natural Language Toolkit for symbolic and statistical NLP.
- **spaCy**: Industrial-strength Natural Language Processing.
- **Scikit-learn**: Machine learning library for classification and feature extraction.
- **Gensim**: Topic modeling and document similarity.
- **TextBlob**: Simplified text processing.
- **Transformers (HuggingFace)**: State-of-the-art pre-trained models.
- **Pandas & Matplotlib/Seaborn**: Data manipulation and visualization.

## Setup & Installation

To replicate this environment, you can use the following Conda commands:

```bash
conda create --name nlp_course_env python=3.11
conda activate nlp_course_env
pip install nltk==3.9.1 pandas==2.2.3 matplotlib==3.10.0 spacy==3.8.3 textblob==0.18.0.post0 vaderSentiment==3.3.2 transformers==4.47.1 scikit-learn==1.6.0 gensim==4.3.3 seaborn==0.13.2 torch==2.5.1 ipywidgets==8.1.5
python -m spacy download en_core_web_sm
```

## License

This project is created by Marouan Taleb for educational purposes.
