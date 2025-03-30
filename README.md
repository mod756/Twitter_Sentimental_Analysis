# Twitter Sentiment Analysis with Model Comparison: Logistic Regression, Naive Bayes, Neural Network, and BERT

## Overview
This project implements a Twitter Sentiment Analysis pipeline using the NLTK `twitter_samples` dataset, preprocesses tweet text, and compares four models—Logistic Regression, Naive Bayes, Neural Network, and BERT—to classify sentiments as positive or negative. The analysis evaluates the effectiveness of traditional machine learning, neural networks, and transformer-based models on Twitter data.

## Features
### Text Preprocessing with NLTK:
- Removes URLs, retweet markers, hashtags, mentions, and stopwords.
- Applies Porter Stemming to normalize words.
- Uses TweetTokenizer for Twitter-specific tokenization.

### Sentiment Analysis Models:
- **Logistic Regression:** A linear classifier using TF-IDF features.
- **Naive Bayes:** A probabilistic model leveraging word frequencies.
- **Neural Network:** A feedforward neural network trained on vectorized text.
- **BERT:** A pre-trained transformer model fine-tuned for sentiment classification.

### Data Visualization:
- **Sentiment Distribution:** Bar plot showing counts of positive (1) and negative (0) tweets.
- **Word Frequency Analysis:** Bar plot of the top 50 most frequent words in the cleaned corpus.
- **Performance Metrics:** Accuracy is used to compare model performance.

## Algorithm Workflow
1. Load the `twitter_samples` dataset from NLTK (`positive_tweets.json` and `negative_tweets.json`).
2. Preprocess tweets with the `process_tweet` function:
   - Remove noise (URLs, hashtags, mentions, etc.).
   - Tokenize, remove stopwords, and apply stemming.
3. Create a Pandas DataFrame with original tweets, cleaned text, and sentiment labels (1 = positive, 0 = negative).
4. Split the data into training and testing sets.
5. Train and evaluate the models:
   - **Logistic Regression:** Fit on TF-IDF vectors.
   - **Naive Bayes:** Fit on TF-IDF features.
   - **Neural Network:** Train on vectorized text (TF-IDF) with dense layers.
   - **BERT:** Fine-tune using tokenized inputs and the Hugging Face Transformers library.
6. Visualize sentiment distribution and word frequency.
7. Evaluate and compare model accuracy.

## Usage
Run the Jupyter Notebook to preprocess the data, train the models, and evaluate their performance:
```bash
jupyter notebook Twitter_Sentiment_Analysis_Model_Comparison.ipynb
```

### Steps in the Notebook:
1. **Data Loading & Preprocessing:** Load tweets, clean text, and store in a DataFrame.
2. **Feature Extraction:** Convert cleaned text into features (TF-IDF for traditional models, tokenized inputs for BERT).
3. **Model Training & Evaluation:**
   - Train Logistic Regression and Naive Bayes using Scikit-Learn.
   - Train a Neural Network using TensorFlow/Keras.
   - Fine-tune BERT using Hugging Face Transformers.
4. **Visualization:** Generate sentiment distribution and word frequency plots.
5. **Results:** Compute and compare accuracy for each model.

## Dataset
### Source:
- NLTK `twitter_samples` corpus.

### Composition:
- **5,000** positive tweets (sentiment = 1).
- **5,000** negative tweets (sentiment = 0).
- **Total Size:** 10,000 tweets, balanced across sentiments.

## Technologies Used
- **Python:** Core programming language.
- **NLTK:** Text preprocessing, tokenization, and stemming.
- **Pandas & NumPy:** Data manipulation and numerical operations.
- **Scikit-Learn:** Logistic Regression, Naive Bayes, and evaluation metrics.
- **TensorFlow/Keras:** Neural Network implementation.
- **Hugging Face Transformers:** BERT model and tokenization.
- **Matplotlib & Seaborn:** Data visualization.
- **Regular Expressions (`re`)**: Text cleaning.
- **Jupyter Notebook:** Interactive environment.

## Results
### Model Performance:
- **Logistic Regression:** Accuracy: **70.05%**
- **Naive Bayes:** Accuracy: **75.4%**
- **Neural Network:** Accuracy: **70.74%**
- **BERT Model:** Accuracy: **49.4%**

### Observations:
- Naive Bayes outperforms all models with an accuracy of **75.4%**, likely due to its effectiveness with sparse text data and the simplicity of the dataset.
- Logistic Regression (**70.05%**) and Neural Network (**70.74%**) perform moderately well, with Logistic Regression slightly ahead.
- BERT (**49.4%**) underperforms compared to expectations, falling below all other models.

### Why BERT Underperformed:
- **Small Dataset Size:** The `twitter_samples` dataset (10,000 tweets) may be insufficient for BERT to leverage its full potential, as transformer models typically excel with larger datasets (e.g., millions of samples).
- **Overfitting Due to Fine-Tuning:** Limited training data and excessive fine-tuning epochs could lead to overfitting, reducing generalization on the test set.

## Future Work
- **Address BERT’s underperformance by:**
  - Using a larger dataset (e.g., real-time Twitter data via API).
  - Avoiding stemming and preserving raw text for BERT input.
  - Tuning hyperparameters (e.g., learning rate, batch size, epochs).
- **Explore additional metrics** (precision, recall, F1-score) for a comprehensive evaluation.
- **Experiment with ensemble methods** combining Naive Bayes and BERT predictions.
- **Test alternative transformer models** (e.g., DistilBERT, RoBERTa) for efficiency and accuracy.

## Contributors
- **KV Modak Prasanna Kumar** (23bcs067@iiitdwd.ac.in)


