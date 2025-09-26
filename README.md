# YouTube Comment Analysis

## Project Overview

This project performs a comprehensive analysis of YouTube comments from an Excel file. The primary goal is to understand viewer reactions and discussion patterns by identifying key themes, recurring phrases, and frequently used words. This is achieved through a combination of Natural Language Processing (NLP) techniques.

The analysis includes:
1.  **Topic Modeling:** Using Non-Negative Matrix Factorization (NMF) to discover abstract topics from the comments.
2.  **N-gram Analysis:** Identifying the most common two-word (bigram) and three-word (trigram) phrases.
3.  **Word Cloud Visualization:** Creating visual representations of the most frequent words, both for the entire comment section and for each individual topic identified.

## Dataset

The dataset is an Excel file (`youtube.xlsx`) containing YouTube comments. The analysis focuses on the column named `" comment"`.

## Technologies Used

*   Python 3.x
*   pandas
*   scikit-learn
*   NLTK
*   WordCloud
*   Matplotlib

## Setup and Installation

1.  Clone the repository:
    ```bash
    git clone https://your-repository-url.git
    cd your-repository-directory
    ```

2.  Install the required Python libraries:
    ```bash
    pip install pandas openpyxl scikit-learn nltk wordcloud matplotlib
    ```

3.  Download necessary NLTK data:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

## Usage and Code Walkthrough

Place your `youtube.xlsx` file in the same directory as the script or provide the correct file path.

### Step 1: Load the Data

First, we load the Excel file into a pandas DataFrame.

```python
import pandas as pd

file_path = r"C:\Users\manis\Downloads\youtube.xlsx"
df = pd.read_excel(file_path)```

### Step 2: Preprocess the Text

The comments are cleaned and prepared for analysis. This involves converting to lowercase, removing non-alphabetic characters, removing common English stop words, and lemmatizing words to their root form.

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

comments = df[' comment'].dropna().astype(str)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

processed_comments = comments.apply(preprocess_text)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf = vectorizer.fit_transform(processed_comments)

num_topics = 5
nmf_model = NMF(n_components=num_topics, random_state=1, max_iter=1000)
nmf_model.fit(tfidf)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx+1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10
feature_names = vectorizer.get_feature_names_out()
display_topics(nmf_model, feature_names, no_top_words)

from sklearn.feature_extraction.text import CountVectorizer

def get_top_ngrams(corpus, n, top_k=10):
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x, reverse=True)
    return words_freq[:top_k]

top_bigrams = get_top_ngrams(processed_comments, n=2, top_k=10)
print("Top 10 Bigrams:")
print(pd.DataFrame(top_bigrams, columns=['Bigram', 'Frequency']))

print("\n" + "="*30 + "\n")

top_trigrams = get_top_ngrams(processed_comments, n=3, top_k=10)
print("Top 10 Trigrams:")
print(pd.DataFrame(top_trigrams, columns=['Trigram', 'Frequency']))

from wordcloud import WordCloud
import matplotlib.pyplot as plt

all_comments_text = " ".join(comment for comment in processed_comments)

wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(all_comments_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

from wordcloud import WordCloud
import matplotlib.pyplot as plt

feature_names = vectorizer.get_feature_names_out()

for topic_idx, topic in enumerate(nmf_model.components_):
    topic_words = {feature_names[i]: topic[i] for i in topic.argsort()[:-50 - 1:-1]}
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(topic_words)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Topic {topic_idx + 1}')
    plt.axis("off")
    plt.show()
