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
