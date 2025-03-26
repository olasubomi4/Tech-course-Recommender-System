import pandas as pd

comments_data = pd.read_csv('Comments.csv')

courses_data = pd.read_csv('Course_info.csv')

courses_data = courses_data.head(len(courses_data)//4)
comments_data=comments_data.head(len(comments_data)//4)



print('Courses Data Shape is', courses_data.shape)
print('Comments Data Shape is', comments_data.shape)


comments_data.head()

courses_data.head()

courses_data['instructor_url'].value_counts()

courses_data['instructor_name'].value_counts()

courses_data['category'].value_counts()

# import neattext.functions as nfx

# Remove stopwords and special characters

courses_data['title'].value_counts()

courses_data['headline'].value_counts()


import nltk
from nltk.corpus import stopwords
import re

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    # Remove special characters (keeping only alphanumeric and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove stopwords
    text = ' '.join(word for word in text.split() if word.lower() not in stop_words)
    return text

courses_data['title_clean'] = courses_data['title'].apply(clean_text)
# courses_data['title_clean'] = courses_data['title_clean'].apply(nfx.remove_special_characters)

courses_data['headline'] = courses_data['headline'].astype(str)
courses_data['headline_clean'] = courses_data['headline'].apply(clean_text)
# courses_data['headline_clean'] = courses_data['headline_clean'].apply(nfx.remove_special_characters)

courses_index = pd.Series(courses_data.index, index=courses_data['title']).drop_duplicates()



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Vectorize course titles and headlines
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

title_count_matrix = model.encode(courses_data['title_clean'].tolist())
# headline_count_matrix = model.encode(courses_data['headline_clean'].tolist())

title_cosine_sim = cosine_similarity(title_count_matrix)
# headline_cosine_sim = cosine_similarity(headline_count_matrix)



title_cosine_sim = cosine_similarity(title_count_matrix)

import numpy as np
def search_courses(query, model,top_n=5):
    # Encode the query using the same model
    query_vector = model.encode([query])

    # Compute similarity between query and all course titles
    similarity_scores = cosine_similarity(query_vector, title_count_matrix)[0]

    # Get top N most relevant courses
    similar_indices = np.argsort(similarity_scores)[::-1][:top_n]

    return courses_data.iloc[similar_indices]['title']


# Example: Search for courses related to "Machine Learning"
print(search_courses("Machine Learning",model, top_n=5))
