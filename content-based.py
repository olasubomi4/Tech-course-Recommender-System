import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
import re
import neattext.functions as nfx

comments_data = pd.read_csv('./Comments.csv')
courses_data = pd.read_csv('./Course_info.csv')

courses_data = courses_data.head(len(courses_data)//4)
comments_data=comments_data.head(len(comments_data)//4)

print('Courses Data Shape is', courses_data.shape)
print('Comments Data Shape is', comments_data.shape)

# Remove stopwords and special characters
courses_data['title'].value_counts()
courses_data['headline'].value_counts()

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

courses_data['headline'] = courses_data['headline'].astype(str)
courses_data['headline_clean'] = courses_data['headline'].apply(clean_text)

courses_index = pd.Series(courses_data.index, index=courses_data['title']).drop_duplicates()

# Vectorize course titles and headlines
model = SentenceTransformer('all-MiniLM-L6-v2')

title_count_matrix = model.encode(courses_data['title_clean'].tolist())
headline_count_matrix = model.encode(courses_data['headline_clean'].tolist())

# Assign weights to title and headline similarities for combined similarity
def search_courses(query, model, top_n=5, title_weight=0.7, headline_weight=0.3):
    query_vector = model.encode([query])

    title_sim_scores = cosine_similarity(query_vector, title_count_matrix)[0]
    headline_sim_scores = cosine_similarity(query_vector, headline_count_matrix)[0]

    # Combine similarities of title and headline
    combined_scores = (title_weight * title_sim_scores) + (headline_weight * headline_sim_scores)

    similar_indices = np.argsort(combined_scores)[::-1][:top_n]

    print("Recommendations done for", query)

    return courses_data.iloc[similar_indices][['title', 'headline']]


def search_courses_by_category(query, model, category=None, sub_category=None, top_n=5, title_weight=0.6, headline_weight=0.4):
    query_vector = model.encode([query])

    filtered_courses = courses_data
    if category:
        filtered_courses = filtered_courses[filtered_courses['category'] == category]
    if sub_category:
        filtered_courses = filtered_courses[filtered_courses['sub_category'] == sub_category]

    if filtered_courses.empty:
        return pd.DataFrame(columns=['title', 'headline'])

    title_matrix = model.encode(filtered_courses['title_clean'].tolist())
    headline_matrix = model.encode(filtered_courses['headline_clean'].tolist())

    title_sim = cosine_similarity(query_vector, title_matrix)[0]
    headline_sim = cosine_similarity(query_vector, headline_matrix)[0]

    combined_score = title_weight * title_sim + headline_weight * headline_sim
    top_indices = np.argsort(combined_score)[::-1][:top_n]

    print("Recommendations done for", query)
    if category:
        print("Category:", category)
    if sub_category:
        print("Sub Category:", sub_category)

    return filtered_courses.iloc[top_indices][['title', 'headline', 'category', 'sub_category']]


# Example: Search for courses related to "Cooking"
print(search_courses("Cooking", model, top_n=5))

# Example: Search for courses related to "Machine Learning"
print(search_courses("Machine Learning", model, top_n=5))

# Example: Search for courses related to "Excel"
print(search_courses("Excel for Experts", model, top_n=5))

# Example: Search for courses related to "Excel"
print(search_courses_by_category("Excel for Experts", model, category='Office Productivity', top_n=5))

# Example: Search for courses related to "Finance and Technology"
print(search_courses_by_category("Learn Algo Trading with Finance Strategies", model, category='Technology', top_n=5))