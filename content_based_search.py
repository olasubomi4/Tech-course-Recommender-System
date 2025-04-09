import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
import re
import joblib

courses_data = pd.read_csv('./Course_info.csv')


recompute_cached_data=False

def search_course(query,language=None,free=None, category=None, sub_category=None,recompute_cached_data=False,top_n=5):

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

    courses_data['headline'] = courses_data['headline'].astype(str)

    courses_data['title_clean'] = courses_data['title'].apply(clean_text)

    courses_data['headline_clean'] = courses_data['headline'].apply(clean_text)

    courses_index = pd.Series(courses_data.index, index=courses_data['title']).drop_duplicates()

    # Vectorize course titles and headlines
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def re_compute_title_and_headline_embedding():
        title_count_embeddings = model.encode(courses_data['title_clean'].tolist())
        headline_count_embeddings = model.encode(courses_data['headline_clean'].tolist())
        course_id_to_index = {course_id: idx for idx, course_id in enumerate(courses_data["id"])}
        index_to_course_id = {idx: course_id for idx, course_id in enumerate(courses_data["id"])}
        joblib.dump(title_count_embeddings, "cache/content-based/search-content/title_embeddings.joblib")
        joblib.dump(headline_count_embeddings, "cache/content-based/search-content/headline_embeddings.joblib")
        joblib.dump(course_id_to_index, "cache/content-based/search-content/course_id_to_index.joblib")
        joblib.dump(index_to_course_id, "cache/content-based/search-content/index_to_course_id.joblib")
        return title_count_embeddings, headline_count_embeddings,course_id_to_index,index_to_course_id

    def retreive_title_and_headline_embedding_from_cache():
        title_count_embeddings=joblib.load("cache/content-based/search-content/title_embeddings.joblib")
        headline_count_embeddings=joblib.load("cache/content-based/search-content/headline_embeddings.joblib")
        course_id_to_index=joblib.load("cache/content-based/search-content/course_id_to_index.joblib")
        index_to_course_id=joblib.load("cache/content-based/search-content/index_to_course_id.joblib")
        return title_count_embeddings, headline_count_embeddings,course_id_to_index,index_to_course_id

    def get_embeddings():
        if recompute_cached_data:
            return re_compute_title_and_headline_embedding()
        else:
            return retreive_title_and_headline_embedding_from_cache()

    title_count_embeddings,headline_count_embeddings,course_id_to_index,index_to_course_id= get_embeddings()


    # Assign weights to title and headline similarities for combined similarity
    # def search_courses(query, model, top_n=5, title_weight=0.7, headline_weight=0.3):
    #     query_vector = model.encode([query])
    #
    #     title_sim_scores = cosine_similarity(query_vector, title_count_embeddings)[0]
    #     headline_sim_scores = cosine_similarity(query_vector, headline_count_embeddings)[0]
    #
    #     # Combine similarities of title and headline
    #     combined_scores = (title_weight * title_sim_scores) + (headline_weight * headline_sim_scores)
    #
    #     similar_indices = np.argsort(combined_scores)[::-1][:top_n]
    #     index_to_filter_courses_index= [index_to_course_id[top_indice] for top_indice in similar_indices]
    #     recommended_courses = courses_data[courses_data['id'].isin(index_to_filter_courses_index)]
    #     print("Recommendations done for", query)
    #     return recommended_courses[['title', 'headline', 'category', 'subcategory']]
    #     # return courses_data.iloc[similar_indices][['title', 'headline']]


    def search_courses_by_category(query, model,language=None,free=None, category=None, sub_category=None, top_n=5, title_weight=0.7, headline_weight=0.3):
        query_vector = model.encode([query])

        filtered_courses = courses_data
        if category:
            filtered_courses = filtered_courses[filtered_courses['category'].str.lower() == category.lower()]
        if sub_category:
            filtered_courses = filtered_courses[filtered_courses['subcategory'].str.lower() == sub_category.lower()]
        if free:
            filtered_courses = filtered_courses[filtered_courses['is_paid'] !=free]
        if language:
            filtered_courses = filtered_courses[filtered_courses['language'].str.lower() == language.lower()]

        if filtered_courses.empty:
            return pd.DataFrame(columns=['title', 'headline'])

        # title_matrix = model.encode(filtered_courses['title_clean'].tolist())
        # headline_matrix = model.encode(filtered_courses['headline_clean'].tolist())
        a=title_count_embeddings
        title_sim = cosine_similarity(query_vector, title_count_embeddings)[0]
        headline_sim =  cosine_similarity(query_vector,headline_count_embeddings)[0]

        combined_score = title_weight * title_sim + headline_weight * headline_sim
        top_indices = np.argsort(combined_score)[::-1][:top_n]

        index_to_filter_courses_index =[index_to_course_id[top_indice] for top_indice in top_indices]

        recommended_courses = filtered_courses[filtered_courses['id'].isin(index_to_filter_courses_index)]

        print("Recommendations done for", query)
        if category:
            print("Category:", category)
        if sub_category:
            print("Sub Category:", sub_category)

        return recommended_courses


    # # Example: Search for courses related to "Cooking"
    # print(search_courses("Cooking", model, top_n=5))
    #
    # # Example: Search for courses related to "Machine Learning"
    # print(search_courses("Machine Learning", model, top_n=5))

    # Example: Search for courses related to "Excel"
    # print(search_courses("Excel for Experts", model, top_n=5))

    # # Example: Search for courses related to "Excel"
    # print(search_courses_by_category("Excel for Experts", model, category='Office Productivity', top_n=5,free=True))
    #
    # Example: Search for courses related to "Excel"
    # print(search_courses_by_category("Machine learning", model, top_n=5,language="english"))
    result=search_courses_by_category(query, model, top_n=top_n,language=language, free=free, category=category, sub_category=sub_category)
    print(result)
    return result

    # # Example: Search for courses related to "Finance and Technology"
    # print(search_courses_by_category("Learn Algo Trading with Finance Strategies", model, category='Technology', top_n=5))

# import time
# start = time.time()
# search_course("Machine learning",language="English",recompute_cached_data=False)
# end = time.time()
# print(f"recompute embedding {end - start}")
#
# start = time.time()
# search_course("Machine learning",language="English",recompute_cached_data=False)
# end = time.time()
# print(f"recompute embedding {end - start}")