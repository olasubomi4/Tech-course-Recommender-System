import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
courses_data = pd.read_csv('./Course_info.csv')
courses_data = courses_data[courses_data["category"].isin(["Development","IT & Software"])]
print('Courses Data Shape is', courses_data.shape)

courses_data = courses_data.head(len(courses_data)//2)

features = ['title', 'is_paid', 'price', 'topic','subcategory','language']

queryId=9463;


def similar_courses(course_id, diversify_recommendations=False,recompute_cached_data=False,top_n=10):
    course_user_likes = courses_data[courses_data['id'] == course_id]["title"]
    def getSimilarityScore():

        if recompute_cached_data:
            return recomputeSimilarityScores()
        return getSimilarityScoresForStorage()


    def getSimilarityScoresForStorage():
        cosine_similarities = joblib.load('cache/content-based/find-similar-contents/cosine_similarity.pkl')
        course_id_to_index= joblib.load('cache/content-based/find-similar-contents/course_id_to_index.pkl')
        index_to_course_id= joblib.load('cache/content-based/find-similar-contents/index_to_course_id.pkl')
        return cosine_similarities, course_id_to_index, index_to_course_id;

    def recomputeSimilarityScores():
        def combine_features(row):
            return row['title'] + " " + str(row['is_paid']) + " " + str(row['price']) + " " + row['topic'] + " " + row['subcategory'] + " " + row['language']

        for feature in features:
            courses_data[feature] = courses_data[feature].fillna('')  # filling all NaNs with blank string

        courses_data["combined_features"] = courses_data.apply(combine_features,
                                           axis=1)  # applying combined_features() method over each rows of dataframe and storing the combined string in "combined_features" column

        # applying combined_features() method over each rows of dataframe and storing the combined string in "combined_features" column

        cv = CountVectorizer()
        count_matrix=cv.fit_transform(courses_data["combined_features"])
        cosine_sim = cosine_similarity(count_matrix)

        course_id_to_index = {course_id: idx for idx, course_id in enumerate(courses_data["id"])}
        index_to_course_id = {idx: course_id for idx, course_id in enumerate(courses_data["id"])}


        joblib.dump(cosine_sim,'cache/content-based/find-similar-contents/cosine_similarity.pkl')
        joblib.dump(index_to_course_id,'cache/content-based/find-similar-contents/index_to_course_id.pkl')
        joblib.dump(course_id_to_index,'cache/content-based/find-similar-contents/course_id_to_index.pkl')

        return cosine_sim,course_id_to_index, index_to_course_id


    import numpy as np


    def get_diverse_recommendations(
            query_index,
            cosine_sim,
            top_n=10,
            similarity_threshold=0.2,
            alpha=0.7
    ):
        """
        Returns a list of `top_n` item indices that are diverse yet similar to the query item.

        Parameters:
        - query_index (int): index of the query item
        - cosine_sim (np.ndarray): cosine similarity matrix
        - top_n (int): number of recommendations to return
        - similarity_threshold (float): minimum similarity required to be considered
        - alpha (float): weight for similarity vs diversity (between 0 and 1)

        Returns:
        - List[int]: indices of recommended items
        """

        # Step 1: Get similarity scores for query item
        similarities_to_query = list(enumerate(cosine_sim[query_index]))

        # Step 2: Filter items based on threshold and exclude self
        candidates = [
            (idx, sim) for idx, sim in similarities_to_query
            if idx != query_index and sim >= similarity_threshold
        ]

        if len(candidates) < top_n:
            print(" Not enough sufficiently similar items. Returning top available.")
            return [index_to_course_id[idx] for idx, _ in sorted(candidates, key=lambda x: x[1], reverse=True)]

        # Step 3: Select the most similar item as c1
        selected = [max(candidates, key=lambda x: x[1])]
        candidate_indices = set(idx for idx, _ in candidates) - {selected[0][0]}

        # Step 4: Pick remaining items using hybrid similarity-diversity scoring
        while len(selected) < top_n:
            best_score = -float('inf')
            best_candidate = None

            for idx in candidate_indices:
                sim_to_query = cosine_sim[query_index][idx]
                sim_to_selected = np.mean([cosine_sim[idx][sel[0]] for sel in selected])
                score = alpha * sim_to_query - (1 - alpha) * sim_to_selected

                if score > best_score:
                    best_score = score
                    best_candidate = idx

            selected.append((best_candidate, cosine_sim[query_index][best_candidate]))
            candidate_indices.remove(best_candidate)

        return [index_to_course_id[idx] for idx, similarity_score in selected]


    def get_similar_recommendation(cosine_sim,top_n=10):
        similar_movies = list(enumerate(cosine_sim[course_index]))

        # accessing the row corresponding to given movie to find all the similarity scores for that movie and then enumerating over it
        sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:]

        sort_size = len(sorted_similar_movies)
        similar_courses_list =[]
        # if sort_size > top_n:
        print(f"Top {top_n} courses similar to " + course_user_likes + " are:\n")
        for element in sorted_similar_movies[:top_n]:
            b=index_to_course_id[element[0]]
            similar_courses_list.append(b)
        # else:
        #     print("Top " + str(sort_size) + "similar movies to " + course_user_likes + " are:\n")
        #     for element in sorted_similar_movies[:sort_size]:
        #         b=index_to_course_id[element[0]]
        #         similar_courses_list.append(b)

        return similar_courses_list

    cosine_sim, course_id_to_index, index_to_course_id=getSimilarityScore()

    def get_title_from_index(index):
        a=courses_data[courses_data.id == index]
        return a["title"].values[0]

    def get_index_from_title(title):
        return courses_data[courses_data.title == title]["index"].values[0]

    course_index= course_id_to_index[course_id]


    recommendations=[]
    if diversify_recommendations:
        recommendations = get_diverse_recommendations(
            query_index=course_index,
            cosine_sim=cosine_sim,
            top_n=top_n,
            similarity_threshold=0.2,
            alpha=0.7
        )

    else:
        recommendations = get_similar_recommendation(cosine_sim,top_n);

    recommend_course_pd=courses_data.copy()
    recommend_course_pd=recommend_course_pd[recommend_course_pd['id'].isin(recommendations)]
    print(recommend_course_pd)

    return recommend_course_pd


# import time
# start = time.time()
# similar_courses(queryId,diversify_recommendations=True,recompute_cached_data=False)
# end = time.time()
# print(f" dont re-compute similarity scores {end - start}")


# start = time.time()
# similar_courses(queryId,diversify_recommendations=True,recompute_similarities=True)
# end = time.time()
# print(f" re-compute similarity scores {end - start}")

