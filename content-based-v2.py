import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
courses_data = pd.read_csv('./Course_info.csv')
courses_data = courses_data[courses_data["category"].isin(["Development","IT & Software"])]
print('Courses Data Shape is', courses_data.shape)

courses_data = courses_data.head(len(courses_data)//2)

features = ['title', 'is_paid', 'price', 'topic','subcategory','language']

queryId=9463;
course_user_likes="Programming Java for Beginners - The Ultimate Java Tutorial"

recompute_similarities=False

def getSimilarityScore():

    if recompute_similarities:
        return recomputeSimilarityScores()

    return getSimilarityScoresForStorage()


def getSimilarityScoresForStorage():
    cosine_similarities = pickle.load(open('cosine_similarity.pkl', 'rb'))
    course_id_to_index= pickle.load(open('course_id_to_index.pkl', 'rb'))
    index_to_course_id= pickle.load(open('index_to_course_id.pkl', 'rb'))

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


    pickle.dump(cosine_sim, open('cosine_similarity.pkl', 'wb'))
    pickle.dump(index_to_course_id, open('index_to_course_id.pkl', 'wb'))
    pickle.dump(course_id_to_index, open('course_id_to_index.pkl', 'wb'))

    return cosine_sim,course_id_to_index, index_to_course_id

cosine_sim, course_id_to_index, index_to_course_id=getSimilarityScore()

def get_title_from_index(index):
    a=courses_data[courses_data.id == index]
    return a["title"].values[0]

def get_index_from_title(title):
    return courses_data[courses_data.title == title]["index"].values[0]

course_index= course_id_to_index[queryId]
similar_movies = list(enumerate(cosine_sim[
                                    course_index]))  # accessing the row corresponding to given movie to find all the similarity scores for that movie and then enumerating over it
sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:]

sort_size = len(sorted_similar_movies)
similar_courses_list =[]
if sort_size > 10:
    print("Top 10 courses similar to " + course_user_likes + " are:\n")
    for element in sorted_similar_movies[:10]:
        b=index_to_course_id[element[0]]
        similar_courses_list.append(b)
else:
    print("Top " + str(sort_size) + "similar movies to " + course_user_likes + " are:\n")
    for element in sorted_similar_movies[:sort_size]:
        b=index_to_course_id[element[0]]
        similar_courses_list.append(b)


print(similar_courses_list)