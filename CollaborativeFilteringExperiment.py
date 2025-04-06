import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

courses_data = pd.read_csv('Course_info.csv')
ratings_df=pd.read_csv("modified_comments_with_author_id.csv")
print(ratings_df.shape)
ratings_df.dropna(inplace=True)
print(ratings_df.shape)

ratings_df["cf"]=None
user_comment_counts = ratings_df.groupby('autor_id').size()  # Counts comments per user
users_with_min_2_comments = user_comment_counts[user_comment_counts >= 5].index  # Get user IDs
a = ratings_df['autor_id'].isin(users_with_min_2_comments)  # Filter ratings_df
ratings_df = ratings_df[a]
print(ratings_df.shape)

ratings_matrix = ratings_df.pivot(index='course_id', columns='autor_id', values='rating')
ratings_matrix = ratings_matrix.fillna(0)

item_similarity = cosine_similarity(ratings_matrix)

# Convert the similarity matrix to a DataFrame for better readability
item_similarity_df = pd.DataFrame(item_similarity, index=ratings_matrix.index, columns=ratings_matrix.index)


# Step 2: Make recommendations based on item similarity
# def get_similar_courses(course_id, top_n=5):
#     # Get the similarity scores for the given course
#     similar_courses = item_similarity_df[course_id].sort_values(ascending=False)
#
#     # Return the top N most similar courses (excluding the course itself)
#     return similar_courses.drop(course_id).head(top_n)
# course_id=33240
# if course_id in ratings_matrix.index:
#     # Example: Get the top 5 most similar courses to a given course (e.g., course_id = 101)
#     recommended_courses = get_similar_courses(course_id, top_n=5)
#     print(recommended_courses)


def get_similar_courses(course_id, top_n=5):
    # Ensure course_id exists
    if course_id not in item_similarity_df.index:
        print(f"Course ID {course_id} not found.")
        return None

    # Get similarity scores for the given course
    similar_courses = item_similarity_df.loc[course_id].sort_values(ascending=False)
    course_title = courses_data.loc[courses_data['id'] == course_id, 'title'].values[0]
    
    # Get the most similar courses (excluding the course itself)
    similar_course_ids = similar_courses.drop(course_id).head(top_n).index  # Extract IDs
    
    # Fetch titles for similar courses
    similar_courses_with_titles = courses_data[courses_data['id'].isin(similar_course_ids)][['id', 'title']]
    
    # Print the course title
    print(f"Course ID {course_id}: {course_title}")
    
    return similar_courses_with_titles


# Example: Get the top 5 most similar courses to a given course_id
recommended_courses = get_similar_courses(15036, top_n=5)
print(recommended_courses)



recommended_courses = get_similar_courses(12226, top_n=5)
print(recommended_courses)



recommended_courses = get_similar_courses(20865, top_n=5)
print(recommended_courses)


recommended_courses = get_similar_courses(22591, top_n=5)
print(recommended_courses)
# print("a")
#
# import matplotlib.pyplot as plt
#
# # Calculate the number of users who reviewed each course
# reviews_per_course = ratings_df.groupby('course_id')['autor_id'].nunique()
#
# # Descriptive statistics
# average_reviews = reviews_per_course.mean()
# median_reviews = reviews_per_course.median()
# print(f"Average number of reviews per course: {average_reviews}")
# print(f"Median number of reviews per course: {median_reviews}")
#
# # Plot the distribution of reviews per course
# plt.hist(reviews_per_course, bins=30, edgecolor='black')
# plt.xlabel('Number of Reviews per Course')
# plt.ylabel('Frequency')
# plt.title('Distribution of Reviews per Course')
# plt.show()
#
#
#
# # Create a binary user-course interaction matrix
# binary_matrix = ratings_df.pivot_table(index='course_id', columns='autor_id', aggfunc='size', fill_value=0)
# binary_matrix = binary_matrix.applymap(lambda x: 1 if x > 0 else 0)  # Convert to binary (1 if commented, 0 otherwise)
#
# # Compute user-user co-occurrence matrix (dot product)
# user_user_matrix = binary_matrix.T.dot(binary_matrix)
#
# # Ensure the diagonal (self-interaction) is 0
# import numpy as np
# np.fill_diagonal(user_user_matrix.values, 0)
#
# print(user_user_matrix)

# from sklearn.neighbors import NearestNeighbors
# import numpy as np
#
# knn = NearestNeighbors(metric='cosine', algorithm='brute')
# knn.fit(ratings_matrix.fillna(0))  # Fill missing values with 0 for KNN
#
# # Find the k nearest neighbors for each user
# distances, indices = knn.kneighbors(ratings_matrix.fillna(0), n_neighbors=5)
#
# # Example: print the 5 nearest neighbors for the first user
# print(indices[0])

# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# import numpy as np
#
# # Assume ratings_matrix is the user-course matrix (rows = users, columns = courses)
# ratings_matrix = ratings_matrix.fillna(0)  # Fill missing values with 0 or another strategy
#
# # Standardize the data (important for distance-based algorithms like K-Means)
# scaler = StandardScaler()
# ratings_matrix_scaled = scaler.fit_transform(ratings_matrix)
#
# # Apply K-Means clustering to group users
# n_clusters = 5  # Number of clusters, adjust as needed
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# user_clusters = kmeans.fit_predict(ratings_matrix_scaled)
#
# # Add the cluster labels to the DataFrame
# ratings_df['user_cluster'] = user_clusters
#
#
# ratings_matrix_T = ratings_matrix.T
#
# # Apply K-Means clustering to group courses
# course_clusters = kmeans.fit_predict(ratings_matrix_T)
#
# # Add the cluster labels to the course dataframe
# ratings_df['course_cluster'] = course_clusters
#
#
# clustered_ratings_matrix = ratings_df.pivot(index='autor_id', columns='course_id', values='rating')
#
# print(clustered_ratings_matrix.shape)



