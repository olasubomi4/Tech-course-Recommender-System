import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load dataset
ratings_df = pd.read_csv("/Users/odekunleolasubomi/PycharmProjects/Recommender/modified_comments_with_author_id.csv")
ratings_df.dropna(inplace=True)

# Filter users with at least 5 comments
user_comment_counts = ratings_df.groupby('autor_id').size()
users_with_min_2_comments = user_comment_counts[user_comment_counts >= 5].index
ratings_df = ratings_df[ratings_df['autor_id'].isin(users_with_min_2_comments)]

# Create USER-ITEM matrix (users as rows, courses as columns)
ratings_matrix = ratings_df.pivot(index='autor_id', columns='course_id', values='rating')
ratings_matrix = ratings_matrix.fillna(0)  # Fill NaN with 0

# Compute user-user similarity (cosine similarity)
user_similarity = cosine_similarity(ratings_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=ratings_matrix.index, columns=ratings_matrix.index)

# Function to recommend courses based on similar users
def get_user_recommendations(user_id, top_n=5):
    if user_id not in user_similarity_df.index:
        print(f"User ID {user_id} not found.")
        return None

    # Find similar users (excluding the user itself)
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).drop(user_id).head(top_n)

    # Get courses rated by similar users
    similar_users_courses = ratings_matrix.loc[similar_users.index]

    # Find top-rated courses from similar users
    avg_ratings = similar_users_courses.mean(axis=0)
    recommended_courses = avg_ratings.sort_values(ascending=False).head(top_n)

    return recommended_courses

# Example recommendations for users
recommended_courses_user_1 = get_user_recommendations(4577072, top_n=5)  # Example user ID
print("Recommended courses for User 1001:")
print(recommended_courses_user_1)

# recommended_courses_user_2 = get_user_recommendations(2002, top_n=5)
# print("\nRecommended courses for User 2002:")
# print(recommended_courses_user_2)
