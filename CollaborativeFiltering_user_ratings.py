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

# Create USER-ITEM matrix
ratings_matrix = ratings_df.pivot(index='autor_id', columns='course_id', values='rating')
ratings_matrix = ratings_matrix.fillna(0)  # Fill NaN with 0

# Compute user-user similarity (cosine similarity)
user_similarity = cosine_similarity(ratings_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=ratings_matrix.index, columns=ratings_matrix.index)


# Function to predict rating for a user on a given course
def predict_rating(user_id, course_id):
    if user_id not in ratings_matrix.index or course_id not in ratings_matrix.columns:
        return None  # User or course not found

    # Find similar users
    similar_users = user_similarity_df[user_id].drop(user_id).sort_values(ascending=False)

    # Ratings given by similar users for the course
    user_ratings = ratings_matrix.loc[similar_users.index, course_id]

    # Filter users who actually rated this course
    valid_ratings = user_ratings[user_ratings > 0]

    if valid_ratings.empty:
        return None  # No similar users have rated this course

    # Compute weighted rating prediction (weighted by similarity)
    weights = similar_users.loc[valid_ratings.index]
    if weights.sum() == 0:  # Avoid division by zero
        return None
    predicted_rating = np.dot(valid_ratings, weights) / weights.sum()

    return predicted_rating


# Function to get recommendations for a user
def get_recommendations(user_id, threshold=2.5, top_n=5):
    if user_id not in ratings_matrix.index:
        print(f"User ID {user_id} not found.")
        return None

    predictions = {}

    for course_id in ratings_matrix.columns:
        if ratings_matrix.loc[user_id, course_id] == 0:  # If user hasn't rated it
            predicted_rating = predict_rating(user_id, course_id)
            if predicted_rating is not None and predicted_rating > threshold:
                predictions[course_id] = predicted_rating

    # Sort recommendations by predicted rating
    recommended_courses = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return recommended_courses


# Example: Get top 5 recommendations for user 1001
recommended_courses_user_1 = get_recommendations(23448456, threshold=2.5, top_n=5)
print("Recommended courses for User 1001:", recommended_courses_user_1)
