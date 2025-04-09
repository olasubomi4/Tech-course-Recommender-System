import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

re_compute_similarity_scores = True

# Load dataset
ratings_df = pd.read_csv("modified_comments_with_author_id.csv")
ratings_df.dropna(inplace=True)

# Filter users with at least 5 comments
user_comment_counts = ratings_df.groupby('autor_id').size()
users_with_min_5_comments = user_comment_counts[user_comment_counts >= 5].index
ratings_df = ratings_df[ratings_df['autor_id'].isin(users_with_min_5_comments)]

# Create USER-ITEM matrix
ratings_matrix = ratings_df.pivot(index='autor_id', columns='course_id', values='rating')
ratings_matrix = ratings_matrix.fillna(0)

# Compute user mean ratings
user_mean_ratings = ratings_matrix.replace(0, np.NaN).mean(axis=1)

# Center the ratings (subtract user mean)
ratings_matrix = ratings_matrix.sub(user_mean_ratings, axis=0).fillna(0)

def reComputerUserSimilarity(centered_matrix):
    user_similarity = cosine_similarity(centered_matrix)
    joblib.dump(user_similarity, 'user_similarity.pkl')
    return user_similarity

def getUserSimilarityScoresForCache():
    return joblib.load('user_similarity.pkl')

def getUserSimilarityScoresForUsers(centered_matrix):
    if re_compute_similarity_scores:
        return reComputerUserSimilarity(centered_matrix)
    else:
        return getUserSimilarityScoresForCache()

# Compute user-user similarity (cosine similarity) on centered ratings
user_similarity = getUserSimilarityScoresForUsers(ratings_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=ratings_matrix.index, columns=ratings_matrix.index)

# Predict rating with user bias adjustment
def predict_rating(user_id, course_id):
    if user_id not in ratings_matrix.index or course_id not in ratings_matrix.columns:
        return None

    similar_users = user_similarity_df[user_id].drop(user_id).sort_values(ascending=False)
    user_ratings = ratings_matrix.loc[similar_users.index, course_id]
    valid_ratings = user_ratings[user_ratings > 0]

    if valid_ratings.empty:
        return None

    weights = similar_users.loc[valid_ratings.index]
    if weights.sum() == 0:
        return None

    # Adjust for user bias (centered ratings)
    mean_ratings = user_mean_ratings.loc[valid_ratings.index]
    centered_ratings = valid_ratings - mean_ratings

    predicted_centered = np.dot(centered_ratings, weights) / weights.sum()

    # Add back the target user's mean rating
    user_bias = user_mean_ratings.loc[user_id]
    predicted_rating = predicted_centered + user_bias

    # Optional: clip to valid rating range
    predicted_rating = np.clip(predicted_rating, 1.0, 5.0)

    return predicted_rating

# Generate top-N recommendations
def get_recommendations(user_id, threshold=-3, top_n=5):
    if user_id not in ratings_matrix.index:
        print(f"User ID {user_id} not found.")
        return None

    # Get courses the user has already rated (seen)
    seen_courses = ratings_matrix.loc[user_id]
    seen_courses = seen_courses[seen_courses > 0].index

    predictions = {}
    for course_id in ratings_matrix.columns:
        if course_id in seen_courses:
            continue  # Skip seen courses

        predicted_rating = predict_rating(user_id, course_id)
        if predicted_rating is not None and predicted_rating > threshold:
            predictions[course_id] = predicted_rating

    # Sort and return top N recommendations
    recommended_courses = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return recommended_courses

# Example usage
recommended_courses_user_1 = get_recommendations(23448456, threshold=-3, top_n=5)
print("Recommended courses for User:", recommended_courses_user_1)
