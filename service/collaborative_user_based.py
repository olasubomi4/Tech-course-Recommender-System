import collaborative

def recommend_courses_for_user(user_id,recompute_cached_data,top_n):

   result= collaborative.search_course(user_id, recompute_cached_data, top_n=top_n)[["id","title","is_paid","price","headline","category","subcategory","topic","language","instructor_name"]]
   return result.to_json(orient='records')
