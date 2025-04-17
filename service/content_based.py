import content_based_search
import content_based_similar_content



def get_similar_course(course_id,diversify_recommendations,recompute_cached_data,top_n):
    result = content_based_similar_content.similar_courses(course_id,diversify_recommendations=diversify_recommendations
                                                         ,recompute_cached_data=recompute_cached_data,top_n=top_n)[["id","title","is_paid","price","headline","category","subcategory","topic","language","instructor_name",]]
    return result.to_json(orient='records')


def search_course(query,language,free,category,sub_category,recompute_cached_data,top_n):
    result= content_based_search.search_course(query,language=language,free=free,category=category,sub_category=
    sub_category,recompute_cached_data=recompute_cached_data,top_n=top_n)[["id","title","is_paid","price","headline","category","subcategory","topic","language","instructor_name"]]
    return result.to_json(orient='records')

