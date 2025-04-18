from flask import Flask, request, jsonify
import flask
from service import collaborative_user_based,content_based
from dto.ResponseObject import ResponseObject
from util.String_util import convert_string_bool_to_bool
app = Flask(__name__)

@app.route("/courses/<int:course_id>/similar",methods=['GET'])
def get_similar_course(course_id):
    responseObject= ResponseObject();
    try:
        diversify_recommendations =convert_string_bool_to_bool(request.args.get('diversify_recommendations',default=None))
        top_n = request.args.get('top_n',default=10,type=int)
        recompute_cached_data = convert_string_bool_to_bool(request.args.get('recompute_cached_data',default=None))

        result=content_based.get_similar_course(course_id,diversify_recommendations,recompute_cached_data,top_n)
        responseObject.setResponseMessage("Action successful")
        responseObject.setResponseStatus(True)
        responseObject.setData(result)
    except Exception as e:
        responseObject.setResponseStatus(False)
        responseObject.setResponseMessage(f"Could not process the request. Please try again later. Due to {e}")

    return jsonify(responseObject.jsonfyResponse())
@app.route("/courses/search",methods=['GET'])
def search_courses():
    responseObject = ResponseObject()
    try:
        query = request.args.get('query')
        language = request.args.get('language',default=None) or None
        free = convert_string_bool_to_bool(request.args.get('free',default=None))
        category = request.args.get('category',default=None) or None
        subcategory = request.args.get('subcategory',default=None) or None
        top_n = request.args.get('top_n',default=10,type=int)
        recompute_cached_data = convert_string_bool_to_bool(request.args.get('recompute_cached_data',default=False))

        result =content_based.search_course(query,language,free,category,subcategory,recompute_cached_data,top_n)
        responseObject.setResponseMessage("Action successful")
        responseObject.setResponseStatus(True)
        responseObject.setData(result)
    except Exception as e:

        responseObject.setResponseStatus(False)
        responseObject.setResponseMessage(f"Could not process the request. Please try again later. Due to {e}")

    return jsonify(responseObject.jsonfyResponse())

@app.route("/courses/<int:user_id>/recommendations", methods=['GET'])
def recommend_courses_for_user(user_id):
    responseObject= ResponseObject();
    try:
        recompute_cached_data = convert_string_bool_to_bool(request.args.get('recompute_cached_data',default=None))
        top_n = request.args.get('top_n',default=10,type=int)
        result=collaborative_user_based.recommend_courses_for_user(user_id,recompute_cached_data,top_n)
        responseObject.setResponseMessage("Action successful")
        responseObject.setResponseStatus(True)
        responseObject.setData(result)
    except Exception as e:
        responseObject.setResponseStatus(False)
        responseObject.setResponseMessage(f"Could not process the request. Please try again later. Due to {e}")

    return jsonify(responseObject.jsonfyResponse())

if __name__ == "__main__":
    app.run(debug=True)
