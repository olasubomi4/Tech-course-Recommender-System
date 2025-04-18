import time

import pandas as pd
import requests
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, wait
from db.PostgreSql import PostgreSql
from dto.Comment import Comment
from udemyResponseObject import UdemyResponse

load_dotenv()
cache_user = os.environ["CACHE_USER"]
cookie = os.environ["COOKIE"]
baseurl = os.environ["BASE_URL"]
db = PostgreSql()
import logging
logger = logging.getLogger(__name__)

logging.basicConfig(filename="app.log",level=logging.INFO,format='[%(filename)s:%(lineno)s - %(funcName)20s() ] %(asctime)s - %(levelname)s - %(message)s'
                    ,datefmt='%Y-%m-%d %H:%M:%S')
def load_important_courses_to_db():
    df = pd.read_csv("/Users/odekunleolasubomi/PycharmProjects/Recommender/Course_info.csv")
    df.dropna(inplace=True)
    df=df[df["category"].isin(["Development","IT & Software"])]
    df["processed"]=False;
    db.insertReleveantCourses(df)

def cordinator():
    numberOfThreads = int(os.getenv("NUMBER_OF_THREADS", 1))
    coursesList=db.retrieveCourses(numberOfThreads)
    try:
        while coursesList is not None:
            processList=[]
            with ThreadPoolExecutor(max_workers=numberOfThreads) as executor:
                for i,course in coursesList.iterrows():
                    processList.append(
                        executor.submit(courseHandler(course)))
                wait(processList)
            time.sleep(15)
            coursesList = db.retrieveCourses(numberOfThreads)

    except Exception as e:
        logger.critical(e)

def courseHandler(course):
    done=False
    course_id = int(course["id"])
    page = "1"
    url = f"{baseurl}{course_id}/reviews/?courseId={course_id}&fields%5Bcourse_review%5D=%40default%2Cresponse%2Ccontent_html%2Ccreated_formatted_with_time_since&fields%5Bcourse_review_response%5D=%40min%2Cuser%2Ccontent_html%2Ccreated_formatted_with_time_since&fields%5Buser%5D=%40min%2Cimage_50x50%2Cinitials%2Cpublic_display_name%2Ctracking_id&is_text_review=1&ordering=course_review_score__rank%2C-created&page={page}"
    header = {"x-udemy-cache-user": f"{cache_user}",
              "Cookie": f'{cookie}'
              }
    comments =pd.DataFrame()
    while not done:
        comments, count, next = scrapeComments(url, header,comments,course_id)
        count= count or 0
        if next is None or (count/12)>=10:
            done=True
        else:
            url = next
    course["processed"] = True
    db.updateCourses(course)
    db.insertComments(comments)
def decode_udemy_response(json_str: str) -> UdemyResponse:
    try:
        return UdemyResponse.parse_raw(json_str)
    except Exception as e:
        logger.critical(e)
def scrapeComments(url:str,header:dict,comments:pd.DataFrame,course_id:int):
    try:
        response = requests.get(url,headers=header)
        logging.log(logging.INFO, f"Scraping {url}")
        if response.status_code == 200:
            # print("Response JSON:", response.json())
            logging.log(logging.INFO, f"response {response.json()}")
            UdemyResponse=decode_udemy_response(response.text)
            for courseReview in UdemyResponse.getCourseList():
                comment=Comment()
                comment.set_id(courseReview.id)
                comment.set_autor_id(courseReview.user.id)
                comment.set_rating(courseReview.rating)
                comment.set_content(courseReview.content)
                comment.set_autor_name(courseReview.user.name)
                comment.set_created_at(courseReview.created)
                comment.set_autor_initials(courseReview.user.initials)
                comment.set_course_id(course_id)
                comments=pd.concat([comments,comment.convertCommentToDataFrame()],ignore_index=True)
            print(len(comments))
            return comments,UdemyResponse.count,UdemyResponse.next
        else:
            logging.log(logging.ERROR, f"failed {response.json()}")
            return None,0,None
    except Exception as e:
        logger.error(e)
        return None, 0, None

cordinator()


