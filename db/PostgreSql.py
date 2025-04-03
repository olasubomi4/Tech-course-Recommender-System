import psycopg2
import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine

import logging
logger = logging.getLogger(__name__)

logging.basicConfig(filename="app.log",level=logging.INFO,format='[%(filename)s:%(lineno)s - %(funcName)20s() ] %(asctime)s - %(levelname)s - %(message)s'
                    ,datefmt='%Y-%m-%d %H:%M:%S')
class PostgreSql:
    load_dotenv()

    def __init__(self,mode="live"):
        if mode != "test":
            self.__dbName=os.environ['DB_NAME']
        else:
            self.__dbName = os.environ['UNIT_TEST_DB_NAME']
        self.__dbPassword=os.environ['DB_PASSWORD']
        self.__dbUser=os.environ['DB_USER']
        self.__dbHost=os.environ['DB_HOST']
        self.__dbPort=os.environ['DB_PORT']

    def __getConnectionEngine(self):
        connection_string = f"postgresql://{self.__dbUser}:{self.__dbPassword}@{self.__dbHost}:{self.__dbPort}/{self.__dbName}"
        engine = create_engine(connection_string)
        return engine

    def __connect(self):
        return psycopg2.connect(database=self.__dbName,host=self.__dbHost,port=self.__dbPort,password=self.__dbPassword,user=self.__dbUser)

    def __getConn(self):
        return self.__connect()
    def execute(self):
        with self.__getConn() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT * FROM demo WHERE "Id" = 1;')
                res = cur.fetchone()
                return res

    def insertReleveantCourses(self, data:pd.DataFrame):
        # with self.__getConn() as conn:
        try:
            data.to_sql("courses", con=self.__getConnectionEngine(), index=True,
                      if_exists="replace")
            return True
        except Exception as e:
            logger.error(e)
            return False

    def insertComments(self, data:pd.DataFrame):
        # with self.__getConn() as conn:
        try:
            data.to_sql("comments", con=self.__getConnectionEngine(), index=False,
                      if_exists="append")
            return True
        except Exception as e:
            logger.error(e)
            return False

    def retrieveTableAsDataFrame(self,tableName:str):
        return pd.read_sql_table(tableName,con=self.__getConnectionEngine(),index_col="id").sort_values(by="id")

    def retrieveCourses(self,count):
        try:
            query = f"SELECT * FROM courses where processed is false ORDER BY id LIMIT {count}"
            return pd.read_sql_query(query,con=self.__getConnectionEngine())
        except Exception as e:
            logger.error(e)


    def updateCourses(self,course):
        courseid=course["index"]
        query = f"UPDATE courses SET processed= True where index ={courseid}"
        try:
            with self.__getConn() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(e)
            return False

    def dropTable(self,tableName:str):
        query=f'DROP TABLE "{tableName}";'
        try:
            with self.__getConn() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    conn.commit()
                    return True

        except Exception as e:
            logger.error(e)
            return False

    def readComments(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM comments",con=self.__getConnectionEngine())











