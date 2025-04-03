import pandas as pd

from db.PostgreSql import PostgreSql

def convertCommentsToCsv():
    db= PostgreSql()
    ratings_df=db.readComments();
    ratings_df.to_csv("modified_comments_with_author_id.csv",index=False)


if __name__ == "__main__":
    convertCommentsToCsv()