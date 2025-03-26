from collections import defaultdict

import pandas as pd

target=["Development","IT & Software"];

df=pd.read_csv("/Users/odekunleolasubomi/PycharmProjects/Recommender/Comments.csv")


# df=pd.read_csv("/Users/odekunleolasubomi/PycharmProjects/Recommender/Comments.csv")


# counter=0
#
# df=df.dropna(subset=['display_name'])
# df['display_name'] = df['display_name'].str.lower().str.strip()
#
# display_name_set= set(df.display_name)
#
# display_name_dict={}
#
# for i in display_name_set:
#     display_name_dict[i]=counter
#     counter+=1
#
#
# df["user_id"]= df.apply(lambda row: display_name_dict[row['display_name']], axis=1)
# # df["user_id"] = pd.factorize(df["display_name"])[0]
#
# # user_counts = df['user_id'].value_counts()
# # df = df[df['user_id'].isin(user_counts[user_counts > 1].index)]
#
# # Remove rows where display_name contains only one word
# df = df[df['display_name'].str.split().str.len() > 2]
#
# df.to_csv("modded_comments.csv",index=False)
#
# # Count how many comments each user has made
# # user_counts = df['display_name'].value_counts()
#
# # Add a column for the number of comments each user has made
#
#
#
# # user_counts = set(df.user_id)
# # user_counts_dic = defaultdict(int)
# # for i,row in df.iterrows():
# #     user_counts_dic[row["user_id"]] += 1
# #
# #
# # df["number_of_comments"]=df.apply(lambda row : user_counts_dic[row["user_id"]], axis=1)
#
# user_counts_dic = df.groupby("user_id").size().to_dict()
# df["number_of_comments"] = df["user_id"].map(user_counts_dic)
#
# df.to_csv("avdd.csv",index=False)




