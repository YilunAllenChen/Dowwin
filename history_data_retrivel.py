'''
This module uploads stock history data to the database.
'''
from apis_mongodb import MongoAPI
from pprint import pprint
mongo = MongoAPI()


# Count how many entries there are in the database
# print(mongo._market_history.find().count())

# Peek one entry in the database
# pprint(mongo.peek_history_data())

# Get all history data from db. NOTE: It's about 700MB in size.
# mongo.get_all_history_data()