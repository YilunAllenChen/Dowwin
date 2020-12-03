#
# Darwin Robotics, 2020
#

'''
This module provides async-compatible utilities to interact with the database.
'''
import pymongo
from __GLOBAL_CONFIGS import DATABASE_URL
from utils_decorators import send_error_to_slack_on_exception



class MongoAPI():
    def __init__(self):
        #TODO: setup environment variables here.
        self.client = pymongo.MongoClient(host=DATABASE_URL())
        self.market = self.client['Dowwin']['v1/Market']
        self.tradebots = self.client['Dowwin']['v1/Tradebots']  

    @send_error_to_slack_on_exception
    def update_stock(self, data: dict, by = 'symb') -> None:
        '''
        Function to update stock market data. Update by name by default.

        :param data: dictionary of stock market data.
        :param by: find the entry in the database that has the same specified field as the passed in data.
        '''
        self.market.update_one({by: data.get(by)}, {'$set': data}, True)

