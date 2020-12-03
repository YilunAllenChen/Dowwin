#
# Darwin Robotics, 2020
#

'''
This module provides async-compatible utilities to interact with the database.
'''
import pymongo
from utils_decorators import send_error_to_slack_on_exception
from utils_config import get_global_config


class MongoAPI():
    def __init__(self, environment='production'):
        GLOBAL_CONFIG = get_global_config(environment)
        # TODO: setup environment variables here.
        self.client = pymongo.MongoClient(host=GLOBAL_CONFIG['DATABASE_URL'])
        self.market = self.client['Dowwin']['v1/Market']
        self.tradebots = self.client['Dowwin']['v1/Tradebots']

    @send_error_to_slack_on_exception
    def update_stock(self, data: dict, by='symb') -> None:
        '''
        Function to update stock market data. Update by name by default.

        :param data: dictionary of stock market data.
        :param by: find the entry in the database that has the same specified field as the passed in data.
        '''
        self.market.update_one({by: data.get(by)}, {'$set': data}, True)


if __name__ == "__main__":
    mongo = MongoAPI()
    mongo.update_stock({
        'symb': "HELLO_WORLD",
        'price': 100,
        'timestamp': -1
    })
