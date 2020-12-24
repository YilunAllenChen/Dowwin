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
    def __init__(self, environment='development'):
        '''
        Mongo API provides functionalities to interact with Darwin Robotics's MongoDB Atlas Cluster.
        
        :param environment: Can be 'production', 'development', etc.
        '''
        GLOBAL_CONFIG = get_global_config(environment)
        # TODO: setup environment variables here.
        self._client = pymongo.MongoClient(host=GLOBAL_CONFIG['DATABASE_URL'])
        self._market = self._client['Dowwin']['v1/_market']
        self.tradebots = self._client['Dowwin']['v1/Tradebots']
        self._market_history = self._client['Dowwin']['v1/MockMarket']

    @send_error_to_slack_on_exception
    def update_stock(self, data: dict, by='symb') -> None:
        '''
        Function to update stock market data. Update by name by default.

        :param data: dictionary of stock market data.
        :param by: find the entry in the database that has the same specified field as the passed in data.
        '''
        self._market.update_one({by: data.get(by)}, {'$set': data}, True)


    @send_error_to_slack_on_exception
    def update_stock_history(self, data: dict, by='symb') -> None:
        '''
        Function to update the *history* stock market data. Update by name by default.

        :param data: dictionary of stock market data.
        :param by: find the entry in the database that has the same specified field as the passed in data.
        '''
        self.mock_market.update_one({by: data.get(by)}, {'$set': data}, True)


if __name__ == "__main__":
    mongo = MongoAPI()
    mongo.update_stock({
        'symb': "HELLO_WORLD",
        'price': 100,
        'timestamp': -1
    })
