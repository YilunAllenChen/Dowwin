#
# Darwin Robotics, 2020
#

'''
This module provides async-compatible utilities to interact with the database.
'''
import pymongo
import asyncio
import motor.motor_asyncio
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
        self._market = self._client['Dowwin']['v1/market']
        self.tradebots = self._client['Dowwin']['v1/Tradebots']
        self._market_history = self._client['Dowwin']['v1/market_history']

    @send_error_to_slack_on_exception
    def update_stock(self, data: dict, by='symbol') -> None:
        '''
        Function will update stock market data. Update by name by default.

        :param data: dictionary of stock market data.
        :param by: find the entry in the database that has the same specified field as the passed in data.
        '''
        self._market.update_one({by: data.get(by)}, {'$set': data}, True)

    def get_all_history_data(self, aggr = {}) -> list:
        '''
        Function fetches all documents in the market_history database
        '''
        cursor = self._market_history.find(aggr)
        return [item for item in cursor]

    def peek_history_data(self, aggr = {}) -> dict:
        '''
        Function fetches the top document in the market_history database
        '''
        cursor = self._market_history.find_one(aggr)
        return cursor


class AIOMongoAPI():
    def __init__(self, environment='development'):
        '''
        AIOMongo API provides asynchronous functionalities to interact with Darwin Robotics's MongoDB Atlas Cluster asynchronously.
        
        :param environment: Can be 'production', 'development', etc.
        '''
        GLOBAL_CONFIG = get_global_config(environment)
        # TODO: setup environment variables here.
        self._client = motor.motor_asyncio.AsyncIOMotorClient(host=GLOBAL_CONFIG['DATABASE_URL'])
        self._market = self._client['Dowwin']['v1/market']
        self.tradebots = self._client['Dowwin']['v1/Tradebots']
        self._market_history = self._client['Dowwin']['v1/market_history']
    
    async def update_stock(self, data: dict, by='symbol'):
        '''
        Function updates stock market data. Update by name by default.

        :param data: dictionary of stock market data. By default, it must have 'symbol' field to specify which stock it is.
        :param by: find the entry in the database that has the same specified field as the passed in data.
        '''
        result = await self._market.update_one({by: data.get(by)}, {'$set': data}, True)
        return result

    async def get(self, stock: str) -> dict:
        '''
        Function gets the data of the specified stock from the database.
        '''
        result = await self._market.find_one({'symbol': stock})
        return result

# if __name__ == "__main__":
#     mongo = MongoAPI()
#     mongo.update_stock({
#         'symb': "HELLO_WORLD",
#         'price': 100,
#         'timestamp': -1
#     })


# async def main():
#     mongo = AIOMongoAPI()
#     data = {
#         'symbol': "HELLO_WORLD",
#         'price': 100,
#         'timestamp': -1
#     }
#     result = await mongo.update_stock(data)
#     print(result)
# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())