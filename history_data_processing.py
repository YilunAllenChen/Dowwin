'''
This module uploads the stock history data from a kaggle notebook to the database.
Used in MEE design stage.


To use this module:

Download stock data from https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs and unzip.
Find the Stock directory and copy over to {this_directory}/Stocks, and then run this script.
'''

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('./Stocks') if isfile(join('./Stocks', f))]

from apis_mongodb import MongoAPI

mongo = MongoAPI()


for fname in onlyfiles:
    with open("./Stocks/" + fname) as f:
        stock = {
            'symbol': fname.split('.')[0],
            'history_data': f.readlines()
        }
        mongo.update_stock_history(stock, by='symbol')