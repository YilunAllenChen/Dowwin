'''
This module uploads stock history data to the database.
To use this module:

Place the stock history data under {this_directory}/Stocks, and then run this script.
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