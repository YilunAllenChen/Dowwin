from apis_slack import SlackAPI
from apis_alpaca import AlpacaAPI
from apis_mongodb import MongoAPI
from data_stock_symbols import stock_symbols
from utils_decorators import send_error_to_slack_on_exception
from time import sleep


alpaca = AlpacaAPI(environment='production')
slack = SlackAPI(environment='production')
mongo = MongoAPI(environment='production')


@send_error_to_slack_on_exception
def data_acquisition_engine():
    for symbol in stock_symbols:
        res = alpaca.get_stock_data_last_traded(symbol)
        try:
            consolidated_data = {
                'symb': res['symbol'],
                'price': res['last']['price'],
                'timestamp': res['last']['timestamp']
            }
            mongo.update_stock(consolidated_data, by='symb')
        except Exception as e:
            raise Exception(f"Unable to acquire market data of [{symbol}]. \
                This is probably because Alpaca API doesn't have data of \
                this stock, or the data is incomplete.")
        sleep(2)  # 0.4


while True:
    data_acquisition_engine()
