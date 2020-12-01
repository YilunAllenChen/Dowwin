'''
This module provides Alpaca APIs to obtain market data, trade with paper money, etc.
It's said that frequency limit for these apis is 200/min per account. 

Reference / API Documentation: https://alpaca.markets/docs/api-documentation/api-v2/
'''

import requests
import json
from pprint import pprint

# Authentication for using Alpaca API.
ALPACA_API_SECRET_KEY = '4CtS8LOFcGI0PJsO9wrU3ddBAHGPPNaAfGsom5jH'
ALPACA_API_KEY = 'PKL7HET7GQ96FGE39LFD'

# Trading API endpoint is used for placing orders, checking account status, etc.
ALPACA_TRADING = 'https://paper-api.alpaca.markets'
# Market API endpoint is used for acquiring market data.
ALPACA_MARKET = 'https://data.alpaca.markets'

# Used to acquire account info.
ALPACA_ACCOUNT = f"{ALPACA_TRADING}/v2/account"
# Used to get the price/size/exchange data of this stock when it's last traded.
ALPACA_STOCK_LAST_TRADE = f"{ALPACA_MARKET}/v1/last/stocks/"


def _alpaca_get(url: str) -> dict:
    '''
    Generalized utility function to send a get http request to the specific url using ALPACA authentications.

    :param url: target url
    :return: dictionary of content. 

    TODO: 401/404 handling
    '''
    res = requests.get(url, headers={
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_API_SECRET_KEY
    })
    return json.loads(res.content)


def alpaca_get_account() -> dict:
    '''
    Function acquires account information.

    :return: dictionary of data about this account.

    Example return: {'id': 'c741afc3-b61a-456b-ac5d-52b520e3beba', 'account_number': 'PA27WLXMQ9LE', 'status': 'ACTIVE', 
    'currency': 'USD', 'buying_power': '400000', 'regt_buying_power': '200000', 'daytrading_buying_power': '400000', 
    'cash': '100000', 'portfolio_value': '100000', 'pattern_day_trader': False, 'trading_blocked': False, 'transfers_blocked': False, 
    'account_blocked': False, 'created_at': '2020-02-19T17:44:20.601067Z', 'trade_suspended_by_user': False, 'multiplier': '4', 
    'shorting_enabled': True, 'equity': '100000', 'last_equity': '100000', 'long_market_value': '0', 'short_market_value': '0', 
    'initial_margin': '0', 'maintenance_margin': '0', 'last_maintenance_margin': '0', 'sma': '0', 'daytrade_count': 0}
    NOTE: Do we need this?
    '''
    return _alpaca_get(ALPACA_ACCOUNT)


def alpaca_get_stock_data_last_traded(symbol: str) -> dict:
    '''
    Function obtains data of a specific stock (price, size, exchange, etc) when it was last traded

    :param symbol: Symbol of the stock we want to get info from. For example, Apple -> AAPL.
    :return: dictionary of data about this stock when it's last traded.

    Example return: dict{'status': 'success', 'symbol': 'AAPL', 'last': {'price': 122.195, 'size': 100, 
    'exchange': 15, 'cond1': 0, 'cond2': 0, 'cond3': 0, 'cond4': 0, 'timestamp': 1606837442640000000}}
    '''
    return _alpaca_get(ALPACA_STOCK_LAST_TRADE + symbol)



if __name__ == "__main__":
    pprint(alpaca_get_account())
    pprint(alpaca_get_stock_data_last_traded('AAPL'))
    
