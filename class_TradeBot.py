from time import time as float_timestamp
from datetime import datetime as timestamp
from data_names import choose_random_name
from utils_config import get_global_config
from class_Trading_Strategy import get_trading_strategy_object
from random import random

STARTING_FUND = 100000

# Good reference from Dowwin Legacy: https://github.com/YilunAllenChen/Dowwin_legacy/blob/master/_Trader.py

class TradeBot():
    def __init__(*args, **kwargs):
        # Timestamp is used as the ID of a Tradebot, hence also marks its date of birth.
        self.id = kwargs.get('id', float_timestamp() * 1000)
        # Name of this Tradebot.
        self.name = kwargs.get('name', choose_random_name())


        # Cash held by the Tradebot that is not used in trading.
        self.cash = kwargs.get('cash', STARTING_FUND)
        # Current value of cash + equity held by the Tradebot.
        self.evaluation = kwargs.get('evaluation', STARTING_FUND)
        # History of evaluations
        self.evaluation_history = kwargs.get('evaluation_history', [])
        # Portfolio of equity held by the tradebot. Example: {'MSFT': 100} (but it will contain more information like when did it buy at which price, etc)
        self.portfolio = kwargs.get('portfolio', {})


        # Strategy ID indicates the strategy this robot uses. For example, 1 could mean CNN, 2 means REINFOCE, 3 means linear regression, etc.
        self.strategy_id = kwargs.get('strategy_id', -1)
        # Strategy Features are passed into the specific evaluation functions based on strategy ID.
        self.strategy_features = kwargs.get('strategy_features', {})

        
        # How frequent this Tradebot performs trading. Randomized between 6 hours to 150 hours.
        self.trading_frequency = kwargs.get('trading_frequency', 6 + 2 * int(random()*72))
        # Last update timestamp.
        self.last_update = kwargs.get('last_update', timestamp())
        # Next udpate timestamp.
        self.next_update = kwargs.get('next_update', timestamp())



    def setup(self):
        self.trading_strategy = get_trading_strategy_object(self.strategy_id, self.strategy_features)

    def evaluate(self, stock: str) -> float:
        '''
        Upon call, this Tradebot should use its trading_strategy to evaluate a given stock and return a normalized intention: -1 for strong sell and 1 for strong buy.

        :param stock: stock symbol to evaluate. e.g., "AAPL" for Apple Inc.
        :return: a float of how good this stock is based on strategy.
        '''
        return self.trading_strategy.evaluate(stock)

    def buy(self, stock: str, num_of_shares: float) -> None:
        '''
        Upon call, this Tradebot will attempt to buy a certain number of shares of a certain stock, and 
        update corresponding info in its portfolio. If it doesn't have enough cash, this action will be aborted.

        :param stock: stock symbol to buy. e.g., "AAPL" for Apple Inc.
        :param num_of_shhares: number of shares. Will be rounded if not integer.
        '''
        pass

    def buy(self, stock: str, num_of_shares: float) -> None:
        '''
        Upon call, this Tradebot will attempt to sell a certain number of shares of a certain stock, and 
        update corresponding info in its portfolio. If it doesn't have enough cash, this action will be aborted.

        :param stock: stock symbol to buy. e.g., "AAPL" for Apple Inc.
        :param num_of_shhares: number of shares. Will be rounded if not integer.
        '''
        pass

    