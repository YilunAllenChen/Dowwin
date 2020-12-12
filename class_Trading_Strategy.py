from utils_logging import log_error

class Trading_Strategy():
    '''
    Abstract class that needs to be inherited.
    '''
    def __init__(self, *args, **kwargs):
        '''
        Initing an instance of a strategy. Include all the setup work here.
        Features of a trading strategy (For example, weights in a neural net) will be passed in as args/kwargs).
        '''
        log_error("You're using abstract trading strategy. STOP.")
    def evaluate(self, stock: str, *args, **kwargs) -> float:
        '''
        Function will attempt to evaluate a given stock based on potentially additional parameters passed in.

        :param stock: symbol of stock to be evaluated. e.g. "AAPL" for Apple Inc.
        :return: Normalized intention value (-1 ~ 1) signifying strong sell (-1) or strong buy (1).
        '''
        log_error("You're using abstract trading strategy. STOP.")
        return -1
    def update(self, *args, **kwargs):
        '''
        Function will update the features of this strategy. Also intepreted as training in some certain circumstances (like NN, REINFORCE).
        '''
    def extract_features(self):
        '''
        After evaluations and updating the strategy and before the Tradebot that has this Trading Strategy to be put back into the database, 
        we need to sync up the features. This function extracts the features in the Trading Strategy and returns it.
        '''
        log_error("You're using abstract trading strategy. STOP.")
        return {}


class Linear_Regression(Trading_Strategy):
    def __init__(self, *args, **kwargs):
        super(Linear_Regression, self).__init__(*args, **kwargs)

    def evaluate(self):
        return 1

class Neural_Network(Trading_Strategy):
    def __init__(self, *args, **kwargs):
        super(Neural_Network, self).__init__(*args, **kwargs)
        print("initing NN")
    def evaluate(self):
        return 1


trading_strategy_dict = {
    -1: Trading_Strategy,
    0:  Linear_Regression
}


def get_trading_strategy_object(id, features):
    return trading_strategy_dict[id](features)


if __name__ == "__main__":
    get_trading_strategy_object(0, {'feature':1})
