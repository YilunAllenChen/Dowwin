
class Trading_Strategy():
    def __init__(self, *args, **kwargs):
        pass
    def setup(self):
        pass
    def evaluate(self):
        return -1


class Linear_Regression(Trading_Strategy):
    def __init__(self, *args, **kwargs):
        super(Linear_Regression, self).__init__(*args, **kwargs)
        print('initing lin reg')
    def setup(self):
        self.some_number = 1
    def evaluate(self):
        return 1



trading_strategy_dict = {
    0: Linear_Regression
}


def get_trading_strategy_object(id, features):
    return trading_strategy_dict[id](features)


if __name__ == "__main__":
    get_trading_strategy_object(0, {'feature':1})
