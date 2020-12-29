import stable_baselines
from stable_baselines import DQN
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.deepq.policies import MlpPolicy

from core import RLStockTradingEnvironment, StockTraderAgent

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


def load_data() -> pd.DataFrame:
    """
    Loads and preprocesses the demo data. This is the Apple US stock information from 1984 to 2017. For simplicity, we
    only consider data as far as 5 years away from the most recent data point [2012, 2017].
    Preprocessing:
        1. Extracts Year, Month, Day from Date
        2. Convert Date to pd.datetime64, Year, Month, Day to int
        3. Convert other data types to the most appropriate
    :return: a pandas DataFrame of preprocessed data containing stock information for APPLE US from 2012 to 2017.
    """
    df = pd.read_csv('data/appl.us.csv')
    # extract year, month, day
    df[['Year', 'Month', 'Day']] = df['Date'].str.split('-', expand=True)
    # convert to appropriate types
    df['Date'] = pd.to_datetime(df['Date'])
    for col_name in ['Year', 'Month', 'Day']:
        df[col_name] = df[col_name].astype(int)
    # truncate to use only the most recent 5 years of data
    most_recent_year = df['Year'].max()
    year_mask = df['Year'] >= most_recent_year - 5
    df = df[year_mask]

    # infer other data types
    df = df.convert_dtypes()

    return df


# A dummy DQNStockTraderAgent

class DummyDQNStockTraderAgent(StockTraderAgent):
    def __init__(self, model: BaseRLModel, **kwargs) -> None:
        super().__init__(model, **kwargs)

    def train(self):
        # do our own stuff
        pass


class DummyRLStockTradingEnvironment(RLStockTradingEnvironment):
    def render(self, mode='human'):
        pass

    def __init__(self) -> None:
        super().__init__()
        self.stock_data: pd.DataFrame = load_data()
        self.data_iterator = self.stock_data.iterrows()

    def seed(self, seed):
        return super().seed(seed=seed)

    def step(self, action):
        return super().step(action)

    def reset(self):
        # return super().reset()
        # get a new data iterator
        self.data_iterator = self.stock_data.iterrows()

    def close(self):
        return super().close()


def main():
    # provide a stock trading demo
    pass


if __name__ == "__main__":
    # run the demo
    main()
