from gym.spaces import space
import stable_baselines
from stable_baselines import DQN, A2C
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.deepq.policies import MlpPolicy

import gym

from core import RLStockTradingEnvironment, StockTraderAgent

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from typing import Tuple


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
    data_path = Path(__file__).parent / 'data/appl.us.csv'
    df = pd.read_csv(str(data_path))
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


def split_train_val() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = load_data()
    # performs a 4:1 split
    most_recent_year = df['Year'].max()
    train_mask = df['Year'] <= most_recent_year - 1
    df_train, df_test = df[train_mask], df[~train_mask]
    return df_train, df_test


# A dummy DQNStockTraderAgent

class DummyDQNStockTraderAgent(StockTraderAgent):
    def __init__(self, model: BaseRLModel, **kwargs) -> None:
        self.model: BaseRLModel
        super().__init__(model, **kwargs)


class DummyRLStockTradingEnvironment(RLStockTradingEnvironment):
    """
    A dummy stock trading environment for demo purposes,
    Description:
        A single stock is available in this environment. The agent starts with
        some initial funding. The goal of the agent is to maximize the profits.
    Source:
        This environment uses a single stock Apple US from the Kaggle stock
        market dataset available at:
        https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
    Observation:
        Type: Discrete(3)
        Num     Observation
        TBD
    Actions:
        Type: Discrete(3)
        Num     Action
        0       Buy 1 stock of the demo stock
        1       Sell 1 stock of the demo stock
        2       Hold
    Reward:

    Starting State:
        The agent start with a fixed pool of funds and no stock purchased.
    Episode Termination:
        We have reached the end of our data.
    """

    ACTION_BUY = 0
    ACTION_SELL = 1
    ACTION_HOLD = 2

    def __init__(self, starting_balance: float, test=False) -> None:
        super().__init__()

        self.is_testing = test

        # load data
        self.train_data, self.test_data = split_train_val()
        # this is the actual data that we use for running this environment
        self.train_data, self.test_data = self.train_data[['Close']], self.test_data[['Close']]
        self.stock_data = self.train_data if not self.is_testing else self.test_data

        # TODO: add some more preprocessing to our data to support multiple stocks
        # TODO: design these
        self.action_space: gym.spaces.Discrete = gym.spaces.Discrete(
            3)  # Stable baselines DQN only accepts discrete actions
        self.observation_space: gym.spaces.Box = gym.spaces.Box(
            low=0.0, high=float('inf'),
            shape=(3,)
        )

        # environment configurations
        self.starting_balance: float = starting_balance
        # assembles the starting shares owned
        self.starting_assets: float = 0.0

        # utilities for updating the environment
        self.data_iterator = self.stock_data.iterrows()

        # updated in run time
        self.balance: float = self.starting_balance
        self.assets: float = self.starting_assets
        # current/previous data in this dummy demo are num stocks x 1 numpy arrays
        self.current_data: float = 0.0
        self.previous_data: float = 0.0
        # load the first piece of stock data
        self.update_current_data()

    def __get_observations__(self) -> np.ndarray:
        return np.array([self.balance, self.assets, self.current_data])

    def update_current_data(self) -> None:
        """
        Updates the environment with the next piece of stock data from the data iterator.
        :return: None
        """
        current_data: pd.DataFrame
        index, current_data = next(self.data_iterator)
        self.previous_data = self.current_data
        self.current_data = current_data.to_numpy().astype(float)

    def seed(self, seed):
        super().seed(seed=seed)

    def step(self, action) -> tuple:
        """
        Work outline:
            - update the environment based on the action taken by the agent
            - check if we should terminate this episode
                terminating condition: we have reached the end of our data
            - look at the action taken by our agent (the method parameter)
                if it is valid:
                    purchase: the agent must have enough balance (self.balance) to purchase their specified number of
                    stocks
                        if not valid, either terminate or give a highly negative reward
                        if valid, change the balance and assets accordingly
                    sell: the agent must have enough shares of that stock to sell
                        if not valid, either teminate or give a highly negative reward
                        if valid, change the balance and assets accordingly
                    hold:
                        seems okay for now, this should probably yield a reward of 0 naturally by our design
        Things to consider:
            if we terminate as soon as the agent makes an illegal move (should be a reasonably common definition for
            terminating conditions),
            we need to increase the number of episodes when training our agent.
        """
        should_terminate = False

        # FIXME: hook stock_id up with other methods to change the buy/sell/hold behavior
        stock_id = 0
        if action == DummyRLStockTradingEnvironment.ACTION_BUY:
            number_change = 1
        elif action == DummyRLStockTradingEnvironment.ACTION_SELL:
            number_change = -1
        elif action == DummyRLStockTradingEnvironment.ACTION_HOLD:
            number_change = 0
        else:
            raise RuntimeError(f'Action {action} is undefined in this environment!')

        # reward is based on how much the total value of the agent's balance and assets has increased
        # this is equivalent to the changes in the total asset values
        # 0. calculate the previous account worth
        # FIXME: rigid implementation
        previous_asset_worth = self.assets * self.previous_data
        current_asset_worth = self.assets * self.current_data
        reward = current_asset_worth - previous_asset_worth + 100.0 # skew towards buying

        # 1. grab the stock information to process the agent action
        # FIXME: rigid implementation
        closing_price = self.current_data
        price_change = number_change * closing_price  # money spent buying the stock is positive

        # 2. update the asset and balance for the agent
        # FIXME: rigid implementation
        self.balance -= price_change
        self.assets += number_change

        # 3. check action validity
        # FIXME: replace this with a method to customize the stopping condition
        if self.balance < 0.0 or self.assets < 0.0:
            # you cannot buy with things you dont have and you cannot sell things you dont have
            # invalid action
            reward = -10000
            should_terminate = True

        # 4. go to the next piece of stock data
        try:
            self.update_current_data()
        except StopIteration:
            # end of our data
            should_terminate = True
        finally:
            # assemble return values
            observations = self.__get_observations__()
            return observations, reward, should_terminate, {}

    def reset(self):

        self.stock_data = self.train_data if not self.is_testing else self.test_data
        # reload the iterator
        self.data_iterator = self.stock_data.iterrows()

        # reset the variables
        self.balance = self.starting_balance
        self.assets = self.starting_assets
        self.current_data = 0.0
        self.previous_data = 0.0
        # load the first piece of stock data
        self.update_current_data()

        return self.__get_observations__()

    def render(self, mode='human'):
        pass

    def close(self):
        return super().close()

    def should(self):
        return self.balance < 0.0

    def set_testing(self, test: bool) -> None:
        """
        Sets if the environment should run a training or testing dataset. Requires reset to take effect.
        :param test: a boolean value indicating if this environment should run its testing dataset
        :return: None
        """
        self.is_testing = test


class DummyEnvironment(RLStockTradingEnvironment):

    ACTION_BUY = 0
    ACTION_SELL = 1
    ACTION_HOLD = 2

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(3)
        # balance, shares owned, current price
        self.observation_space = gym.spaces.Box(low=0.0, high=float('inf'), shape=(3,))
        self.starting_balance = 1e3
        self.starting_shares = 0.0

        self.balance = self.starting_balance
        self.shares = self.starting_shares

        # load data
        self.train_data, self.test_data = split_train_val()
        self.train_data, self.test_data = self.train_data[['Close']], self.test_data[['Close']]
        self.stock_data: pd.DataFrame = self.train_data

        # variables
        # FIXME: rigid implementation
        self.current_closing_price = 0.0
        self.previous_closing_price = 0.0

        # utilities
        self.data_iterator = self.stock_data.iterrows()

        # load in the first sample
        self.update_current_data()

    def update_current_data(self) -> None:
        # FIXME: rigid implementation
        index, data = next(self.data_iterator)
        self.previous_closing_price = self.current_closing_price
        self.current_closing_price = data.item()

    def step(self, action):
        should_terminate = False

        if action == DummyRLStockTradingEnvironment.ACTION_BUY:
            number_change = 1
        elif action == DummyRLStockTradingEnvironment.ACTION_SELL:
            number_change = -1
        elif action == DummyRLStockTradingEnvironment.ACTION_HOLD:
            number_change = 0
        else:
            raise RuntimeError(f'Action {action} is undefined in this environment!')
        # calculate change in total account worth
        # FIXME: rigid implementation
        # NOTE: we have not processed the action yet. The environment is still in the "previous" state at this point
        #       only the current and previous closing price have been update to reflect the "current" state
        previous_worth = self.shares * self.previous_closing_price + self.balance
        current_worth = self.shares * self.current_closing_price + self.balance

        # process action
        self.shares += number_change
        self.balance -= number_change * self.previous_closing_price

        # calculate reward
        # reward = positive increase in total account value
        if self.balance < 0.0 or self.shares < 0.0:
            # invalid state
            reward = -100.0
            should_terminate = True
        else:
            reward = current_worth - previous_worth

        try:
            self.update_current_data()
        except StopIteration:
            should_terminate = True
        finally:
            return np.array([self.balance, self.shares, self.previous_closing_price]), reward, should_terminate, {}

    def reset(self):
        self.data_iterator = self.stock_data.iterrows()
        self.balance = self.starting_balance
        self.shares = self.starting_shares
        self.update_current_data()
        return np.array([self.balance, self.shares, self.previous_closing_price])

    def render(self, mode='human'):
        pass


def main():

    import datetime
    import time
    def log_message(message: str) -> None:
        print(f'{datetime.datetime.now().isoformat()}::{message}')

    # persist running log
    save_dir = Path(__file__).parent / 'experiments'
    if not save_dir.exists():
        save_dir.mkdir()

    log_message('Initializing environment...')
    starting_balance = 1e5
    # initialize the environment
    env: DummyEnvironment = DummyEnvironment()
    model = A2C('MlpPolicy', env, learning_rate=1e-3)
    agent = DummyDQNStockTraderAgent(model)

    start_time = time.time()
    log_message('Training agent...')
    agent.model.learn(total_timesteps=int(2e5))
    end_time = time.time()
    log_message(f'Done in {end_time - start_time} seconds!')
    log_message(f'Playing agent against the environment...')
    # play the agent on our training data
    num_episodes, max_timesteps = (3, int(1e5))
    save_filename = save_dir / f'{datetime.date.today().isoformat()}-{num_episodes}-{max_timesteps}.csv'
    save_data = []  # episode, timestep, action taken, reward, shares owned
    for episode in range(num_episodes):
        log_message(f'{"=" * 11}\nEpisode {episode}')
        obs = env.reset()
        print(obs)
        for timestep in range(max_timesteps):
            log_message(f'Timestep: {timestep}')
            action, _ = model.predict(obs)
            obs, reward, dones, info = env.step(action)
            save_data.append([
                episode, timestep, action, float(reward), env.shares, float(env.balance)
            ])
    log_message(f'Saving playback to {save_filename}')
    pd.DataFrame(save_data, columns=['Episode', 'Timestep', 'Action', 'Reward', 'Shares owned', 'Balance']).to_csv(str(save_filename.absolute()))
    log_message(f'Done!')
    #
    # # TODO: run our agent thorugh our validation market
    pass


if __name__ == "__main__":
    main()
