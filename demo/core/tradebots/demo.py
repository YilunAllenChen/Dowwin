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
        self.stock_data = self.stock_data[
            ['Close']]  # use only closing price, make sure to pass in a list to get a DataFrame and not a Series
        # TODO: design these
        num_total_stocks = 1
        self.num_total_stocks = num_total_stocks  # FIXME: this should be read from the data
        self.ASSET_OFFSET = 1
        self.CURRENT_DATA_OFFSET = self.num_total_stocks + self.ASSET_OFFSET

        self.action_space: gym.spaces.Discrete = gym.spaces.Discrete(
            3)  # Stable baselines DQN only accepts discrete actions
        self.observation_space: gym.spaces.Box = gym.spaces.Box(
            low=0.0, high=float('inf'),
            shape=(1 + num_total_stocks * 2, 1)
        )
        # TODO: remove
        # self.observation_space: gym.spaces.Dict = gym.spaces.Dict({
        #     'balance': gym.spaces.Box(low=0.0, high=float('inf'), shape=(1,)),  # this is the cash balance for our agent
        #     # assets: for each stock
        #     #   # shares owned, adjusted total value (= closing price * # shares owned)
        #     # we only need to know how many shares we are owning
        #     'assets': gym.spaces.Box(low=0.0, high=float('inf'), shape=(self.num_total_stocks, 1)),
        #     # stock data: closing price of our stocks for this observation
        #     'current_data': gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(self.num_total_stocks, 1))
        # })

        # environment configurations
        self.starting_balance: float = starting_balance
        # assembles the starting assets (no stocks owned)
        self.starting_assets: np.ndarray = np.zeros(shape=(self.num_total_stocks, 1))

        # utilities for updating the environment
        self.data_iterator = self.stock_data.iterrows()

        # updated in run time
        self.balance: float = self.starting_balance
        self.assets: np.ndarray = self.starting_assets.copy()  # copy to prevent modifications on the original
        # current/previous data in this dummy demo are num stocks x 1 numpy arrays
        self.current_data: np.ndarray = np.zeros(shape=(self.num_total_stocks, 1))  # this will be populated immediately
        self.previous_data: np.ndarray = np.zeros(shape=(self.num_total_stocks, 1))
        # load the first piece of stock data
        self.update_current_data()

    def __get_observations__(self) -> dict:
        observation = np.zeros(shape=self.observation_space.shape)
        observation[0] = self.balance
        observation[self.ASSET_OFFSET: self.CURRENT_DATA_OFFSET] = self.assets
        observation[self.CURRENT_DATA_OFFSET:] = self.current_data

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
        if action is DummyRLStockTradingEnvironment.ACTION_BUY:
            number_change = 1
        elif action is DummyRLStockTradingEnvironment.ACTION_SELL:
            number_change = -1
        elif action is DummyRLStockTradingEnvironment.ACTION_HOLD:
            number_change = 0
        else:
            raise RuntimeError(f'Action {action} is undefined in this environment!')

        # reward is based on how much the total value of the agent's balance and assets has increased
        # this is equivalent to the changes in the total asset values
        # 0. calculate the previous account worth
        # FIXME: rigid implementation
        previous_asset_worth = (self.assets[:, 0].reshape((-1, 1)) * self.previous_data).sum()
        current_asset_worth = (self.assets[:, 0].reshape((-1, 1)) * self.current_data).sum()
        reward = current_asset_worth - previous_asset_worth

        # 1. grab the stock information to process the agent action
        # FIXME: rigid implementation
        closing_price = self.current_data[stock_id].item()
        price_change = number_change * closing_price  # money spent buying the stock is positive

        # 2. update the asset and balance for the agent
        # FIXME: rigid implementation
        self.balance -= price_change
        self.assets[stock_id] += number_change

        # 3. check action validity
        # FIXME: replace this with a method to customize the stopping condition
        if self.balance < 0.0:
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
        self.assets = self.starting_assets.copy()
        self.current_data = np.ndarray([])
        self.previous_data = np.ndarray([])
        # load the first piece of stock data
        self.update_current_data()

        return self.__get_observations__()

    def render(self, mode='human'):
        pass

    def close(self):
        return super().close()

    def shoud_stop(self):
        return self.balance < 0.0

    def set_testing(self, test: bool) -> None:
        """
        Sets if the environment should run a training or testing dataset. Requires reset to take effect.
        :param test: a boolean value indicating if this environment should run its testing dataset
        :return: None
        """
        self.is_testing = test


# FIXME: remove
# class DummyRLEnvironment(gym.Env):
#     goal_position = np.array([0.0, 0.0])
#
#     """
#     A simple move rule. The intended effect is to teach our agent to take action 1
#     because that is the only possible action to let our agent reach the goal position
#     """
#     move_rule = [
#         np.array([-0.05, -0.05]),
#         np.array([0.05, 0.0]),
#         np.array([0.0, 0.05]),
#         np.array([0.05, 0.05])
#     ]
#
#     def __init__(self):
#         self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
#         self.action_space = gym.spaces.Discrete(4)
#
#         self.initial_position = np.array([1.0, 1.0])
#         self.agent_position = self.initial_position
#         self.previous_agent_position = self.initial_position
#         self.steps_taken = 0.0
#
#     def step(self, action):
#         # goal: train the agent to stay at (0, 0)
#
#         # update agent position
#         assert action in self.action_space
#         self.previous_agent_position = self.agent_position  # record previous
#         # a simple move rule
#         move_delta = self.move_rule[action]
#         self.agent_position += move_delta
#
#         out_of_bounds = not self.observation_space.contains(self.agent_position)
#         current_distance = np.linalg.norm(self.agent_position - self.goal_position)
#         if not out_of_bounds:
#             # reward is the reduction (positive indicates reduction) in distance between the agent position and the
#             goal position
#             # previous_distance = np.linalg.norm(self.previous_agent_position - self.goal_position)
#             #
#             # reward = -(current_distance - previous_distance)
#             reward = np.linalg.norm(np.array([1.0, 1.0]) - self.agent_position) - self.steps_taken
#         else:
#             reward = -100
#
#         is_done = out_of_bounds or current_distance <= 1e-5
#         observations = self.agent_position
#
#         return observations, reward, is_done, {}
#
#     def reset(self):
#         self.steps_taken = 0.0
#         self.initial_position = np.array([1.0, 1.0])
#         self.agent_position = self.initial_position
#         self.previous_agent_position = self.initial_position
#
#         return self.agent_position
#
#     def render(self, mode='human'):
#         pass


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
    env: DummyRLStockTradingEnvironment = DummyRLStockTradingEnvironment(starting_balance)
    model = DQN('MlpPolicy', env, learning_rate=1e-3)
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
        for timestep in range(max_timesteps):
            log_message(f'Timestep: {timestep}')
            action, _ = model.predict(obs)
            obs, reward, dones, info = env.step(action)
            save_data.append([
                episode, timestep, action, reward, env.assets.item()
            ])
    log_message(f'Saving playback to {save_filename}')
    pd.DataFrame(save_data, columns=['Episode', 'Timestep', 'Action', 'Reward', 'Shares owned']).to_csv(str(save_filename.absolute()))
    log_message(f'Done!')

    # FIXME: remove
    # # FIXME: maybe find a better way to specify the initial balance? MAYBE?
    # starting_balance = 1e5
    # # load stock data
    # train_data, test_data = split_train_val()
    # # initialize an environment
    # env = DummyRLStockTradingEnvironment(stock_data=train_data, starting_balance=starting_balance)
    # # initialize a model and an agent
    # model = DQN(MlpPolicy, env, learning_rate=1e-3, prioritized_replay=True)
    # agent = DummyDQNStockTraderAgent(model)
    #
    # # run the environment
    # total_learning_timesteps = 2e5
    # agent.model.learn(total_timesteps=total_learning_timesteps)
    #
    # # TODO: run our agent through our stock market data
    # observation_timesteps = int(1e5)
    # obs = env.reset()
    # for i in range(observation_timesteps):
    #     action, states = model.predict(obs)
    #     obs, rewards, is_done, info = env.step(action)
    #     print(f'Observations: {obs}\nRewards: {rewards}')
    #     if is_done:
    #         break
    # print('Done!')
    #
    # # TODO: run our agent thorugh our validation market
    pass


if __name__ == "__main__":
    main()
