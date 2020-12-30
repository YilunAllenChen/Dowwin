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


def split_train_val() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = load_data()
    # performs a 4:1 split
    most_recent_year = df['Year'].max()
    train_mask = df['Year'] <= most_recent_year - 1
    df_train, df_test = df[train_mask], df[~train_mask]
    return (df_train, df_test)


# A dummy DQNStockTraderAgent

class DummyDQNStockTraderAgent(StockTraderAgent):
    def __init__(self, model: BaseRLModel, **kwargs) -> None:
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
        0       Buy
        1       Sell
        2       Hold
    Reward:

    Starting State:
        The agent start with a fixed pool of funds and no stock purchased.
    Episode Termination:
        We have reached the end of our data.
    """

    def __init__(self, stock_data: pd.DataFrame, starting_balance: float) -> None:
        super().__init__()

        # TODO: design these
        self.observation_space = None
        self.action_space = gym.spaces.Discrete(3)

        # FIXME: environment configurations
        self.stock_data: pd.DataFrame = load_data()
        self.balance = starting_balance
        """
        Assets is a python dictionary of Stock Name -> [closing price, # shares owned, adjusted total value]
        """
        self.assets = pd.DataFrame([[0.0, 0.0, 0.0]], columns=['Closing Price', 'Shares Owned', 'Adjusted Total Value'])

        # utilities
        self.data_iterator = self.stock_data.iterrows()

        # state: [balance, ]
        # self.state =

        # action space {-k,... -1, 0, 1, ... k}, k=number of shares, size=2k+1
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1, ))

        # observation space - shape = price, balance, closing price
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(3,))

    def seed(self, seed):
        super().seed(seed=seed)

    def step(self, actions) -> None:
        """
        Work outline:
            - check if we should terminate this episode
                terminating condition: we have reached the end of our data
            - look at the action taken by our agent (the method parameter)
                if it is valid:
                    purchase: the agent must have enough balance (self.balance) to purchase their specified number of stocks
                        if not valid, either terminate or give a highly negative reward
                        if valid, change the balance and assets accordingly
                    sell: the agent must have enough shares of that stock to sell
                        if not valid, either teminate or give a highly negative reward
                        if valid, change the balance and assets accordingly
                    hold:
                        seems okay for now, this should probably yield a reward of 0 naturally by our design
        Things to consider:
            if we teminate as soon as the agent makes an illegal move (should be a reasonably common definition for terminating conditions),
            we need to increase the number of episodes when training our agent.
        """

        pass

    def reset(self):
        # return super().reset()
        # get a new data iterator
        self.data_iterator = self.stock_data.iterrows()

    def render(self, mode='human'):
        pass

    def close(self):
        return super().close()

    def calculate_total_value(self) -> float:
        return self.balance + self.assets['Adjusted Total Value'].sum()


class DummyRLEnvironment(gym.Env):
    goal_position = np.array([0.0, 0.0])

    """
    A simple move rule. The intended effect is to teach our agent to take action 1
    because that is the only possible action to let our agent reach the goal position
    """
    move_rule = [
        np.array([-0.05, -0.05]),
        np.array([0.05, 0.0]),
        np.array([0.0, 0.05]),
        np.array([0.05, 0.05])
    ]

    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.action_space = gym.spaces.Discrete(4)

        self.initial_position = np.array([1.0, 1.0])
        self.agent_position = self.initial_position
        self.previous_agent_position = self.initial_position
        self.steps_taken = 0.0

    def step(self, action):
        # goal: train the agent to stay at (0, 0)

        # update agent position
        assert action in self.action_space
        self.previous_agent_position = self.agent_position  # record previous
        # a simple move rule
        move_delta = self.move_rule[action]
        self.agent_position += move_delta

        out_of_bounds = not self.observation_space.contains(self.agent_position)
        current_distance = np.linalg.norm(self.agent_position - self.goal_position)
        if not out_of_bounds:
            # reward is the reduction (positive indicates reduction) in distance between the agent position and the goal position
            # previous_distance = np.linalg.norm(self.previous_agent_position - self.goal_position)
            #
            # reward = -(current_distance - previous_distance)
            reward = np.linalg.norm(np.array([1.0, 1.0]) - self.agent_position) - self.steps_taken
        else:
            reward = -100

        is_done = out_of_bounds or current_distance <= 1e-5
        observations = self.agent_position

        return observations, reward, is_done, {}

    def reset(self):
        self.steps_taken = 0.0
        self.initial_position = np.array([1.0, 1.0])
        self.agent_position = self.initial_position
        self.previous_agent_position = self.initial_position

        return self.agent_position

    def render(self, mode='human'):
        pass


def main():
    # FIXME: maybe find a better way to specify the initial balance? MAYBE?
    starting_balance = 1e5
    # load stock data
    train_data, test_data = split_train_val()
    # initialize an environment
    env = DummyRLStockTradingEnvironment(stock_data=train_data, starting_balance=starting_balance)
    # initialize a model and an agent
    model = DQN(MlpPolicy, env, learning_rate=1e-3, prioritized_replay=True)
    agent = DummyDQNStockTraderAgent(model)

    # run the environment
    total_learning_timesteps = 2e5
    agent.model.learn(total_timesteps=total_learning_timesteps)

    # TODO: run our agent through our stock market data
    observation_timesteps = int(1e5)
    obs = env.reset()
    for i in range(observation_timesteps):
        action, states = model.predict(obs)
        obs, rewards, is_done, info = env.step(action)
        print(f'Observations: {obs}\nRewards: {rewards}')
        if is_done:
            break
    print('Done!')

    # TODO: run our agent thorugh our validation market


if __name__ == "__main__":
    # run the demo
    env = DummyRLEnvironment()
    model = A2C('MlpPolicy', env, learning_rate=1e-3, verbose=1)
    model.learn(total_timesteps=20000)

    # show the trained model
    max_timesteps = int(1e3)
    obs = env.reset()
    time_step = (0, 0)
    while True:
        print(env.initial_position)
        action, model_state = model.predict(obs)
        print(f'Time step: {(time_step)}\tObservations: {obs}\tAction Taken: {action}')
        obs, reward, is_done, _ = env.step(action)
        if is_done:
            obs = env.reset()
            time_step = (time_step[0] + 1, 0)
        else:
            time_step = (time_step[0], time_step[1] + 1)
    print(f'Terminated:\tObservations: {obs}')

