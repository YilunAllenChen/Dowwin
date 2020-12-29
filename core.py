""" Core Module of Dowwin

"""

from pathlib import Path
from uuid import uuid4

import gym
import stable_baselines
from stable_baselines.common.base_class import BaseRLModel


class TraderAgent:
    """Base class for all TraderAgents. TraderAgents must act in a given environment."""
    def save(self, save_path=None) -> None:
        """Saves the information contained in this RLTraderAgent"""
        pass


class RLTraderAgent(TraderAgent):

    """
    RLTraderAgent is an Agent that makes decision using reinforcement learning techniques.
    RLTraderAgent uses stable-baselines models for decision making.
    RLTraderAgent also wraps some auxiliary functionality.
    """

    def __init__(self, model: BaseRLModel, **kwargs) -> None:

        self.model = model  # a stable-baseline compatible model
        # TODO: kwargs is a temporary fix for extensibility, replace with optional arguments.
        # TODO: support additional functionality here.
    pass

    # TODO: support additional functionality below
    def save_model(self, save_path=None, save_name=None) -> None:
        if save_path is None:
            save_path = Path.cwd()
        if not save_path.exists():
            save_path.mkdir(parents=True)
        if save_name is None:
            save_name = f'Unnamed Model {uuid4()}'
        self.model.save(str(save_path / save_name))

    def save(self, save_path=None, model_save_name=None) -> None:
        if save_path is None:
            save_path = Path.cwd()
        if not save_path.exists():
            save_path.mkdir(parents=True)
        super().save(save_path=save_path)
        # also save the model
        self.save_model(save_path, model_save_name)

    def train(self):
        pass
    def evaluate(self):
        pass

    # NOTE: load the model using stable-baseline method. example: self.model.load(...) or self.model.load_parameters(...)



class StockTraderAgent(RLTraderAgent):
    pass


class ForeignExchangeTraderAgent(RLTraderAgent):
    pass


class ProfileManagerAgent(RLTraderAgent):
    pass


class TradingEnvironment:
    pass


class RLTradingEnvironment(gym.Env):

    def __init__(self) -> None:
        super().__init__()

        # TODO: support additional functionality here

    pass


class RLStockTradingEnvironment(RLTradingEnvironment):
    pass


class RLForeignExchangeTradingEnvironment(RLTradingEnvironment):
    pass


class RLProfileManagementTradingEnvironment(RLTradingEnvironment):
    pass
