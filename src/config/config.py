'''
Author: Allen Chen

This module handles the global config of the environment, and should be able to toggle between dev and prod when instructed.

TODO: There must be a better way to do this...
'''


import yaml
from src.util.logging import log_error

def get_global_config(postfix: str) -> object:
    '''
    function obtains the global config variables and return as a dict.

    :param postfix: postfix that specifies the environment. For example, postfix='production' corresponds to config_production.yaml.
    :return: object(usually dict) that contains all global config.
    '''
    try:
        with open(f"config_{postfix}.yaml") as f:
            obj = yaml.full_load(f)
        return obj
    except Exception as e:
        log_error(f"Unable to open config file in [{postfix}] environment. Using [development] as a default")
        with open(f"config_development.yaml") as f:
            obj = yaml.full_load(f)
        return obj