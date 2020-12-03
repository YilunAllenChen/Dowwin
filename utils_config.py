import yaml

def get_global_config(postfix: str) -> object:
    '''
    function obtains the global config variables and return as a dict.

    :param postfix: postfix that specifies the environment. For example, postfix='production' corresponds to config_production.yaml.
    :return: object(usually dict) that contains all global config.
    '''
    with open(f"config_{postfix}.yaml") as f:
        obj = yaml.full_load(f)
    return obj

