import yaml

def get_global_config(postfix='production') -> object:
    ''
    with open(f"config_{postfix}.yaml") as f:
        obj = yaml.full_load(f)
    return obj



print(get_config('production'))