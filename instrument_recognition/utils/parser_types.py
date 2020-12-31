
def str2bool(value):
    """
    argparse type
    allows you to set bool flags in the following format:
        python program.py --bool1 True --bool2 false ... etc
    """              
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    elif value.lower() in {'none'}:
        return None
    raise ValueError(f'{value} is not a valid boolean value')

def noneint(value):
    if isinstance(value, int):
        return value
    elif isinstance(value, str):
        if value.lower() in {'none',}:
            return None
        else:
            return int(value)
        
    raise ValueError(f'bad input to noneint type')