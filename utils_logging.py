from datetime import datetime

def log_ok(msg: str) -> None:
    print('[\033[92m  OK   \033[0m] ' + str(msg))
    
def log_error(msg: str) -> None:
    print('[\033[91m ERROR \033[0m] ' + str(msg))

def log_info(msg: str) -> None:
    print('[\033[94m INFO  \033[0m] ' + str(msg))

def log(msg: str) -> None:
    log_info(msg)