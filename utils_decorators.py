from utils_logging import log_error
from apis_import import slack
from datetime import datetime

def send_error_to_slack_on_exception(func):
    '''
    Decorator prevents program from crashing by returning None on exception thrown. 
    At the same time, it will post to dowwin channel signifying that program was going to crash. 
    '''
    def decorated_send_error_to_slack_on_exception(*args, **kwargs):
        res = None
        try:
            res = func(*args, **kwargs)
        except Exception as e:
            slack.send_to_dowwin_channel(f"*[ERROR]* [{datetime.now().strftime('%Y%m%d %H:%M:%S')}] Unable to run [{func.__name__}]: {str(e)}")
    return decorated_send_error_to_slack_on_exception

def async_send_error_to_slack_on_exception(func):
    '''
    Decorator prevents program from crashing by returning None on exception thrown. 
    At the same time, it will post to dowwin channel signifying that program was going to crash. 
    '''
    async def async_decorated_send_error_to_slack_on_exception(*args, **kwargs):
        res = None
        try:
            res = await func(*args, **kwargs)
        except Exception as e:
            slack.send_to_dowwin_channel(f"*[ERROR]* [{datetime.now().strftime('%Y%m%d %H:%M:%S')}] Unable to run [{func.__name__}]: {str(e)}")
    return async_decorated_send_error_to_slack_on_exception
