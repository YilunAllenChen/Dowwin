from src.util.logging import log_error
from src.apis.slack import SlackAPI
from datetime import datetime


slack = SlackAPI()

# Reference
# def failsafe(func):
#     def decorated_failsafe(*args, **kwargs):
#         res = None
#         try:
#             res = func(*args, **kwargs)
#         except Exception as e:
#             log_error("Exception thrown running function [\033[93m {} \033[0m], error message: {}".format(str(func.__name__),str(e)))
#         return res
#     return decorated_failsafe

# def async_failsafe(func):
#     async def decorated_async_failsafe(*args, **kwargs):
#         res = None
#         try:
#             res = await func(*args, **kwargs)
#         except Exception as e:
#             log_error("Exception thrown running function [\033[93m {} \033[0m], error message: {}".format(str(func.__name__),str(e)))
#         return res
#     return decorated_async_failsafe


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
