from _utils_logging import log_error

def failsafe(func):
    def decorated_failsafe(*args, **kwargs):
        res = None
        try:
            res = func(*args, **kwargs)
        except Exception as e:
            log_error("Exception thrown running function [\033[93m {} \033[0m], error message: {}".format(str(func.__name__),str(e)))
        return res
    return decorated_failsafe


def async_failsafe(func):
    async def decorated_async_failsafe(*args, **kwargs):
        res = None
        try:
            res = await func(*args, **kwargs)
        except Exception as e:
            log_error("Exception thrown running function [\033[93m {} \033[0m], error message: {}".format(str(func.__name__),str(e)))
        return res
    return decorated_async_failsafe
