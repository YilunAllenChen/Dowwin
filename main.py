from __GLOBAL_CONFIGS import DATABASE_URL, debug_env
from _utils_decorators import async_failsafe, failsafe 
import asyncio

@failsafe
def throw_shit():
    raise Exception("SHIT")

@async_failsafe
async def throw_this():
    print("YO")
    raise Exception("SHIT")



if __name__ == "__main__":
    pass