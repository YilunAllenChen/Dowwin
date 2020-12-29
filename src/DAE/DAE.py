'''
Author: Allen Chen

Data Acquitision Engine entry point. Currently under development.
'''

from src.DAE.yahoofinance import Source
from src.util.logging import *
from src.data.stock_symbols import stock_symbols
import asyncio
import datetime as dt



starting_time = dt.datetime.now()

# number of sources defines how many asynchronous web scraping coroutines will run at the same time.
num_of_sources = 10
idle_time = 0.5
rate_max = 8

# A task lets a source fetches information on its dedicated segment of stocks and udpates them to the database.
async def task(source):
    portion = int(len(stock_symbols)/num_of_sources)
    dedicated_starting_ndx = int(portion * source.id)
    dedicated_ending_ndx = dedicated_starting_ndx + portion
    dedicated_list = stock_symbols[dedicated_starting_ndx:dedicated_starting_ndx+portion]
    log_ok(f"Source [{source.id}] is assigned stocks: {dedicated_list}")
    while(True):
        for stock in dedicated_list:
            try:
                await source.fetch(stock)
                await asyncio.sleep(idle_time)
            except Exception as e:
                print(f"Exception Encountered with Source [{source.id}] fetching {stock}: {e}")
                pass

async def moderator(sources):
    global idle_time
    while(True):
        calls = sum([source.successful_fetch_counter for source in sources])
        time_passed = (dt.datetime.now() - starting_time).total_seconds()
        rate = calls / time_passed
        if rate > rate_max:
            idle_time += 0.05
        else:
            if idle_time > 0:
                idle_time -= 0.05
        await asyncio.sleep(10)




async def cli(sources):
    while(True):
        command = await loop.run_in_executor(None, input, "Please input command ('help' for help): ")
        if 'status' in command:
            for source in sources:
                msg = f"Source {source.id} : "
                if source.status == 'stopped':
                    msg += '[\033[91m STOPPED \033[0m] '
                elif source.status == 'running':
                    msg += '[\033[92m   OK   \033[0m] '
                else:
                    msg += '[\033[94m UNKNOWN \033[0m] '
                print(msg)

        if 'report' in command:
            calls = sum([source.successful_fetch_counter for source in sources])
            time_passed = (dt.datetime.now() - starting_time).total_seconds()
            rate = calls / time_passed
            log_info(f"********REPORT********\nCumulative calls: {calls} \nTime passed: {time_passed}\nRate at {rate}/sec\nIdle Time: {idle_time}")

        if 'help' in command:
            print("Current support: 'report' and 'status'")

async def main():

    # Initialize a number of sources. Each source is given an asyncio http client session. 
    sources = [Source() for _ in range(num_of_sources)]

    # Initialize tasks and assign each task a source to work with.
    tasks = [asyncio.ensure_future(task(sources[ndx])) for ndx in range(num_of_sources)]

    # Finally add a report task to monitor the process.
    tasks.append(asyncio.ensure_future(cli(sources)))
    tasks.append(asyncio.ensure_future(moderator(sources)))


    # If there's a critical error, stop the process and log the final result.
    try:
        returns = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        log_error(e)
        calls = sum([source.counter for source in sources])
    ending_time = dt.datetime.now()
    time_diff = ending_time - starting_time
    log_info(time_diff)
    log_info(calls)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
