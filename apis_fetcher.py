'''
*WIP* Prototyping fetcher classes used in DAE.

Currently supporting Yahoo Finance API backed by aiohttp client session. Rate is about 8 calls/sec with 99% successful attempts.
'''

import aiohttp
import asyncio
import re
import json
import datetime as dt

from random import choice
from data_stock_symbols import stock_symbols
from utils_decorators import send_error_to_slack_on_exception

class Problem():
    '''
    Generic Struct-like class to store timestamped exceptions.
    '''
    def __init__(self, error_msg: str):
        self.timestamp = dt.datetime.now()
        self.error_msg = error_msg

class Source():
    '''
    Class should be designed in inheritance-based structure.
    All 'Sources' should be able to acquire relative information from its url, process them and store them into our database.

    '''
    id = 0
    def __init__(self):

        # Assign unique id.
        self.id = Source.id
        Source.id += 1

        # Start an asyncio http session
        self.session = aiohttp.ClientSession()

        # counter for successful fetches. Used in system statistics
        self.successful_fetch_counter = 0

        # Temperary cache for timestamped problems. 'maximum_problems_per_minute' defines the max number of 
        # problems that can be stored in problems before an error is thrown and handled.
        self.problems = []
        self.maximum_problems_per_minute = 10

        # Current status.
        self.status = 'stopped'

        # If in poor health, how much time we're giving this source to recover/rest
        self.suspension_time_on_poor_health = 300

        # When will it be available next.
        self.next_available_time = dt.datetime.now()

    @send_error_to_slack_on_exception
    def _health_check(self) -> None:
        '''
        TODO
        '''
        # an error will be thrown when _health_check is executed, indicating that this source has encountered too many errors in
        # the last minute and is hence in poor health. 
        def only_keep_problems_in_last_min(problem):
            threshold = dt.datetime.now() - dt.timedelta(minutes=1)
            return problem.timestamp > threshold

        filtered = filter(only_keep_problems_in_last_min ,self.problems)
        self.problems = [problem for problem in filtered]
        if len(self.problems) >= 10:
            error_message = [f"{problem.timestamp.strftime('%m%d%Y %H:%M:%S')} | {type(problem.error_msg)} : {problem.error_msg}" \
                for problem in self.problems]
            self.pause()
            raise Exception(f"Source [{self.id}] is in poor health and is hence suspended for {self.suspension_time_on_poor_health} seconds. Details: {error_message}")
            
    def _is_ready_to_fetch(self) -> bool:
        '''
        TODO
        '''
        if self.status == 'running':
            return True
        if self.status == 'stopped':
            if dt.datetime.now() > self.next_available_time:
                self.status = 'running'
                return True
        return False

    def pause(self) -> None:
        '''
        TODO
        '''
        if self.status == 'running':
            self.status = 'stopped'
            self.next_available_time = dt.datetime.now() + dt.timedelta(seconds=self.suspension_time_on_poor_health)

    async def fetch(self, symbol: str):
        '''
        TODO
        '''
        self._health_check()
        while not self._is_ready_to_fetch():
            await asyncio.sleep(1)

        try:
            async with self.session.get(f'https://finance.yahoo.com/quote/{symbol}') as resp:
                html = await resp.text()
                json_str = html.split('root.App.main =')[1].split(
                    '(this)')[0].split(';\n}')[0].strip()
                data = json.loads(json_str)[
                    'context']['dispatcher']['stores']['QuoteSummaryStore']
                # return data
                data = json.dumps(data).replace('{}', 'null')
                data = re.sub(
                    r'\{[\'|\"]raw[\'|\"]:(.*?),(.*?)\}', r'\1', data)
                data = json.loads(data)
                self.successful_fetch_counter += 1
        except Exception as e:
            self.problems.append(Problem(e))


starting_time = dt.datetime.now()


async def task(source):
    while(True):
        try:
            random_stock = choice(stock_symbols)
            await source.fetch(choice(stock_symbols))
        except Exception as e:
            # print(f"Exception Encountered with Source [{source.id}] fetching {random_stock}: {e}")
            pass
    
async def report(sources):
    while(True):
        calls = sum([source.successful_fetch_counter for source in sources])
        time_passed = (dt.datetime.now() - starting_time).total_seconds()
        rate = calls / time_passed
        print(f"Cumulative calls:  {calls}, time passed: {time_passed}, rate at {rate}/sec")
        await asyncio.sleep(30)


async def main():
    sources = [Source() for _ in range(1)]

    tasks = [asyncio.ensure_future(task(sources[ndx])) for ndx in range(1)]
    tasks.append(asyncio.ensure_future(report(sources)))

    try:
        returns = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        print(e)
        calls = sum([source.counter for source in sources])
    ending_time = dt.datetime.now()

    time_diff = ending_time - starting_time
    print(time_diff)
    print(calls)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
