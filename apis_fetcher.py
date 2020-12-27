import aiohttp
import asyncio
import re
import json
import datetime as dt

from random import choice
from data_stock_symbols import stock_symbols

from utils_decorators import send_error_to_slack_on_exception

class Source():


    id = 0

    def __init__(self):
        self.id = Source.id
        Source.id += 1
        self.counter = 0
        self.session = aiohttp.ClientSession()
        self.status = 'stopped'
        self.resting_time = 60
        self.next_available_time = dt.datetime.now()

    def status_check(self):
        if self.status == 'running':
            return True
        if self.status == 'stopped':
            if dt.datetime.now() > self.next_available_time:
                self.status = 'running'
                return True
        return False


    def pause(self):
        if self.status == 'running':
            self.status = 'stopped'
            self.next_available_time = dt.datetime.now() + dt.timedelta(seconds=self.resting_time)
            # report to slack.

    async def fetch(self, symbol: str):
        # Check status first

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
            if resp.status != 200:
                raise Exception()
            self.counter += 1


starting_time = dt.datetime.now()
problematic_stock_list = set()


async def task(source):
    while(True):
        try:
            random_stock = choice(stock_symbols)
            await source.fetch(choice(stock_symbols))
        except Exception as e:
            print(f"Exception Encountered with Source [{source.id}] fetching {random_stock}: {e}")
            problematic_stock_list.add(random_stock)
            print(f"Current problematic stock list: {problematic_stock_list}")
    
async def report(sources):
    while(True):
        calls = sum([source.counter for source in sources])
        time_passed = (dt.datetime.now() - starting_time).total_seconds()
        rate = calls / time_passed
        # print(f"Cumulative calls:  {calls}, time passed: {time_passed}, rate at {rate}/sec")
        await asyncio.sleep(3)


async def main():
    sources = [Source() for _ in range(10)]

    tasks = [asyncio.ensure_future(task(sources[ndx])) for ndx in range(10)]
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
