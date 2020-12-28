'''
Author: Allen Chen

This is an example of entry point to CORE. Pay close attention to the import syntax - they're relative to this repo.
Don't try to run this by doing 'python3 main.py' under this directory. Try to add your Target in Makefile under the root dir,
and call './run YOUR_TARGET_NAME' from root. 
'''

from src.CORE.class_TradeBot import TradeBot
from src.util.logging import log_ok, log_info, log_error


bot = TradeBot()
log_info(f"Just initialized a bot named {bot.name}")

log_ok(f"Bot is given cash: {bot.cash}")

log_error("Nothing else to do ! :(")