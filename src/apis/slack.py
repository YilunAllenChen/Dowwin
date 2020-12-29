'''
Author: Allen Chen

This module provides APIs to interact with our Slack bot.
'''

import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from pprint import pprint
from src.util.logging import log_error
from src.config.config import get_global_config


class SlackAPI():
    def __init__(self, environment='development'):
        '''
        Slack API provides functionalities to interact with Darwin Robotic's Slack Bot, Niwrad.
        
        :param environment: Can be 'production', 'development', etc.
        '''
        GLOBAL_CONFIG = get_global_config(environment)
        self.client = WebClient(token=GLOBAL_CONFIG['SLACK_BOT_TOKEN'])

    def send_to_dowwin_channel(self, content: str) -> None:
        '''
        Sends a message to the project_dowwin channel on Slack.

        :param content: the content you want to send.
        '''
        try:
            if len(content) > 0:
                response = self.client.chat_postMessage(
                    channel="C01G1FJUVC2",
                    text=content
                )
        except SlackApiError as e:
            log_error(e.response['error'])
        except Exception as e:
            log_error(e)

if __name__ == "__main__":
    slack = SlackAPI()
    slack.send_to_dowwin_channel("Test message")