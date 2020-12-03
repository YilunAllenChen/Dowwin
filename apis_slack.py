import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from pprint import pprint
from utils_logging import log_error


SLACK_BOT_TOKEN = "xoxb-1518310205110-1544392027987-rY62XGdCfY9dFtN5sII2OTRm"


slack_token = SLACK_BOT_TOKEN
client = WebClient(token=slack_token)

class SlackAPI():
    def __init__(self):
        #TODO: Setup environment variables here. Don't forget to point function variables to these instead of global ones.
        pass

    def send_to_dowwin_channel(self, content: str) -> None:
        try:
            if len(content) > 0:
                response = client.chat_postMessage(
                    channel="C01G1FJUVC2",
                    text=content
                )
        except SlackApiError as e:
            log_error(e.response['error'])
        except Exception as e:
            log_error(e)
