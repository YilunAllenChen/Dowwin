from apis_slack import SlackAPI
from apis_alpaca import AlpacaAPI
from apis_mongodb import MongoAPI

environment = 'production'


alpaca = AlpacaAPI(environment=environment)
slack = SlackAPI(environment=environment)
mongo = MongoAPI(environment=environment)
