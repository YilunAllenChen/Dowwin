
_database_url = "mongodb+srv://trading_agent:Lc2gwI5aUjnVXYoU@cluster0.y7n4w.mongodb.net/Dowwin?retryWrites=true&w=majority"


def set_env_to_debug():
    global _database_url
    _database_url = "INSERT DEBUG URL HERE"

def DATABASE_URL():
    return _database_url