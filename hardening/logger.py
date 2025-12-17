import json
from datetime import datetime

def log_event(**kwargs):
    record = {
        "ts": datetime.utcnow().isoformat(),
        **kwargs,
    }
    print(json.dumps(record, default=str))
