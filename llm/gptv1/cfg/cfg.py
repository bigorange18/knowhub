import torch
from pathlib import Path

WORKFOLDER = Path.cwd().resolve()
FILEPATH = Path(__file__)

DATASETMAP = {
    "SMS":{
        "url": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        "name": "SMSSpamCollection"}
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # NEW











