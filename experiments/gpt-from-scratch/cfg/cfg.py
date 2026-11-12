import torch
from pathlib import Path

# 以本文件位置定位项目根目录（cfg/ 的上一级），不依赖从哪个目录执行脚本
WORKFOLDER = Path(__file__).resolve().parent.parent
FILEPATH = Path(__file__)

DATASETMAP = {
    "SMS":{
        "url": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        "name": "SMSSpamCollection"}
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # NEW











