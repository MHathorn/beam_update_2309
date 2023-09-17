import logging

from colorlog import ColoredFormatter
from datetime import datetime
from os import makedirs
from os.path import exists, join

log_dir = "../log"
if not exists(log_dir):
    makedirs(log_dir)
now = datetime.now()  # current date and time
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
logging.basicConfig(filename=join(log_dir, f"{date_time}_log.txt"))
LOG_LEVEL = logging.DEBUG
LOGFORMAT = (
    "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
)
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(LOGFORMAT)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
log = logging.getLogger("pythonConfig")
log.setLevel(LOG_LEVEL)
log.addHandler(stream)
