"""config.py"""

from pathlib import Path
import os
import getpass
getpass.getuser()

CHROMEDRIVER_FULL_PATH__ = os.environ.get('CHROMEDRIVER_FULL_PATH')
KOYFIN_URL = "https://app.koyfin.com/"
KOYFIN_PWD = os.environ.get('KOYFIN_PASSWORD')
KOYFIN_EMAIL = os.environ.get('KOYFIN_EMAIL')
DOWNLOADS_PATH = Path(r'C:\Users') / getpass.getuser() / 'Downloads'
KOYFIN_DATA_FOLDER_PATH = DOWNLOADS_PATH
