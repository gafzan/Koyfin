"""config.py"""

from pathlib import Path
import os
import getpass
getpass.getuser()


def get_project_root() -> Path:
    return Path(__file__).parent.parent


CHROMEDRIVER_FULL_PATH__ = os.environ.get('CHROMEDRIVER_FULL_PATH')
KOYFIN_URL = "https://app.koyfin.com/"
KOYFIN_PWD = os.environ.get('KOYFIN_PASSWORD')
KOYFIN_EMAIL = os.environ.get('KOYFIN_EMAIL')
DOWNLOADS_PATH = Path(r'C:\Users') / getpass.getuser() / 'Downloads'
KOYFIN_DATA_FOLDER_PATH = DOWNLOADS_PATH
FINANCIAL_REPORTS_FOLDER_PATH = get_project_root() / 'data/financial_reports'

if not FINANCIAL_REPORTS_FOLDER_PATH.exists():
    os.makedirs(FINANCIAL_REPORTS_FOLDER_PATH)

