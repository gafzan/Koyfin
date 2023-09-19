"""koyfinbot.py"""

import pandas as pd
from pathlib import Path
import datetime
from time import sleep
import logging
import random

from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from tools.general_tools import get_newest_file_paths
from tools.general_tools import DOWNLOADS_PATH

from automation.config import CHROMEDRIVER_FULL_PATH__
from automation.config import KOYFIN_URL
from automation.config import KOYFIN_PWD
from automation.config import KOYFIN_EMAIL
from automation.config import KOYFIN_DATA_FOLDER_PATH

# logger
formatter = logging.Formatter('%(asctime)s : %(module)s : %(funcName)s : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


class WebBot:
    def __init__(self):
        logger.debug('Initializing a web bot')
        service = Service(executable_path=CHROMEDRIVER_FULL_PATH__)
        self.driver = webdriver.Chrome(service=service)

    def click_button(self, x_path: str):
        logger.debug(f'Find element with x path: {x_path}')
        btn = self.driver.find_element(by=By.XPATH, value=x_path)
        logger.debug(f'Click button with x path: {x_path}')
        btn.click()

    def type_into_box(self, x_path: str, input_txt: str):
        logger.debug(f'Find element with x path: {x_path}')
        box = self.driver.find_element(by=By.XPATH, value=x_path)
        logger.debug(f'Input text "{input_txt}" in box x path: {x_path}')
        box.send_keys(input_txt)

    @staticmethod
    def bot_sleep(sleep_time: int = None):
        if not sleep_time:
            sleep_time = random.randint(1, 5)
        logger.debug(f'Sleep for {sleep_time} seconds...')
        sleep(sleep_time)
        logger.debug('Done sleeping')


class KoyfinBot(WebBot):
    def __init__(self, remove_min_market_cap: bool = False):
        """
        Initiator for the KoyfinBot class
        :param remove_min_market_cap: bool by default there is a minimum USD 50 million market cap filter applied
        """
        super().__init__()
        self.driver.get(KOYFIN_URL)
        self._screen_number = 0
        self.remove_min_market_cap = remove_min_market_cap

    def login(self):
        """
        Logs in with pre-defined credentials
        :return: None
        """
        logger.info('Login to koyfin.com')
        # click the 'login' button
        self.click_button(x_path='//*[@id="root"]/div[1]/section/div[1]/div[2]/button[2]/label')
        self.bot_sleep()
        # type the email
        self.type_into_box(x_path='//*[@id="root"]/div[1]/div/div[2]/div/form/div[2]/div[1]/div/div[2]/input',
                           input_txt=KOYFIN_EMAIL)
        # type the password
        self.type_into_box(x_path='//*[@id="root"]/div[1]/div/div[2]/div/form/div[2]/div[2]/div/div[2]/input',
                           input_txt=KOYFIN_PWD)
        self.click_button(x_path='//*[@id="root"]/div[1]/div/div[2]/div/form/div[3]/button/label')
        self.bot_sleep(4)

    def go_to_my_screens(self):
        """
        Clicks 'My Screens' and maximizes the window (if not some widgets will not be able to be activated later)
        :return:
        """
        logger.info('Go to my screen page')
        self.driver.maximize_window()
        # click the "screens" button
        self.bot_sleep(2)
        self.click_button('//*[@id="root"]/div[1]/section/div[2]/div[1]/div/div/div[1]/div/div[2]/a[5]/div[2]/div')
        self.bot_sleep(2)

    def add_new_screen(self):
        """
        Assumes that the "Screens" page is loaded.
        :return:
        """
        logger.info('Add new screen')
        # click the '+' symbol to create a new screener
        self.click_button(
            '//*[@id="root"]/div[1]/section/div[2]/div[2]/div[1]/div[1]/div/div/div[1]/div[2]/div/button/span/i')
        self.bot_sleep(sleep_time=2)

        # remove 'Trading Region' criteria
        self.click_button('/html/body/div[3]/div/div[2]/div/div[2]/div/div/div[3]/div[1]/div/div[1]/div[3]/div/div/div/div[1]/div/div[1]/span[2]/i')
        self.bot_sleep(2)

    def name_screen(self, name: str):
        """
        Set the name of the screen
        :param name: str
        :return: None
        """
        logger.info(f'Name the screen "{name}"')
        screen_name_bar_element = self.driver.find_element('xpath', '/html/body/div[3]/div/div[2]/div/div[2]/div/div/div[3]/div[1]/div/div[3]/div[3]/div/div/div[1]/div')
        screen_name_bar_element.click()
        active_element = self.driver.execute_script("return document.activeElement")
        active_element.send_keys(Keys.CONTROL + "a")
        active_element.send_keys(Keys.DELETE)
        active_element.send_keys(name)

    def add_universe_criteria(self, criteria: dict):
        """
        Assumes that the criteria page is loaded. Configures a criteria filter on the screen page.
        :param criteria: dict key = criteria (str), values = criteria items (list of str)
        e.g. {'country': ['Sweden']}
        :return:
        """

        # check that the specified dictionary has the correct keys
        uni_criteria_list = ['country', 'description', 'etf constituents', 'exchange', 'industry', 'sector',
                             'trading country', 'trading region']
        self.check_criteria(criteria=criteria, eligible_criteria=uni_criteria_list)

        logger.info(f'Adding universe criteria\n{self.str_uni_criteria(criteria=criteria)}')

        if self.remove_min_market_cap:
            # by default there is a minimum USD 50 million market cap filter applied
            self.click_button('/html/body/div[3]/div/div[2]/div/div[2]/div/div/div[3]/div[1]/div/div[2]/div[3]/div/div/div/div/div/div[1]/span[1]/div/div/div[1]/span[1]/i')

        # loop through each criteria (e.g. 'country') and add the specified items (e.g. 'sweden') per criteria
        criteria_num = 1
        for criteria, items in criteria.items():
            # add a universe criteria
            self.click_button(
                f'/html/body/div[{min(3 + self._screen_number * 2, 5)}]/div/div[2]/div/div[2]/div/div/div[3]/div[1]/div/div[1]/div[3]/div/div/button/label'
            )
            self.bot_sleep(3)

            self.click_button(f'/html/body/div[{min(4 + self._screen_number * 2, 6)}]/div/div/div/div[2]/div/div[{uni_criteria_list.index(criteria.lower()) + 1}]/div/label')  # select the type of criteria

            # add the items for the specified criteria
            self._add_universe_criteria_items(criteria_num=criteria_num, items=items)

            # need to remove the criteria from the list since the XPATH is dynamic based on the already selected
            # universe criteria
            uni_criteria_list.remove(criteria.lower())
            criteria_num += 1
        self.bot_sleep(5)

    def _add_universe_criteria_items(self, criteria_num: int, items: list):
        """
        Selects items from a drop down list after entering text in a search box. The first item filtered will be
        selected.
        :param criteria_num: int
        :param items: list of str
        :return: None
        """
        # click the drop down list to add the criteria items
        self.click_button(
            f'/html/body/div[{min(3 + self._screen_number * 2, 5)}]/div/div[2]/div/div[2]/div/div/div[3]/div[1]/div/div[1]/div[3]/div/div/div/div[{criteria_num + 1}]/div/div[1]/span[1]/div/button/label')
        self.bot_sleep(1)

        if not isinstance(items, list):
            items = [items]

        # loop through each items to add
        for item in items:
            # search the drop down list by entering text in a search box
            search_bar_element = self.driver.find_element('xpath',
                                                          f'/html/body/div[{min(4 + self._screen_number * 2, 6) + criteria_num}]/div/div/div/div[2]/div/div[2]/div/div/div')

            search_bar_element.click()
            active_element = self.driver.execute_script("return document.activeElement")
            active_element.send_keys(item)

            # select the items filtered based on the search
            self.bot_sleep(1)
            try:
                # this will work if only one item is displayed after the search
                self.click_button(f'/html/body/div[{min(4 + self._screen_number * 2, 6) + criteria_num}]/div/div/div/div[2]/div/div[3]/div/div/div/label')
            except:
                # if there are more than one item after the search for example searching for 'Airlines' under
                # Industry will return two items: Airlines, Passenger Airlines
                self.click_button(
                    f'/html/body/div[{min(4 + self._screen_number * 2, 6) + criteria_num}]/div/div/div/div[2]/div/div[3]/div/div[1]/div/label')

            # clear the content in the search bar by clicking, select all and delete
            search_bar_element.click()
            active_element = self.driver.execute_script("return document.activeElement")
            active_element.send_keys(Keys.CONTROL + "a")
            active_element.send_keys(Keys.DELETE)
            search_bar_element.click()

        # click one of the criteria box to remove the drop down list
        # this will also refresh the number of stocks that has passed the filter
        self.click_button(f'/html/body/div[{min(3 + self._screen_number * 2, 5)}]/div/div[2]/div/div[2]/div/div/div[3]/div[1]/div/div[1]/div[3]/div/div/div/div[1]/div/div[1]/div/label')

    def click_modify_criteria(self, prev_num_uni_criteria: int):
        """
        Click the 'Modify Criteria' button that will return a screen for changing the criteria
        :param prev_num_uni_criteria: int
        :return: None
        """
        logger.info('Modify criteria')
        # click the 'Modify Criteria' button
        self.click_button('//*[@id="root"]/div[1]/section/div[2]/div[2]/div[1]/div[1]/div/div/div[2]/div[1]/button/label')

        self.bot_sleep(sleep_time=1)
        # delete the previous universe criteria
        for i in range(prev_num_uni_criteria):
            self.click_button('/html/body/div[5]/div/div[2]/div/div[2]/div/div/div[3]/div[1]/div/div[1]/div[3]/div/div/div/div[2]/div/div[1]/span[2]/i')

    def run_screener(self, max_number: int = 2000):
        """
        Clicks 'Create Screen' unless the number of filtered stocks exceeds a specified maximum (raises an error if too
        many stocks are filtered)
        :param max_number: int
        :return: None
        """
        filtered_stocks_num_str = self.driver.find_element('xpath',
                                                           f'/html/body/div[{min(3 + self._screen_number * 2, 5)}]/div/div[2]/div/div[2]/div/div/div[3]/div[1]/div/div[2]/div[3]/div/div/div/div/div/div[2]/label[1]').text
        filtered_stocks_num = int(float(filtered_stocks_num_str.replace(',', '')))
        if filtered_stocks_num > max_number:
            raise ValueError(f"number of filtered stocks ({filtered_stocks_num}) exceeds {max_number}."
                             f"\nOnly {max_number} stocks can be displayed and downloaded")
        else:
            logger.info(f"Run screener with {filtered_stocks_num} stocks")
            # click 'Create Screen'
            self.click_button(
                x_path=f'/html/body/div[{min(3 + self._screen_number * 2, 5)}]/div/div[2]/div/div[2]/div/div/div[3]/div[2]/div[2]/button[2]/label')
        self._screen_number += 1
        self.bot_sleep(5)

    def add_data_columns(self, data_columns_config: dict):
        """
        Edits the data columns and adds relevant data specified in a dictionary
        :param data_columns_config: dict
            keys = str name of the data
            value = str or list of str format of the data (e.g. 'Last Twelve Months'). If not format (e.g. 'Industry'
            set value to None)
        :return:
        """
        logger.info('Edit data columns')
        # click 'edit columns'
        self.click_button(
            '//*[@id="root"]/div[1]/section/div[2]/div[2]/div[1]/div[1]/div/div/div[2]/div[2]/div[2]/button[1]/span/i')

        search_bar = self.driver.execute_script("return document.activeElement")

        # loop through each data and type
        for data_name, data_metas in data_columns_config.items():
            logger.info(f"Lookup '{data_name}'")
            search_bar.send_keys(data_name)  # insert the name of the data in the search bar
            self.bot_sleep(3)
            self.click_button('//*[@id="koy-data-source-item-0"]/div[1]/div[2]/div[1]')  # select the top element

            # if we are only adding data that does not have any transformation (like 'Sector') col_vals is None
            if data_metas:
                # first convert data meta to a list if not already
                if not isinstance(data_metas, list):
                    data_metas = [data_metas]
                for data_meta in data_metas:
                    logger.info(f"{data_meta} ({data_name})")
                    data_meta_element = self.driver.find_element(By.XPATH, f"//*[contains(text(),'{data_meta}')]")
                    data_meta_element.click()
                    self.bot_sleep(2)

                # press the 'back' button to go back to the search bar (only used when the data requires additional
                # inputs like data format)
                self.click_button(
                    '/html/body/div[4]/div/div[2]/div/div[2]/div/section/div[2]/div[2]/div/div[1]/div/button/label')

            # clear all elements in search bar
            search_bar.send_keys(Keys.CONTROL + "a")
            search_bar.send_keys(Keys.DELETE)

        # click 'x'
        self.click_button('/html/body/div[4]/div/div[2]/div/div[2]/button/div/span/i')

    def download_screener_result(self):
        """
        Clicks 'Download' on the Screens page and moves the file from 'Downloads' to the defined download path
        :return:
        """
        logger.info('Wait for 30 seconds then download the results in excel')
        self.bot_sleep(30)
        self.click_button('//*[@id="root"]/div[1]/section/div[2]/div[2]/div[1]/div[1]/div/div/div[1]/div[1]/div[2]/div/button[2]/label')
        self.bot_sleep(3)

    @staticmethod
    def check_criteria(criteria, eligible_criteria):
        if any(key.lower() not in eligible_criteria for key in criteria.keys()):
            raise ValueError("the keys in 'criteria' dict needs to be one of '%s'" % "', '".join(eligible_criteria))

    @staticmethod
    def str_uni_criteria(criteria: dict):
        """
        Returns a string that describes the universe criteria
        :param criteria: dict
        :return: str
        """
        s = ""
        i = 1
        for key, value in criteria.items():
            if not isinstance(value, list):
                value = [value]
            s += f'Criteria {i}: {key.upper()} = %s\n' % ", ".join(value)
            i += 1
        return s


class KoyfinDataDownloader:

    def __init__(self, screen_criteria: {dict, list}):
        self.screen_criteria = screen_criteria

    def download_data(self, data_config: dict = None) -> None:
        """
        Launches a Koyfin bot and downloads data according to specified criteria screener. The result will be in csv
        files in the downloads folder
        :return: None
        """

        # set up the Koyfin bot, login and go to the screen page
        kf_bot = KoyfinBot()
        kf_bot.login()
        kf_bot.go_to_my_screens()

        # loop though each screen configuration and download the result as a csv file
        for i, screen in enumerate(self.screen_criteria):
            logger.info(f'Setting up screen #{i + 1}')
            if i == 0:
                kf_bot.add_new_screen()
                kf_bot.name_screen(name=f'Koyfin Bot Screen')
            else:
                kf_bot.click_modify_criteria(prev_num_uni_criteria=len(self.screen_criteria[i - 1]))
            kf_bot.add_universe_criteria(criteria=screen)
            kf_bot.run_screener()
            kf_bot.add_data_columns(data_columns_config=data_config)
            kf_bot.download_screener_result()

    def get_latest_downloaded_data_df(self) -> pd.DataFrame:
        """
        Looks up the latest downloaded csv files and merges the result into a DataFrame
        :return: DataFrame
        """
        # list of file paths to the latest downloaded csv files
        newest_files = get_newest_file_paths(path=DOWNLOADS_PATH, suffix='csv', num_files=len(self.screen_criteria))
        df = None  # initialize result
        for file_path in newest_files:  # for each file path load the data into a DataFrame and merge it
            if df is None:
                df = pd.read_csv(file_path, index_col=0)
            else:
                df = pd.concat([df, pd.read_csv(file_path, index_col=0)])
        return df

    def save_data(self, file_name: str = None, save_folder_path: str = None) -> None:
        """
        Saves the merged data from the latest downloaded csv files in a specified folder
        :param file_name: str (if not specified the name of the file will be 'koyfin_data')
        :param save_folder_path: str
        :return: None
        """

        downloaded_data_df = self.get_latest_downloaded_data_df()

        # save the new file
        if not save_folder_path:
            save_folder_path = KOYFIN_DATA_FOLDER_PATH
        else:
            save_folder_path = Path(save_folder_path)
        if not file_name:
            file_name = f'koyfin_data_{datetime.date.today().strftime("%Y%m%d")}'
        save_file_path = save_folder_path / f'{file_name}.csv'
        downloaded_data_df.to_csv(save_file_path)
        logger.info(f"Saved in {save_file_path}")

    @property
    def screen_criteria(self):
        return self._screen_criteria

    @screen_criteria.setter
    def screen_criteria(self, screen_criteria: {dict, list}):
        if isinstance(screen_criteria, list):
            self._screen_criteria = screen_criteria
        elif isinstance(screen_criteria, dict):
            self._screen_criteria = [screen_criteria]
        else:
            raise ValueError("screen_criteria can only be specified as a dict or list of dict")

