"""koyfinbot.py"""

import pandas as pd
import numpy as np
from time import sleep
import logging
import random

from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains

from tools.general_tools import get_newest_file_paths
from tools.general_tools import DOWNLOADS_PATH

from automation.config import CHROMEDRIVER_FULL_PATH__
from automation.config import KOYFIN_URL
from automation.config import KOYFIN_PWD
from automation.config import KOYFIN_EMAIL

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
        self.driver.maximize_window()
        self.bot_sleep(2)
        # click the 'login' button
        self.driver.find_elements(By.XPATH, "//*[text()='Log In']")[-1].click()
        self.bot_sleep()
        # type the email
        self.type_into_box(x_path='//*[@id="root"]/div[1]/div/div[2]/div/form/div[2]/div[1]/div/div[2]/input',
                           input_txt=KOYFIN_EMAIL)
        # type the password
        self.type_into_box(x_path='//*[@id="root"]/div[1]/div/div[2]/div/form/div[2]/div[2]/div/div[2]/input',
                           input_txt=KOYFIN_PWD)
        self.click_button(x_path='//*[@id="root"]/div[1]/div/div[2]/div/form/div[3]/button/label')
        self.bot_sleep(4)

    # __________________________________________________________________________________________________________________
    # Financial analysis
    def fa_get_balance_sheet_df(self, company: str, report_type: {str, list}) -> {pd.DataFrame, dict}:
        """
        Returns a balance sheet DataFrame copied from koyfin.com. If 'report_type' is a list of str then returns a dict
        of DataFrames whit keys as report type and DataFrame as values
        :param company: str
        :param report_type: str or list of str 'fy', 'fq', 'ltm'
        :return: DataFrame or dict of DataFrame
        """
        return self.fa_get_financial_report_df(company=company, financial_analysis_code='FA.BS',
                                               report_type=report_type)

    def fa_get_income_statement_df(self, company: str, report_type: {str, list}) -> {pd.DataFrame, dict}:
        """
        Returns a income statement DataFrame copied from koyfin.com. If 'report_type' is a list of str then returns a dict
        of DataFrames whit keys as report type and DataFrame as values
        :param company: str
        :param report_type: str or list of str 'fy', 'fq', 'ltm'
        :return: DataFrame or dict of DataFrame
        """
        return self.fa_get_financial_report_df(company=company, financial_analysis_code='FA.IS',
                                               report_type=report_type)

    def fa_get_cash_flow_statement_df(self, company: str, report_type: {str, list}) -> {pd.DataFrame, dict}:
        """
        Returns a cash flow statement DataFrame copied from koyfin.com. If 'report_type' is a list of str then returns a dict
        of DataFrames whit keys as report type and DataFrame as values
        :param company: str
        :param report_type: str or list of str 'fy', 'fq', 'ltm'
        :return: DataFrame or dict of DataFrame
        """
        return self.fa_get_financial_report_df(company=company, financial_analysis_code='FA.CF',
                                               report_type=report_type)

    def fa_get_financial_report_df(self, company: str, financial_analysis_code: str, report_type: {str, list}) -> {pd.DataFrame, dict}:
        """
        Returns a financial report DataFrame copied from koyfin.com. If 'report_type' is a list of str then returns a dict
        of DataFrames whit keys as report type and DataFrame as values
        :param company: str
        :param financial_analysis_code: str
        :param report_type: str or list of str 'fy', 'fq', 'ltm'
        :return: DataFrame or dict of DataFrame
        """
        # click the main search bar, search for a ticker then a financial data code like 'FA.IS' for income statement
        # and copy the result to the clipboard
        self._click_main_search_bar()
        self._fa_search_for_data(company=company, financial_analysis_code=financial_analysis_code)

        # make the reporting type a list of str
        if isinstance(report_type, str):
            report_type = [report_type]

        # loop through all the reporting types and store each financial account as a DataFrame in a dict
        result = {}  # initialize the result
        for rep_type in report_type:
            self._choose_data_observation_period(data_observation_period=rep_type)
            self.fa_copy_to_clipboard()  # click the 'Copy To Clipboard' button

            # store the clipboard str in a DataFrame and clean up the result
            raw_financial_report_df = pd.read_clipboard("\t")
            financial_report_df = self._adjust_fa_df(raw_financial_report_df=raw_financial_report_df)
            result[rep_type] = financial_report_df

        # return result
        if len(result) == 1:
            return result[report_type[0]]
        else:
            return result

    def _choose_data_observation_period(self, data_observation_period: str) -> None:
        """
        Picks the reporting type for the accounting statement like annual, quarterly or last 12 months rolling
        :param data_observation_period: str
        :return: None
        """
        data_observation_period = data_observation_period.lower()
        if data_observation_period not in ['ltm', 'fq', 'fy']:
            raise ValueError("'data_observation_period' can only be 'LTM', 'FQ' or 'FY")
        text_converter = {
            'ltm': 'Last 12 Months (LTM)',
            'fq': 'Quarterly (FQ)',
            'fy': 'Annual (FY)'
        }
        logger.info(f"reporting type: {text_converter[data_observation_period]}")
        data_obs_element = self.driver.find_element(By.XPATH, f"//*[contains(text(),'{text_converter[data_observation_period]}')]")
        data_obs_element.click()
        self.bot_sleep(2)
        return

    def _click_main_search_bar(self) -> None:
        """
        Clicks on the main search bar in order to search for stocks
        :return: None
        """
        # click the search bar on the main page to get the underlying search pop-up window
        self.driver.find_element(By.CLASS_NAME, "console__label___FRSkP").click()
        self.bot_sleep(2)
        return

    def _fa_search_for_data(self, company: str, financial_analysis_code: str) -> None:
        """
        Searches for a company and then the data for the company
        :param company: str
        :param financial_analysis_code: str
        :return: None
        """
        logger.info(f"Searches for '{company.upper()}' and '{financial_analysis_code}' data")
        # in the pop-up window, enter the company name or ticker and click on the first search result
        # search_box = self.driver.find_element(By.XPATH, "/html/body/div[3]/div/div[2]/div/div[2]/div/div/div[2]/input")
        search_box = self.driver.find_elements(By.CLASS_NAME, 'console-popup__input____f4OM')[-1]
        search_box.send_keys(company)
        self.bot_sleep(2)
        self.click_button(x_path='//*[@id="cb-item-0"]/div[1]/div[1]')

        # input the code for the desired financial analysis page
        search_box.send_keys(f' {financial_analysis_code}')
        self.click_button(x_path='//*[@id="cb-item-0"]/div[2]')
        self.bot_sleep(10)  # the data needs to load first
        return

    def fa_copy_to_clipboard(self) -> None:
        # click the 'Copy To Clipboard' button
        self.click_button(
            x_path='//*[@id="root"]/div[1]/section/div[2]/div[2]/div[1]/div[1]/div[2]/div/div[1]/div[2]/button[1]/label')
        return

    def _adjust_fa_df(self, raw_financial_report_df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the raw DataFrame copied from koyfin.com like removing empty rows and creating a section column
        :param raw_financial_report_df: DataFrame
        :return: DataFrame
        """
        # make the DataFrame more clean by removing empty rows and add a section column
        sections = raw_financial_report_df[raw_financial_report_df.iloc[:, 1:].isnull().all(1)]
        financial_report_split_df = np.split(raw_financial_report_df, sections.index)
        financial_report_df = None
        section_name_map = {'Liabilities': 'Liabilities & Equity', 'Report Date': 'Date'}
        for df in financial_report_split_df:
            if not df.empty:
                # add a column with the sub-section name
                section_name = df.iloc[0, 0]
                section_name = section_name_map.get(section_name, section_name)
                df['Section'] = section_name
                # combine the result into one DataFrame
                if financial_report_df is None:
                    financial_report_df = df
                else:
                    financial_report_df = pd.concat([financial_report_df, df], axis=0)
        financial_report_df.dropna(inplace=True)  # remove the rows with nan
        financial_report_df.reset_index(drop=True, inplace=True)
        self._clean_fa_data(df=financial_report_df)
        return financial_report_df

    @staticmethod
    def _clean_fa_data(df: pd.DataFrame) -> None:
        """
        Reorders the columns and converts the amounts to floats
        :param df: DataFrame
        :return: None
        """
        # adjust DataFrame content (data to floats and move the str columns)
        df.rename(columns={df.columns[0]: 'Item'},
                  inplace=True)  # rename first column
        section_column = df.pop('Section')  # column to be moved
        df.insert(1, 'Section', section_column)  # put the section column after 'Item' column
        df.replace('-', np.nan, inplace=True)
        df.loc[~(df.Section == 'Date'), df.columns[2:]] = df.loc[~(df.Section == 'Date'), df.columns[2:]].astype(float)
        return

    # __________________________________________________________________________________________________________________
    # Screening
    def download_screening_result(self, universe_criteria: {list, dict}, data_columns: dict,
                                  return_dataframe: bool = False, screen_name: str = 'Koyfin Bot Screen',
                                  delete_screen: bool = True) -> {pd.DataFrame, None}:
        """
        Set up a screen in koyfin according to some criteria (e.g. country or industry) and specify the data that is to
        be downloaded for each screen.
        :param universe_criteria: dict or list of dict
        :param data_columns: dict
        :param return_dataframe: bool if True a DataFrame is returned
        :param screen_name: str
        :param delete_screen: bool if True each screen gets deleted after the data has been downloaded
        :return: None or DataFrame
        """
        # always convert to a list so the input can be iterated
        if not isinstance(universe_criteria, list):
            universe_criteria = [universe_criteria]

        self._go_to_my_screens()
        self._screen_existence_check(screen_name=screen_name)

        # loop though each screen configuration and download the result as a csv file
        for i, screen_criteria in enumerate(universe_criteria):
            data_columns_i = self._adjust_data_columns(screen_criteria=screen_criteria, data_columns=data_columns)
            logger.info(f'Setting up screen #{i + 1}/{len(universe_criteria)}')
            self._add_new_screen()
            self._name_screen(name=screen_name)
            self._set_universe_criteria(criteria_config=screen_criteria)
            self._create_screen()
            self._add_data_columns(data_columns_config=data_columns_i)
            self._download_screener_result()
            if delete_screen:
                self._delete_screen(screen_name=screen_name)
        if return_dataframe:
            return self.get_latest_downloaded_data_df(num_files=len(universe_criteria))
        return

    @staticmethod
    def get_latest_downloaded_data_df(num_files: int) -> pd.DataFrame:
        """
        Looks up the latest downloaded csv files and merges the result into a DataFrame
        :return: DataFrame
        """
        # list of file paths to the latest downloaded csv files
        newest_files = get_newest_file_paths(path=DOWNLOADS_PATH, suffix='csv', num_files=num_files)
        df = None  # initialize result
        for file_path in newest_files:  # for each file path load the data into a DataFrame and merge it
            if df is None:
                df = pd.read_csv(file_path, index_col=0)
            else:
                df = pd.concat([df, pd.read_csv(file_path, index_col=0)])
        return df

    @staticmethod
    def _adjust_data_columns(screen_criteria: dict, data_columns: dict) -> dict:
        """
        Removes data column keys that are already in the screen criteria
        :param screen_criteria: dict
        :param data_columns: dict
        :return: None
        """
        # all the universe criteria types does not need to be in the data columns since they will be included
        # automatically. i.e. if 'industry' is part of the universe criteria it does not need to be in data columns
        # otherwise 'industry' will be de-selected
        for key in data_columns.copy().keys():
            if key.lower() in [k.lower() for k in screen_criteria]:
                data_columns.pop(key)
        return data_columns

    def _screen_existence_check(self, screen_name: str) -> None:
        """
        Raises an error if there is already a screen with a specified name
        :param screen_name: str
        :return: None
        """
        screen_tab_elements = self.driver.find_elements(By.XPATH, f"//*[text()='{screen_name}']")
        if len(screen_tab_elements):
            raise ValueError(f"There already exists a screen with the name '{screen_name}'")
        return

    def _go_to_my_screens(self):
        """
        Clicks 'My Screens' and maximizes the window (if not some widgets will not be able to be activated later)
        :return:
        """
        logger.info('Go to my screen page')

        # click the "screens" button
        self.bot_sleep(2)
        self.click_button('//*[@id="root"]/div[1]/section/div[2]/div[1]/div/div/div[1]/div/div[2]/a[5]/div[2]/div')
        self.bot_sleep(2)

    def _add_new_screen(self) -> None:
        """
        Assumes that the "Screens" page is loaded.
        :return: None
        """
        logger.info('Add new screen')
        # click the '+' symbol to create a new screener
        self.click_button(
            '//*[@id="root"]/div[1]/section/div[2]/div[2]/div[1]/div[1]/div/div/div[1]/div[2]/div/button/span/i')
        self.bot_sleep(sleep_time=2)

        # remove 'Trading Region' criteria
        self.click_button('/html/body/div[4]/div/div[2]/div/div[2]/div/div/div[3]/div[1]/div/div[1]/div[3]/div/div/div/div[1]/div/div[1]/span[2]/i')
        self.bot_sleep(2)
        return

    def _name_screen(self, name: str) -> None:
        """
        Set the name of the screen
        :param name: str
        :return: None
        """
        logger.info(f'Name the screen "{name}"')
        screen_name_bar_element = self.driver.find_element(By.NAME, "screen-title")
        screen_name_bar_element.click()
        active_element = self.driver.execute_script("return document.activeElement")
        active_element.send_keys(Keys.CONTROL + "a")
        active_element.send_keys(Keys.DELETE)
        active_element.send_keys(name)
        return

    def _set_universe_criteria(self, criteria_config: dict) -> None:
        """
        Assumes that the criteria page is loaded. Configures a criteria filter on the screen page.
        :param criteria_config: dict key = criteria (str), values = criteria items (list of str)
        e.g. {'country': ['Sweden']}
        :return:
        """

        # check that the specified dictionary has the correct keys
        uni_criteria_list = ['Country', 'Exchange', 'Industry', 'Sector',
                             'Trading Country', 'Trading Region']
        self._check_criteria(criteria=criteria_config, eligible_criteria=uni_criteria_list)

        logger.info(f'Adding universe criteria_config\n{self._str_uni_criteria(criteria=criteria_config)}')

        if self.remove_min_market_cap:
            # by default there is a minimum USD 50 million market cap filter applied
            self.click_button('/html/body/div[3]/div/div[2]/div/div[2]/div/div/div[3]/div[1]/div/div[2]/div[3]/div/div/div/div/div/div[1]/span[1]/div/div/div[1]/span[1]/i')

        # loop through each criteria_config (e.g. 'country') add the specified items (e.g. 'sweden') per criteria_config
        for criteria, items in criteria_config.items():
            criteria = criteria.title()
            # add a universe criteria_config
            self.driver.find_element(By.XPATH, "//*[text()='Add Universe Criteria']").click()
            self.bot_sleep(2)
            self.driver.find_elements(By.XPATH, f"//*[text()='{criteria}']")[-1].click()

            # add the items for the specified criteria_config
            use_contains_text = criteria in ['Country', 'Trading Country', 'Trading Region']
            self._add_universe_criteria_items(items=items, use_contains_text=use_contains_text)
        self.bot_sleep(5)
        return

    def _add_universe_criteria_items(self, items: list, use_contains_text: bool) -> None:
        """
        Selects items from a drop down list after entering text in a search box. The first item filtered will be
        selected.
        :param items: list of str
        :return: None
        """
        # click the drop down list to add the criteria items
        selection_element = self.driver.find_element(By.XPATH, f"//*[text()='No items selected']")
        selection_element.click()
        self.bot_sleep(1)

        if not isinstance(items, list):
            items = [items]

        # loop through each items to add it as a criteria
        for item in items:
            item = item.title()
            # search the drop down list by entering text in a search box
            search_bar_element = self.driver.find_elements(By.CLASS_NAME, 'input-field__inputField__container___w4bAA')[-1]
            search_bar_element.click()
            active_element = self.driver.execute_script("return document.activeElement")
            active_element.send_keys(item)
            self.bot_sleep(1)

            # select the items filtered based on the search
            if use_contains_text:
                self.driver.find_elements(By.XPATH, f"//*[contains(text(), '{item}')]")[-1].click()
            else:
                self.driver.find_elements(By.XPATH, f"//*[text()='{item}']")[-1].click()

            # clear the content in the search bar by clicking, select all and delete
            search_bar_element.click()
            active_element = self.driver.execute_script("return document.activeElement")
            active_element.send_keys(Keys.CONTROL + "a")
            active_element.send_keys(Keys.DELETE)
            search_bar_element.click()

        # click one of the criteria box to remove the drop down list
        # this will also refresh the number of stocks that has passed the filter
        selection_element.click()
        return

    def _create_screen(self) -> None:
        """
        Clicks 'Create Screen'
        :return: None
        """
        logger.info(f"Run screener")
        # click 'Create Screen'
        self.driver.find_element(By.XPATH, f"//*[text()='Create Screen']").click()
        self.bot_sleep(5)
        return

    def _add_data_columns(self, data_columns_config: dict) -> None:
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
            self.bot_sleep(1)
            try:
                self.driver.find_element(By.XPATH, f"//*[text()='{data_name}']").click()  # click the data name
            except:
                self.driver.find_element(By.XPATH, '//*[@id="koy-data-source-item-0"]/div[1]/div[2]/div[1]').click()  # click first element
            self.bot_sleep(1)

            # if we are only adding data that does not have any transformation (like 'Sector') data_metas is None
            if data_metas:
                # first convert data meta to a list if not already
                if not isinstance(data_metas, list):
                    data_metas = [data_metas]
                for data_meta in data_metas:
                    logger.info(f"{data_meta} ({data_name})")
                    data_meta_element = self.driver.find_element(By.XPATH, f"//*[contains(text(),'{data_meta}')]")
                    # data_meta_element = self.driver.find_element(By.XPATH, f"//*[text()={data_meta}]")
                    data_meta_element.click()
                    self.bot_sleep(2)

                # press the 'back' button to go back to the search bar (only used when the data requires additional
                # inputs like data format)
                self.driver.find_element(By.XPATH, f"//*[text()='Go Back']").click()

            # clear all elements in search bar
            search_bar.send_keys(Keys.CONTROL + "a")
            search_bar.send_keys(Keys.DELETE)

        # click 'x'
        self.driver.find_element(By.CLASS_NAME, 'dialog__closeIcon___bJnzL').click()
        return

    def _download_screener_result(self) -> None:
        """
        Clicks 'Download' on the Screens page and moves the file from 'Downloads' to the defined download path
        :return:
        """
        logger.info('Wait for 30 seconds then download the results in excel')
        self.bot_sleep(30)
        self.click_button('//*[@id="root"]/div[1]/section/div[2]/div[2]/div[1]/div[1]/div/div/div[1]/div[1]/div[2]/div/button[2]/label')
        self.bot_sleep(3)
        return

    def _delete_screen(self, screen_name: str):
        """
        Deletes all screens with a specific name
        :param screen_name: str case sensitive
        :return:
        """
        action = ActionChains(self.driver)

        # list of all elements with the name of the screen
        screen_tab_elements = self.driver.find_elements(By.XPATH, f"//*[text()='{screen_name}']")
        for e in screen_tab_elements:
            # right click on the tab
            action.context_click(e).perform()
            # click 'Delete Screen' after right-click
            self.driver.find_element(By.XPATH, f"//*[text()='Delete Screen']").click()
            self.bot_sleep(2)
            # confirm by clicking 'Delete Permanently'
            self.driver.find_element(By.XPATH, f"//*[text()='Delete Permanently']").click()
            self.bot_sleep(2)
            logger.info(f"Delete screen named '{screen_name}'")
        return

    @staticmethod
    def _check_criteria(criteria, eligible_criteria) -> None:
        """
        Raises an error if the critieria is not in the list of eligible criterias
        :param criteria:
        :param eligible_criteria:
        :return:
        """
        if any(key.title() not in eligible_criteria for key in criteria.keys()):
            raise ValueError("the keys in 'criteria' dict needs to be one of '%s'" % "', '".join(eligible_criteria))
        return

    @staticmethod
    def _str_uni_criteria(criteria: dict) -> str:
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

