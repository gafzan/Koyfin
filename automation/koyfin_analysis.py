"""koyfin_analysis.py"""

import pandas as pd
import numpy as np
import datetime
from dateutil import parser
import matplotlib.pyplot as plt
import seaborn as sns

import os
import logging

from automation.koyfinbot import KoyfinBot
from automation.config import FINANCIAL_REPORTS_FOLDER_PATH
from automation.config import get_project_root
from tools.general_tools import ask_user_yes_or_no
from tools.general_tools import user_picks_element_from_list

# logger
formatter = logging.Formatter('%(asctime)s : %(module)s : %(funcName)s : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


sns.set_style('darkgrid')
CROSS_SECTIONAL_DATA_FILE_PATH = get_project_root() / "data/company_fundamentals.csv"


class KoyfinAnalyst:
    """Class definition of KoyfinAnalyst
    Downloads and performs analysis of accounting statements
    * Make accounting adjustments like capitalizing R&D and calculating non cash working capital
    * Analysis of growth, profitability and reinvestments
    * Normalizing balance sheet and income statement with total assets and revenues
    * Get various metrics like capital-to-sales ratio, interest coverage ratio and EBIT margin adjusted for R&D
    """

    min_num_price_target = 5

    def __init__(self, company: str = None):
        self.balance_sheet_fy_df = None
        self.balance_sheet_fq_df = None
        self.balance_sheet_ltm_df = None
        self.income_statement_fy_df = None
        self.income_statement_fq_df = None
        self.income_statement_ltm_df = None
        self.cash_flow_fy_df = None
        self.cash_flow_fq_df = None
        self.cash_flow_ltm_df = None
        self._rnd_is_capitalized = False
        self.comparables_df = None
        self.ticker = None
        self.company = company
        self._koyfin_bot = KoyfinBot

    def set_financial_reports(self) -> None:
        """
        Sets financial report attributes if the data folder exists else an error is raised
        :return: None
        """
        if self.company is None:
            raise ValueError("'company' needs to be specified")

        company_folder_path = FINANCIAL_REPORTS_FOLDER_PATH / self.company
        if not company_folder_path.exists():
            raise ValueError(f"No data folder exists for '{self.company}'\nRun download_financial_reports() to get the "
                             f"fiancial reports from koyfin")
        self._read_financial_reports_csv(company_folder_path=company_folder_path)
        days = self.days_since_last_report_date()
        if days > 120:
            logger.warning(f"Number of days since last reporting date is {days}. Consider download new data.")
        return

    def _read_financial_reports_csv(self, company_folder_path) -> None:
        """
        Reads CSV files as DataFrames and stores them as attributes. The numeric columns are converted to float from str
        :param company_folder_path: path to the folder containing the financial report files
        :return: None
        """
        self.balance_sheet_fy_df = self._convert_report_to_numeric_df(pd.read_csv(company_folder_path / 'balance_sheet_fy.csv', index_col=0))
        self.income_statement_fy_df = self._convert_report_to_numeric_df(pd.read_csv(company_folder_path / 'income_statement_fy.csv', index_col=0))
        self.cash_flow_fy_df = self._convert_report_to_numeric_df(pd.read_csv(company_folder_path / 'cash_flow_fy.csv', index_col=0))
        self.balance_sheet_fq_df = self._convert_report_to_numeric_df(pd.read_csv(company_folder_path / 'balance_sheet_fq.csv', index_col=0))
        self.income_statement_fq_df = self._convert_report_to_numeric_df(pd.read_csv(company_folder_path / 'income_statement_fq.csv', index_col=0))
        self.cash_flow_fq_df = self._convert_report_to_numeric_df(pd.read_csv(company_folder_path / 'cash_flow_fq.csv', index_col=0))
        self.balance_sheet_ltm_df = self._convert_report_to_numeric_df(pd.read_csv(company_folder_path / 'balance_sheet_ltm.csv', index_col=0))
        self.income_statement_ltm_df = self._convert_report_to_numeric_df(pd.read_csv(company_folder_path / 'income_statement_ltm.csv', index_col=0))
        self.cash_flow_ltm_df = self._convert_report_to_numeric_df(pd.read_csv(company_folder_path / 'cash_flow_ltm.csv', index_col=0))
        return

    @staticmethod
    def _convert_report_to_numeric_df(report_df: pd.DataFrame):
        """
        Assumes report_df is a DataFrame with first two columns having str and once column named 'Section'
        :param report_df: DataFrame
        :return: DataFrame
        """
        report_df.loc[~(report_df.Section == 'Date'), report_df.columns[2:]] = report_df.loc[~(report_df.Section == 'Date'), report_df.columns[2:]].astype(float)
        return report_df

    def download_financial_reports(self, overwrite: bool = False) -> None:
        """
        Downloads accounting statements from koyfin and saves them in a CSV file in the specified company folder. If
        data already exists it is either overwritten or added to the existing data (i.e. to update the data after a new
        reporting date)
        :param overwrite: bool if True, the csv file is overwritten by the new data, else the new data is added
        :return: None
        """
        # launch the KoyfinBot and login
        bot = self._koyfin_bot()
        bot.login()
        report_types = ['fy', 'fq', 'ltm']
        report_codes = {'FA.BS': 'balance_sheet', 'FA.IS': 'income_statement', 'FA.CF': 'cash_flow'}
        reports = {}

        # loop through each accounting statement and each type ('FY', 'FQ' and 'LTM') and store the result in a dict
        for code, report_name in report_codes.items():
            # download from koyfin
            # dict format with the report type as keys and DataFrame as values
            report = bot.fa_get_financial_report_df(company=self.company, financial_analysis_code=code, report_type=report_types)
            reports[report_name] = report
        bot.driver.close()

        # create a data folder if it does not exist
        company_folder_path = FINANCIAL_REPORTS_FOLDER_PATH / self.company
        if not company_folder_path.exists():
            os.makedirs(company_folder_path)
            overwrite = True

        # loop through the financial reports and store them as csv in a folder
        for report_name in report_codes.values():
            for report_type in report_types:
                # look up the report name (e.g. 'balance_sheet') and type (e.g. 'ltm') drop the column with the LTM data
                report_df = reports[report_name][report_type].drop('Current/LTM', axis=1)
                save_file_path = company_folder_path / f"{report_name}_{report_type}.csv"
                if not overwrite:
                    # get the existing report and merge the new data columns
                    existing_df = pd.read_csv(save_file_path, index_col=0)
                    report_df = self._add_new_data_to_financial_report(new_report_df=report_df,
                                                                       existing_report_df=existing_df)
                # save the report DataFrame as a csv file
                report_df.to_csv(save_file_path)
        return

    @staticmethod
    def _add_new_data_to_financial_report(new_report_df: pd.DataFrame, existing_report_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame that has added the new columns in 'new_report_df' to 'existing_report_df' merged on 'Item'
        and 'Section' columns
        :param new_report_df: DataFrame
        :param existing_report_df: DataFrame
        :return: DataFrame
        """
        if existing_report_df.shape[0] < new_report_df.shape[0]:
            new_items = set(new_report_df['Item']).difference(existing_report_df['Item'])
            raise ValueError(f"The new report has new items: '%s'" % "', '".join(new_items))
        is_new_col = ~new_report_df.columns.isin(existing_report_df.columns)  # boolean array
        is_new_col[0] = True  # 'Item' column
        is_new_col[1] = True  # 'Section' column
        result = pd.merge(existing_report_df, new_report_df.loc[:, is_new_col], on=('Item', 'Section'), how='left')
        return result

    def delete_financial_reports(self) -> None:
        """
        Sets all financial account attributes to None
        :return: None
        """
        self.balance_sheet_fy_df = None
        self.balance_sheet_fq_df = None
        self.balance_sheet_ltm_df = None
        self.income_statement_fy_df = None
        self.income_statement_fq_df = None
        self.income_statement_ltm_df = None
        self.cash_flow_fy_df = None
        self.cash_flow_fq_df = None
        self.cash_flow_ltm_df = None
        self._rnd_is_capitalized = False
        self.ticker = None
        return

    def days_since_last_report_date(self) -> int:
        """
        Returns the number of days between today and the last reporting date as documented in the income statement for
        the last trailing 12 months
        :return: int
        """
        if self.income_statement_ltm_df is None:
            raise ValueError('No income statement T12M has been set')
        latest_report_date_str = self.income_statement_ltm_df[self.income_statement_ltm_df.Item == 'Report Date'].iloc[0, -1]
        days = (datetime.datetime.today() - parser.parse(latest_report_date_str)).days
        return days

    def set_cross_sectional_company_data(self) -> None:
        """
        Reads a csv file and store it as a DataFrame
        :return: None
        """
        self.comparables_df = pd.read_csv(CROSS_SECTIONAL_DATA_FILE_PATH)
        self._add_additional_columns()
        return

    def _add_additional_columns(self) -> None:
        """
        Adds EBIT estimates, capital to sales ratio and reivestment to the comparables DataFrame
        :return: None
        """
        # calculate EBIT margin estimates
        self.comparables_df['EBIT Margin - Est Avg (NTM)'] = self.comparables_df['EBIT - Est Avg (NTM)'] / self.comparables_df['Revenues - Est Avg (NTM)'] * 100
        self.comparables_df['EBIT Margin - Est Avg (FY1E)'] = self.comparables_df['EBIT - Est Avg (FY1E)'] / self.comparables_df['Revenues - Est Avg (FY1E)'] * 100
        self.comparables_df['EBIT Margin - Est Avg (FY2E)'] = self.comparables_df['EBIT - Est Avg (FY2E)'] / self.comparables_df['Revenues - Est Avg (FY2E)'] * 100
        self.comparables_df['EBIT Margin - Est Avg (FY3E)'] = self.comparables_df['EBIT - Est Avg (FY3E)'] / self.comparables_df['Revenues - Est Avg (FY3E)'] * 100

        # capital to sales ratio as the ratio of the sum of net debt (cash subtracted) and equity and revenues
        self.comparables_df['Capital Sales Ratio'] = self.comparables_df[['Net Debt (LTM)', 'Total Equity (LTM)']].sum(axis=1) / self.comparables_df['Total Revenues (LTM)'].values

        # reinvestments as CAPEX + Acquisitions - D&A + Change in WC
        # 'Capital Expenditure (LTM)' and 'Cash Acquisitions (LTM)' are already negative so they are here subtracted
        self.comparables_df['Reinvestment'] = self.comparables_df['Chg in NWC (LTM)'] - self.comparables_df[['Capital Expenditure (LTM)', 'Cash Acquisitions (LTM)', 'D&A for EBITDA (LTM)']].fillna(0).sum(
            axis=1).values
        self.comparables_df['Reinvestment Per Revenue'] = self.comparables_df['Reinvestment'] / self.comparables_df['Total Revenues (LTM)'].values
        self.comparables_df['Reinvestment Per Revenue'] *= 100
        return

    # __________________________________________________________________________________________________________________
    # accounting adjustments
    def add_amortized_research_and_development(self, amortization_life: int) -> None:
        """
        Adds the R&D Asset and R&D Amortization to the income statement. The R&D Asset is the capitalized R&D expenses with
        a specified amortization life. R&D Amortization is the total amortization stemming from the R&D Asset
        :param amortization_life: int
        :return: None
        """

        if self.income_statement_fy_df is None:
            raise ValueError('income_statement_10k_df is None')

        # remove the rows for R&D asset and amortization if they exists
        self.balance_sheet_fy_df = self.balance_sheet_fy_df.loc[~(self.balance_sheet_fy_df.Item == 'R&D Asset')]
        self.income_statement_fy_df = self.income_statement_fy_df.loc[~(self.income_statement_fy_df.Item == 'R&D Amortization')]

        # extract the R&D expenses into a DataFrame and transpose the result
        rnd_df = self.income_statement_fy_df[self.income_statement_fy_df.Item == 'R&D Expenses'].iloc[:, 2:].T
        rnd_df.fillna(0, inplace=True)
        rnd_df.columns = ['R&D Expenses']

        if amortization_life >= rnd_df.shape[0]:
            raise ValueError(f"'amortization_life' ({amortization_life}) can't be longer than the number of available "
                             f"reporting periods ({rnd_df.shape[0]})")

        def _calc_rnd_asset(expenses):
            """
            Returns a sum product of the expenses and the share of amortization
            :param expenses:
            :return: float
            """
            share_not_amortized = np.linspace(0, 1, amortization_life + 1)
            return np.dot(expenses, share_not_amortized)

        # calculate the R&D asset as well as the R&D amortization and add the rows to the income statement DataFrame
        rnd_df['R&D Asset'] = rnd_df['R&D Expenses'].rolling(window=amortization_life + 1).apply(_calc_rnd_asset,
                                                                                                 raw=True)
        rnd_df['R&D Amortization'] = rnd_df['R&D Expenses'].rolling(
            window=amortization_life).sum().shift() / amortization_life

        # replace the last row which is LTM with the last FY values
        rnd_df.iloc[-1, :] = np.nan
        rnd_df.fillna(method='ffill', inplace=True)

        # reformat the DataFrame so that it can be merged with the other financial reports
        rnd_df = rnd_df.T.drop('R&D Expenses').reset_index().rename(columns={'index': 'Item'})
        rnd_df['Section'] = 'Accounting Adjustments'

        # add the new data to the income statement and balance sheet
        self.income_statement_fy_df = pd.concat([self.income_statement_fy_df, rnd_df[rnd_df.Item == 'R&D Amortization']], ignore_index=True, sort=False)
        self.balance_sheet_fy_df = pd.concat([self.balance_sheet_fy_df, rnd_df[rnd_df.Item == 'R&D Asset']], ignore_index=True, sort=False)
        self._add_research_and_development_to_quarterly_reports()
        self._rnd_is_capitalized = True
        return

    def _add_research_and_development_to_quarterly_reports(self) -> None:
        """
        Adds an R&D Asset and R&D Amortization row to the fiscal quarter balance sheet and income statement
        :return: None
        """
        if self.balance_sheet_fq_df is not None:
            self.balance_sheet_fq_df = self._add_research_and_development_to_quarterly_report(quarterly_report_df=self.balance_sheet_fq_df,
                                                                                              fiscal_year_report_df=self.balance_sheet_fy_df,
                                                                                              item='R&D Asset')
        if self.balance_sheet_ltm_df is not None:
            self.balance_sheet_ltm_df = self._add_research_and_development_to_quarterly_report(quarterly_report_df=self.balance_sheet_ltm_df,
                                                                                              fiscal_year_report_df=self.balance_sheet_fy_df,
                                                                                              item='R&D Asset')
        if self.income_statement_fq_df is not None:
            self.income_statement_fq_df = self._add_research_and_development_to_quarterly_report(quarterly_report_df=self.income_statement_fq_df,
                                                                                              fiscal_year_report_df=self.income_statement_fy_df,
                                                                                              item='R&D Amortization')
            # approximate the cost per quarter by dividing the annual amortization expense by 4
            self.income_statement_fq_df.loc[
                self.income_statement_fq_df.Item == 'R&D Amortization',
                self.income_statement_fq_df.columns[2:]
            ] /= 4
        if self.income_statement_ltm_df is not None:
            self.income_statement_ltm_df = self._add_research_and_development_to_quarterly_report(quarterly_report_df=self.income_statement_ltm_df,
                                                                                              fiscal_year_report_df=self.income_statement_fy_df,
                                                                                              item='R&D Amortization')
        return

    @staticmethod
    def _add_research_and_development_to_quarterly_report(quarterly_report_df: pd.DataFrame,
                                                          fiscal_year_report_df: pd.DataFrame, item: str) -> pd.DataFrame:
        """
        Returns a DataFrame with a new row relaed to R&D based on the annual values from a fiscal year financial report
        DataFrame
        :param quarterly_report_df: DataFrame
        :param fiscal_year_report_df: DataFrame
        :param item: str
        :return: DataFrame
        """
        # extract the R&D item from the fiscal year financial report and change the columns to be like the fiscal
        # quarter financial report
        rnd_fq_df = fiscal_year_report_df[fiscal_year_report_df.Item == item].copy()
        new_columns = ['Item', 'Section']
        new_columns.extend(['4Q ' + col_n.replace(' ', '') for col_n in rnd_fq_df.columns[2:]])
        # new_columns.append('Current/LTM')
        rnd_fq_df.columns = new_columns

        # if the R&D item already exists, remove it
        mask = quarterly_report_df.Item.isin([item])
        quarterly_report_df = quarterly_report_df.loc[~mask]

        # combine the R&D DataFrame with the original fiscal quarter report
        quarterly_report_df = pd.concat([quarterly_report_df, rnd_fq_df], ignore_index=True,
                                                    sort=False)
        # forward fill the empty cells for Q1, Q2 and Q3 to be the previous Q4
        quarterly_report_df.loc[
            quarterly_report_df.Item == item,
            quarterly_report_df.columns[2:]
        ] = quarterly_report_df[quarterly_report_df.Item == item].iloc[0, 2:].fillna(method='ffill').values
        return quarterly_report_df

    def add_non_cash_working_capital(self) -> None:
        """
        Adds Non Cash Current Assets, Non Debt Current Liabilities and Non Cash Working Capital to the balance sheet
        :return: None
        """
        self.balance_sheet_fy_df = self._add_non_cash_working_capital(balance_sheet_df=self.balance_sheet_fy_df)
        if self.balance_sheet_fq_df is not None:
            self.balance_sheet_fq_df = self._add_non_cash_working_capital(balance_sheet_df=self.balance_sheet_fq_df)
        if self.balance_sheet_fq_df is not None:
            self.balance_sheet_ltm_df = self._add_non_cash_working_capital(balance_sheet_df=self.balance_sheet_ltm_df)
        return

    @staticmethod
    def _add_non_cash_working_capital(balance_sheet_df) -> pd.DataFrame:
        """
        Returns a DataFrame with added rows for 'Non Cash Current Asset', 'Non Debt Current Liabilities' and 'Non Cash Working Capital'
        :param balance_sheet_df: DataFrame
        :return: DataFrame
        """
        # remove the item rows in case the they already exists
        mask = balance_sheet_df.Item.isin(['Non Cash Current Asset', 'Non Debt Current Liabilities', 'Non Cash Working Capital'])
        balance_sheet_df = balance_sheet_df.loc[~mask]

        # list the names of accounting items classified as non cash current assets and non debt current liabilities
        non_cash_ca_items = ['Total Receivables', 'Inventory', 'Prepaid Expenses', 'Other Current Assets / Total']
        non_debt_cl_items = ['Accounts Payable / Total', 'Accrued Expenses / Total', 'Current Income Taxes Payable',
                             'Unearned Revenue Current / Total', 'Other Current Liabilities']

        # calculate working capital and store it in a DataFrame
        non_cash_wc_df = balance_sheet_df[balance_sheet_df.Item.isin(non_cash_ca_items)].iloc[:, 2:]\
            .sum().to_frame('Non Cash Current Asset')
        non_cash_wc_df['Non Debt Current Liabilities'] = \
            balance_sheet_df[balance_sheet_df.Item.isin(non_debt_cl_items)].iloc[:, 2:].sum()  # sum the numeric value columns
        non_cash_wc_df['Non Cash Working Capital'] = non_cash_wc_df['Non Cash Current Asset'].values - non_cash_wc_df[
            'Non Debt Current Liabilities'].values
        non_cash_wc_df = non_cash_wc_df.T.reset_index().rename(columns={'index': 'Item'})
        non_cash_wc_df['Section'] = 'Accounting Adjustments'

        # add the new data to the balance sheet
        balance_sheet_df = pd.concat([balance_sheet_df, non_cash_wc_df], ignore_index=True, sort=False)
        return balance_sheet_df

    # __________________________________________________________________________________________________________________
    # get data
    def get_company_data(self) -> pd.Series:
        """
        Returns a Series with data field names as index
        :return: Series
        """
        if self.comparables_df is None:
            raise ValueError("comparables_df is not set. Run set_cross_sectional_company_data")
        company_data = self.comparables_df.copy()
        company_data.set_index('Ticker', inplace=True)
        # use ticker attribute if it has been specified, else use company attribute
        ticker = self.ticker if self.ticker else self.company
        try:
            company_data = company_data.loc[ticker]
        except KeyError:
            raise ValueError(f"'{ticker}' does not exist as a ticker in the comparable data")

        # check so that the ticker is unique
        if isinstance(company_data, pd.DataFrame):  # is only a DataFrame if the ticker is not unique
            print(f"There are {company_data.shape[0]} stocks that has the ticker '{ticker}'")
            company_data = self._get_company_data_non_unique_ticker(company_data=company_data)
        return company_data

    @staticmethod
    def _get_company_data_non_unique_ticker(company_data: pd.DataFrame) -> pd.Series:
        """
        Asks the user to choose between different companies that has the same tickers
        :param company_data: DataFrame
        :return: Series
        """
        # first create a list of concatenated info for each ticker row
        ticker_info_list = [f"Name: {row['Name']}, Country: {row['Country']}, Sector: {row['Sector']}" for _, row in
                            company_data.iterrows()]
        chosen_ticker_info = user_picks_element_from_list(list_=ticker_info_list)
        i = ticker_info_list.index(chosen_ticker_info)
        return company_data.iloc[i, :].copy()

    def get_filtered_comparables_df(self, filter_by: {str, None}) -> pd.DataFrame:
        """
        Returns a DataFrame with cross sectional data after filtering based on a specific column
        :param filter_by: str e.g. 'Industry' will only include stocks in the same industry as the specified ticker
        :return: DataFrame
        """
        if filter_by:
            filter_by = filter_by.title()
            company_data = self.get_company_data()
            filter_value = company_data[filter_by]
            return self.comparables_df[self.comparables_df[filter_by] == filter_value].copy()
        else:
            return self.comparables_df.copy()

    def get_percentile_data_for_company(self, comparables_df: pd.DataFrame, data_col_name: str, as_quartile: bool) -> float:
        """
        Returns the percentile for the company for a specified data
        :param comparables_df: DataFrame
        :param data_col_name: str
        :param as_quartile: bool if True the result is rounded to quartiles
        :return: float
        """
        # make str have capitalized first letters
        comparables_df = comparables_df.copy()
        comparables_df.columns = comparables_df.columns.str.title()
        hist_proxy_data_name = data_col_name.title()
        # log a warning if there are few historical data
        if comparables_df[hist_proxy_data_name.title()].count() < 5:
            logger.warning(
                f"Only has {comparables_df[hist_proxy_data_name].count()} stocks with '{hist_proxy_data_name}'")

        # get the historical percentile for the ticker
        ticker_idx = comparables_df[comparables_df.Ticker == self.ticker].index[0]
        hist_pct_ile = comparables_df[hist_proxy_data_name].rank(pct=True).loc[ticker_idx]
        if as_quartile:
            hist_pct_ile = max(0.25, min(0.75, round(hist_pct_ile / 0.25, 0) * 0.25))  # round to quartiles
        return hist_pct_ile

    def get_capital_to_revenue_ratio(self, capitalize_rnd: bool, reporting_type: str) -> pd.DataFrame:
        """
        Returns a DataFrame with the ratio of capital to revenues adjusted for R&D is specified
        :param capitalize_rnd: bool
        :param reporting_type: str
        :return:
        """
        capital_items = ['Net Debt', 'Total Equity']
        if capitalize_rnd:
            capital_items.append('R&D Asset')
        capital = self.get_balance_sheet_item(item=capital_items, report_type=reporting_type)
        result = capital.sum(axis=1).div(self.get_income_statement_item(item='Total Revenues', report_type=reporting_type)['Total Revenues'])
        result.columns = ['Capital to Sales Ratio']
        return result

    def get_ebit_margin(self, capitalize_rnd: bool, reporting_type: str) -> pd.DataFrame:
        """
        Returns a DataFrame with the EBIT margin adjusted for R&D if specified
        :param capitalize_rnd: bool if True adds back the net R&D expense
        :param reporting_type: str
        :return: DataFrame
        """
        is_df = self.get_income_statement_item(item=['EBIT', 'Total Revenues'], report_type=reporting_type)
        if capitalize_rnd:
            rnd_df = self.get_income_statement_item(item=['R&D Expenses', 'R&D Amortization'], report_type=reporting_type)
            is_df = pd.concat([is_df, rnd_df], axis=1)
            is_df['EBIT'] += (is_df['R&D Expenses'] - is_df['R&D Amortization']).fillna(0)
        is_df['EBIT Margin'] = is_df['EBIT'] / is_df['Total Revenues'].values
        return is_df[['EBIT Margin']]

    def get_effective_tax_rate(self, reporting_type: str) -> pd.DataFrame:
        """
        Returns a DataFrame with the effective tax rate here defined as income tax expense divided by EBIT. The actual
        'effective' tax rate should be with respect to EBIT after subtracting interest expenses but the effective tax
        rate used in the valuation
        :param reporting_type: str
        :return: DataFrame
        """
        is_df = self.get_income_statement_item(item=['EBT / Incl. Unusual Items', 'Income Tax Expense'], report_type=reporting_type)
        is_df['Effective Tax Rate'] = is_df['Income Tax Expense'] / is_df['EBT / Incl. Unusual Items'].values
        return is_df[['Effective Tax Rate']]

    def get_reinvestment(self, capitalize_rnd: bool, reporting_type: str, with_details: bool = False) -> pd.DataFrame:
        """
        Returns a DataFrame with gross and net reinvestment and its components if specified
        :param capitalize_rnd: bool if True, treat R&D as a capital and not an operational expense
        :param reporting_type: str
        :param with_details: bool if True returns a larger DataFrame with the components
        :return: DataFrame
        """
        result_df = pd.concat(
            [
                -self.get_cash_flow_item(item=['Capital Expenditure', 'Cash Acquisitions'], report_type=reporting_type),
                self.get_change_non_cash_working_capital(report_type=reporting_type),
                self.get_income_statement_item(item='D&A for EBITDA', report_type=reporting_type)
            ],
            axis=1
        )

        # calculate reinvestment
        result_df['Gross Reinvestment'] = result_df[['Capital Expenditure', 'Cash Acquisitions', 'Change Non Cash Working Capital']].sum(axis=1)
        result_df['Reinvestment'] = result_df['Gross Reinvestment'] - result_df['D&A for EBITDA'].fillna(0).values

        if capitalize_rnd:
            result_df = pd.concat(
                [
                    result_df,
                    self.get_income_statement_item(item=['R&D Expenses','R&D Amortization'], report_type=reporting_type)
                ],
                axis=1
            )
            result_df['Gross Reinvestment'] += result_df['R&D Expenses'].fillna(0).values
            result_df['Reinvestment'] += result_df['R&D Expenses'].fillna(0).values - result_df['R&D Amortization'].fillna(0).values

            # move reinvestment columns to the right
            result_df.insert(result_df.shape[1] - 1, 'Gross Reinvestment', result_df.pop('Gross Reinvestment'))
            result_df.insert(result_df.shape[1] - 1, 'Reinvestment', result_df.pop('Reinvestment'))

        if with_details:
            return result_df
        else:
            return result_df[['Reinvestment']]

    def get_reinvestments_per_revenue(self, capitalize_rnd: bool, reporting_type: str, with_details: bool = False) -> pd.DataFrame:
        """
        Returns a DataFrame with gross and net reinvestment and its components if specified with respect to revenues
        :param capitalize_rnd: bool if True, treat R&D as a capital and not an operational expense
        :param reporting_type: str
        :param with_details: bool if True returns a larger DataFrame with the components
        :return: DataFrame
        """
        reinv_df = self.get_reinvestment(capitalize_rnd=capitalize_rnd, reporting_type=reporting_type, with_details=with_details)
        reinv_df /= self.get_income_statement_item(item='Total Revenues', report_type=reporting_type).values
        return reinv_df

    def get_change_non_cash_working_capital(self, report_type: str) -> pd.DataFrame:
        """
        Returns a DataFrame with the change in non cash working capital
        :param report_type: str
        :return: DataFrame
        """
        # balance sheet items
        result = self.get_balance_sheet_item(item='Non Cash Working Capital', report_type=report_type)
        if report_type == 'ltm':
            result = result.diff(4)
        else:
            result = result.diff()
        result.columns = ['Change Non Cash Working Capital']
        return result

    def get_interest_coverage_ratio(self, report_type: str, avg_window: int = None, with_details: bool = False) -> pd.DataFrame:
        """
        Returns a DataFrame with the interest coverage ratio (IRC) as well as the components is specified
        :param report_type: str
        :param avg_window: int window for calculating an average of both EBIT and interest expense before calculating
        the ICR
        :param with_details: bool if True returns a larger DataFrame with the components
        :return:
        """
        result = self.get_income_statement_item(item=['EBIT', 'Interest Expense / Total'], report_type=report_type)
        result['Interest Expense / Total'] *= -1
        if avg_window:
            result = result.rolling(window=avg_window).mean()
        result['ICR'] = result['EBIT'] / result['Interest Expense / Total']
        if with_details:
            return result
        else:
            return result[['ICR']]

    def get_roic(self, capitalize_rnd: bool, report_type: str) -> pd.DataFrame:
        """
        Returns a DataFrame with Return on Invested Capital (ROIC) defined as EBIT minus taxes divided by the sum of
        Net Total Debt, Total Equity and R&D asset if applicable
        :param capitalize_rnd: bool add R&D asset to equity
        :param report_type: str
        :return: DataFrame
        """
        ic = self.get_balance_sheet_item(item=['Net Debt', 'Total Equity'], report_type=report_type)
        ebit = self.get_income_statement_item(item='EBIT', report_type=report_type) - self.get_income_statement_item(item='Income Tax Expense', report_type=report_type).values
        if capitalize_rnd:
            rnd_asset_df = self.get_balance_sheet_item(item='R&D Asset', report_type=report_type)
            ic = ic.join(rnd_asset_df)
            ebit += self.get_income_statement_item(item='R&D Expenses', report_type=report_type).fillna(0).values * rnd_asset_df.notnull().astype(int).values
            ebit -= self.get_income_statement_item(item='R&D Amortization', report_type=report_type).fillna(0).values * rnd_asset_df.notnull().astype(int).values
        result = ebit.div(ic.sum(axis=1), axis=0)
        result.columns = ['ROIC']
        return result

    def get_revenue_ytd_sum(self) -> float:
        """
        Returns the total revenues earned during so far during the latest fiscal year
        :return: float
        """
        return self.get_accounting_item_ytd_sum(accounting_statement_name='income_statement', item='Total Revenues')

    def get_accounting_item_ytd_sum(self, accounting_statement_name: str, item: {str, list}) -> {float, pd.Series}:
        """
        Returns the year-to-date sum of accounting items i.e. the sum of all the fiscal quarters for the latest fiscal
        year
        :param accounting_statement_name: str
        :param item: str
        :return: float or Series
        """
        if accounting_statement_name.lower() == 'balance_sheet':
            raise ValueError("This script should only be used for 'income_statement' and 'cash_flow'")
        reporting_item_fq_df = self._get_financial_report_item(financial_report_name=accounting_statement_name, report_type='fq',
                                                               item=item, normalize=False)
        reporting_item_fq_df['FY'] = [s.split(' ')[1] for s in reporting_item_fq_df.index]
        result = reporting_item_fq_df.groupby('FY').agg('sum').iloc[-1]
        if result.shape[0] == 1:
            return result[0]
        else:
            return result

    def get_fcff_components(self, reporting_type: str) -> pd.DataFrame:
        """
        Returns a DataFrame with the various components to Free Cash Flow to Firm (FCFF)
        :param reporting_type: str
        :return: DataFrame
        """
        # combine all the relevant items from the income statement, balance sheet and cash flows into a DataFrame
        total_df = pd.concat(
            [
                self.get_income_statement_item(item=['EBIT', 'Income Tax Expense', 'D&A for EBITDA'], report_type=reporting_type),
                self.get_change_non_cash_working_capital(report_type=reporting_type),
                self.get_cash_flow_item(item=['Capital Expenditure', 'Cash Acquisitions'], report_type=reporting_type)
            ],
            axis=1
        )
        total_df['Income Tax Expense'] *= -1  # tax expense has a negative impact on cash flows
        total_df['FCFF'] = total_df.sum(axis=1)  # sum the components
        return total_df

    def get_balance_sheet_item(self, item: {str, list}, report_type: str, normalize: bool = False):
        """
        Returns one or more items from the balance sheet as a DataFrame
        :param item: str or list of str
        :param report_type: str
        :param normalize: bool if True each amount is divided by total assets
        :return: DataFrame
        """
        return self._get_financial_report_item(financial_report_name='balance_sheet', item=item, report_type=report_type, normalize=normalize)

    def get_income_statement_item(self, item: {str, list}, report_type: str, normalize: bool = False):
        """
        Returns one or more items from the income statement as a DataFrame
        :param item: str or list of str
        :param report_type: str
        :param normalize: bool if True each amount is divided by total revenues
        :return: DataFrame
        """
        return self._get_financial_report_item(financial_report_name='income_statement', item=item, report_type=report_type, normalize=normalize)

    def get_cash_flow_item(self, item: {str, list}, report_type: str):
        """
        Returns one or more items from the cash flow statement as a DataFrame
        :param item: str or list of str
        :param report_type: str
        :return: DataFrame
        """
        return self._get_financial_report_item(financial_report_name='cash_flow', item=item, report_type=report_type, normalize=False)

    def _get_financial_report_item(self, financial_report_name: str, item: {str, list}, report_type: str, normalize: bool) -> pd.DataFrame:
        """
        Returns one or more items from a specified financial account as a DataFrame
        :param financial_report_name: str
        :param item: str or list of str
        :param report_type: str
        :param normalize: bool if True balance sheet gets normalized by total revenues and income statement by revenues
        :return: DataFrame
        """
        # get the financial account DataFrame
        financial_report_df = self._get_financial_report(financial_report_name=financial_report_name, report_type=report_type, normalize=normalize)

        # convert to list of str with lower characters
        if isinstance(item, str):
            item = [item]
        item = [s.lower() for s in item]

        # raise an error if an item does not exist
        for i in item:
            if i not in financial_report_df.Item.str.lower().values:
                raise ValueError(f"'{i}' does not exists as a {financial_report_name.replace('_', ' ')} item")

        # transpose result and exclude the section name
        item_idx = [list(financial_report_df.Item.str.lower()).index(i) for i in item]  # find the row index for each item
        result_df = financial_report_df.iloc[item_idx].copy()
        result_df.set_index('Item', inplace=True)
        result_df.drop('Section', inplace=True, axis=1)
        return result_df.T

    def get_normalized_balance_sheet(self, reporting_type: str) -> pd.DataFrame:
        """
        Returns a balance sheet DataFrame where every monetary amount normalized by total assets
        :param reporting_type: str
        :return: DataFrame
        """
        return self._get_financial_report(financial_report_name='balance_sheet', report_type=reporting_type, normalize=True)

    def get_normalized_income_statement(self, reporting_type: str):
        """
        Returns a income statement DataFrame where every monetary amount normalized by total revenues
        :param reporting_type: str
        :return: DataFrame
        """
        return self._get_financial_report(financial_report_name='income_statement', report_type=reporting_type, normalize=True)

    def _get_financial_report(self, financial_report_name: str, report_type: str, normalize: bool) -> pd.DataFrame:
        """
        Returns the financial report attribute
        :param financial_report_name: str
        :param report_type: str
        :param normalize: bool if True the balance sheet will be normalized by the total assets and the income statement
        by the total revenues
        :return: DataFrame
        """
        financial_reports = {
            'balance_sheet': {
                'fy': self.balance_sheet_fy_df,
                'fq': self.balance_sheet_fq_df,
                'ltm': self.balance_sheet_ltm_df
            },
            'income_statement': {
                'fy': self.income_statement_fy_df,
                'fq': self.income_statement_fq_df,
                'ltm': self.income_statement_ltm_df
            },
            'cash_flow':
                {
                    'fy': self.cash_flow_fy_df,
                    'fq': self.cash_flow_fq_df,
                    'ltm': self.cash_flow_ltm_df
                }
        }
        result = financial_reports[financial_report_name][report_type.lower()]
        if result is None:
            raise ValueError(f"'{financial_report_name}_{report_type.lower()}_df' is None for {self.company.upper()}\nCall set_financial_reports() to download the financial accounts.")

        if normalize and financial_report_name in ['income_statement', 'balance_sheet']:
            # a dictionary with all the item names that should not be normalized (e.g. per share items)
            report_excl_item_map = {
                'income_statement': ["Report Date", 'Period Ending', "Total Revenues / CAGR 1Y", "Gross Profit / CAGR 1Y",
                                   "Net EPS - Basic", "Basic EPS - Continuing Operations",
                                   "Basic Weighted Average Shares Outstanding", "Net EPS - Diluted",
                                   "Diluted EPS - Continuing Operations",
                                   "Diluted Weighted Average Shares Outstanding", "Normalized Basic EPS",
                                   "Normalized Diluted EPS",
                                   "Dividend Per Share", "Payout Ratio", "Effective Tax Rate - (Ratio)"],
                'balance_sheet': ["Report Date", 'Period Ending', "ECS Total Shares Outstanding on Filing Date",
                                  "ECS Total Common Shares Outstanding", "Book Value / Share", "Tangible Book Value Per Share",
                                  "Tangible Book Value Per Share"]
            }
            normalizing_item = {
                'income_statement': 'Total Revenues',
                'balance_sheet': 'Total Assets'
            }
            result.loc[~result.Item.isin(report_excl_item_map[financial_report_name]), list(result.columns)[2:]] /= \
                result[result.Item == normalizing_item[financial_report_name]].iloc[:, 2:].values
        return result

    # __________________________________________________________________________________________________________________
    # analysis
    def analyse_growth(self, report_type: str = 'ltm', median_groupby: str = None) -> None:

        # calculate historical Revenue growth
        if report_type.lower() == 'fy':
            lag = 1
        else:
            lag = 4
        growth_df = self.get_income_statement_item(item='Total Revenues', report_type=report_type).pct_change(lag) * 100
        growth_df.dropna(inplace=True)
        g_col_name = f'Revenue Growth {report_type.upper()} {round(growth_df.iloc[-1, 0], 2)}%'
        growth_df.columns = [g_col_name]

        # plot the result
        _, ax = plt.subplots(figsize=(10, 5))

        if self.comparables_df is not None:
            # filter out the row for the specific ticker
            company_df = self.comparables_df[self.comparables_df.Ticker == self.company.upper()].copy()
            if company_df.empty:
                raise ValueError(f"'{self.company}' is not part of the cross sectional data")

            # plot comparable median for 5Y CAGR together with filling the area between the 25 and 75%-ile
            col_name = 'Total Revenues/CAGR (5Y TTM)'
            stats = self._get_comparison_stats(df=self.comparables_df,
                                               data_name=col_name, classifier=median_groupby,
                                               agg_method=[self.percentile_75, 'median', self.percentile_25])
            col_name = f"5Y CAGR ({median_groupby + '' if median_groupby else ''}median) {stats['median']}%"
            growth_df[col_name] = stats['median']
            color = 'blue'
            growth_df.plot(ax=ax, y=col_name, color=color)
            ax.fill_between(growth_df.index, stats['percentile_25'], stats['percentile_75'], alpha=0.1, color=color)

            # plot 5/10/15-year CAGR
            lt_g_col_name_map = {'5Y CAGR': 'Total Revenues/CAGR (5Y TTM)', '7Y CAGR': 'Total Revenues/CAGR (7Y TTM)',
                                 '10Y CAGR': 'Total Revenues/CAGR (10Y TTM)'}
            lt_g_styles = ['-', '-.', ':']
            i = 0
            for k, v in lt_g_col_name_map.items():
                value = company_df[v].values[0]
                if not np.isnan(value):
                    col_name = f"{k} {value}%"
                    growth_df[col_name] = value
                    growth_df.plot(ax=ax, y=col_name, color='darkorange', style=lt_g_styles[i])
                i += 1
            # plot the estimated 2Y CAGR if there is enough analyst coverage
            if company_df['Price Target - #'].values[0] >= 3:
                est_growth_df = company_df[['Est Rev CAGR (1Y)', 'Est Rev CAGR (2Y)', 'Est Rev CAGR (3Y)']].dropna(axis=1)
                if not est_growth_df.empty:
                    est_col_name = est_growth_df.columns[-1]
                    est_col_name.replace('Rev ', '')
                    est_col_name += f" {est_growth_df.iloc[0, -1]}%"
                    growth_df[est_col_name] = est_growth_df.iloc[0, -1]
                    growth_df.plot(ax=ax, y=est_col_name, style='--', color='grey')

        colors = ['g' if g > 0 else 'r' for g in growth_df[g_col_name].values]  # conditional color: red if negative, else green
        growth_df.plot(ax=ax, y=g_col_name, kind='bar', color=colors, title='Revenue Growth (%)')
        return

    def analyse_profitability(self, comparison: bool, capitalize_rnd: bool, classifier: str = None, report_type: str = 'ltm'):
        """
        Generate a plot with EBIT margin as a bar chart together with comparable median as well as 25 and 75%-iles
        Also if there are enough analysts covering the stock (proxy by number of price targets) also plot the expected
        profitability
        :param comparison: bool
        :param capitalize_rnd: bool if True adjust the EBIT margin when assuming that R&D gets capitalized
        :param classifier: str
        :param report_type: str
        :return:
        """
        # get the data
        ebit_df = self.get_ebit_margin(capitalize_rnd=capitalize_rnd, reporting_type=report_type) * 100

        # plot the result
        _, ax = plt.subplots(figsize=(10, 5))
        colors = ['g' if g > 0 else 'r' for g in ebit_df.values]  # conditional color: red if negative, else green
        ebit_df.plot(ax=ax, y='EBIT Margin', kind='bar', color=colors, title='EBIT Margin (%)')

        if comparison:
            if self.comparables_df is None:
                raise ValueError("'comparables_df' attribute has not been specified")
            stats = self._get_comparison_stats(df=self.comparables_df,
                                               data_name='EBIT Margin % (LTM)', classifier=classifier,
                                               agg_method=[self.percentile_75, 'median', self.percentile_25])
            median_name = 'Median'
            if classifier:
                grouping = classifier.lower()
            else:
                grouping = 'all'
            median_name += f"({grouping}) {round(stats['median'], 2)}%"
            # plot the median
            pd.DataFrame(data={median_name: ebit_df.shape[0] * [stats['median']]}).plot(ax=ax, kind='line')
            # make an area between the 25 and 75%-ile
            ax.fill_between(ebit_df.index, stats['percentile_25'], stats['percentile_75'], alpha=0.2)
            plt.xticks(rotation=90)  # rotate the x-axis labels

        if self.comparables_df is not None:
            min_price_target_num = 3
            analyst_filtered_df = self.comparables_df[
                (self.comparables_df['Price Target - #'] >= min_price_target_num) &
                (self.comparables_df.Ticker.str.lower() == self.company.lower())]
            if not analyst_filtered_df.empty:
                est_ebit = analyst_filtered_df['EBIT - Est Avg (NTM)'].values[0] / analyst_filtered_df['Revenues - Est Avg (NTM)'].values[0]
                est_ebit *= 100
                pd.DataFrame(data={f'EBIT Margin (NTM) {round(est_ebit, 2)}%': ebit_df.shape[0] * [est_ebit]}).plot(ax=ax,
                                                                                                                 kind='line',
                                                                                                                 linestyle='dashed')
                plt.xticks(rotation=90)  # rotate the x-axis labels
        return

    def analyse_reinvestment(self, capitalize_rnd: bool, comparison: bool = True, classifier: str = None,
                             reporting_type: str = 'fy') -> None:
        """
        Makes a 2x2 subplot with Gross Reinvestment, Maintenance and Reinvestment as a % of Revenue as well as the
        Capital-to-Sales ratio
        :param capitalize_rnd: bool if True, R&D is capitalized
        :param comparison: bool if True add the median as well as 25 and 75%-ile to the Reinvestment and Capital to
        Sales plots
        :param classifier: str filter out comparable stocks based on e.g. 'Sector'
        :param reporting_type: str
        :return: None
        """
        df = self.get_reinvestments_per_revenue(capitalize_rnd=capitalize_rnd, reporting_type=reporting_type, with_details=True) * 100
        gross_inv_items = ['Capital Expenditure', 'Cash Acquisitions', 'Change Non Cash Working Capital']
        maintenance_items = ['D&A for EBITDA']
        if capitalize_rnd:
            gross_inv_items.append('R&D Expenses')
            maintenance_items.append('R&D Amortization')

        # create a plot window with 4 subplots: gross reinvestment, maintenance, reinvestment (net) and capital-to-sales ratio
        _, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,5))
        plt.tight_layout(pad=1.5, w_pad=0.8, h_pad=5.0)

        df[gross_inv_items].plot(ax=axes[0 ,0], kind='bar', legend=True, title='Gross Reinvestment (% Revenue)')
        df[maintenance_items].plot(ax=axes[0 ,1], kind='bar', legend=True, title='Maintenance (% Revenue)')
        df['Reinvestment'].plot(ax=axes[1, 0], kind='line', title='Reinvestment (% Revenue)')
        self.get_capital_to_revenue_ratio(capitalize_rnd=capitalize_rnd, reporting_type=reporting_type).plot(ax=axes[1, 1], kind='line', title='Capital-to-Sales ratio')

        # if applicable add comparison looking at the median for reinvestment and capital to sales ratio using the
        # cross sectional data
        if comparison:
            if self.comparables_df is None:
                raise ValueError("'comparables_df' attribute has not been specified")
            # add a column with the relevant metric and add the median together with 25 and 75%-ile
            reinv_comp_df = self._get_comparison_reinvestment_df()
            capital_sales_comp_df = self._get_comparison_capital_sales_ratio_df()
            self._add_comparison_to_plot(plot=axes[1, 0], x_axis=df.index, df=reinv_comp_df, data_col_name='Reinvestment', classifier=classifier)
            self._add_comparison_to_plot(plot=axes[1, 1], x_axis=df.index, df=capital_sales_comp_df, data_col_name='Capital Sales Ratio', classifier=classifier, median_legend_suffix='')
        return

    def analyse_cash_flows(self, reporting_type: str):
        """
        Creates a plot with the cash flow components as a stacked bar chart together with a line for the Free Cash Flow
        to the Firm (FCFF) as a line
        :param reporting_type:
        :return:
        """
        # combine all the relevant items from the income statement, balance sheet and cash flows into a DataFrame
        total_df = self.get_fcff_components(reporting_type=reporting_type)

        # plot the components as a stacked bar chart and add the sum as a line
        _, ax = plt.subplots()
        total_df.loc[:, total_df.columns != 'FCFF'].plot(ax=ax, stacked=True, kind='bar', title='Free Cash Flow to Firm (FCFF)')
        total_df[['FCFF']].plot(ax=ax, color='black', style='.--')
        plt.xticks(rotation=90)  # rotate the x-axis labels
        return

    def _add_comparison_to_plot(self, plot, x_axis: pd.Index, df:pd.DataFrame, data_col_name: str, classifier: str, median_legend_suffix: str = '%') -> None:
        """
        Adds the median among comparable stocks together with the 25 and 75%-ile
        :param plot:
        :param x_axis: pd.Index
        :param df: DataFrame
        :param data_col_name: str
        :param classifier: str
        :return: None
        """
        # calculate the stats
        stats = self._get_comparison_stats(df=df, data_name=data_col_name, classifier=classifier,
                                           agg_method=[self.percentile_75, 'median', self.percentile_25])
        median_name = 'Median'
        if classifier:
            median_name += f" ({classifier})"
        median_name += f" {round(stats['median'], 2)}{median_legend_suffix}"
        # plot the median
        pd.DataFrame(index=x_axis, data={median_name: len(x_axis) * [stats['median']]}).plot(ax=plot, kind='line')
        # make an area between the 25 and 75%-ile
        plot.fill_between(x_axis, stats['percentile_25'], stats['percentile_75'], alpha=0.2)
        return None

    def _get_comparison_reinvestment_df(self):  # TODO change so that this is not called
        """
        Returns a DataFrame containing the cross sectional data and a new column named 'Reinvestment'
        :return: DataFrame
        """
        df = self.comparables_df.copy()
        df[['Capital Expenditure (LTM)', 'Cash Acquisitions (LTM)']] *= -1
        df['Reinvestment'] = df[['Capital Expenditure (LTM)', 'Cash Acquisitions (LTM)', 'Chg in NWC (LTM)']].sum(
            axis=1) - df['D&A for EBITDA (LTM)'].fillna(0).values
        df['Reinvestment'] /= df['Total Revenues (LTM)'].values
        df['Reinvestment'] *= 100
        return df

    def _get_comparison_capital_sales_ratio_df(self):  # TODO change so that this is not called
        """
        Returns a DataFrame containing the cross sectional data and a new column named 'Capital Sales Ratio'
        :return: DataFrame
        """
        df = self.comparables_df.copy()
        df['Capital Sales Ratio'] = df[['Net Debt (LTM)', 'Total Equity (LTM)']].sum(axis=1) / df['Total Revenues (LTM)'].fillna(0).values
        return df

    def _get_comparison_stats(self, df: pd.DataFrame, data_name: str, classifier: str, agg_method) -> pd.Series:
        """
        Returns a Series with 25%-ile, median and 75%-ile for a specific data column
        :param df: DataFrame
        :param data_name: str name of the data column
        :param classifier: str name of the column to filter on based on the corresponding value of the ticker
        :param agg_method:
        :return: Series
        """
        if classifier:
            if self.company is None:
                raise ValueError("'ticker' attribute is None")
            try:
                col_idx = list(df.columns.str.lower()).index(classifier.lower())
            except ValueError:
                raise ValueError(f"'{classifier}' is not a valid classifier for comparables")
            classifier = df.columns[col_idx]
            # filter out the row for the specific ticker
            company_df = df[df.Ticker == self.company.upper()].copy()
            if company_df.empty:
                raise ValueError(f"'{self.company}' is not part of the cross sectional data")
            classifier_for_ticker = company_df[classifier].values[0]
            df = df[df[classifier] == classifier_for_ticker].copy()
        stats = df[data_name].replace(np.inf, np.nan).agg(agg_method)
        return stats

    def sector_comparision(self, data_columns: {str, list}) -> pd.DataFrame:
        """
        Returns a DataFrame containing data for a specified ticker and a summary of sector aggregates (mean, max,
        75%-ile, median, 25%-ile and count)
        :param data_columns: str or list of str
        :return: DataFrame
        """
        return self._comparision(data_columns=data_columns, classifier_column='Sector')

    def industry_comparision(self, data_columns: {str, list}) -> pd.DataFrame:
        """
        Returns a DataFrame containing data for a specified ticker and a summary of industry aggregates (mean, max,
        75%-ile, median, 25%-ile and count)
        :param data_columns: str or list of str
        :return: DataFrame
        """
        return self._comparision(data_columns=data_columns, classifier_column='Industry')

    def _comparision(self, data_columns: {str, list}, classifier_column: str) -> pd.DataFrame:
        """
        Returns a DataFrame containing data for a specified ticker and a summary of aggregates (mean, max, 75%-ile,
        median, 25%-ile and count) for a specific str column (e.g. 'Sector' or 'Industry')
        :param data_columns: str or list of str (case insensitive)
        :param classifier_column: str (e.g. 'Sector' or 'Industry')
        :return: DataFrame
        """
        # filter rows to have the same classifier value as the ticker and aggregate the data from the specified columns
        comparable_df = self.get_filtered_comparables_df(filter_by=classifier_column)
        comparable_df.columns = comparable_df.columns.str.title()  # first letter is capitalized in each word

        # first letter is capitalized for each data column name
        if not isinstance(data_columns, list):
            data_columns = [data_columns]
        data_columns = [d_col.title() for d_col in data_columns.copy()]

        # aggregate the result for the comparable firms
        result_df = comparable_df[data_columns].agg(
            ['mean', 'max', self.percentile_75, 'median', self.percentile_25, 'min', 'count']
        )

        # change the index to be more informative
        result_df.index = [f'{idx.replace("_", " ").title()} ({classifier_column})' for idx in result_df.index]

        # combine the aggregate data with the data for the ticker
        ticker_data = self.get_company_data()
        ticker_data.index = ticker_data.index.str.title()
        result_df = pd.concat([pd.DataFrame(data=ticker_data[data_columns].values, index=data_columns, columns=[self.company]).T,
                               result_df])
        return result_df

    @staticmethod
    def percentile_75(x):
        return x.quantile(0.75)

    @staticmethod
    def percentile_25(x):
        return x.quantile(0.25)

    # __________________________________________________________________________________________________________________
    # forecasts
    def enough_analyst_coverage(self) -> bool:
        """
        Returns True if 'Price Target - #' is larger than min_num_price_target attribute else False
        :return: bool
        """
        return self.get_company_data()['Price Target - #'] >= self.min_num_price_target

    def revenue_growth_forecast_analyst(self, terminal_growth_rate: float) -> list:
        """
        Returns a list of floats with annual revenue growth forecast based on analyst estimates that are extrapolated to
        the 5th fiscal year by the CAGR over first 3 years that later converges to a terminal growth rate at 10y
        :param terminal_growth_rate: float
        :return: list
        """
        # data for the specific ticker
        company_data = self.get_company_data()

        # log a warning if not enough price targets (proxy for analyst coverage)
        if company_data['Price Target - #'] < self.min_num_price_target:
            logger.warning(f"Only has {company_data['Price Target - #']} price targets")

        # enough price targets make the use of analyst estimates
        est_fy1_g = company_data['Est Rev CAGR (1Y)'] / 100
        est_fy2_g = company_data['Revenues - Est YoY % (FY2E)'] / 100
        est_fy3_g = company_data['Revenues - Est YoY % (FY3E)'] / 100

        # extrapolate the estimated growth rates for the first 3 years to 5
        estimates = np.array([est_fy1_g, est_fy2_g, est_fy3_g])
        cagr = ((1 + estimates).prod()) ** (1 / len(estimates)) - 1
        return self._three_stage_model(year_1=est_fy1_g, year_2=est_fy2_g, year_3=est_fy3_g, mid_year=cagr,
                                       terminal_year=terminal_growth_rate)

    def revenue_growth_forecast_analyst_proxy(self, terminal_growth_rate: float, hist_proxy_data_name: str = 'Total Revenues/CAGR (5Y TTM)',
                                              proxy_filter_by: {None, str}='Industry'):
        """
        Returns a list of floats with annual revenue growth forecast based on proxy analyst estimates that are
        extrapolated to the 5th fiscal year by the CAGR over first 3 years that later converges to a terminal growth
        rate at 10y. Proxy estimates are calculated by looking at the historical distribution of growth and use that
        same percentile (rounded to quarters) to approximate the analyst growth estimates. I.e. if the stock has been
        growing as the median firm, the median growth estimate is used.
        :param hist_proxy_data_name: str name of the data column to be used in the historical estimation
        :param proxy_filter_by: str filter comparables based on this (e.g. 'Industry') if None use the netire universe
        :param terminal_growth_rate: float
        :return: list
        """
        # get the filtered comparable data
        comparables_df = self.get_filtered_comparables_df(filter_by=proxy_filter_by)

        # calculate the historical percentile for the specified data to be used to estimate the proxy
        hist_pct_ile = self.get_percentile_data_for_company(comparables_df=comparables_df,
                                                            data_col_name=hist_proxy_data_name, as_quartile=True)

        # use the historical percentile to proxy the analyst estimates
        est_fy1_g = comparables_df['Est Rev CAGR (1Y)'].quantile(hist_pct_ile) / 100
        est_fy2_g = comparables_df['Revenues - Est YoY % (FY2E)'].quantile(hist_pct_ile) / 100
        est_fy3_g = comparables_df['Revenues - Est YoY % (FY3E)'].quantile(hist_pct_ile) / 100

        # extrapolate the estimated growth rates for the first 3 years to 5
        estimates = np.array([est_fy1_g, est_fy2_g, est_fy3_g])
        cagr = ((1 + estimates).prod()) ** (1 / len(estimates)) - 1
        return self._three_stage_model(year_1=est_fy1_g, year_2=est_fy2_g, year_3=est_fy3_g, mid_year=cagr,
                                       terminal_year=terminal_growth_rate)

    def ebit_mrg_forecast_analyst(self, terminal_ebit_mrg: float):
        """
        Returns a list of floats with EBIT margin forecast based on analyst estimates that are extrapolated to the 5th
        fiscal year by the mean over first 3 years that later converges to a terminal margin at 10y
        :param terminal_ebit_mrg: float
        :return: list
        """
        # data for the specific ticker
        company_data = self.get_company_data()

        # log a warning if not enough price targets (proxy for analyst coverage)
        if company_data['Price Target - #'] < self.min_num_price_target:
            logger.warning(f"Only has {company_data['Price Target - #']} price targets")

        # enough price targets make the use of analyst estimates
        est_fy1_mrg = company_data['EBIT Margin - Est Avg (FY1E)'] / 100
        est_fy2_mrg = company_data['EBIT Margin - Est Avg (FY2E)'] / 100
        est_fy3_mrg = company_data['EBIT Margin - Est Avg (FY3E)'] / 100
        est_mrg_mid = np.mean([est_fy1_mrg, est_fy2_mrg, est_fy3_mrg])
        return self._three_stage_model(year_1=est_fy1_mrg, year_2=est_fy2_mrg, year_3=est_fy3_mrg, mid_year=est_mrg_mid,
                                       terminal_year=terminal_ebit_mrg)

    def ebit_mrg_forecast_analyst_proxy(self, terminal_ebit_mrg: float, proxy_filter_by: {None, str}='Industry') -> list:
        """
        Returns a list of floats with EBIT margin forecast based on proxy analyst estimates that are extrapolated to the
        5th fiscal year by the mean over first 3 years that later converges to a terminal margin at 10y.
        Proxy estimates are calculated by looking at the historical distribution of margins and use that same percentile
        (rounded to quarters) to approximate the analyst estimates. I.e. if the stock has been profitable as the median
        firm, the median margin estimate is used.
        :param terminal_ebit_mrg: float
        :param proxy_filter_by: str
        :return: list
        """
        # get the filtered comparable data
        comparables_df = self.get_filtered_comparables_df(filter_by=proxy_filter_by)

        # calculate the historical percentile for the specified data to be used to estimate the proxy
        hist_pct_ile = self.get_percentile_data_for_company(comparables_df=comparables_df,
                                                            data_col_name='EBIT Margin % (LTM)', as_quartile=True)

        # use the historical percentile to proxy the analyst estimates
        est_fy1_mrg = comparables_df['EBIT Margin - Est Avg (FY1E)'].quantile(hist_pct_ile) / 100
        est_fy2_mrg = comparables_df['EBIT Margin - Est Avg (FY2E)'].quantile(hist_pct_ile) / 100
        est_fy3_mrg = comparables_df['EBIT Margin - Est Avg (FY3E)'].quantile(hist_pct_ile) / 100
        est_mrg_mid = np.mean([est_fy1_mrg, est_fy2_mrg, est_fy3_mrg])
        return self._three_stage_model(year_1=est_fy1_mrg, year_2=est_fy2_mrg, year_3=est_fy3_mrg, mid_year=est_mrg_mid,
                                       terminal_year=terminal_ebit_mrg)

    def capital_sales_ratio_forecast(self) -> list:
        """
        Returns a list of floats. Takes the LTM capital to sales ratio, lets it converge to the historical mean over 5
        years then converge to sector median at 10 years
        :return: list of floats
        """
        hist_cap_rev = self.get_capital_to_revenue_ratio(capitalize_rnd=self.rnd_is_capitalized, reporting_type='ltm')
        initial_cap_rev_ratio = hist_cap_rev.iloc[-1]
        mid_cap_rev_ratio = hist_cap_rev.iloc[-12:].median()
        terminal_cap_rev_ratio = self.get_filtered_comparables_df(filter_by='Industry')['Capital Sales Ratio'].median()
        return pd.Series([initial_cap_rev_ratio, None, None, None, mid_cap_rev_ratio, None, None, None, None, terminal_cap_rev_ratio, terminal_cap_rev_ratio]).interpolate().values.tolist()

    def tax_rate_forecast(self, marginal_tax_rate: float) -> list:
        """
        Returns a list of floats. Takes the average effective tax rate seen for the past 12 months, floors it at zero and
        converge the rate to the marginal tax rate over the coming 5 years and stays there after that
        :param marginal_tax_rate: float
        :return: list
        """
        effective_tax_rate = self.get_effective_tax_rate(reporting_type='fq').iloc[-4:, 0].mean()
        tax_rate_list = [effective_tax_rate, None, None, None, marginal_tax_rate, marginal_tax_rate, marginal_tax_rate,
                         marginal_tax_rate, marginal_tax_rate, marginal_tax_rate]
        return pd.Series(tax_rate_list).interpolate().values.tolist()

    @staticmethod
    def _three_stage_model(year_1: float, year_2: float, year_3: float, mid_year: float, terminal_year: float) -> list:
        """
        Returns a list of length 11 with interpolated values between the provided ones
        :param year_1: float
        :param year_2: float
        :param year_3: float
        :param mid_year: float
        :param terminal_year: float
        :return: list
        """
        return pd.Series([year_1, year_2, year_3, None, mid_year, None, None, None, None, terminal_year, terminal_year]).interpolate().values.tolist()

    @property
    def ticker(self):
        return self._ticker

    @ticker.setter
    def ticker(self, ticker: str):
        if ticker:
            self._ticker = ticker.replace(' ', '').upper()
        else:
            self._ticker = ticker

    @property
    def company(self):
        return self._company

    @company.setter
    def company(self, company: str):
        """
        Set the financial reports to None when changing the company name
        :param company: str
        :return:
        """
        self.delete_financial_reports()
        if company:
            self._company = company.replace(' ', '').upper()
        else:
            self._company = company
        self.ticker = self.company

    @property
    def rnd_is_capitalized(self):
        return self._rnd_is_capitalized
