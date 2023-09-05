"""fundamental_data_screener.py"""
import pandas as pd
import numpy as np
from datetime import date

from percentage_rank_score import calculate_percent_ranking
from tools.excel_tools import save_and_format_excel
from tools.general_tools import ask_user_yes_or_no

from config import KOYFIN_DATA_FOLDER_PATH
from config import KOYFIN_ANALYSIS_RESULT_FOLDER_PATH

from automation.download_us_swe_data import download_us_swe_data


KOYFIN_DATA_FILE_NAME = "usa_swe_fundamental_value_data"
MIN_MARKET_CAP_USD_M = None
MIN_NUM_PRICE_TARGET = 3
EXCL_SECTORS = ['Financials', 'Real Estate']
PROFITABLE = True
USE_QUARTILES = False
VALUE_MULTIPLE_EST = 'EV/EBIT (NTM)'
VALUE_MULTIPLE_HIST = 'EV/EBIT (LTM)'
FUNDAMENTALS_EST = ['Est Rev CAGR', 'Est EPS CAGR', 'EBIT Margin % (NTM)', 'Return on Capital (%) (LTM)']
FUNDAMENTALS_HIST = ['Total Revenues/CAGR (3Y FQ)', 'EBIT Margin % (LTM)', 'Return on Capital (%) (LTM)']
POSITIVE_EST_EPS_REV = True
EST_EPS_REV_VAR = ['EPS Est Avg Rev % (FY1E - 1M)', 'EPS Est Avg Rev % (FY1E - 3M)', 'EPS Est Avg Rev % (FY1E - 6M)']
MAX_HIST_VAL_METRIC_DIST_PCT = 50
VALUE_METRIC_PCT = 'EV/EBIT LTM (10Y R)'


def _assign_non_nan_value(row, col_names: list):
    """
    Takes a row and first checks if a column (specified by the given column names) is non nan. If yes, the value is
    returned, else the next column is checked until there are no more columns and returns nan
    :param row:
    :param col_names: list of strings
    :return:
    """
    for col_name in col_names:
        if not np.isnan(row[col_name]):
            return row[col_name]
        elif col_names.index(col_name) == len(col_names) - 1:
                return np.nan
        else:
            pass


def assign_value_est_rev_cagr(row):
    """
    Assigns the 3Y, 2Y or 1Y Est. Revenue CAGR depending on if there is a value available
    :param row: row
    :return:
    """
    column_names = ['Est Rev CAGR (3Y)', 'Est Rev CAGR (2Y)', 'Est Rev CAGR (1Y)']
    return _assign_non_nan_value(row, col_names=column_names)


def assign_value_est_eps_cagr(row):
    """
    Assigns the 3Y, 2Y or 1Y Est. EPS CAGR depending on if there is a value available
    :param row: row
    :return:
    """
    column_names = ['Est EPS CAGR (3Y)', 'Est EPS CAGR (2Y)', 'Est EPS CAGR (1Y)']
    return _assign_non_nan_value(row, col_names=column_names)


def add_additional_columns(df: pd.DataFrame) -> None:
    """
    Adds additional custom columns like estimated EBIT margin and CAGR depending on the available time horizon
    :param df: DataFrame
    :return: None
    """
    # add est. EBIT margin
    df['EBIT Margin % (NTM)'] = df['EBIT - Est Avg (NTM)'] / df['Revenues - Est Avg (NTM)'] * 100
    df['Est Rev CAGR'] = df.apply(assign_value_est_rev_cagr, axis=1)
    df['Est EPS CAGR'] = df.apply(assign_value_est_eps_cagr, axis=1)


def ranking_value_vs_fundamental(df: pd.DataFrame, value_metric: str, fundamentals: list, use_quartiles: bool) -> pd.DataFrame:
    """
    Returns a DataFrame with a score equal to the difference in ranking (percentiles or quartiles) between a value
    metric and fundamental data
    :param df: DataFrame
    :param value_metric: str
    :param fundamentals: list of str
    :param use_quartiles: bool convert percentiles to quartiles when True
    :return: DataFrame
    """

    # define the ranking configuration
    kpi_list = [value_metric]
    kpi_list.extend(fundamentals)
    ranking_config = [
        {
            'kpi': kpi,
            'ascending': True,
            'group_by': 'Industry',
        }
        for kpi in kpi_list
    ]

    # calculate the percentage rating and return result as a DataFrame
    pct_rank_df = calculate_percent_ranking(df=df, ranking_config=ranking_config)

    # convert percentiles to quartiles if applicable
    if use_quartiles:
        pct_rank_df = round_pct_to_quartiles_df(pct_df=pct_rank_df)

    # calculate the difference in ranking between the fundamental and value metric
    for i, fund_metric in enumerate(fundamentals):
        # the ranking of the value metric will be in the first column (i=0)
        pct_rank_df[f'{fund_metric} vs {value_metric}'] = pct_rank_df.iloc[:, i + 1] - pct_rank_df.iloc[:, 0]

    pct_rank_df['score'] = pct_rank_df.iloc[:, -len(fundamentals):].sum(axis=1)

    return pct_rank_df


def round_pct_to_quartiles_df(pct_df: pd.DataFrame)->pd.DataFrame:
    """
    Rounds the percentiles in the specified DataFrame to quartiles (e.g. 33 percentile -> 2nd quartile i.e. 50%)
    :param pct_df: DataFrame
    :return: DataFrame
    """
    rounding_logic = pd.Series([0.25, 0.5, 0.75, 1.0])
    labels = rounding_logic.to_list()
    rounding_logic = pd.Series([-np.inf]).append(rounding_logic)
    quart_rank_df = pct_df.copy()
    for col in pct_df.copy():
        quart_rank_df[col] = pd.to_numeric(pd.cut(pct_df[col], rounding_logic, labels=labels))
    return quart_rank_df


def eligibility_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out rows from a DataFrame and only returns the rows that has passed certain filters
    :param df: DataFrame
    :return: DataFrame
    """
    if MIN_MARKET_CAP_USD_M:
        df = df[df['MarketCap'] >= MIN_MARKET_CAP_USD_M]
    df = df[~df['Sector'].isin(EXCL_SECTORS)]
    if PROFITABLE:
        df = df[~df['P/E (LTM)'].isnull()]
    if POSITIVE_EST_EPS_REV:
        for eps_rev_col in EST_EPS_REV_VAR:
            df = df[df[eps_rev_col] > 0]
    if MAX_HIST_VAL_METRIC_DIST_PCT:
        df = df[df[f"{VALUE_METRIC_PCT}"] < MAX_HIST_VAL_METRIC_DIST_PCT]
    return df


def main():

    if ask_user_yes_or_no('Download new data from koyfin?'):
        download_us_swe_data(file_name=KOYFIN_DATA_FILE_NAME)

    df = pd.read_csv(KOYFIN_DATA_FOLDER_PATH / f'{KOYFIN_DATA_FILE_NAME}.csv', index_col=0)
    add_additional_columns(df=df)
    est_rank_df = ranking_value_vs_fundamental(df=df, value_metric=VALUE_MULTIPLE_EST, fundamentals=FUNDAMENTALS_EST,
                                               use_quartiles=USE_QUARTILES)
    hist_rank_df = ranking_value_vs_fundamental(df=df, value_metric=VALUE_MULTIPLE_HIST, fundamentals=FUNDAMENTALS_HIST,
                                                use_quartiles=USE_QUARTILES)

    # add the score depending on the number of price targets (proxy for analyst coverage)
    # if not enough analyst coverage, only use historical data else use estimates
    df['Use Estimate'] = df['Price Target - #'] >= MIN_NUM_PRICE_TARGET
    df['Score'] = np.where(df['Use Estimate'], est_rank_df['score'], hist_rank_df['score'])
    df.sort_values('Score', inplace=True, ascending=False)

    unfiltered_df = df.copy()
    df = eligibility_filter(df=df)
    result_df = df[['Name', 'Trading Country', 'Sector', 'Industry', 'Market Cap', 'Use Estimate', 'Score']]

    # save the result in Excel
    save_file_path = KOYFIN_ANALYSIS_RESULT_FOLDER_PATH / f"fundamental_value_score_results_{date.today().strftime('%Y%m%d')}.xlsx"
    print('Saving the result in excel...')
    save_and_format_excel(data={'result': result_df, 'detailed result': df, 'unfiltered': unfiltered_df,
                                'est. score': est_rank_df, 'hist. score': hist_rank_df}, save_file_path=save_file_path)
    print(f'Saved in {save_file_path}')


if __name__ == '__main__':
    main()
