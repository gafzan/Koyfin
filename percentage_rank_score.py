"""percentage_rank_score"""

import pandas as pd


def calculate_percent_ranking_score(df: pd.DataFrame, ranking_config: {list, dict}=None, sort_score=False)->pd.DataFrame:
    """
    Calculates a score as a weighted sum of percent/normalized rankings
    :param df: DataFrame
    :param ranking_config: list dict
    :param sort_score: bool
    :return: DataFrame
    """

    if ranking_config is None:
        # asks user to customize the ranking configuration
        ranking_config = _user_create_ranking_config(kpi_l=list(df), group_by_l=list(df))
    elif isinstance(ranking_config, dict):
        ranking_config = [ranking_config]

    # loop through each ranking and calculate a score a weighted score at the end
    result_df = pd.DataFrame(index=df.index)  # initialize the result
    for rank_conf in ranking_config:  # loop through rankings
        ranking_df = _perform_percent_ranking(df=df,
                                              kpi=rank_conf['kpi'],
                                              high_is_good=rank_conf['ascending'],
                                              group_by=rank_conf.get('group_by', None))

        result_df = result_df.join(ranking_df)

    # on the first column calculate the score as the weighted sum of normalized ranks
    result_df.insert(0, 'score',
                     (result_df * [d['weight'] for d in ranking_config]).sum(axis=1)
                     )
    # add the relevant kpis for reference
    used_kpis = set([d['kpi'] for d in ranking_config])
    result_df = result_df.join(df[used_kpis])

    if sort_score:
        result_df.sort_values(by=['score'], ascending=False, inplace=True)

    return result_df


def _perform_percent_ranking(df: pd.DataFrame, kpi: str, high_is_good: bool, group_by: str = None)->pd.DataFrame:
    """
    Returns a DataFrame with the normalized ranking (rank / count) for a specified column
    :param df: DataFrame
    :param kpi: str
    :param high_is_good: bool
    :param group_by: str
    :return: DataFrame
    """
    col_name = kpi
    if group_by:
        col_name += f" group by '{group_by}'"
        ranking = df.copy().groupby(group_by)[kpi].rank(pct=True, ascending=high_is_good)
    else:
        ranking = df.copy()[kpi].rank(pct=True, ascending=high_is_good)

    # convert to DataFrame
    col_name += (' high ' if high_is_good else ' low ') + 'is good score'
    return pd.DataFrame({col_name: ranking.values}, index=ranking.index)

# ______________________________________________________________________________________________________________________
# User input functions


def _user_create_ranking_config(kpi_l: list, group_by_l: list)->list:
    """
    Asks user to define the ranking configuration
    :param kpi_l: list of str
    :param group_by_l: list of str
    :return: list of dict
    """

    group_by_l.append(None)
    result = []
    ask_user = True
    while ask_user:
        # asks user for kpi, if high values are prefered, the variable the kpi should be grouped by (if any) and the
        # weight of the score
        kpi_input = _user_choose_from_list(_list=kpi_l, msg=f'Chose a KPI (1 - {len(kpi_l)})\n> ')  # user chose a kpi
        high_is_good = _user_input_yes_or_no(msg=f"\nHigh values for '{kpi_input}' is good?\n> ")
        group_by = _user_choose_from_list(_list=group_by_l, msg=f"Chose a variable to group '{kpi_input}' by "
            f"(1 - {len(group_by_l)})\n> ")
        weight = float(input(f"\nWeight for '{kpi_input}' grouped by '{group_by}'\n> "))

        # add configuration to the result list
        result.append(
            {
                'kpi': kpi_input,
                'ascending': high_is_good,
                'group_by': group_by,
                'weight': weight
            }
        )

        # print current ranking configuration
        print('\n', '*' * 5, 'Current rankings', '*' * 5)
        for idx, rank_conf_d in enumerate(result):
            print(idx + 1, f": {rank_conf_d['kpi']}, {'high' if high_is_good else 'low'} values are good, grouped by "
            f"{group_by}, weight = {round(100 * rank_conf_d['weight'], 2)}%")

        ask_user = _user_input_yes_or_no(msg='\nAdd another ranking?\n> ')
    return result


def _list_print(_list: list)->None:
    """
    Prints the elements of the list together with an index starting at 1
    :param _list: list
    :return: None
    """
    for idx, e in enumerate(_list):
        print(idx + 1, f' : {e}')
    return


def _user_input_yes_or_no(msg: str)->bool:
    """
    Asks user to answer a yes/no question and returns the corresponding bool
    :param msg: str
    :return: bool
    """
    yes_ans = ['y', 'yes', '1', 'true', 't']
    no_ans = ['n', 'no', '0', 'false', 'f']
    ans = input(msg).lower()
    print('')
    if ans in yes_ans:
        return True
    elif ans in no_ans:
        return False
    else:
        raise ValueError(f"'{ans}' is not recognized answer")


def _user_choose_from_list(_list: list, msg: str):
    """
    Asks user to pick an element from a list and returns the element
    :param _list: list
    :param msg: str
    :return:
    """
    _list_print(_list=_list)
    return _list[int(input(msg)) - 1]

