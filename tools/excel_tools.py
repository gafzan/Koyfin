"""excel_tools.py"""

import pandas as pd


def save_and_format_excel(data: {dict, pd.DataFrame}, save_file_path: str, first_col_width: int = 20,
                          other_col_width: int = 20)->None:
    """
    Saves each DataFrame in Excel and formats the results
    :param data: DataFrame or dict of DataFrames (with sheet names as keys)
    :param save_file_path: str
    :return: None
    """
    # create a pandas excel writer using xlsx writer as the engine
    xl_writer = pd.ExcelWriter(save_file_path, engine='xlsxwriter')
    workbook = xl_writer.book
    worksheets = xl_writer.sheets

    header_format_map = {'font_color': 'white', 'fg_color': '#002060', 'bottom': 5, 'top': 5, 'align': 'right',
                         'bold': True}
    row_format_map = {'font_color': 'black', 'align': 'right', 'num_format': '#,##0.00'}

    # always have the data as a dict with sheet names as keys and DataFrames as values
    if isinstance(data, pd.DataFrame):
        df_dict = {'result': data}
    else:
        df_dict = data

    # loop through the dict and store the DataFrame and format the headers and rows
    for name, df in df_dict.items():
        df.to_excel(xl_writer, sheet_name=name)
        values = df.values
        col_headers = df.columns.values
        first_row_values = df.index.values
        for row in range(df.shape[0] + 1):
            for col in range(df.shape[1] + 1):
                if row == 0:  # header row should have a different color and borders
                    format_map = header_format_map
                    if col == 0:
                        value = df.index.name
                    else:
                        value = col_headers[col - 1]
                else:
                    format_map = row_format_map
                    if row % 2 == 0:
                        format_map['fg_color'] = 'white'
                    else:
                        format_map['fg_color'] = '#F2F2F2'

                    if col == 0:  # first row
                        format_map['bold'] = True
                        value = first_row_values[row - 1]
                    else:  # values inside the table
                        format_map['bold'] = False
                        value = values[row - 1, col - 1]
                cell_format = workbook.add_format(format_map)
                # formatting does not work on nan
                try:
                    worksheets[name].write(row, col, value, cell_format)
                except TypeError:
                    worksheets[name].write(row, col, '', cell_format)

        # set the column width and freeze a cell
        worksheets[name].set_column(0, 0, first_col_width)  # first column
        worksheets[name].set_column(1, df.shape[1], other_col_width)  # other columns
        worksheets[name].freeze_panes(1, 1)
    xl_writer.close()



