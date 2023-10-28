"""demo_1.py"""

from automation.koyfinbot import KoyfinBot

criteria = {
    'country': ['United States', 'Sweden'],
    'industry': ['Banks', 'Energy']
}

data = {
    'Industry': None,
    'Total Debt': 'Last Twelve Months',
    'Revenue Estimate CAGR': ['1 Year', '2 Year', '3 Year'],
}

bot = KoyfinBot()  # initialize an instance of a KoyfinBot object
bot.login()  # insert your credentials
bot._go_to_my_screens()  # goes to the 'My Screens' page
bot._add_new_screen()
bot._name_screen(name='KoyfinBot demo')
bot._set_universe_criteria(criteria_config=criteria)
bot._create_screen()
bot._add_data_columns(data_columns_config=data)
bot._download_screener_result()  # the csv file will be stored under 'Downloads'
bot.driver.close()


