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
bot.go_to_my_screens()  # goes to the 'My Screens' page
bot.add_new_screen()
bot.name_screen(name='KoyfinBot demo')
bot.add_universe_criteria(criteria=criteria)
bot.run_screener()
bot.add_data_columns(data_columns_config=data)
bot.download_screener_result()  # the csv file will be stored under 'Downloads'
bot.driver.close()


