"""demo_2.py"""

from koyfinbot import KoyfinDataDownloader

criteria = [
    {
        'country': 'United States',
        'industry': 'Banks'
    },
    {
        'country': 'Sweden',
        'industry': 'Energy'
    }
]

data = {
    'Industry': None,
    'Total Debt': 'Last Twelve Months',
    'Revenue Estimate CAGR': ['1 Year', '2 Year', '3 Year'],
}

bot = KoyfinDataDownloader(screen_criteria=criteria)
bot.download_data(data_config=data)
bot.save_data()
