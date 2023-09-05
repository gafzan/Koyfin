"""download_us_swe_data.py"""

from automation.koyfinbot import KoyfinDataDownloader

SCREEN_CRITERIA_CONFIG = [
        {
            'country': ['United States', 'Sweden'],
            'sector': ['Communication Services', 'Consumer Discretionary', 'Consumer Staples', 'Energy']
        },
        {
            'country': ['United States', 'Sweden'],
            'sector': ['Financials']
        },
        {
            'country': ['United States', 'Sweden'],
            'sector': ['Health Care', 'Industrials', 'Information Technology']
        },
        {
            'country': ['United States', 'Sweden'],
            'sector': ['Materials', 'Real Estate', 'Utilities']
        }
    ]


def download_us_swe_data(file_name: str = None):
    if not file_name:
        file_name = 'usa_swe_data'
    kf_downloader = KoyfinDataDownloader(screen_criteria=SCREEN_CRITERIA_CONFIG)
    kf_downloader.download_data()
    kf_downloader.save_data(file_name=file_name)


def main():
    download_us_swe_data()


if __name__ == '__main__':
    main()
