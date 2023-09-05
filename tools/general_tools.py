"""general_tools.py"""

from pathlib import Path
import os
import getpass
getpass.getuser()

DOWNLOADS_PATH = Path(r'C:\Users') / getpass.getuser() / 'Downloads'


def get_newest_file_paths(path: str, suffix: {str, list, tuple}=('xlsx', 'csv'), num_files: int = None):
    """
    Returns a list with the newest files created in a specified location
    :param path: str path location of the folder
    :param suffix: str or iterable with str
    :param num_files: int (if not specified will return all ordered by age)
    :return:
    """
    os.chdir(path)
    files = sorted(os.listdir(os.getcwd()), key=os.path.getmtime)
    if isinstance(suffix, str):
        suffix = [suffix]
    files = [f for f in files.copy() if f.split('.')[-1] in suffix]
    if num_files:
        return files[-num_files:]
    else:
        return files


def rename_latest_downloaded_file(new_path: str, new_name: str = None):
    """
    Moves the latest downloaded file to a new location and saved under a new name. Overwrites an existing file with the
    same name in the new location if it exists.
    :param new_path: str
    :param new_name: str (excluding file type)
    :return: None
    """
    os.chdir(DOWNLOADS_PATH)
    files = sorted(os.listdir(os.getcwd()), key=os.path.getmtime)
    newest = files[-1]
    if new_name:
        new_file_path = f'{new_path}/{new_name}.{newest.split(".")[-1]}'
    else:
        new_file_path = f'{new_path}/{newest.split("/")[-1]}'
    if os.path.isfile(new_file_path):
        os.remove(new_file_path)
    os.rename(newest, new_file_path)
    return


def user_picks_element_from_list(list_: list):
    """Assumes that list_ is a list. Script will print a list of all the elements and then ask user to pick one.
    Returns the chosen element."""
    if len(list_) == 0:
        raise ValueError('List is empty.')
    for i in range(len(list_)):
        print('{}: {}'.format(i + 1, list_[i]))
    ask_user = True
    while ask_user:
        try:
            list_index = int(input('Enter a number between 1 and {}:'.format(len(list_))))
            assert 1 <= list_index <= len(list_)
            ask_user = False
        except (ValueError, AssertionError):
            pass
    return list_[list_index - 1]


def ask_user_yes_or_no(question: str)->bool:
    """
    Asks a question to user and user needs to say 'yes' or 'no' (several versions are accepted)
    :param question: str
    :return: bool
    """
    accpetable_yes = ['sure', 'yeah', 'yes', 'y']
    accpetable_no = ['no', 'n', 'nope']

    while True:
        answer = input(question + '\nYes or No?: ').lower()
        if answer in accpetable_yes:
            return True
        elif answer in accpetable_no:
            return False
        else:
            print("'{}' is not an acceptable answer...\n".format(answer))

