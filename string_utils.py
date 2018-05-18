import re

from unidecode import unidecode

numeric = re.compile('[0-9]')
not_alphanumeric = re.compile('[^0-9a-zA-Z ]+')


def normalize(string):
    return replace_not_alphanumeric(unidecode(remove_multi_spaces(string.strip().lower())))


def remove_multi_spaces(string):
    return ' '.join(string.strip().lower().split())


def create_filename(string):
    return replace_not_alphanumeric(string).replace(' ', '_').lower()


def replace_not_alphanumeric(string, repl=''):
    return re.sub(not_alphanumeric, repl, string)


def replace_numeric(string, repl='#'):
    return re.sub(numeric, repl, string)


def contains_numeric(string):
    return any(char.isdigit() for char in string)
