import string
from time import gmtime, strftime


def word_count(text):
    for char in string.punctuation:
        text = text.replace(char, ' ')

    return len(text.split())


def get_timestamp():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())