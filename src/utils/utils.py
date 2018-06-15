import string


def word_count(text):
    for char in string.punctuation:
        text = text.replace(char, ' ')

    return len(text.split())
