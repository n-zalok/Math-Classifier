article = {'a': 'apple', 'b': 'ball', 'c': 'cat', 'd': ''}

text = ''
for art in article.values():
    if art:
        text += f'{art} '


print(len(text.split()))