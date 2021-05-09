import json
import string

printable = set(string.printable)

with open('datasets/quotes.json', encoding="utf-8") as f:
    rawData = json.load(f)

data = []

for item in rawData:
    data.append(item['Quote'])

data = list(dict.fromkeys(data))

with open('datasets/quotes.txt', 'w', encoding="utf-8") as f:
    for item in data:
        f.write(''.join(filter(lambda x: x in printable, item)))
        f.write("\n")