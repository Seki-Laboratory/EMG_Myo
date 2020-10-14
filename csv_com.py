import csv
with open('./usui.csv') as f:
    reader = csv.reader(f)
    l = [row for row in reader]


print(l)