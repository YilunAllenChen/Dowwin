from pprint import pprint, pformat
import csv

sectors = {}
industries = {}
list_of_all_stocks = set()


def add_to(dic: dict, data: dict, by: str):
    if data[by] not in dic:
        dic[data[by]] = [data]
    else:
        dic[data[by]].append(data)


csvfiles = ['nyse.csv', 'nasdaq.csv', 'amex.csv']

for csvfile in csvfiles:
    with open(csvfile, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row.pop('')
            row.pop('IPOyear')
            row.pop("LastSale")
            # print(row['Symbol'], row['Name'], row['Sector'], row['MarketCap'], row['industry'])
            add_to(sectors, row, 'Sector')
            add_to(industries, row, 'industry')
            list_of_all_stocks.add(row['Symbol'])

with open('data_stock_basicinfo.py', 'w') as f:
    f.writelines('info_by_sectors = ' + str(sectors) + '\n\n')
    f.writelines('info_by_industries = ' + str(industries) + '\n\n')
    f.writelines('all_stocks_symbol = ' + str(list_of_all_stocks) + '\n\n')