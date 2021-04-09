import csv

def file_open(filename):
    with open(filename,'r',encoding='utf-8',newline='') as file:
        data = list(csv.reader(file))
    return data

def columns_from_file(data):
    user = [i[0] for i in data]
    article = [i[1] for i in data]
    return user, article

if __name__ == "__main__":
    a = file_open('readers.csv')
    # print(a)
    user, article = columns_from_file(a)
    # print(user)