#name : TAISHI ROY
#roll : 19CS60R26
#assignment no : 2

from __future__ import print_function
import csv
import sys

def dataManip( filename ):
    dataSet = []
    fields = []
    rows = [] 
    data = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)  
        fields = csvreader.next()  
        for row in csvreader: 
            rows = []
            for col in row:
                rows.append(col.replace(',',''))
            dataSet.append(rows)

    for i in range(len(dataSet)):
        x = []
        for j in dataSet[i][0]:
            x.append(int(j))
        data.append(x)

    return data



def getAccuracy (data, p0, p1, tset) :
    count = 0
    for i in range(len(data)):
        pno = p0
        pyes = p1
        for j in range(1,len(data[i])):
            pno*=tset[j-1][(data[i][j]-1)][0]
            pyes*=tset[j-1][(data[i][j]-1)][1]


        if pno >= pyes: 
            if data[i][0] == 0:
                count+=1
        else:
            if data[i][0] == 1:
                count+=1

    return (count/(len(data)*1.0))*100




def main():
    filename = "data2_19.csv"
    data = dataManip(filename)


    # data is a list of list containing the dataset
    X= []

    for i in range(7):
        row = []
        for j in range(len(data)):
            row.append(data[j][i])
        X.append(row)


    # X of size 7 contains formatted data
    p0 = X[0].count(0)/(len(X[0]) * 1.0)
    p1 = X[0].count(1)/(len(X[0]) * 1.0)

    tset = []
    for l in range(1,7):
        p = []
        for i in range(5):
            row = []
            for j in range(2):
                count = 0
                for k in range(len(X[0])):
                    if X[0][k]==j and X[l][k]==i+1:
                        count+=1
                if count==0 :
                    row.append((count+1)/((X[0].count(1)+2)*1.0))
                else:
                    row.append(count/(X[0].count(1)*1.0))
            p.append(row)
        tset.append(p)

    # in tset first -> attr 0 5 second -> type 0 4 third -> D 0 1

    filename = "test2_19.csv"
    test = dataManip(filename)

    acc = getAccuracy(test,p0,p1,tset)

    print("Accuracy (test data) = ", end=' ')
    print(acc, end=' ')
    print("%")

    

if __name__ == '__main__':
	main()
