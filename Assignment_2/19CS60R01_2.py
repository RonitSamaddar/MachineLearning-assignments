#Name 				:	Ronit Samaddar
#Roll				:	19CS60R01
#Assignment Number	:	2


import pandas as pd
import numpy as np

#Reading and seeing the database

DataFrame=pd.read_csv("data2_19.csv")
#print(DataFrame.head(10))

#Function for getting the original csv and restoring it to a columnwise separated csv

def restoreCSV(inputfile,outputfile):
    DataFrame=pd.read_csv(inputfile)
    cols=DataFrame.columns.values
    cols_new=cols[0].split(',')#Getting columns name separated by comma delimiter
    #print(cols_new)
    data=DataFrame.values
    data_new=np.zeros((data.shape[0],len(cols_new)),int)
    #print(data_new.shape)
    for i in range(0,data.shape[0]):
        lst=[int(values) for values in data[i][0].split(',')]#Getting attribute values separated by comma delimiter
        data_new[i]=np.array(lst)
    #print(data_new)
    DataFrame_new=pd.DataFrame(data_new,columns=cols_new)#Generating new dataframe
    DataFrame_new.to_csv(outputfile,index=False)


restoreCSV("data2_19.csv","data.csv")
restoreCSV("test2_19.csv","test.csv")





df_train=pd.read_csv("data.csv")
df_test=pd.read_csv("test.csv")
#Getting column names
cols=df_train.columns.values


#function to get number of rows having a particular class value and attribute value
def get_frame_size(dataframe,class_name,class_value,attr_name="",attr_value=-1):
    
    #Getting dataframe consisting of rows where class=class value and attr_name==attr_value 
    d=dataframe.loc[dataframe[class_name]==class_value]

    if(attr_name!=""):
        d2=d.loc[d[attr_name]==attr_value]
        d=d2
    return len(d)



#function to predict a given data row based on a train dataframe
def predict(cols,data,dataframe):
    num_class=len(df_train[cols[0]].unique())
    num_rows=len(dataframe)
    #print(num_class)
    max_value=-1
    max_class=''
    for c in list(df_train[cols[0]].unique()):
        n_c=get_frame_size(dataframe,cols[0],c)#Getting number of rows with the particular class value
        p_ac=1
        for i in range(1,len(cols)):        
            cname=cols[i]
            cvalue=data[i]
            n_ic=get_frame_size(dataframe,cols[0],c,cname,cvalue)#Getting number of rows with data[cname]=cvalue
            #and class = cname

            p=(n_ic+1)*1.0/(n_c+num_class)
            p_ac=p_ac*p
        p_ac=p_ac*(n_c*1.0/num_rows)
        if(p_ac>max_value):
            max_value=p_ac
            max_class=c
    return max_class
        
            

#FINDING ACCURACY ON TRAIN DATASET
count1=0
count2=0
for i in range(0,len(df_train)):
    c1=df_train.iloc[i][0]
    c2=predict(cols,df_train.iloc[i],df_train)
    #print(c1,c2)
    if(c1==c2):
        count1+=1
    count2+=1
print("Accuracy on train data set= "+str((count1*100.0)/count2))


#FINDING ACCURACY ON TEST DATASET
count1=0
count2=0
for i in range(0,len(df_test)):
    c1=df_test.iloc[i][0]
    c2=predict(cols,df_test.iloc[i],df_train)
    #print(c1,c2)
    if(c1==c2):
        count1+=1
    count2+=1
print("Accuracy on test data set= "+str((count1*100.0)/count2))




