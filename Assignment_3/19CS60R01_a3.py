#MAIN GIVEN IN THE END
#Accuracy coming different at different times as everytime the sampling of the database
#affects the consecutive decisions trees constructed and so model and predictions varies
#on multiple runs of this code

##RUN MULTIPLE TIMES TO SEE THE ACCURACIES


import numpy as np
import pandas as pd
import math
import sys




class Node:
    def __init__(self):
        self.attr=""
        self.class_val=""
        self.child_nodes=[]

def log2(x):
    return math.log(x)/math.log(2)
def pos_neg_count(df):
    num_rows=len(df.axes[0])
    num_cols=len(df.axes[1])-1#neglecting the class column
    
    pos_df=df[df["survived"]=="yes"]
    pos_count=len(pos_df.axes[0])
    neg_count=num_rows-pos_count
    return pos_count,neg_count
def entropy(pos,neg,num_rows):
    if(pos==0 or neg==0):
        return 0
    pos_pr=pos/num_rows
    neg_pr=neg/num_rows
    E=-pos_pr*log2(pos_pr)-neg_pr*log2(neg_pr)
    return E


def form_tree(df,nd,attr_list,level,print_flag):
   
    #print("ATTR LIST = "+str(attr_list))
    #df = dataframe
    #nd=Current Node
    
    #print(df["survived"].unique()) //Two classes = "yes" and "no"
    num_rows=len(df.axes[0])
    num_cols=len(df.axes[1])-1#neglecting the class column
    pos,neg=pos_neg_count(df)
    
    t="\t"*level
    if(pos==0):
        nd.class_val="no"
        if(print_flag==1): print(t+"##Class = no")
        
        return nd
    if(neg==0):
        nd.class_val="yes"
        if(print_flag==1): print(t+"##Class = yes")
        return nd
    
    
    #Finding Prior Entropy
    E1=entropy(pos,neg,num_rows)
    #print(E1)
    #E1==prior Entropy
    
    max_attr=""
    max_diff=-99999
    #Finding Posterior Entropy
    for c in attr_list:
        u=list(df[c].unique())
        if(len(u)==1):
            attr_list.remove(c)
            continue
        else:
            E2=0
            for val in u:
                dfv=df[df[c]==val]
                num_rowsv=len(dfv.axes[0])
                posv,negv=pos_neg_count(dfv)
                E=entropy(posv,negv,num_rowsv)
                E2=E2+num_rowsv*1.0*E/num_rows
            Ediff=E1-E2
            if(Ediff>max_diff):
                max_diff=Ediff
                max_attr=c
                
    
    if(max_diff!=-99999):
        #If a attribute is Chosen, we create child nodes and recursively call function on them
        nd.attr=max_attr
        attr_list.remove(nd.attr)
        u=list(df[max_attr].unique())
        for val in u:
            df1=df[df[max_attr]==val]
            
            if(print_flag==1): print(t+"##"+max_attr+" = "+str(val))
            RN=Node()
            nd.child_nodes.append([val,RN])
            form_tree(df1,RN,attr_list,level+1,print_flag)
        if(print_flag==1): print(t+"##"+max_attr+" = DEFAULT")
        RN=Node()
        if(pos>=neg):
            if(print_flag==1): print(t+"        ##Class = yes")
            RN.class_val="yes"
        else:
            if(print_flag==1): print(t+"        ##Class = no")
            RN.class_val="no"
        nd.child_nodes.append(["DEFAULT",RN])
        attr_list.append(nd.attr)
        return nd
    else:
        #If no more attribute to choose, then we assign class as majority class
        if(pos>=neg):
            nd.class_val="yes"
            if(print_flag==1): print(t+"##Class = yes")
            return nd
        else:
            nd.class_val="no"
            if(print_flag==1): print(t+"##Class = no")
            return nd
            
        

def form_dec_tree(df,print_flag):
    columns=list(df.columns)
    attr_list=columns[0:len(columns)-1] # attr_list = possible attributes to choose from
    RN=Node() # Node created for root node
    RN=form_tree(df,RN,attr_list,0,print_flag)
    return RN
def predict(print_flag,df,row,tree):
    node=tree
    #print(r)
    #print("\n\n")
    index=0
    while(node.class_val!="yes" and node.class_val!="no"):
        if(print_flag==1):
            print("    PREDICTION INDEX = "+str(index))
            index+=1
        attr=node.attr
        val=df.at[row,attr]
        flag=0
        for l in node.child_nodes:
            if(print_flag==1):
                print(l[0],val)
                c=sys.stdin.read(1)
            if(l[0]==val):
                if(print_flag==1):print("    GETTING NEW NODE")
                node=l[1]
                flag=1
                break
            if(l[0]=="DEFAULT"):
                cnode=l[1]
        if(flag==0):
            node=cnode
    return node.class_val
    
    



def AdaBoost_train(df):
    
    #Will return list of classifiers in the form [ (Tree object 1,alpha 1),   (Tree object 2,alpha 2),...]
    num_rows=len(df.axes[0])
    classifiers=[] # will be of the form classifiers=[[Tree1,Alpha1],[Tree2,Alpha2],[Tree3,Alpha3]]
    weight_rows=np.ones(num_rows)
    weight_rows=weight_rows/num_rows
    for i in range(0,3):
        #print("Classifier "+str(i))
        Tree=form_dec_tree(df,0)
        #print("Got Classifier "+str(i))
        #print(df.iloc[0])
        target=[[predict(0,df,int(row),Tree),df.at[row,"survived"]] for row in df.index]
        #each element of target = [predicted class,actual class]
        #print(target)
        eq=[(target[r][0]==target[r][1])*1 for r in range(0,num_rows)]
        acc=sum(eq)*100.0/num_rows
        print("Classifier "+str(i+1)+" trained with accuracy "+str(acc))
        rows=[index for index,value in enumerate(eq) if value==0] # Misclassified rows
        mis_weights=[weight_rows[r] for r in rows] # Weights of misclassified rows
        error_rate=(sum(mis_weights)*1.0+1*math.exp(-6))/num_rows
        alpha=0.5*math.log((1-error_rate)/(error_rate*1.0))
        classifiers.append([Tree,alpha])
        print("Classifier "+str(i+1)+" has error rate,alpha = "+str(error_rate)+","+str(alpha))
        
        
        #print(pos_neg_count(df))
        if(i<2):
            for j in range(0,num_rows):
                if(eq[j]==1):#Correctly Classified
                    weight_rows[j]=weight_rows[j]*math.exp(-alpha)
                else:
                    weight_rows[j]=weight_rows[j]*math.exp(alpha)
            weight_rows=weight_rows*1.0/np.sum(weight_rows)
            df=df.sample(num_rows,replace=True,axis=0,weights=weight_rows)
            df=df.reset_index(drop=True)   
    
    
    return classifiers


def predict_test(df,trees):
    #print(trees)
    #print("HELLO")
    pred=[]
    for row in df.index:
        #print("PREDICTING ROW "+str(row))
        pred1=[]
        #print("PREDICTING FOR TREE 1")
        pred1.append(predict(0,df,int(row),trees[0][0]))
        #print("PREDICTING FOR TREE 2")
        pred1.append(predict(0,df,int(row),trees[1][0]))
        #print("PREDICTING FOR TREE 3")
        pred1.append(predict(0,df,int(row),trees[2][0]))
        pred.append(pred1)
    #print("HELLO2")
    alphas=[trees[i][1] for i in range(0,3)]
    #print(alphas)
    new_pred=[]
    for row in df.index:
        pred_row=pred[row]
        sum_yes=0
        sum_no=0
        for i in range(0,3):
            if pred_row[i]=="yes":
                sum_yes+=alphas[i]
            elif pred_row[i]=="no":
                sum_no+=alphas[i]
        #print(pred_row)
        #print(alphas)
        #print(sum_yes,sum_no)
        #print("\n")
        if(sum_yes>=sum_no):
            new_pred.append("yes")
        else:
            new_pred.append("no")
    tar=[[new_pred[r],df.at[r,"survived"]] for r in df.index]
    eq=[(tar[r][0]==tar[r][1])*1 for r in df.index]
    #print(sum(eq),len(df.index))
    acc=sum(eq)*100.0/len(df.index)
    return acc
    #print(tar)
    


def main():
    df=pd.read_csv("data3_19.csv")
    cols=df.columns
    Classifiers=AdaBoost_train(df)
    #print(Classifiers)
    df_test=pd.read_csv("test3_19.csv",names=cols)
    acc=predict_test(df_test,Classifiers)
    print("\nFinal test accuracy = "+str(acc))

if __name__=="__main__":
    main()
    
