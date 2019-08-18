# Roll 				= 	19CS60R01
# Name 				= 	Ronit Samaddar
# Assignment Number = 	1


import csv
import numpy as np
import math
import random
import sys

class Node:

	def __init__(self):
		self.attr=""
		self.rows=[]
		self.level=-1
		self.count=0
		self.class_val=""
		self.child_nodes=[]

	




def log(val,base):
	return math.log(val)/math.log(base)
def readInput(file):
	with open(file,"r") as f:
		data=csv.reader(f)
		column=[]
		matrix=[]
		index=0
		for row in data:
			if index==0:
				column=row
			else:
				matrix.append(row)
			index+=1
	return column,matrix
def get_class_count(class_label):
	pos_rows=np.where(class_label=='yes')
	pos=len(pos_rows[0])
	neg_rows=np.where(class_label=='no')
	neg=len(neg_rows[0])
	return pos,neg

def get_values(att,cols,data):
	for i in range(0,len(cols)):
		if(cols[i]==att):
			return data[:,i]



def form_tree(index,node,cols,npdata,npclass):
	#print(str(index)+"-th iteration of form_tree")
	#print("Level of present node = "+str(node.level))
	rows=node.rows
	data=npdata[rows,:]
	target=npclass[rows,:]
	pos,neg=get_class_count(target)
	#print("Rows = "+str(pos)+"+, "+str(neg)+"-")

	if(pos==0 or neg>=9*pos):
		node.class_label="no"
		#print("LEAF NODE!!!! Class Label = "+node.class_label)
		return -1
		
	elif(neg==0 or pos>=9*neg):
		node.class_label="yes"
		#print("LEAF NODE!!!! Class Label = "+node.class_label)
		return 1
		
	else:
		#calculating initial entropy
		E1=0
		classes=list(np.unique(list(target)))
		num_class=len(classes)
		for c in classes:
			#print("Class = "+c)
			#print("Rows = ")
			l=np.where(target==c)
			#print("Length of class = "+str(len(l[0])))

			p=len(l[0])*1.0/len(rows)
			if(p!=0):
				E1=E1+p*-log(p,num_class)
		#End of calculating E1

		#calculating next Entropy
		flag=0
		E2=0
		maxE=-99999
		maxattr=""
		for attr in cols[0:len(cols)-1]:
			col_list=get_values(attr,cols,data)
			values=list(np.unique(list(col_list)))
			if(len(values)==1):
				continue
			else:
				for j in range(0,len(values)):
					val=values[j]
					rows1=np.where(col_list==val)
					frac=len(rows1[0])*1.0/len(rows)
					data1=data[rows1[0],:]
					target1=target[rows1[0]]
					s=0;
					for c in classes:
						r=np.where(target1==c)
						#print(attr,val,c,len(r[0]),len(rows1[0]))
						p=len(r[0])*1.0/len(rows1[0])
						if(p!=0):
							s=s+p*-log(p,num_class)
						#print(attr,val,c,len(r[0]),len(rows1[0]),p,p*-log(p,num_class))
					s=s*frac
					E2=E2+s
					#print(attr,val,s)
					#print(val,s)
				#print(attr,E1,E2)
				E=E1-E2
				if(E>maxE):
					maxE=E
					maxattr=attr
		if(maxattr==""):
			if(pos>=neg):
				node.class_label="yes"
				#print("LEAF NODE!!!! Class Label = "+node.class_label)
				return 1
				
			else:
				node.class_label="no"
				#print("LEAF NODE!!!! Class Label = "+node.class_label)
				return -1
				
		else:
			#print("NON-LEAF NODE!!!! Attribute chosen = "+maxattr)
			node.attr=maxattr
			col_list=get_values(maxattr,cols,data)
			values=list(np.unique(list(col_list)))
			for i in range(0,len(values)):
				v=values[i]
				node1=Node()
				rw=np.where(col_list==v)
				node1.rows=list(rw[0])
				node1.count=len(node1.rows)
				node1.level=node.level+1
				node.child_nodes.append([v,node1])
			for i in range(0,len(values)):
				#print("Value chosen for next iteration : "+node.child_nodes[i][0])
				#print("Press any key for next iteration\n")
				#sys.stdin.read(1)
				p=form_tree(index+1,node.child_nodes[i][1],cols,data,target)
				if(p==1):
					node.child_nodes[i][1].class_val="yes"
				elif(p==-1):
					node.child_nodes[i][1].class_val="no"
				#print("For "+str(index)+"-th iteration "+str(i)+"-th child node = "+node.child_nodes[i][1].class_val+" class")
	return 0;

def print_tree(node,s):
	attr=node.attr
	num_child=len(node.child_nodes)
	for i in range(0,num_child):
		child=node.child_nodes[i]
		value=child[0]
		cnode=child[1]
		
		if(cnode.class_val==""):
			print(s+attr+" = "+value)
			print_tree(cnode,s+"\t")
		else:
			print(s+attr+" = "+value+" : "+cnode.class_val)
			
def predict(root,cols,row):
	col_val={}
	for i in range(0,len(cols)-1):
		col_val[cols[i]]=row[0,i]
	node=root
	while(node.class_val!="yes" and node.class_val!="no"):
		attr=node.attr
		val=col_val[attr]
		for i in range(0,len(node.child_nodes)):
			if(val==node.child_nodes[i][0]):
				node=node.child_nodes[i][1]
				break
	return node.class_val




		





def main():
	cols,matrix=readInput("data1_19.csv")
	#print(matrix)
	#print(cols)
	npmatrix=np.matrix(matrix)

	npdata=npmatrix[:,0:npmatrix.shape[1]-1]
	npclass=npmatrix[:,npmatrix.shape[1]-1]
	#print(npclass)

	num_rows=npdata.shape[0]
	num_features=npdata.shape[1]
	classes=np.unique(list(npclass))
	num_classes=classes.shape[0]
	print("Number of rows of data = "+str(num_rows))
	print("Number of features/attributes = "+str(num_features))
	print("Number of classes = "+str(num_classes))
	print("Classes = "+str(classes))

	print("\n\nFeatures :=\n")

	for i in range(0,num_features):
		print("Feature Name: "+cols[i])
		print("Feature Values: "+str(list(np.unique(list(npdata[:,i]))))+"\n")

	rows=list(range(0,num_rows))
	test_rows=list(random.sample(rows,10))
	train_rows=rows
	for x in test_rows:
		train_rows.remove(x)

	#print(train_rows)

	
	root=Node()
	root.level=0
	root.rows=train_rows
	root.count=len(train_rows)
	form_tree(0,root,cols,npdata,npclass)
	print("\n\nDECISION TREE := ")
	print_tree(root,"")

	print("\n\nTEST SAMPLES :=")
	
	sample=npmatrix[test_rows,:]
	for i in range(0,sample.shape[0]):
		m=sample[i,0:sample.shape[1]]
		l=sample[i,0:sample.shape[1]-1]
		print(str(i)+"-th test sample = "+str(m))
		pred=predict(root,cols,l)
		print("Prediction = "+str(pred))
		
	
		#print(str(i)+"-th test samples"+str(npmatrix[test_rows[i],:])))
		#pred=predict(root,)
		#print("Prediction = "+pred)


	

	
	






if __name__=="__main__":
	main()
