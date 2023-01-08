import numpy as np
import math
import matplotlib
import copy
import csv
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import warnings



"""
"""
def isFloat(val):

	try:
	
		num = float(val)
		return (True)
		
	except:
	
		return(False)

"""
input :
"""
def writeData(filename,data):

	try:

		outputfile = open(filename,"a")
	
	except:
	
		print("\nUnable to open file : " + filename + "\n")
		
		exit(1)
	
	outputfile.write("\n===============================================================================\n\n")	
	outputfile.write(data)
	outputfile.write("\n===============================================================================\n\n")
	outputfile.close()


"""
"""
def remove_null_lines(list_of_lines,filename):

		new_list_of_lines = []
		
		new_list_of_lines.append(list_of_lines[0].split(","))
		
		reject = []
		
		for i in range(1,len(list_of_lines)):
		
			templist = list_of_lines[i].split(",")
			
			if("Y" in templist[len(templist)-1].upper() or "N" in templist[len(templist)-1].upper()):
			
				new_list_of_lines.append(templist)
				
			else:
			
				reject.append(i)
		
		tempdata = ""
		
		for i in reject:
		
			tempdata += (str(i) + ", ")

		if(tempdata == ""):
			tempdata = "None"
		
		data = "\nData sets removed due to absence of outcome : \n" + tempdata; 
				
		writeData(filename,data)		
				
		return (new_list_of_lines)


"""
input :
"""
def cleandata(datamatrix,filename):

	updated = ""

	for line in datamatrix[1:]:
	
		if(not isFloat(line[8])):
		
			updated += (line[0] + ", ")
			line[8] = "4"
	
	if(updated == ""):
		updated = "None"
	
	data = "\nFollowing datasets had null values of Fasciculation score : " + "\n" + updated 
	
	for line in datamatrix:
	
		line[len(line)-1] = line[len(line)-1].split("\n")[0]
			
	writeData(filename,data)		
			
	return(datamatrix)


"""
input : 
function :
"""
def write_to_csv(filename,data):

	try:

		csvfile = open(filename,'w', newline='')
	
	except:
	
		print("Unable to open file : " + filename)
		
	obj = csv.writer(csvfile)
	
	obj.writerows(data)


"""
input :
"""
def readFile(filename):

    try:
    
        datafile = open(filename,"r")
        
        list_of_lines = datafile.readlines()
        
    except:
    
        print("\nUnable to open file : " + filename + "\n")
        exit(1)
        
    return(list_of_lines)





"""
input :
"""    
def createListforattribute(list_of_lines):

	templist = list_of_lines[0].split(",")
	
	ans = []
	
	for i in range(len(templist)):
	
		ans.append( [] )
		
	for line in list_of_lines:
	
		templist = line.split(",")
		
		for i in range(len(templist)):
			
			if(isFloat(templist[i])):
				ans[i].append(float(templist[i]))
	
			else:
				ans[i].append(templist[i])
				
	return (ans)


		
"""
input :
"""
def findMean(datalist,size):

	return (sum(datalist)/size)


	
"""
input :
"""
def findMedian(datalist,size):

	datalist.sort()
	
	if(size%2 == 0):

		med1 = size//2
		med2 = med1 - 1
	
		return ( (datalist[med1] + datalist[med2]) / 2)
	
	else:
	
		return(datalist[size//2])
		
		
		
"""
input : 		
"""
def findMode(datalist,size):

	tempdict = {}
	
	for value in datalist:
	
		if(value in tempdict):
		
			tempdict[value] += 1
			
		else:
		
			tempdict[value] = 1
			
	
	templist = list(tempdict.values())
	
	templist.sort(reverse = True)
	
	maxval = templist[0]
	
	modelist = []
	
	for key in list(tempdict.keys()):
	
		if(tempdict[key] == maxval):
			
			modelist.append(key)			
		
	return (modelist)
		
		

"""
input :
"""
def findVariance(datalist,size,mean):

	tempsum = 0
	
	for value in datalist:
	
		tempsum += ((value - mean)**2)
		
	return (tempsum/size)



	
"""
input :
"""
def findStandardDeviation(datalist,size,mean):

	return (math.sqrt(findVariance(datalist,size,mean)))
	
	
	
"""
input :
"""
def findQuartile(datalist,size):

	datalist.sort()

	qlist = []
	
	if(size % 2 == 0):
	
		med1 = size//2
		med2 = med1 -1
		qlist.append( findMedian( copy.deepcopy ( datalist[0:med2] ) , len(datalist[0:med2]) ) )
		qlist.append( findMedian( copy.deepcopy( datalist) , size ) )
		qlist.append( findMedian( copy.deepcopy ( datalist[med1 + 1 : ] ) , len(datalist[med1 + 1 : ]) ) )
		
	else:
	
		qlist.append( findMedian( copy.deepcopy ( datalist[0:size//2] ) , len(datalist[0:size//2]) ) )
		qlist.append( findMedian( copy.deepcopy( datalist ) , size ) )
		qlist.append( findMedian( copy.deepcopy ( datalist[(size//2)+1 : ] ) , len(datalist[(size//2)+1 : ]) ) )
		
	return(qlist)
	


"""
input :
"""	
def findFrequency(datalist):

	tempdict = {}
	
	for value in datalist:
		
		if(value.lower() in tempdict):
			
			tempdict[value.lower()] += 1
				
		else:
			
			tempdict[value.lower()] = 1


	return(tempdict)



"""
input : 
"""
def analyze(list_of_lines,outputfilename):

	os.system("mkdir graph")

	for line in list_of_lines:
	
		if(type(line[1]) == str):
			
			data = "Parameter Name : " + str(line[0]) + "\n"
			data += str(findFrequency(line[1:]))
			
		else:
	
			mean = findMean(line[1:],len(line)-1)
			maxval = max(line[1:])
			minval = min(line[1:])
			
			data = "Parameter Name : " + str(line[0]) + "\n"
			data += "Maximum Value : " + str(maxval) + "\n"
			data += "Minimum Value : " + str(minval) + "\n"
			data += "Mean : " + str(mean) + "\n"
			data += "Median : " + str(findMedian(copy.deepcopy(line[1:]),len(line)-1)) + "\n"
			data += "Mode : " + str(findMode(line[1:],len(line)-1)) + "\n"
			data += "Variance : " + str(findVariance(line[1:],len(line)-1,mean)) + "\n"
			data += "Standard : " + str(findStandardDeviation(line[1:],len(line)-1,mean)) + "\n"
			data += "Quartile [25%, 50%, 75%] : " + str(findQuartile(copy.deepcopy(line[1:]),len(line)-1)) + "\n"
	
			
			plt.hist(line[1:])
			plt.title("Histogram")
			plt.xlabel(str(line[0]))
			plt.ylabel("Frequency")
			plt.savefig("graph/" + str(line[0]) + ".png")
			plt.clf()
		
		
		writeData(outputfilename,data)
		
		
"""
input :
"""			
def preprocess(list_of_attributevalues,reportfile,inputgenfile):

	for line in list_of_attributevalues[6:]:
	
		if(not (type(line[1]) == str)):
		
			mean = findMean(line[1:],len(line[1:]))
			sd = findStandardDeviation(line[1:],len(line[1:]),mean)
			
			for i in range(1,len(line)):
			
				line[i] = (line[i]-mean)/sd
				
	abgline = list_of_attributevalues[10]
	
	for i in range(1,len(abgline)):
		if(abgline[i].lower() == "ma"):
			abgline[i] = 0
		else:
			abgline[i] = 1				
	
	
	popline = list_of_attributevalues[12]
	
	for i in range(1,len(popline)):
		if(popline[i].lower() == "sev"):
			popline[i] = 2
		elif(popline[i].lower() == "mod"):
			popline[i] = 1
		else:
			popline[i] = 0
			
	intuline = list_of_attributevalues[16]
	
	for i in range(1,len(intuline)):
		if("Y" in intuline[i].upper()):
			intuline[i] = 1
		else:
			intuline[i] = 0
	
	matrix = list_of_attributevalues[6:]
	
	transpose = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))] 
	
	write_to_csv(inputgenfile,transpose)
	
	abgdict = {"0" : "ma", "1" : "wnl"}
	popdict = {"2" : "sev", "1" : "mod", "0" : "mild"}
	intudict = {"1" : "YES", "0" : "NO"}
	
	
	data = "\nData preprocessing completed\n" + "Notations used : " + "\n" + str(abgdict) + "\n" + str(popdict) + "\n" + str(intudict)
	
	writeData(reportfile,data)	



def naivebayes(filename):


	data = pd.read_csv(filename)

	X = data.iloc[:,:-1]
	Y = data.iloc[:,-1]

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

	clf = GaussianNB()
	clf.fit(X_train, Y_train)

	Y_pred = clf.predict(X_test)

	writeData("Classification_Report.txt","Naive Bayes : \n" + classification_report(Y_test, Y_pred))

	cm = confusion_matrix(Y_test, Y_pred)
	fig = sns.heatmap(cm,annot = True)
	plt.savefig('graph/naivebayes_ConfusionMatrix.png')
	plt.clf()

				
def svm_KFold(filename):

	data = pd.read_csv(filename)

	X = data.iloc[:,:-1].values
	Y = data.iloc[:,-1].values

	kf = KFold(n_splits = 10, shuffle = True)
   
	scores = []
    
	for train_idx, test_idx in kf.split(X):
	
		X_train,X_test = X[train_idx],X[test_idx]
		Y_train,Y_test = Y[train_idx],Y[test_idx]
		
		clf = SVC(kernel = "rbf")
		clf.fit(X_train, Y_train)
		scores.append(clf.score(X_test, Y_test))
	
	return(float(np.mean(scores)))
	
	
def decisiontree(filename):

	data = pd.read_csv(filename)

	X = data.iloc[:,:-1]
	Y = data.iloc[:,-1]

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
	
	clf = DecisionTreeClassifier(random_state=45)
	
	clf.fit(X_train, Y_train)

	Y_pred = clf.predict(X_test)


	writeData("Classification_Report.txt","Decision Tree : \n" + classification_report(Y_test, Y_pred))

	cm = confusion_matrix(Y_test, Y_pred)
	fig = sns.heatmap(cm,annot = True)
	plt.savefig('graph/Decision Tree_ConfusionMatrix.png')
	plt.clf()
		
warnings.filterwarnings("ignore")



#MAIN


list_of_lines = readFile(input("\nEnter name of thesis file : "))
datamatrix = remove_null_lines(list_of_lines,"cleaning_preprocessing_report.txt")
datamatrix = cleandata(datamatrix,"cleaning_preprocessing_report.txt")
write_to_csv("cleandata.csv",datamatrix)



list_of_lines = readFile("cleandata.csv")
list_of_attributevalues = createListforattribute(list_of_lines)
analyze(list_of_attributevalues,"analyze.txt")
preprocess(list_of_attributevalues,"cleaning_preprocessing_report.txt","input.csv",)



naivebayes("input.csv")
svm("input.csv")
decisiontree("input.csv")
