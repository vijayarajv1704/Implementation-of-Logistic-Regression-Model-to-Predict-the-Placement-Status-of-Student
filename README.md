# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary Libraries.
2. Read the CSV file using pd.read_csv.
3. Print the data status using LabelEncoder().
4. Find the accuracy , confusion ,classification report .
5. End the Program.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Vijayaraj V
RegisterNumber: 212222230174  
*/
import pandas as pd
data=pd.read_csv("/content/Placement_Data(1).csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:


![image](https://github.com/vijayarajv1704/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121303741/f1fffbf6-4397-4103-960b-0eeedee8a077)

![image](https://github.com/vijayarajv1704/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121303741/03ec63e2-20df-4053-954d-a5952c8d1e17)

![image](https://github.com/vijayarajv1704/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121303741/9807e900-90c0-4bb7-ac9e-5657cf468458)

![image](https://github.com/vijayarajv1704/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121303741/a805a230-212c-461a-bd1d-1c4985806ebf)

![image](https://github.com/vijayarajv1704/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121303741/7d2a5930-098d-4380-a86a-651c91d415d6)

![image](https://github.com/vijayarajv1704/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121303741/fbf1033e-7990-4dbf-b0e5-6bdeb935f24d)

![image](https://github.com/vijayarajv1704/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121303741/f002b0e2-84e5-49ba-b7ef-e02b3f63b3be)

![image](https://github.com/vijayarajv1704/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121303741/2624ce7f-9d71-48a8-aba6-6f90bca548ee)

![image](https://github.com/vijayarajv1704/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121303741/28970978-acb2-4429-84c5-13aad01b500d)

![image](https://github.com/vijayarajv1704/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121303741/931a904e-52ac-4846-b428-b59459378007)

![image](https://github.com/vijayarajv1704/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121303741/e9310701-0041-487b-b0d9-02eead8bc31b)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
