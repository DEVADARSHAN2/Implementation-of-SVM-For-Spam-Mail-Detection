# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. start.
2. Import chardet.
3. Read the dataset.
4. Import SVC from sklearn.
5. Fit the data in the model and run the algorithm.
6. stop.

## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: DEVADARSHAN A S
RegisterNumber:  212222110007
```
```
import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy


```

## Output:

#### data.head()
![image](https://github.com/DEVADARSHAN2/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119432150/57fbae8e-6807-48d4-9034-edc419a80890)

#### data.tail()
![image](https://github.com/DEVADARSHAN2/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119432150/9f9adafc-12c1-4e3a-997c-fb58ec8e6ff1)


#### data.info()
![image](https://github.com/DEVADARSHAN2/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119432150/79a377f9-6035-4a89-beca-f592836d127f)

#### data.isnull().sum()
![image](https://github.com/DEVADARSHAN2/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119432150/8ac87fdb-09cf-4cb3-b3ae-b7266c62f4fe)

#### Y_prediction value
![image](https://github.com/DEVADARSHAN2/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119432150/ff357943-4fba-4e94-be68-db47463ef498)

#### Accuracy value
![image](https://github.com/DEVADARSHAN2/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119432150/6a849465-4464-49d5-94da-a139327243fa)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
