
# Data Description

해당 데이터는 KDD Cup 1999 Dataset을 가공한 네트워크 칩입 감지 시스템용 데이터이다. 각 Feature들은 어떤 protocol type을 사용하는 지 혹은 어떤 service를 사용하는 지 등의 내용를 담고 있다. Class는 현재 네트워크가 침입을 당한 상태인 지 만약 침입을 당하였다면 어떤 종류의 침입을 당했는 지를 나타낸다.

# Class Description

"Normal", "dos", "u2r", "r2l", "probe" 5개의 Class가 존재한다.

Normal은 정상을 의미하며, 나머지 4개는 네트워크상 침입 기법들의 이름들이다.

# Pandas로 데이터 불러오기
## 테스트데이터 첫번째 열에 인덱스가 그대로 적혀있어 지워주었다


```python
import pandas as pd
data = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')
data_all_X = data.iloc[:,:-1]
data_all_y = data.iloc[:,-1]
#테스트 데이터의 첫번제열 제외하고 받아오기
test_all_X = test.iloc[:,1:-1]
test_all_y = test.iloc[:,-1]
```

### protocol_type만 현재 label로 되어있음. 0,1,2,3으로 바꿀지 없앨지 각각을 attribute로 붙일지 결정
##### 일단 protocol_type은 지우자


```python
train_X = data_all_X.drop(["protocol_type"],axis=1)
test_X = test_all_X.drop(["protocol_type"],axis=1)
```

# 데이터 전처리 1

# y를 0, 1, 2, 3 으로


```python
from sklearn.preprocessing import LabelEncoder
```


```python
le = LabelEncoder()
y_encoded = le.fit_transform(data_all_y)
```


```python
y_encoded
```




    array([1, 1, 0, ..., 1, 0, 1], dtype=int64)



# Y를 행렬로


```python
data2 = data.apply(le.fit_transform)
```


```python
test2 = test.apply(le.fit_transform)
```


```python
train_y_encoded = data2.iloc[:,-1]
```


```python
test_y_encoded = test2.iloc[:,-1]
```


```python
from sklearn.preprocessing import OneHotEncoder
```


```python
ohe = OneHotEncoder()
```


```python
train_y_to_array = ohe.fit_transform(train_y_encoded.values.reshape(-1,1)).toarray()
```


```python
test_y_to_array = ohe.fit_transform(test_y_encoded.values.reshape(-1,1)).toarray()
```

# 트레이닝 시작

## 학습에 대한 평가는 10-fold Cross-Validation과, Test set의 점수 2가지로 시행하였다.


```python
from sklearn.model_selection import cross_val_score
```

## Decision Tree


```python
from sklearn.tree import DecisionTreeClassifier
```


```python
DT = DecisionTreeClassifier(max_depth=10)
```


```python
DT.fit(train_X,train_y_to_array)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')




```python
scores = cross_val_score(DT,train_X,train_y_to_array,cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
```

    Accuracy: 0.99 (+/- 0.00)
    


```python
DT.score(test_X,test_y_to_array)
```




    0.73450000000000004



## RandomForest


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
RFC = RandomForestClassifier()
```


```python
RFC.fit(train_X,train_y_to_array)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)




```python
scores = cross_val_score(RFC,train_X,train_y_to_array,cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
```

    Accuracy: 1.00 (+/- 0.00)
    


```python
RFC.score(test_X,test_y_to_array)
```




    0.73319999999999996



## KNN


```python
from sklearn.neighbors import KNeighborsClassifier
```


```python
knn = KNeighborsClassifier()
```


```python
knn.fit(train_X,train_y_to_array)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
               weights='uniform')




```python
scores = cross_val_score(knn,train_X,train_y_to_array,cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
```

    Accuracy: 0.99 (+/- 0.00)
    


```python
knn.score(test_X,test_y_to_array)
```




    0.73529999999999995



## Naive Bayes


```python
from sklearn.naive_bayes import GaussianNB
```


```python
gnb = GaussianNB()
```


```python
gnb.fit(train_X,data_all_y)
```




    GaussianNB(priors=None)




```python
scores = cross_val_score(gnb,train_X,data_all_y,cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
```

    Accuracy: 0.39 (+/- 0.01)
    


```python
gnb.score(test_X,test_all_y)
```




    0.28949999999999998



## Neural Network


```python
from sklearn.neural_network import MLPClassifier
```


```python
mlp = MLPClassifier(hidden_layer_sizes=(25,15,8),max_iter=10000,activation="logistic")
```


```python
mlp.fit(train_X,train_y_to_array)
```




    MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(25, 15, 8), learning_rate='constant',
           learning_rate_init=0.001, max_iter=10000, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=None,
           shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False)




```python
scores = cross_val_score(mlp,train_X,train_y_to_array,cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
```

    Accuracy: 0.98 (+/- 0.01)
    


```python
mlp.score(test_X,test_y_to_array)
```




    0.72219999999999995



## Linear Regression


```python
from sklearn.linear_model import LinearRegression
```


```python
LR = LinearRegression()
```


```python
LR.fit(train_X,train_y_to_array)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
scores = cross_val_score(LR,train_X,train_y_to_array,cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
```

    Accuracy: 0.81 (+/- 0.01)
    


```python
LR.score(test_X,test_y_to_array)
```




    0.2643998123958512



## Bagging


```python
from sklearn.ensemble import BaggingClassifier
```


```python
bagging = BaggingClassifier()
```


```python
bagging.fit(train_X,data_all_y)
```




    BaggingClassifier(base_estimator=None, bootstrap=True,
             bootstrap_features=False, max_features=1.0, max_samples=1.0,
             n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
             verbose=0, warm_start=False)




```python
scores = cross_val_score(bagging,train_X,data_all_y,cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
```

    Accuracy: 1.00 (+/- 0.00)
    


```python
bagging.score(test_X,test_all_y)
```




    0.75939999999999996



# 결과

Naive Bayes, Linear Regression을 제외한 DT, RF, KNN, MLP, Bagging 에서 75%정도의 성능을 볼 수 있었다.
좀 더 성능을 올리기 위해 무엇이 문제일까 생각해 보았다. 

우선 학습용 데이터의 분포를 보면, 


```python
data_all_y.value_counts()
```




    normal    67343
    dos       45927
    probe     11656
    r2l         995
    u2r          52
    Name: xAttack, dtype: int64



u2r, r2l에 비해 normal과 dos가 압도적으로 많음을 볼 수 있다.
실제로 분류된 결과의 confusion matrix를 보면, 


```python
from sklearn.metrics import confusion_matrix
```


```python
test_all_y.value_counts()
```




    normal    4329
    dos       3332
    r2l       1199
    probe     1053
    u2r         87
    Name: xAttack, dtype: int64




```python
confusion_matrix(test_all_y,bagging.predict(test_X))
```




    array([[2710,  583,   39,    0,    0],
           [  43, 4188,   98,    0,    0],
           [  86,  301,  666,    0,    0],
           [   2, 1088,   79,   29,    1],
           [   0,   78,    5,    3,    1]], dtype=int64)



위는 성능이 가장 좋았던 bagging classifier의 confusion matrix이다. 76%의 성능중 대부분은 normal과 dos가 차지하고있고, r2l과 u2r은 거의 분류를 못하고 있음을 볼 수 있다.

# 개선방향

## 1. 데이터의 균등화

r2l에 대한 학습이 전혀 이루어지지 않다고 판단하여, 먼저 학습 데이터에서 각 class의 비율을 맞춰주기로 하였다. 
u2r은 학습용 데이터에서도 이미 너무 적으므로, 995개 존재하는 r2l과 비슷한 수로 맞추기로 하였다.
아래 코드는 원래 data에서 각 class별로 임의의 개수 씩 뽑아 새로운 dataFrame을 만든다.



```python
import random 
balanced_data = pd.DataFrame()
stop=0

for i in range (0,100000):
    if(data.iloc[i,-1]=='normal'):
        balanced_data[i] = data.iloc[i,:]
        i = i + random.randrange(1,5)
        stop+=1
        if(stop==700):
            break
stop=0
for i in range (0,100000):
    if(data.iloc[i,-1]=='dos'):
        balanced_data[i] = data.iloc[i,:]
        i = i + random.randrange(1,5)
        stop+=1
        if(stop==700):
            break
stop=0
for i in range (0,100000):
    if(data.iloc[i,-1]=='probe'):
        balanced_data[i] = data.iloc[i,:]
        i = i + random.randrange(1,5)
        stop+=1
        if(stop==900):
            break
stop=0
for i in range (0,100000):
    if(data.iloc[i,-1]=='r2l'):
        balanced_data[i] = data.iloc[i,:]
        
        #data5[i+100000]= data.iloc[i,:]
        stop+=1
        if(stop==1000):
            break
stop=0
for i in range (0,100000):
    if(data.iloc[i,-1]=='u2r'):
        balanced_data[i] = data.iloc[i,:]
       
        stop+=1
        if(stop==90):
            break
                

```


```python
balanced_data = balanced_data.transpose()
```

총 3831개의 instance로 줄어든 것을 볼 수 있다.

## 2. 삭제했던 protocol_type을 추가


```python
Xlist = list()
Xlist=balanced_data['protocol_type']
icmplist = list()
udplist = list()
tcplist = list()
```


```python
for i in range (0,len(Xlist)):
    if(Xlist[Xlist.index[i]]=='icmp'):
        icmplist.append(1)
    else:
        icmplist.append(0)
        
for i in range (0,len(Xlist)):
    if(Xlist[Xlist.index[i]]=='udp'):
        udplist.append(1)
    else:
        udplist.append(0)
        
for i in range (0,len(Xlist)):
    if(Xlist[Xlist.index[i]]=='tcp'):
        tcplist.append(1)
    else:
        tcplist.append(0)
```


```python
balanced_data_with_protocol = balanced_data
balanced_data_with_protocol['icmp'] = icmplist
balanced_data_with_protocol['tcp'] = tcplist
balanced_data_with_protocol['udp'] = udplist

```


```python
balanced_data_with_protocol = balanced_data_with_protocol.drop(['protocol_type'],axis=1)
```

기존의 protocol_type속성을 삭제하는 대신, tcp, udp, icmp속성을 만들어 해당되는곳에 1이 체크되도록 하였다.
이는 test data에도 똑같이 적용된다.


```python
X_test_list = list()
X_test_list=test['protocol_type']
icmptestlist = list()
udptestlist = list()
tcptestlist = list()
```


```python
for i in range (0,len(X_test_list)):
    if(X_test_list[X_test_list.index[i]]=='icmp'):
        icmptestlist.append(1)
    else:
        icmptestlist.append(0)
        
for i in range (0,len(X_test_list)):
    if(X_test_list[X_test_list.index[i]]=='udp'):
        udptestlist.append(1)
    else:
        udptestlist.append(0)
        
for i in range (0,len(X_test_list)):
    if(X_test_list[X_test_list.index[i]]=='tcp'):
        tcptestlist.append(1)
    else:
        tcptestlist.append(0)
```


```python
test_data_with_protocol = test
test_data_with_protocol['icmp'] = icmptestlist
test_data_with_protocol['tcp'] = udptestlist
test_data_with_protocol['udp'] = tcptestlist

```


```python
test_data_with_protocol = test_data_with_protocol.drop(['protocol_type'],axis=1)
```

지금 train_set을 보면, class가 섞여있지 않고 순서대로 나열되어 있으므로, shuffle해 주었다.


```python
from sklearn.utils import shuffle
balanced_data_with_protocol = shuffle(balanced_data_with_protocol)
test_data_with_protocol = shuffle(test_data_with_protocol)
```

# 변경된 데이터를 통한 학습


```python
newTrain_X = balanced_data_with_protocol.drop(["xAttack"],axis=1)
newTrain_y = balanced_data_with_protocol["xAttack"]
balanced_data_with_protocol_trans = balanced_data_with_protocol.apply(le.fit_transform)
newTrain_y_to_array = ohe.fit_transform(balanced_data_with_protocol_trans.iloc[:,-4].values.reshape(-1,1)).toarray()
```


```python
newTest_X = test_data_with_protocol.iloc[:,1:].drop(["xAttack"],axis=1)
newTest_y = test_data_with_protocol["xAttack"]
test_data_with_protocol_trans = test_data_with_protocol.apply(le.fit_transform)
newTest_y_to_array = ohe.fit_transform(test_data_with_protocol_trans.iloc[:,-4].values.reshape(-1,1)).toarray()
```


```python
DT = DecisionTreeClassifier()
DTfit = DT.fit(newTrain_X,newTrain_y_to_array)
DT.score(newTest_X,newTest_y_to_array)
```




    0.68100000000000005




```python
knn = KNeighborsClassifier()
knn.fit(newTrain_X,newTrain_y_to_array)
knn.score(newTest_X,newTest_y_to_array)
```




    0.75419999999999998




```python
RFC = RandomForestClassifier()
RFC.fit(newTrain_X,newTrain_y_to_array)
RFC.score(newTest_X,newTest_y_to_array)
```




    0.70069999999999999




```python
bagging = BaggingClassifier()
bagging.fit(newTrain_X,newTrain_y)
bagging.score(newTest_X,newTest_y)  
```




    0.72370000000000001




```python
mlp=MLPClassifier(hidden_layer_sizes=(30,15,8),max_iter=10000,activation="logistic")
mlp.fit(newTrain_X,newTrain_y)
mlp.score(newTest_X,newTest_y)  
```




    0.74470000000000003




```python
confusion_matrix(newTest_y,bagging.predict(newTest_X))
```




    array([[2541,  483,  298,   10,    0],
           [ 466, 3705,  137,   17,    4],
           [  76,  129,  848,    0,    0],
           [   1,  958,  104,  133,    3],
           [   1,   47,   18,   11,   10]], dtype=int64)



학습 데이터의 비율을 조절하여 전반적인 성능의 향상과 r2l이 조금 더 잘 분류되고 있음을 보았다.

# 추가 변경

지금까지는 각 모델들을 scikit learn에서 디폴트로 제공하는 parameter 그대로 사용하였다.
성능 향상을 위해, 각 모델들의 parametmer를 조절해 보았다.

## Decision Tree

Decision Tree모델에서는 split의 기준으로 gini를 쓸지 entropy를 쓸지를 정할 수 있고,
max_depth, min_sample_split, min_samples_leaf등을 조절할 수 있다.


```python
DT = DecisionTreeClassifier(criterion="entropy",max_depth=None,min_samples_split=15,min_samples_leaf=2)
DTfit = DT.fit(newTrain_X,newTrain_y_to_array)
DT.score(newTest_X,newTest_y_to_array)
```




    0.77110000000000001




```python
DT = DecisionTreeClassifier(criterion="entropy",max_depth=None,min_samples_split=30,min_samples_leaf=7)
DTfit = DT.fit(newTrain_X,newTrain_y_to_array)
DT.score(newTest_X,newTest_y_to_array)
```




    0.78769999999999996




```python
DT = DecisionTreeClassifier(criterion="entropy",max_depth=10,min_samples_split=3,min_samples_leaf=2)
DTfit = DT.fit(newTrain_X,newTrain_y_to_array)
DT.score(newTest_X,newTest_y_to_array)
```




    0.77890000000000004



gini에서 entropy로 바꾸고, parameter에 조금씩 변화를 주는것으로 획기적인 변화를 보여주고 있다.

# Random Forest


```python
RFC = RandomForestClassifier(criterion = "entropy",n_estimators=20,max_depth=None,min_samples_split=30,min_samples_leaf=7,n_jobs=-1)
RFC.fit(newTrain_X,newTrain_y_to_array)
RFC.score(newTest_X,newTest_y_to_array)
```




    0.73629999999999995




```python
RFC = RandomForestClassifier(criterion = "entropy",n_estimators=20,max_depth=10,min_samples_split=3,min_samples_leaf=2,n_jobs=-1)
RFC.fit(newTrain_X,newTrain_y_to_array)
RFC.score(newTest_X,newTest_y_to_array)
```




    0.76770000000000005



Random Forest에서는 큰 변화를 볼 수 없었다.

# KNN


```python
knn = KNeighborsClassifier(n_neighbors=3,algorithm="kd_tree",weights = "distance",leaf_size = 30)
knn.fit(newTrain_X,newTrain_y_to_array)
knn.score(newTest_X,newTest_y_to_array)
```




    0.75680000000000003




```python
knn = KNeighborsClassifier(n_neighbors=1,algorithm="kd_tree",weights = "uniform",leaf_size = 30)
knn.fit(newTrain_X,newTrain_y_to_array)
knn.score(newTest_X,newTest_y_to_array)
```




    0.75660000000000005



KNN도 큰 변화는 없었다.

## MLP


```python
mlp=MLPClassifier(hidden_layer_sizes=(20,20,20),max_iter=20000,activation="logistic")
mlp.fit(newTrain_X,newTrain_y_to_array)
mlp.score(newTest_X,newTest_y_to_array)
```




    0.70309999999999995



## Bagging

Bagging 에서는 sample과 feature의 비율과 estimator를 지정해 줄 수 있다.


```python
bagging = BaggingClassifier(base_estimator=DT,max_samples=0.9,max_features=0.9)
bagging.fit(newTrain_X,newTrain_y)
bagging.score(newTest_X,newTest_y) 
```




    0.7833




```python
bagging = BaggingClassifier(base_estimator=knn,max_samples=0.9,max_features=0.9,n_jobs=-1)
bagging.fit(newTrain_X,newTrain_y)
bagging.score(newTest_X,newTest_y) 
```




    0.76329999999999998



DT를 통한 bagging이 가장 높은 성능을 보였다.

# Feature selection

마지막으로 성능을 높이기 위해 Feature selection을 수행하였다.


```python
from sklearn.feature_selection import VarianceThreshold
```


```python
from sklearn.feature_selection import RFE
```


```python
VT = VarianceThreshold(threshold=(.95 * (1 - .95)))
```


```python
rfe = RFE(estimator=DT,n_features_to_select=39)
```

RFE와 VarianceThreshold를 선택하여 수행할 수 있게 하였다.


```python
which_feature = False
if(which_feature==False):
    rfeinfo = rfe.fit(newTrain_X,newTrain_y)
else:
    rfeinfo=VT.fit(newTrain_X)
```

삭제되는 feature를 list로 받아, train과 test에서 모두 지워질 수 있도록 하였다.


```python
rfeindex = rfeinfo.get_support()
X_train_rfe = list()
for i in range(0,40):
    if rfeindex[i] == False :
        X_train_rfe.append(newTrain_X.columns[i])
    

```


```python
X_train_rfe_done = newTrain_X

for i in range (0,len(X_train_rfe)):
    X_train_rfe_done = X_train_rfe_done.drop(labels=[X_train_rfe[i]],axis=1)
    
```


```python
test_X_rfe_done = newTest_X
for i in range (0,len(X_train_rfe)):
    test_X_rfe_done = test_X_rfe_done.drop(labels=[X_train_rfe[i]],axis=1)
```

선택한 Feature만을 담고 있는 데이터로 학습을 진행하였다.


```python
DT2 = DecisionTreeClassifier(criterion="entropy",max_depth=10,min_samples_split=3,min_samples_leaf=2)
DTfit = DT2.fit(X_train_rfe_done,newTrain_y_to_array)
DT2.score(test_X_rfe_done,newTest_y_to_array)
```




    0.79320000000000002




```python
RFC2 = RandomForestClassifier(criterion = "entropy",n_estimators=20,max_depth=None,min_samples_split=30,min_samples_leaf=7,n_jobs=-1)
RFC2.fit(X_train_rfe_done,newTrain_y_to_array)
RFC2.score(test_X_rfe_done,newTest_y_to_array)
```




    0.6976




```python
bagging = BaggingClassifier(base_estimator=DT,max_samples=0.9,max_features=0.9)
bagging.fit(X_train_rfe_done,newTrain_y)
bagging.score(test_X_rfe_done,newTest_y) 
```




    0.79620000000000002



대체적으로 성능이 상승됨을 볼 수 있었다.

# 최종 모델

최종 모델은, 가장 성능이 높게 나왔던 DT를 vote시켜보도록 하였다.


```python
from sklearn.ensemble import VotingClassifier
```


```python
DT = DecisionTreeClassifier(criterion="entropy",max_depth=10,min_samples_split=3,min_samples_leaf=2)
```


```python
minsplit =3
minleaf = 2
depth = 10
criter = "entropy"
DT1 = DecisionTreeClassifier(criterion = criter,max_depth=depth,min_samples_split=minsplit,min_samples_leaf=minleaf)
DT2 = DecisionTreeClassifier(criterion = criter,max_depth=depth,min_samples_split=minsplit+1,min_samples_leaf=minleaf)
DT3 = DecisionTreeClassifier(criterion = criter,max_depth=depth,min_samples_split=minsplit,min_samples_leaf=minleaf+1)
DT4 = DecisionTreeClassifier(criterion = criter,max_depth=depth,min_samples_split=minsplit+1,min_samples_leaf=minleaf+1)
DT5 = DecisionTreeClassifier(criterion = criter,max_depth=depth,min_samples_split=minsplit,min_samples_leaf=minleaf)
DT6 = DecisionTreeClassifier(criterion = criter,max_depth=depth,min_samples_split=minsplit,min_samples_leaf=minleaf)
DT7 = DecisionTreeClassifier(criterion = criter,max_depth=depth,min_samples_split=minsplit,min_samples_leaf=minleaf)

```


```python
vote = VotingClassifier(estimators=[('DT1',DT1),('DT2',DT2),('DT3',DT3),('DT4',DT4),('DT5',DT5),('DT6',DT6),('DT7',DT7)],voting="hard")
```


```python
vote.fit(X_train_rfe_done,newTrain_y)
```




    VotingClassifier(estimators=[('DT1', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=10,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=2, min_samples_split=3,
                min_weight_fraction_le...      min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best'))],
             flatten_transform=None, n_jobs=1, voting='hard', weights=None)




```python
vote.score(test_X_rfe_done,newTest_y)
```




    0.81189999999999996



81%정도의 성능을 보았다.


```python
confusion_matrix(newTest_y,vote.predict(test_X_rfe_done))
```




    array([[2889,  373,   70,    0,    0],
           [  43, 4112,  147,   23,    4],
           [ 100,  188,  765,    0,    0],
           [   5,  671,  179,  342,    2],
           [   1,   52,   14,    9,   11]], dtype=int64)



confusion matrix를 보면 기존보다 향상된 성능을 보여주고 있음을 알 수 있다.
특히 r2l이 좀 더 제자리를 찾아간 것을 볼 수 있다.
