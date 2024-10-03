from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score


def create_design_matrix(x, y, degree):
    N = len(x)
    num_terms = (degree + 1) * (degree + 2) // 2  # Number of polynomial terms up to the given degree
    X = np.ones((N, num_terms))  # Initialize the design matrix
    index = 1
    for i in range(1, degree+1):
        for j in range(i+1):
            X[:, index] = (x ** (i-j)) * (y ** j)
            index += 1
    return X

x = np.arange(0, 1, 0.01)
y = np.arange(0, 1, 0.01)
x, y = np.meshgrid(x, y)

x_flat = x.flatten()
y_flat = y.flatten()

x_train, x_test, y_train, y_test, = train_test_split(
    x_flat, y_flat,test_size=0.2, random_state=42) 

x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train_scaled = scaler_x.fit_transform(x_train)
x_test_scaled = scaler_x.transform(x_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

X = create_design_matrix(x_train_scaled.flatten(), y_train_scaled.flatten(), 1)
##X_test = create_design_matrix(x_test_scaled.flatten(), y_test_scaled.flatten(), 1)

clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
print(scores)

X = create_design_matrix(x_train_scaled.flatten(), y_train_scaled.flatten(), 1)
print(X)
print(X.size)
print(y.size)
##hva faen er det som foregår? ahahahah

""" 
bruker design matrix X (stor x) og y. (det er her jeg får dataen min fra)
deretter shufler jeg dataen 
deretter deler jeg opp i treningsdata og testdata
deretter ... ???

def cross_validation(x, y, function, folds = 10): #number of folds defaults to 10
    reshuffle data (føler det finnes en slik funksjon)
    x.split(folds) (??) kan bruke roll, men redd for å plagiere maiken, lol
    scores = []
    for k in range(folds) :
        x2 = x[....]
        y2 = y[....]
        scores.append(function(x2,y2))

    del opp datasettet
    gjør funksjonen på alle datasettene
    samle resultatet av funksjonen
    finn gjennomsnittet av resultatet??
    returnerer gjennomsnittet av resultatene

 """