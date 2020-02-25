from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

K_S = [2, 3, 5, 10, 15, 20, 30]
# Quasi Identifier Amount values
# 2 - AGE/SEX
# 3 - AGE/SEX/OUTCOME
QI_S = [2, 3, 4, 5]

ALGOS = [
    LogisticRegression(solver='liblinear'),
    DecisionTreeClassifier(),
    KNeighborsClassifier(),
    GaussianNB(),
    SVC(gamma='auto')
]
