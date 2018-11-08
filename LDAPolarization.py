import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Import your data set
ldaDf = pd.read_excel('lda_Data.xlsx')

identity_Dictionary = {
    'M0-set 1': 0,
    'CDβGeo-set 1': 1,
    'V14-set 1': 2,
    'TD-set 1': 3,
    'M0-set 2': 4,
    'CDβGeo-set 2': 5,
    'V14-set 2': 6,
    'TD-set 2': 7,
}

identity_list = ldaDf['Identity'].tolist()

tag_list = []
for item in identity_list:
    tag_list.append(identity_Dictionary[item])

loo = LeaveOneOut()
lda = LinearDiscriminantAnalysis(solver='eigen', n_components=2, tol=0.001)
test_fold_predictions = []

np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)

X = np.array(ldaDf[['C1', 'C2', 'C3', 'C4', 'C5']][0:44])
X_sub = np.array(ldaDf[['C1', 'C2', 'C3', 'C4', 'C5']])
y = np.array(tag_list[0:44])
y_sub = np.array(tag_list)
print(X)
# print(y.shape)
print(X.shape)
print(X_sub[45].shape)
X_data = np.concatenate((X, X_sub[45].reshape(1, 5), X_sub[56].reshape(1, 5),
                         X_sub[67].reshape(1, 5), X_sub[78].reshape(1, 5)))

y_data = np.concatenate((y.reshape(44, 1), y_sub[45].reshape(1, 1), y_sub[56].reshape(1, 1),
                         y_sub[67].reshape(1, 1), y_sub[78].reshape(1, 1)))

print(np.delete(X_data, 2, 0).shape)
lda.fit(np.delete(X_data, 1, 0), np.delete(y_data, 1, 0))
lda.fit(X, y)

# print(lda.predict([[1.19757962, 0.95700722, 1.65597815, 0.98985271, 1.10466738]]))
i = 0
for item in np.array(ldaDf[['C1', 'C2', 'C3', 'C4', 'C5']][44:88]):

    print(lda.predict([item.tolist()]))

if lda.predict([X_test.tolist()]) == y_test:
    predict_Score += 1

print(predict_Score)

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    lda.fit(X_train, y_train)
    test_fold_predictions.append(lda.predict(X_test))

print(test_fold_predictions)

print(np.sum(cross_val_score(estimator=lda, X=X, y=y, cv=loo)))
print(30/44)


print(lda.score(X, y))
print(cross_val_score(estimator=lda, X=X, y=y, cv=loo))
