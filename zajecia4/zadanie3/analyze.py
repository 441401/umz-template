import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

train = pd.read_csv('train.csv', sep=',')
test = pd.read_csv('test.csv', sep=',')

train_X = pd.DataFrame(train, columns=train.columns[:-1])
train_Y = train['Class']

test_X = pd.DataFrame(test, columns=test.columns[:-1])
test_Y = test['Class']

#I
#clf = KNeighborsClassifier(n_neighbors=0)
#clf = clf.fit(train_X, train_Y)

#II
clf = KNeighborsClassifier(n_neighbors=1)
clf = clf.fit(train_X, train_Y)

#III
clf = KNeighborsClassifier(n_neighbors=2)
clf = clf.fit(train_X, train_Y)

#IV
clf = KNeighborsClassifier(n_neighbors=3)
clf = clf.fit(train_X, train_Y)

#V
clf = KNeighborsClassifier(n_neighbors=5)
clf = clf.fit(train_X, train_Y)

#VI
clf = KNeighborsClassifier(n_neighbors=8)
clf = clf.fit(train_X, train_Y)

#VII
clf = KNeighborsClassifier(n_neighbors=13)
clf = clf.fit(train_X, train_Y)

#VIII
clf = KNeighborsClassifier(n_neighbors=21)
clf = clf.fit(train_X, train_Y)

#IX
clf = KNeighborsClassifier(n_neighbors=34)
clf = clf.fit(train_X, train_Y)

#X
clf = KNeighborsClassifier(n_neighbors=55)
clf = clf.fit(train_X, train_Y)

#XI
clf = KNeighborsClassifier(n_neighbors=89)
clf = clf.fit(train_X, train_Y)

#XI
clf = KNeighborsClassifier(n_neighbors=144)
clf = clf.fit(train_X, train_Y)

print('TRAIN SET')
print(confusion_matrix(clf.predict(train_X), train_Y))
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
print(confusion_matrix(clf.predict(test_X), test_Y))
print(sum(clf.predict(test_X) == test_Y) / len(test_X))
