import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

train = pd.read_csv('train.csv', sep=',')
test = pd.read_csv('test.csv', sep=',')

train_X = pd.DataFrame(train, columns=train.columns[:-1])
train_Y = train['Class']

test_X = pd.DataFrame(test, columns=test.columns[:-1])
test_Y = test['Class']


clf = RandomForestClassifier()
clf = clf.fit(train_X, train_Y)

#max_depth=4
clf = RandomForestClassifier(max_depth=4)
clf = clf.fit(train_X, train_Y)

#I
clf = RandomForestClassifier(max_features=3, oob_score=1, n_estimators=100)
clf = clf.fit(train_X, train_Y)

#II
clf = RandomForestClassifier(max_features=0.333, oob_score=1, n_estimators=100)
clf = clf.fit(train_X, train_Y)

#III
clf = RandomForestClassifier(max_features=None, oob_score=1, n_estimators=100)
clf = clf.fit(train_X, train_Y)

#IV
clf = RandomForestClassifier(max_features="sqrt", oob_score=1, n_estimators=100)
clf = clf.fit(train_X, train_Y)

#V
clf = RandomForestClassifier(max_features='log2', oob_score=1, n_estimators=100)
clf = clf.fit(train_X, train_Y)


print('TRAIN SET')
print(confusion_matrix(clf.predict(train_X), train_Y))
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
print(confusion_matrix(clf.predict(test_X), test_Y))
print(sum(clf.predict(test_X) == test_Y) / len(test_X))
