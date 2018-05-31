import pandas as pd
import graphviz
from sklearn import tree
from sklearn.metrics import confusion_matrix

train = pd.read_csv('train.csv', sep=',')
test = pd.read_csv('test.csv', sep=',')

train_X = pd.DataFrame(train, columns=train.columns[:-1])
train_Y = train['Class']

test_X = pd.DataFrame(test, columns=test.columns[:-1])
test_Y = test['Class']

#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(train_X, train_Y)


def show_plot():
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=train.columns[:-1],
                                    class_names=[str(x)
                                                 for x in [1, 2, 3, 4, 5, 6, 7]],
                                    filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    graph.view()


clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_X, train_Y)
show_plot()

#MAX DEPTH 4
clf = tree.DecisionTreeClassifier(max_depth=4)
clf = clf.fit(train_X, train_Y)
show_plot()

#I
clf = tree.DecisionTreeClassifier(presort=1, min_samples_leaf=3)
clf = clf.fit(train_X, train_Y)
show_plot()

#II
clf = tree.DecisionTreeClassifier(presort=1, min_samples_split=3)
clf = clf.fit(train_X, train_Y)
show_plot()

#III
clf = tree.DecisionTreeClassifier(presort=1, min_samples_leaf=10, min_samples_split=10)
clf = clf.fit(train_X, train_Y)
show_plot()

#IV
clf = tree.DecisionTreeClassifier(presort=1, criterion='entropy', splitter='random')
clf = clf.fit(train_X, train_Y)
show_plot()

#V
clf = tree.DecisionTreeClassifier(presort=1, max_features='log2')
clf = clf.fit(train_X, train_Y)
show_plot()

#VI
#clf = tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,             #min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=None, splitter='best')
#clf = clf.fit(train_X, train_Y)
#show_plot()


print('TRAIN SET')
print(confusion_matrix(clf.predict(train_X), train_Y))
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
print(confusion_matrix(clf.predict(test_X), test_Y))
print(sum(clf.predict(test_X) == test_Y) / len(test_X))
