import pandas as pd
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

plt.interactive(True)

#DATA
##TRAIN
rtrain = pd.read_csv(os.path.join('train', 'in.tsv'), sep='\t', header=None)
##DEV
rdev = pd.read_csv(os.path.join('dev-0', 'in.tsv'), sep='\t', header=None)
rdev_expected = pd.read_csv(os.path.join('dev-0', 'expected.tsv'), sep='\t', names=['y'], header=None)
##PROD
rprod = pd.read_csv(os.path.join('test-A', 'in.tsv'), sep='\t', header=None)

lr_full = LogisticRegression()
X = pd.DataFrame(rtrain.loc[:, 1:34])
lr_full.fit(X, rtrain[0])


#VARIABLES
##TRAIN
occupancy_v = str((sum(rtrain[0] == 'g') / len(rtrain)))
zero_rule_v = str(1 - sum(rtrain[0] == 'g') / len(rtrain))

tp_v = sum((lr_full.predict(X) == rtrain[0]) & (lr_full.predict(X) == 'g'))
tn_v = sum((lr_full.predict(X) == rtrain[0]) & (lr_full.predict(X) == 'b'))
fp_v = sum((lr_full.predict(X) != rtrain[0]) & (lr_full.predict(X) == 'g'))
fn_v = sum((lr_full.predict(X) != rtrain[0]) & (lr_full.predict(X) == 'b'))

accuracy_v = str((tp_v + tn_v) / len(rtrain))
sensivity_v = str(tp_v / (tp_v + fn_v))
specifity_v = str(tn_v / (fp_v + tn_v))

##DEV
dev_occupancy_v = str(sum(rdev_expected['y'] == 'g') / len(rdev_expected))
dev_zero_rule_v = str(1 - sum(rdev_expected['y'] == 'g') / len(rdev))

dev_tp_v = sum((lr_full.predict(rdev) == rdev_expected['y']) & (lr_full.predict(rdev) == 'g'))
dev_tn_v = sum((lr_full.predict(rdev) == rdev_expected['y']) & (lr_full.predict(rdev) == 'b'))
dev_fp_v = sum((lr_full.predict(rdev) != rdev_expected['y']) & (lr_full.predict(rdev) == 'g'))
dev_fn_v = sum((lr_full.predict(rdev) != rdev_expected['y']) & (lr_full.predict(rdev) == 'b'))

dev_accuracy_v = str((dev_tp_v + dev_tn_v) / len(rdev))
dev_sensivity_v = str(dev_tp_v / (dev_tp_v + dev_fn_v))
dev_specifity_v = str(dev_tn_v / (dev_fp_v + dev_tn_v))



#FILE
##TRAIN
output = open('notes.txt', 'w')
output.write('Training data\n')
output.write('=' * 100 + '\n')
output.write('Occupancy: ' + occupancy_v +'%\n')
output.write('Zero rule model accuracy on training set is: ' + zero_rule_v +'\n') 
output.write('=' * 100 + '\n')
output.write('lr model on full model only accuracy on training data: ' + accuracy_v +'\n')
output.write('lr model on full model only sensivity on training data: ' + sensivity_v +'\n')
output.write('lr model on full model only specifity on training data: ' + specifity_v +'\n')
output.write('=' * 100 + '\n')
output.write('Matrix\n')
output.write('True Positives: ' + str(tp_v) +'\n')
output.write('True Negatives: ' + str(tn_v) + '\n')
output.write('False Positives: ' + str(fp_v) + '\n')
output.write('False Negatives: ' + str(fn_v) + '\n')
output.write('=' * 100 + '\n\n\n\n')
##DEV
output.write('Development data\n')
output.write('=' * 100 + '\n')
output.write('Occupancy: ' + dev_occupancy_v +'%\n')
output.write('Zero rule model accuracy on development set is: ' + dev_zero_rule_v +'\n') 
output.write('=' * 100 + '\n')
output.write('lr model on full model only accuracy on development data: ' + dev_accuracy_v +'\n')
output.write('lr model on full model sensivity on development data: ' + dev_sensivity_v +'\n')
output.write('lr model on full model only specifity on development data: ' + dev_specifity_v +'\n')
output.write('=' * 100 + '\n')
output.write('Matrix\n')
output.write('True Positives: ' + str(dev_tp_v) +'\n')
output.write('True Negatives: ' + str(dev_tn_v) + '\n')
output.write('False Positives: ' + str(dev_fp_v) + '\n')
output.write('False Negatives: ' + str(dev_fn_v) + '\n')
output.write('=' * 100 + '\n\n')
output.close()



#OUT

##DEV
file = open(os.path.join('dev-0', 'out.tsv'), 'w')
for line in list(lr_full.predict(rdev)):
	file.write(str(line)+'\n')
file.close()

##PROD
file = open(os.path.join('test-A', 'out.tsv'), 'w')
for line in list(lr_full.predict(rprod)):
	file.write(str(line)+'\n')
file.close()


#PLOT
#replace_dict = {'g': 1, 'b': 0}
#rtrain[0] = rtrain[0].map(replace_dict)
#rdev_expected["y"] = rdev_expected["y"].map(replace_dict)

#sns.regplot(x=rdev, y=rdev_expected.y, logistic=True, y_jitter=.1)
#plt.show()

#sns.pairplot(rdev, hue=1, size=1.75, aspect=1.5, logistic=True)






