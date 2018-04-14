import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

plt.interactive(True)

data = pd.read_csv('./train/train.tsv', sep = '\t', names= ['price','isNew','rooms','floor','location','sqrMetres'], lineterminator = '\n')

#wypelnia NaN wartosciami w ()
data = data.fillna(0)

#data.describe()

reg = linear_model.LinearRegression()

df = pd.DataFrame(data, columns=['sqrMetres'])
reg.fit(df, data['price'])

a = reg.coef_[0]
b = reg.intercept_

#dev
pd_dev = pd.read_csv('./dev-0/in.tsv', sep = '\t', names = ['isNew','rooms','floor','location','sqrMetres'], lineterminator = '\n')

pd_dev = pd_dev.fillna(0)

df_dev = pd.DataFrame(pd_dev, columns=['sqrMetres'])
df_dev_out = pd.DataFrame(columns=['out'])

for index, row in df_dev.iterrows():
	df_dev_out.loc[index] = (a*row['sqrMetres'])+b

df_dev_out.to_csv('./dev-0/out.tsv', sep='\t', header=False, index=False)

#prod
pd_prod = pd.read_csv('./test-A/in.tsv', sep = '\t', names = ['isNew','rooms','floor','location','sqrMetres'], lineterminator = '\n')

pd_prod = pd_prod.fillna(0)

df_prod = pd.DataFrame(pd_prod, columns=['sqrMetres'])
df_prod_out = pd.DataFrame(columns=['out'])

for index, row in df_prod.iterrows():
	df_prod_out.loc[index] = (a*row['sqrMetres'])+b

df_prod_out.to_csv('./test-A/out.tsv', sep='\t', header=False, index=False)


#Wykres

data_cleared = data[data['price'] <= 1000000]
sorted_data_cleared = data_cleared.sort_values('sqrMetres')
sns.regplot(y = sort_data_cleared['price'], x = sort_data_cleared['sqrMetres'])
plt.show()

