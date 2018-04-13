import pandas as pd
from sklearn import linear_model

data = pd.read_csv('./train/train.tsv', sep = '\t', names= ['price','isNew','rooms','floor','location','sqrMetres'], lineterminator = '\n')

#wypelnia NaN wartosciami w ()
data = data.fillna(0)
data['isNew'] = data['isNew'].astype(int)

#data.describe()

reg = linear_model.LinearRegression()

df = pd.DataFrame(data, columns=['sqrMetres', 'isNew', 'rooms'])

reg.fit(df, data['price'])

a1 = reg.coef_[0]
a2 = reg.coef_[1]
a3 = reg.coef_[2]
b = reg.intercept_

#dev
pd_dev = pd.read_csv('./dev-0/in.tsv', sep = '\t', names = ['isNew','rooms','floor','location','sqrMetres'], lineterminator = '\n')
pd_dev = pd_dev.fillna(0)
pd_dev['isNew'] = pd_dev['isNew'].astype(int)
df_dev = pd.DataFrame(pd_dev, columns=['sqrMetres', 'isNew', 'rooms'])

df_dev_out = reg.predict(df_dev)

pd.Series(df_dev_out.flatten()).to_csv('./dev-0/out.tsv', sep='\t', header=False, index=False)

#prod
pd_prod = pd.read_csv('./test-A/in.tsv', sep = '\t', names = ['isNew','rooms','floor','location','sqrMetres'], lineterminator = '\n')
pd_prod = pd_prod.fillna(0)
pd_prod['isNew'] = pd_prod['isNew'].astype(int)
df_prod = pd.DataFrame(pd_prod, columns=['sqrMetres', 'isNew', 'rooms'])

df_prod_out = reg.predict(df_prod)

pd.Series(df_dev_out.flatten()).to_csv('./test-A/out.tsv', sep='\t', header=False, index=False)
