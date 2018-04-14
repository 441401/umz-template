import pandas as pd
import numpy as np
from sklearn import linear_model

data = pd.read_csv('./train/in.tsv', sep = '\t', names= ['price', 'mileage', 'year', 'brand', 'engingeType', 'engineCapacity'], lineterminator = '\n')

#data.describe()
#data.engingeType.unique()
#data.brand.unique()
#wypelnia NaN wartosciami w ()

data['isBenzyna'] = np.where(data['engingeType'] == 'benzyna', 1, 0)
data['isDiesel'] = np.where(data['engingeType'] == 'diesel', 1, 0)
data['isGaz'] = np.where(data['engingeType'] == 'gaz', 1, 0)

data = data.fillna(0)

reg = linear_model.LinearRegression()

df = pd.DataFrame(data, columns=['mileage', 'year', 'engineCapacity', 'isBenzyna', 'isDiesel', 'isGaz'])

reg.fit(df, data['price'])

#dev
pd_dev = pd.read_csv('./dev-0/in.tsv', sep = '\t', names = ['mileage', 'year', 'brand', 'engingeType', 'engineCapacity'], lineterminator = '\n')
pd_dev['isBenzyna'] = np.where(pd_dev['engingeType'] == 'benzyna', 1, 0)
pd_dev['isDiesel'] = np.where(pd_dev['engingeType'] == 'diesel', 1, 0)
pd_dev['isGaz'] = np.where(pd_dev['engingeType'] == 'gaz', 1, 0)
pd_dev = pd_dev.fillna(0)

df_dev = pd.DataFrame(pd_dev, columns=['mileage', 'year', 'engineCapacity', 'isBenzyna', 'isDiesel', 'isGaz'])

df_dev_out = reg.predict(df_dev)
df_dev_out_abs = pd.DataFrame(df_dev_out, columns=['price'])
df_dev_out_abs['price'] = df_dev_out_abs['price'].abs()

pd.Series(df_dev_out.flatten()).to_csv('./dev-0/out.tsv', sep='\t', header=False, index=False)



#prod
pd_prod = pd.read_csv('./test-A/in.tsv', sep = '\t', names = ['mileage', 'year', 'brand', 'engingeType', 'engineCapacity'], lineterminator = '\n')
pd_prod['isBenzyna'] = np.where(pd_prod['engingeType'] == 'benzyna', 1, 0)
pd_prod['isDiesel'] = np.where(pd_prod['engingeType'] == 'diesel', 1, 0)
pd_prod['isGaz'] = np.where(pd_prod['engingeType'] == 'gaz', 1, 0)
pd_prod = pd_prod.fillna(0)

df_prod = pd.DataFrame(pd_prod, columns=['mileage', 'year', 'engineCapacity', 'isBenzyna', 'isDiesel', 'isGaz'])

df_prod_out = reg.predict(df_prod)
df_prod_out_abs = pd.DataFrame(df_prod_out, columns=['price'])
df_prod_out_abs['price'] = df_prod_out_abs['price'].abs()

pd.Series(df_dev_out.flatten()).to_csv('./test-A/out.tsv', sep='\t', header=False, index=False)
