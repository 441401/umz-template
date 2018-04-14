import pandas as pd
from sklearn import linear_model

data = pd.read_csv('./train/in.tsv', sep = '\t', names= ['price', 'mileage', 'year', 'brand', 'engingeType', 'engineCapacity'], lineterminator = '\n')

#data.describe()
#data.engingeType.unique()
#data.brand.unique()
#wypelnia NaN wartosciami w ()

replace_dict_engingeType = {'benzyna' : 0, 'diesel' : 1, 'gaz' : 2}
data['engingeType']=data['engingeType'].map(replace_dict_engingeType)

replace_dict_brand = {'Volvo' : 0, 'Kia' : 1, 'Toyota' : 2, 'Skoda' : 3, 'Renault' : 4, 'Opel' : 5,
       'Mercedes-Benz' : 6, 'Ford' : 7, 'SEAT' : 8, 'Suzuki' : 9, 'Mazda' : 10, 'Volkswagen' : 11,
       'Audi' : 12, 'Hyundai' : 13, 'Citroen' : 14, 'Nissan' : 15, 'Honda' : 16, 'Chevrolet' : 17,
       'Jeep' : 18, 'Alfa' : 19, 'Fiat' : 20, 'Peugeot' : 21, 'Mitsubishi' : 22, 'Infiniti' : 23, 'BMW' : 24,
       'Daewoo' : 25, 'Jaguar' : 26, 'FSM' : 27, 'Dacia' : 28, 'Gaz' : 29, 'Subaru' : 30, 'MINI' : 31,
       'Saab' : 32, 'Lexus' : 33, 'Lincoln' : 34, 'Porsche' : 35, 'Dodge' : 36, 'Chrysler' : 37, 'Land' : 38,
       'Pilgrim' : 39, 'Cadillac' : 40, 'Ssangyong' : 41, 'Hummer' : 42, 'Lancia' : 43, 'Maserati' : 44,
       'Daihatsu' : 45, 'Rover' : 46, 'Grecav' : 47, 'Uaz' : 48, 'Smart' : 49, 'Microcar' : 50, 'Tata' : 51,
       'Pontiac' : 52, 'FSO' : 53, 'Aixam' : 54, 'Ligier' : 55, 'Vauxhall' : 56, 'Brilliance' : 57,
       'Proton' : 58, 'Austin' : 59, 'CHERY' : 60, 'Mercury' : 61, 'Ferrari' : 62, 'Chatenet' : 63,
       'MG' : 64, 'Shuanghuan' : 65, 'dla' : 66, 'Syrena' : 67, 'Warszawa' : 68, 'Iveco' : 69, 'Lada' : 70,
       'Scion' : 71, 'GMC' : 72, 'Triumph' : 73, 'Bentley' : 74, 'Trabant' : 75, 'Rolls-Royce' : 76,
       'Minauto' : 77, 'Aston' : 78, 'star' : 79, 'Barkas' : 80, 'Simca' : 81, 'Abarth' : 82, 'Polonez' : 83,
       'Mahindra&Mahindra' : 84, 'Oldsmobile' : 85, 'Isuzu' : 86, 'Bedford' : 87, 'Buick' : 88,
       'Tarpan' : 89}
data['brand']=data['brand'].map(replace_dict_brand)

data = data.fillna(0)

reg = linear_model.LinearRegression()

df = pd.DataFrame(data, columns=['mileage', 'year', 'brand', 'engingeType', 'engineCapacity'])

reg.fit(df, data['price'])

#dev
pd_dev = pd.read_csv('./dev-0/in.tsv', sep = '\t', names = ['mileage', 'year', 'brand', 'engingeType', 'engineCapacity'], lineterminator = '\n')
pd_dev['engingeType']=pd_dev['engingeType'].map(replace_dict_engingeType)
pd_dev['brand']=pd_dev['brand'].map(replace_dict_brand)
pd_dev = pd_dev.fillna(0)

df_dev = pd.DataFrame(pd_dev, columns=['mileage', 'year', 'brand', 'engingeType', 'engineCapacity'])

df_dev_out = reg.predict(df_dev)
df_dev_out_abs = pd.DataFrame(df_dev_out, columns=['price'])
df_dev_out_abs['price'] = df_dev_out_abs['price'].abs()

pd.Series(df_dev_out.flatten()).to_csv('./dev-0/out.tsv', sep='\t', header=False, index=False)



#prod
pd_prod = pd.read_csv('./test-A/in.tsv', sep = '\t', names = ['mileage', 'year', 'brand', 'engingeType', 'engineCapacity'], lineterminator = '\n')
pd_prod['engingeType']=pd_prod['engingeType'].map(replace_dict_engingeType)
pd_prod['brand']=pd_prod['brand'].map(replace_dict_brand)
pd_prod = pd_prod.fillna(0)

df_prod = pd.DataFrame(pd_prod, columns=['mileage', 'year', 'brand', 'engingeType', 'engineCapacity'])

df_prod_out = reg.predict(df_prod)
df_prod_out_abs = pd.DataFrame(df_prod_out, columns=['price'])
df_prod_out_abs['price'] = df_prod_out_abs['price'].abs()

pd.Series(df_dev_out.flatten()).to_csv('./test-A/out.tsv', sep='\t', header=False, index=False)
