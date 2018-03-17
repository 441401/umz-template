import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.interactive(True)

report = pd.read_csv('train.tsv', sep = '\t', names= ['price','isNew','rooms','floor','location','sqrMetres'])

from sklearn import linear_model

reg = linear_model.LinearRegression()

df = pd.DataFrame(report, columns=['sqrMetres','rooms','isNew'])
df_out = pd.DataFrame(columns=['out'])

reg.fit(df, report['price'])

for index, row in df.iterrows():
	#print(reg.predict(row['sqrMetres']))
	df_out.loc[index] = reg.predict(row['sqrMetres','rooms','isNew'])


#Wyswietlenie wykresu
#sns.regplot( y= report['price'], x=report['sqrMetres'])
#plt.show()

#Zapis do pliku
df_out.to_csv('out.tsv', sep='\t', header=False, index=False)



reg.coef_
#Out[11]: array([ 1898.49917374, 86260.98155537,  3297.93958317])


reg.intercept_
#Out[12]: 23408.307793239073

#y = (1898,49917374*x1) + (86260,98155537*x2) + (3297,93958317*x3) + 23408,307793239073

