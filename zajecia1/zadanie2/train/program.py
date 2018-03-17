import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.interactive(True)

report = pd.read_csv('train.tsv', sep = '\t', names= ['price','isNew','rooms','floor','location','sqrMetres'])

from sklearn import linear_model

reg = linear_model.LinearRegression()

df = pd.DataFrame(report, columns=['sqrMetres'])
df_out = pd.DataFrame(columns=['out'])

reg.fit(df, report['price'])

for index, row in df.iterrows():
	#print(reg.predict(row['sqrMetres']))
	df_out.loc[index] = reg.predict(row['sqrMetres'])


#Wyswietlenie wykresu
#sns.regplot( y= report['price'], x=report['sqrMetres'])
#plt.show()

#Zapis do pliku
df_out.to_csv('out.tsv', sep='\t', header=False, index=False)



reg.coef_[0]
#Out[15]: 2997,289255577755

reg.intercept_
#Out[16]: 193805,52901125335

#y = 2997.289255577755*x + 193805.52901125335

