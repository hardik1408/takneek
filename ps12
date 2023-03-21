import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Reading the csv file
df=pd.read_csv('IPL 2022 Batters.csv')
print(df)

# Dropping rows having NULL values
newdf=df.dropna()
print(newdf)

# Creating graphs using matplotlib library
newdf.plot(x='Runs',y='4s')
plt.grid()
plt.title('RUNS VS 4s')
plt.xlabel('RUNS')
plt.ylabel('4S')
plt.show()

newdf.plot(x='HS',y='Avg',color = "hotpink")
plt.grid()
plt.title('Avg vs HS')
plt.show()

newdf.plot(x='Runs',y='50')
plt.bar(x='50',height='Runs')
plt.title('NO OF 50 VS RUNS')
plt.show()
