import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv('Ticketsales_s.csv')
#evt erste line entfernen
df['Relative show day'] = df['Relative show day'].astype(int)
df.set_index(pd.PeriodIndex(year=2023, quarter=1, day=df['Relative show day'], freq='D'), inplace=True)


# Create a PeriodIndex for the "Relative show day" column
df['Relative show day'] = df['Relative show day'].astype(int)
df.set_index(pd.PeriodIndex(year=2023, quarter=1, day=df['Relative show day'], freq='D'), inplace=True)


https://www.youtube.com/watch?v=S8tpSG6Q2H0