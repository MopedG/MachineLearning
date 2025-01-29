import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Ticketsales.csv')
#evt erste line entfernen
df.index.freq = '53D'

plt.figure(figsize=(12, 8))
plt.plot(df.index, df['Sum Tickets sold'], marker='o', linestyle='-', color='b')
plt.title('Ticket Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sum Tickets Sold')
plt.grid(True)
plt.show()