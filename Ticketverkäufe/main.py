import pandas as pd
import matplotlib.pyplot as plt

# CSV-Daten einlesen
df = pd.read_csv("Ticketsales.csv")

# Show-Jahre in numerische Werte umwandeln (YEAR 1, YEAR 2, YEAR 3)
df['Event_Year'] = df['Event Name'].apply(lambda x: int(x.split()[-1]))

# Daten nach Event-Jahr sortieren
df = df.sort_values(by=['Event_Year', 'Relative show day'])

# Für jedes Show-Jahr einen Graphen plotten
for year in df['Event_Year'].unique():
    # Daten für das aktuelle Jahr filtern
    df_year = df[df['Event_Year'] == year]

    # Plot für das aktuelle Jahr
    plt.figure(figsize=(10, 6))
    plt.plot(df_year['Relative show day'], df_year['Sum Tickets sold'], marker='o', label=f'ART SHOW YEAR {year}')
    plt.xlabel('Relative show day')
    plt.ylabel('Sum Tickets sold')
    plt.title(f'Ticketverkäufe für ART SHOW YEAR {year}')
    plt.grid(True)
    plt.legend()
    plt.show()
