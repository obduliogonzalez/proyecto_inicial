import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv(r'C:\\Users\\nestor.gonzalez\\Documents\\GitHub\\proyecto_inicial\\teams_and_leagues.csv')

print(df.head());
# Rellenar los campos vacíos en 'league_name' con 'N/A'
df['league_name'] = df['league_name'].fillna('N/A')

# Guardar el archivo limpio
df.to_csv('C:\\Users\\nestor.gonzalez\\Documents\\GitHub\\proyecto_inicial\\teams_and_leagues_dos.csv', index=False)
