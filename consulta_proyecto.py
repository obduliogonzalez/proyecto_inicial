import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sqlalchemy import create_engine

# Configura tu conexión usando SQLAlchemy
db_url = "postgresql+psycopg2://postgres:123456@localhost:5432/proyectoInicial"
engine = create_engine(db_url)

# Lista de consultas a ejecutar
queries = [
    ("teams_leagues", "SELECT * FROM public.teams_leagues LIMIT 10;"),
    ("player_quince", "SELECT * FROM public.player_quince LIMIT 10;"),
    ("player_dieciseis", "SELECT * FROM public.player_dieciseis LIMIT 10;"),
    ("player_diecisiete", "SELECT * FROM public.player_diecisiete LIMIT 10;"),
    ("player_dieciocho", "SELECT * FROM public.player_dieciocho LIMIT 10;"),
    ("player_diecinueve", "SELECT * FROM public.player_diecinueve LIMIT 10;"),
    ("player_veinte", "SELECT * FROM public.player_veinte LIMIT 10;")
]

# Ejecuta cada consulta y muestra los resultados
for name, query in queries:
    print(f"Resultados de la tabla: {name}")
    df = pd.read_sql(query, engine)
    print(df.head())
    print("\n" + "-"*50 + "\n")

# Análisis Exploratorio de Datos (EDA)
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], kde=True, color='skyblue')
plt.title('Distribución de Edad de los Jugadores')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# relacion entre el valor del mercado  y el rating
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='overall', y='value_eur', hue='potential', palette='cool')
plt.title('Valor de mercado vs Rating general')
plt.xlabel('Rating General')
plt.ylabel('Valor en Euros')
plt.grid(True)
plt.show()


### DATA CLEANING, limpieza de datos
# Verificar valores nulos
print(df.isnull().sum())

# Eliminar duplicados
df.drop_duplicates(inplace=True)



####DATA WRANGLING: tranformacion y manipulacion de datos

# Convertir la columna 'value_eur' a numérico
df['value_eur'] = pd.to_numeric(df['value_eur'], errors='coerce')
df = df.dropna(subset=['value_eur'])


# Rellenar valores nulos en columnas clave
df['club'] = df['club'].fillna('N/A')
df['nationality'] = df['nationality'].fillna('N/A')
df['value_eur'] = df['value_eur'].fillna(df['value_eur'].median())
print(df['value_eur'].head())
print(df['value_eur'].dtype)


# Convertir 'contract_valid_until' a numérico
df['contract_valid_until'] = pd.to_numeric(df['contract_valid_until'], errors='coerce')

# Rellenar valores nulos en 'contract_valid_until' con un valor por defecto, pondremos -> 2020
df['contract_valid_until'] = df['contract_valid_until'].fillna(2020)

# Crear una nueva columna para el valor por año de contrato
df = df[df['contract_valid_until'] > 2020]
df['value_per_year'] = df['value_eur'] / (df['contract_valid_until'] - 2020)
df['value_per_year'] = df['value_per_year'].fillna(0)

print(df[['value_eur', 'contract_valid_until', 'value_per_year']].head())
print(df['contract_valid_until'].dtype)

# Mostrar las primeras filas
print(df[['short_name', 'value_eur', 'contract_valid_until', 'value_per_year']].head())


##### DATA TRANFORMATION:  normalizacion y escalado

from sklearn.preprocessing import StandardScaler

# Seleccionar las columnas numéricas
features = ['age', 'height_cm', 'weight_kg', 'overall', 'potential', 'value_eur', 'wage_eur']
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# Crear un nuevo DataFrame con los datos escalados
df_scaled = pd.DataFrame(df_scaled, columns=features)
print(df_scaled.head())


#### GENERACION DEL INFORME FINAL
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Análisis del FIFA 2020', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)

pdf = PDF()
pdf.add_page()
pdf.chapter_title('Distribución de Edad')
pdf.chapter_body('El análisis mostró que la mayoría de los jugadores están en el rango de 20-30 años...')
pdf.output('informe_fifa.pdf')




# Entrenamiento de un modelo de regresion lineal para predecir value_eur
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Selecciona características y variable objetivo
features = ['age', 'height_cm', 'weight_kg', 'overall', 'potential', 'wage_eur']
X = df[features]
y = df['value_eur']

# Divide los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrena el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones y evaluación
y_pred = model.predict(X_test)
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
print(f'R^2 Score: {r2_score(y_test, y_pred)}')


# generacion de un informe pdf con fpdf

mean_value_per_year = df['value_per_year'].mean()
print(f'Promedio de valor por año: {mean_value_per_year:.2f} EUR')


from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font('Arial', 'B', 16)
pdf.cell(200, 10, txt="Análisis del FIFA 2020", ln=True, align='C')
pdf.set_font('Arial', '', 12)

# Revisar si el valor es finito antes de imprimir
if not pd.isnull(mean_value_per_year) and mean_value_per_year != float('inf'):
    pdf.cell(200, 10, txt=f"Promedio de valor por año: {mean_value_per_year:.2f} EUR", ln=True, align='L')
else:
    pdf.cell(200, 10, txt="Promedio de valor por año: No disponible", ln=True, align='L')

pdf.output("informe_fifa2.pdf")


# Asumimos que ya tienes un DataFrame llamado 'df' con la información de los jugadores

# Filtramos para obtener los 10 jugadores más valiosos
top_players = df.nlargest(10, 'value_eur')

# Configuramos el tamaño de la figura para graficar
fig, axes = plt.subplots(5, 2, figsize=(12, 15))
fig.suptitle('Análisis Individual de los Jugadores', fontsize=16)

# Convertimos la lista de ejes en un solo arreglo para iterar
axes = axes.flatten()

# Generamos una gráfica pequeña para cada jugador

# Convertir columnas a numéricas (si no lo son)
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['overall'] = pd.to_numeric(df['overall'], errors='coerce')
df['potential'] = pd.to_numeric(df['potential'], errors='coerce')
df['value_eur'] = pd.to_numeric(df['value_eur'], errors='coerce')

# Verificar si hay valores nulos después de la conversión
print(df[['age', 'overall', 'potential', 'value_eur']].isnull().sum())

# Eliminar filas con valores nulos en las columnas seleccionadas
df.dropna(subset=['age', 'overall', 'potential', 'value_eur'], inplace=True)



for i, player in enumerate(top_players.itertuples()):
    # Configurar cada gráfico
    ax = axes[i]
    ax.bar(['Edad', 'Overall', 'Potential', 'Altura (cm)', 'Peso (kg)'], 
           [player.age, player.overall, player.potential, player.height_cm, player.weight_kg],
           color='skyblue')
    ax.set_title(f'{player.short_name}')
    ax.set_ylim(0, 100)  # Ajustar según las métricas
    ax.grid(True)

# Eliminamos los ejes vacíos si hay menos jugadores que subplots

# Seleccionar los 10 jugadores más valiosos
top_players = df.nlargest(10, 'value_eur')

# Crear el gráfico `pairplot`
sns.pairplot(top_players, hue='short_name', vars=['age', 'overall', 'potential', 'value_eur'])
plt.suptitle('Comparación de Jugadores', y=1.02)
plt.show()


# Verifica los tipos de datos de las columnas
print(df[['age', 'overall', 'potential', 'value_eur']].dtypes)


sns.pairplot(top_players, hue='short_name', vars=['age', 'overall', 'potential', 'value_eur'], diag_kind='hist')
