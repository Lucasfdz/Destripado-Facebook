


import streamlit as st 
import io
import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sn
import json
from IPython.display import Image
import cufflinks as cf
from IPython.display import display,HTML
import seaborn as sns
import plotly.express as px
import time
from datetime import time
from spellchecker import SpellChecker
spell = SpellChecker()
import re
import nltk
from pprint import pprint
from urllib.parse import urlparse
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go




# Configuración inicial de la aplicación
st.set_page_config(page_title="Destripando Facebook", layout="wide")

# ## 0.2  Funciones y otro Código a importar


def analisis_DF(x):
    a = x.shape
    b = x.size
    c = x.ndim
    d = x.info()
    return f"{a} filas por culumna, {b} total datos, {c} num. dimensiones. \n Estructura: \n {d}" 

def analisis_DF_2(x):
    a = print("INDICES: \n",x.index,"\n")
    b = print("COLUMNAS: \n",x.columns,"\n")
    c = print("VALORES: \n",x.values,"\n)")


def quitar_lista_columna (DF, col): ### Importante: introducir con como str "col"
    df_sin_lista = DF[col].apply(lambda x: x[0] if len(x) > 0 else x)
    df_sin_lista = df_sin_lista.to_frame()
    return df_sin_lista

def quitar_diccionario_primario (DF, col): ### Importante: introducir con como str "col"
    resultado = DF[col].apply(lambda x : x.get("reaction"))
    resultado = resultado.to_frame()
    return resultado

def descomprimir_diccionario (DF, col): ### Importante: introducir con como str "col"
    resultado = DF[col].apply(pd.Series)
    return resultado

def atributos_fechas(x):
    x['año'] = x.index.year
    x['mes'] = x.index.month_name()
    x['dia_mes'] = x.index.day
    x['dia_semana'] = x.index.day_name()
    x['hora'] = x.index.hour
    return(x)

def fechas_a_numeros(df,colsemana,colmes):
    df[colsemana].replace({"Monday": 1,
                                 "Tuesday": 2,
                                 "Wednesday": 3,
                                 "Thursday": 4,
                                 "Friday": 5,
                                 "Saturday": 6,
                                 "Sunday": 7}, inplace= True) 
    df[colmes].replace({"January": 1,
                              "February": 2,
                              "March": 3,
                              "April": 4,
                              "May": 5,
                              "June": 6,
                              "July": 7,
                             "August": 8,
                             "September": 9,
                             "October": 10,
                              "November": 11,
                               "December": 12}, inplace = True)  
    return df


# Codigo pagina 1

# 1. Carga de datos eficiente.
# 2. Definición de funciones de extracción y limpieza de datos.
# 3. Obtención de nuevas variables.
# 4. Visualización efectiva de la información.
# 5. Describir las principales tendencias de likes y reactions del usuario.
# 6. Exposición clara y concisa de principales insights y hallazgos.

# ## 0.4. Observaciones

# 1. La información de las bases de datos utilizadas cubren solo hasta 2016, lo que posiblemenete sea debido a la entrada del RGPD (REGLAMENTO (UE) 2016/679 DEL PARLAMENTO EUROPEO Y DEL CONSEJO de 27 de abril de 2016 relativo a la protección de las personas físicas en lo que respecta al tratamiento de datos personales y a la libre circulación de estos datos y por el que se deroga la Directiva 95/46/CE (Reglamento general de protección de datos).
# 
# 2. Los datos empleados en este ejemplo corresponden con los de un estudiante de edad media.
# 

# # 1. CARGA DE DATOS

# ## 1.1. Likes y Reacciones




dfl0 = pd.read_json("likes_and_reactions_1.json")
dfl1 = pd.read_json("likes_and_reactions_2.json")
dfl2 = pd.read_json("likes_and_reactions_2.json")


# # 2. ANÁLISIS EXPLORATORIO


# %time
df_likes = pd.concat([dfl0, dfl1, dfl2], axis=0)



# Dado que solo conteiene información redundante, vamos a eliminar la columna Title


df_likes = df_likes.drop(columns = "title")



analisis_DF(df_likes)



analisis_DF_2(df_likes)


# ## 2.1 Detección de nulos y duplicados


nulos = df_likes.isnull().sum()




# ## 2.3 Conclusiones
# Resulta necesario un trabajo previo de extracción de datos para la creación de nuevas variables.

# # 3 . CREACIÓN DE NUEVAS VARIABLES

# ## 3.1  Likes y Reacciones 

# De la columna de df_likes vamos a "desencapsular" la información  con una función lambda. También comprobaremos que la longitud del DF resultante coincide con el original.
# 
# Para ello el primer paso es eliminar la primera columna. Aplicaremos una función en vez de acudir a un bucle. (Esta función es cortesía de mi profe del bootcamp :))


def quitar_lista_columna (DF, col): ### Importante: introducir con como str "col"
    df_sin_lista = DF[col].apply(lambda x: x[0] if len(x) > 0 else x)
    df_sin_lista = df_sin_lista.to_frame()
    return df_sin_lista




df_likes1 = quitar_lista_columna(df_likes,"data")






# El siguiente paso es pasar los diccionarios a columnas, para ello eliminamos el "diccionario externo" de cada columna. 


def quitar_diccionario_primario (DF, col): ### Importante: introducir con como str "col"
    resultado = DF[col].apply(lambda x : x.get("reaction"))
    resultado = resultado.to_frame()
    return resultado




df_likes2 = quitar_diccionario_primario(df_likes1,"data")



def descomprimir_diccionario (DF, col): ### Importante: introducir con como str "col"
    resultado = DF[col].apply(pd.Series)
    return resultado



df_likes3 = descomprimir_diccionario(df_likes2, "data")






df_likes3 = df_likes3.reset_index()
df_likes3.rename({"Index":"level_0"}, axis=1)


# ### Observación
# Conviene señalar que se conserva el INDEX original como precaución ante los posteriores tratamientos de datos.

# ### 3.1.1.  Obtención de variables temporales



df_likes["timestamp"] = pd.to_datetime(df_likes["timestamp"])


# También creamos un nuevo DF que almacene la fecha original y un contador.



df_likesT = df_likes



df_likesT["num"] = 1


# Seguimos generando las fechas. 



df_likes_t = df_likes.reset_index()
df_likes_t = df_likes_t.set_index("timestamp")




df_likes_tiempo = atributos_fechas(df_likes_t)


# De cara a optimizar la visualización y el análisis de datos, se procede a renombrar los días de la semana y meses con la siguiente función basada en replace:


df_likes_tiempo_def = fechas_a_numeros(df_likes_tiempo,"dia_semana","mes")


# ## 3.3 Creación DFs consolidados

# Cabe reseñar que los DF de distintos grupos de información no pueden unirse, dado que tienen índices distintos. 


likes_DEF = pd.merge(left =df_likes3, right = df_likes_tiempo_def, on = "index", how = "outer" )


# Ahora procedemos a eliminar las columnas inservibles con DROP.


likes_ = likes_DEF.drop(columns= ["index","actor","data"])
likes_T = likes_.groupby(["año","mes","hora"])["num"].count()
likes_T = likes_T.reset_index()

# Código pagina 2
# Cargar el archivo JSON en una variable Python
filename = 'your_posts_1.json'
with open(filename, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Convertir los datos JSON en un DataFrame de pandas
df = pd.json_normalize(data)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
first_attachment = df.at[30, 'attachments']

# Reemplaza los NaN por listas vacías en la columna 'attachments'
df['attachments'] = df['attachments'].apply(lambda x: x if isinstance(x, list) else [])

# Ahora puedes crear df_adjuntos sin el error de tipo
df_adjuntos = pd.DataFrame(df['attachments'].values.tolist(), columns=['attachments'])

for index, row in df_adjuntos.iterrows():
    attachment = row['attachments']
    if pd.isna(attachment) or not attachment or len(str(attachment)) <= 4:
        df_adjuntos.at[index, 'tipo'] = 'None'
        df_adjuntos.at[index, 'titulo'] = 'None'
        df_adjuntos.at[index, 'enlace'] = 'None'
    else:
        # Extraemos el primer valor de la lista que es un diccionario
        x0 = list(attachment.values())[0]
        # Extraemos el primer valor de x0 si x0 es una lista no vacía, de lo contrario None
        x1 = x0[0] if isinstance(x0, list) and x0 else None
        # Procedemos solo si x1 es un diccionario
        if isinstance(x1, dict) and x1:
            tipo = list(x1.keys())[0]
            # Extraemos el valor correspondiente a la clave tipo si está disponible
            x2 = x1.get(tipo)
            # Procedemos solo si x2 es un diccionario
            if isinstance(x2, dict) and x2:
                titulo = x2.get('title') or x2.get('name') or 'None'  # Verifica múltiples claves para el título
                enlace = x2.get('url', 'None')
            else:
                titulo = 'None'
                enlace = 'None'
        else:
            tipo = 'None'
            titulo = 'None'
            enlace = 'None'
        
        df_adjuntos.at[index, 'tipo'] = tipo
        df_adjuntos.at[index, 'titulo'] = titulo
        df_adjuntos.at[index, 'enlace'] = enlace

df_merge = pd.merge(df, df_adjuntos, left_index=True, right_index=True, how='left')
df_def = df_merge[['timestamp', 'data', 'tipo', 'titulo', 'enlace']]

 # Crear un nuevo DataFrame con el índice y la columna 'data' de df_def
new_df = df_def[['data']].copy()

# Añadir una nueva columna 'count' que tiene el recuento del número de elementos en las listas de la columna 'data'
new_df['count'] = new_df['data'].apply(lambda x: len(x) if isinstance(x, list) else 0)

# Conservar el index original
new_df.reset_index(inplace=True) 

# Inicializa la columna 'comentario' con valores por defecto None o puedes usar np.nan
new_df['comentario'] = None

# Itera a través de las filas con iterrows()
for index, row in new_df.iterrows():
    # Comprueba si la cuenta es 2
    if row['count'] >= 1:
        # Extrae el primer valor de la lista, que es un diccionario
        data_dict = row['data'][0]
        # Asume que el diccionario tiene solo un par clave-valor y obten el valor
        comentario = list(data_dict.values())[0]
        # Asigna el valor a la nueva columna 'comentario'
        new_df.at[index, 'comentario'] = comentario

df_publicaciones = pd.merge(df_def, new_df, left_index=True, right_index=True, how='left')

# Eliminar columnas no deseadas
df_publicaciones = df_publicaciones.drop(['data_x', 'data_y', 'index', 'count'], axis=1)

# Renombrar la columna 'timestamp' a 'fecha'
df_publicaciones = df_publicaciones.rename(columns={'timestamp': 'fecha'})

# Reordenar las columnas
df_publicaciones = df_publicaciones[['fecha', 'comentario', 'tipo', 'titulo', 'enlace']]

# Asumimos que 'df_publicaciones' es tu DataFrame y que 'fecha' es la columna que contiene las fechas en formato datetime
# Si 'fecha' no es un objeto datetime, descomenta y ejecuta la siguiente línea:
# df_publicaciones['fecha'] = pd.to_datetime(df_publicaciones['fecha'])

# Extraer año, mes, día y hora
df_publicaciones['año'] = df_publicaciones['fecha'].dt.year
df_publicaciones['mes'] = df_publicaciones['fecha'].dt.month
df_publicaciones['día'] = df_publicaciones['fecha'].dt.day
df_publicaciones['hora'] = df_publicaciones['fecha'].dt.hour

# Ahora df_publicaciones tiene columnas adicionales para año, mes, día y hora


#webs y analsisid e sentimiento : las filas donde la columna 'enlace' no está vacía y no es una cadena vacía
df_publicaciones_no_vacias = df_publicaciones[(df_publicaciones['enlace'].notna()) & (df_publicaciones['enlace'] != '')]

# Extraer el dominio de cada URL en la columna 'enlace'
df_publicaciones_no_vacias['dominio'] = df_publicaciones_no_vacias['enlace'].apply(lambda x: urlparse(x).netloc)

# Quitar 'www.' para mantener consistencia
df_publicaciones_no_vacias['dominio'] = df_publicaciones_no_vacias['dominio'].replace(r'^www\.', '', regex=True)

# Filtrar los dominios vacíos
df_publicaciones_no_vacias = df_publicaciones_no_vacias[df_publicaciones_no_vacias['dominio'] != '']

# Contar la frecuencia de cada dominio y obtener los 20 más frecuentes
conteo_dominios = df_publicaciones_no_vacias['dominio'].value_counts().nlargest(20).reset_index()
conteo_dominios.columns = ['dominio', 'conteo']

# Crear un gráfico de barras con los conteos de los dominios
figURL = px.bar(conteo_dominios, x='dominio', y='conteo', title='Top 20 de URLs por Dominio')


# Filtrar el DataFrame para obtener solo las filas con 'docs subidos'
docs_subidos_df = df_publicaciones[df_publicaciones['tipo'] == 'docs subidos']

# Contar las ocurrencias de cada 'titulo' único
conteo_titulos = docs_subidos_df['titulo'].value_counts()

# Crear el gráfico de queso
figext = px.pie(conteo_titulos, values=conteo_titulos.values, names=conteo_titulos.index, title='Distribución de Títulos para Documentos Subidos')

# Agregar el conteo total en las etiquetas
figext.update_traces(textinfo='percent+label+value')

# ANALISIS DE SENTIMIENTO Filtrar para obtener solo las filas de 'Comentario o modificación' y donde 'comentario' no esté vacío
comentarios_df = df_publicaciones[(df_publicaciones['tipo'] == 'Comentario o modificación') & (df_publicaciones['comentario'].notnull())]

# Seleccionar las columnas de interés incluyendo las temporales ya existentes
comentarios_df = comentarios_df[['fecha', 'comentario', 'año', 'mes', 'día', 'hora']]

# Asumiendo que 'comentarios_df' es tu DataFrame

# Función para verificar si el contenido es numérico
def es_numero(comentario):
    try:
        float(comentario) # Intenta convertir a float
        return True
    except ValueError:
        return False

# Aplicar la función para marcar los comentarios numéricos como NaN
comentarios_df['comentario'] = comentarios_df['comentario'].apply(lambda x: np.nan if es_numero(x) else x)

# Eliminar las filas con NaN en la columna 'comentario'
comentarios_df = comentarios_df.dropna(subset=['comentario'])

def normalize_text(text):
    words = re.findall(r'\w+', text.lower())
    corrected_text = []
    for word in words:
        corrected_word = spell.correction(word)
        # Asegúrate de que solo se añaden strings
        if isinstance(corrected_word, str):
            corrected_text.append(corrected_word)
        else:
            corrected_text.append(word)  # O puedes decidir omitir la palabra o manejar de otra manera
    return ' '.join(corrected_text)

comentarios_df['comentario_normalizado'] = comentarios_df['comentario'].apply(normalize_text)









# Aquí empieza código de STREAMLIT







def main():
    # Creación de la barra de navegación con pestañas
    tab1, tab2, = st.tabs(["Likes y Reacciones", "Comentarios y Publicaciones"])

    # Contenido de la primera pestaña
    with tab1:
        st.header("Likes y Reacciones")
        st.write("Visualización interactiva de las reacciones en Facebook.")
        # Gráfico pagina uno

        likes_['fecha'] = pd.to_datetime(likes_[['año', 'mes', 'dia_mes']].astype(str).agg('-'.join, axis=1))

        # Sidebar - Selección de tipo de reacción
        unique_reactions = likes_['reaction'].unique()
        selected_reaction = st.sidebar.multiselect('Selecciona las reacciones', unique_reactions, default=unique_reactions)

        # Sidebar - Selección de rango de fechas
        min_date = likes_['fecha'].min().date()
        max_date = likes_['fecha'].max().date()
        start_date, end_date = st.sidebar.date_input('Selecciona el rango de fechas', [min_date, max_date], min_value=min_date, max_value=max_date)

        # Filtrado del DataFrame basado en selección
        filtered_likes = likes_[(likes_['reaction'].isin(selected_reaction)) & (likes_['fecha'].dt.date >= start_date) & (likes_['fecha'].dt.date <= end_date)]

        # Agrupar los datos por fecha y tipo de reacción para el gráfico
        grouped_data = filtered_likes.groupby(['fecha', 'reaction']).agg({'num': 'sum'}).reset_index()

        # Crear el gráfico de barras con Plotly
        fig = px.bar(grouped_data, x='fecha', y='num', color='reaction', title='Histórico de Reacciones')

        # Ocultar los títulos de los ejes
        fig.update_layout(showlegend=True, xaxis={'title': ''}, yaxis={'title': ''})
        
        # Gráfico 2 pagina 1 

        # Agrupar los datos por mes y tipo de reacción para el segundo gráfico
        grouped_by_month = filtered_likes.groupby([filtered_likes['fecha'].dt.to_period('M'), 'reaction']).agg({'num': 'sum'}).reset_index()

        # Convertir el período a fecha para el eje x
        grouped_by_month['mes'] = grouped_by_month['fecha'].dt.to_timestamp()

        # Crear el gráfico de barras con Plotly para el segundo gráfico
        fig2 = px.bar(grouped_by_month, x='mes', y='num', color='reaction', title='Número de Reacciones por Mes')

        # Ocultar los títulos de los ejes para el segundo gráfico
        fig2.update_layout(showlegend=True, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig)
        st.plotly_chart(fig2)
        
        


    # Contenido de la segunda pestaña
    with tab2:
        st.header("Comentarios y Publicaciones")
        st.write("Aquí puedes agregar contenido relacionado con comentarios y publicaciones.")
        #Gráfico 1 pagina 2
        #Sidebar - Filtros similares a los de la primera imagen para fecha y tipo de publicación
        unique_types = df_publicaciones['tipo'].unique()
        selected_types = st.sidebar.multiselect('Selecciona los tipos de publicación', unique_types, default=unique_types)

        min_year = int(df_publicaciones['fecha'].dt.year.min())
        max_year = int(df_publicaciones['fecha'].dt.year.max())
        start_year, end_year = st.sidebar.select_slider('Selecciona el rango de años', options=list(range(min_year, max_year + 1)), value=(min_year, max_year))

        # Filtrado del DataFrame basado en selección
        filtered_df = df_publicaciones[
            (df_publicaciones['tipo'].isin(selected_types)) &
            (df_publicaciones['fecha'].dt.year >= start_year) &
            (df_publicaciones['fecha'].dt.year <= end_year)]

        # Agrupa por año y mes, y cuenta las publicaciones por tipo
        df_agrupado = filtered_df.groupby([filtered_df['fecha'].dt.to_period('M'), 'tipo']).size().reset_index(name='conteo')

        # Crea un gráfico de barras
        fig3 = px.bar(df_agrupado, x=df_agrupado['fecha'].dt.strftime('%Y-%m'), y='conteo', color='tipo',
                    title='Número de publicaciones por tipo cada mes')

        # Oculta los títulos de los ejes y apila las barras
        fig3.update_layout(
            xaxis_title='',
            yaxis_title='',
            barmode='stack',
            legend_title_text='Tipo de Publicación')
        
        # FIg4
        fig4 = px.pie(df_publicaciones, names='tipo', title='Distribución de Tipos de Publicaciones')
        fig5 = px.bar(conteo_dominios, x='dominio', y='conteo', title='Top 20 de URLs por Dominio')
        fig6 = px.pie(conteo_titulos, values=conteo_titulos.values, names=conteo_titulos.index, title='Distribución de Títulos para Documentos Subidos')



        st.plotly_chart(fig3)
        st.plotly_chart(fig4)
        st.plotly_chart(fig5)
        

    # Contenido de la tercera pestaña
    
  

# Ejecutar la función principal
if __name__ == "__main__":
    main()
