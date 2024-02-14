import pandas as pd
from funciones import *

'''
NOTA: El fichero trainDef.csv, viene del dataframe procesado que he generado en la practica 4,
para realizar la prediccion, tomo como referencia el fichero trainDef.csv, lo utilizo como modelo de entrenamiento.
Dado que para predecir si un pasajero sobrevive o no, una de las variables dependientes que tomo es la edad,
sin embargo, en la practica 4 realize una transformacion de edad (originalmente era numerico), cuyo cambio fue
a agrupacion por rangos, por lo tanto, mantengo la edad como estaba y agrego una columna nueva llamada rango_edades
'''

df_train_original = pd.read_csv('./ficherosCSV/train.csv')

print("COLUMNAS (antes de filtrar)")

print(df_train_original.columns)

print("______________________________________")

# elimino las columnas irrelevantes, es decir, selecciono las que considero importantes

df_train_original = df_train_original[['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age']]

print("COLUMNAS (despues de filtrar)")

print(df_train_original.columns)

print("________________________")

print("Porcentaje valores faltantes")
porcentaje = get_porcentaje_faltantes(dataframe=df_train_original)
print(porcentaje)

print("________________________")

promedioEdad = df_train_original['Age'].mean().round()

print("Promedio Edad")
print(promedioEdad)

print("___________________________________")

print("Tras haber rellenado los valores faltantes con el promedio en la edad")
df_train_original['Age'].fillna(promedioEdad, inplace=True)

print("Porcentaje valores faltantes")
porcentaje = get_porcentaje_faltantes(dataframe=df_train_original)
print(porcentaje)

# en vez de sustituir la columna de edades por rangos, agrego una columna adidional llamada categoria, para facilitar el modelo de prediccion en el test, ademas de la edad especifica
rangos = [0, 18, 30, 60, 100]

'''
Categoria edad:
Joven (entre 18 y 29 años)
Mediana edad (entre 30 y 59 años)
Tercera edad (a partir de 60 años)
'''
etiquetas = ['Menor de edad', 'Joven', 'Mediana edad', 'Tercera edad']

df_train_original['Categoria_edad'] = pd.cut(df_train_original['Age'], bins=rangos, labels=etiquetas)

df_train_original['Age'] = df_train_original['Age'].astype(int)

# agrupo niveles socioeconomicos
clases = [1, 2, 3]

etiquetas = ['Superior', 'Medio', 'Inferior']

df_train_original['Pclass'] = df_train_original['Pclass'].map(
    {valor: etiqueta for valor, etiqueta in zip(clases, etiquetas)})

#en este caso, no es necesario transformar Survives (de 0 a No y de 1 a Si), ya que, en vez de visualizar los datos, simplemente realizamos la preciccion


# con todas las modificaciones, las guardo en un fichero nuevo .csv
df_train_original.to_csv('./ficherosCSV/trainDef.csv', index=False)
