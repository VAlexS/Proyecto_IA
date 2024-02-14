import pandas as pd
from funciones import *

'''
Tras haber realizado las transformaciones con respecto al fichero de entrenamiento, 
realizo el preprocesamiento y las transformaciones necesarias en el fichero de prueba, proporcionado por el enunciado
'''

# obtengo el fichero de prueba
test_file = pd.read_csv('./ficherosCSV/test_kaggle.csv', delimiter=';')  # para las columnas separadas por ;

#PREPROCESAMIENTOS Y TRANSFORMACIONES EN EL FICHERO TEST

# elimino las columnas irrelevantes, es decir, selecciono las que considero importantes

test_file = test_file[['PassengerId', 'Pclass', 'Sex', 'Age']]

print("COLUMNAS (despues de filtrar)")

print(test_file.columns)

print("________________________")

print("Porcentaje valores faltantes")
porcentaje = get_porcentaje_faltantes(dataframe=test_file)
print(porcentaje)

print("________________________")

promedioEdad = test_file['Age'].mean().round()

print("Promedio Edad")
print(promedioEdad)

print("___________________________________")

print("Tras haber rellenado los valores faltantes con el promedio en la edad")
test_file['Age'].fillna(promedioEdad, inplace=True)

print("Porcentaje valores faltantes")
porcentaje = get_porcentaje_faltantes(dataframe=test_file)
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

test_file['Categoria_edad'] = pd.cut(test_file['Age'], bins=rangos, labels=etiquetas)

test_file['Age'] = test_file['Age'].astype(int)

# agrupo niveles socioeconomicos
clases = [1, 2, 3]

etiquetas = ['Superior', 'Medio', 'Inferior']

test_file['Pclass'] = test_file['Pclass'].map(
    {valor: etiqueta for valor, etiqueta in zip(clases, etiquetas)})

print(test_file)

#tranfiero todo a un nuevo fichero .csv de test, con el procesamiento
test_file.to_csv('./ficherosCSV/test_kaggle_Def.csv', index=False)