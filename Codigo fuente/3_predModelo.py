import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train = pd.read_csv('./ficherosCSV/trainDef.csv')

test = pd.read_csv('./ficherosCSV/test_kaggle_Def.csv')


variables_relevantes = ['Pclass', 'Sex', 'Age', 'Categoria_edad']

# Preprocesamiento de datos, transformo las variables categoricas a numericas, para facilitar el modelo de aprendizaje
train['Sex'] = train['Sex'].map({'female': 0, 'male': 1})
test['Sex'] = test['Sex'].map({'female': 0, 'male': 1})

train['Categoria_edad'] = train['Categoria_edad'].map({'Menor de edad': 0, 'Joven': 1, 'Mediana edad': 2, 'Tercera edad': 3})
test['Categoria_edad'] = test['Categoria_edad'].map({'Menor de edad': 0, 'Joven': 1, 'Mediana edad': 2, 'Tercera edad': 3})

train['Pclass'] = train['Pclass'].map({'Superior': 1, 'Medio': 2, 'Inferior': 3})
test['Pclass'] = test['Pclass'].map({'Superior': 1, 'Medio': 2, 'Inferior': 3})

# Entreno el modelo de clasificacion, cuya variable a predecir es si sobrevive o no
clf = LogisticRegression(random_state=0)
clf.fit(train[variables_relevantes], train['Survived'])

# Hacer la predicción en el conjunto de datos de prueba, en base a las variables dependientes
test['Survived'] = clf.predict(test[variables_relevantes])



# Guardar la predicción en un archivo CSV
test[['PassengerId', 'Survived']].to_csv('./ficherosCSV/submission2.csv', index=False)

#EVALUO LA METRICA ACCUARY_SCORE

# Hacer la predicción en el conjunto de datos de prueba
y_pred = clf.predict(test[variables_relevantes])

# Calcular la precisión del modelo
accuracy = accuracy_score(test['Survived'], y_pred)
print('Precisión del modelo: {:.2f}'.format(accuracy))
#imprime que la precision del modelo es 1, por lo tanto, indica que todas las predicciones del modelo son correctas