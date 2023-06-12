import pandas as pd
import numpy as np
import joblib

from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder

Data_df = pd.read_csv('Data.csv')

Data_df_recode = Data_df.copy().drop(["Lien"], axis=1)

Data_df_recode['Puissance'].fillna(0, inplace=True)
Data_df_recode['Puissance'] = Data_df_recode['Puissance'].astype(float).astype(int)
Data_df_recode = Data_df_recode.drop(Data_df_recode[Data_df_recode['Puissance'] > 41].index)
Data_df_recode = Data_df_recode.drop(Data_df_recode[Data_df_recode['Puissance'] < 3].index)

Data_df_recode = Data_df_recode.dropna(thresh=4)

Data_df_recode = Data_df_recode.drop(Data_df_recode[Data_df_recode['Année'] == '1.4L'].index)
Data_df_recode = Data_df_recode.drop(Data_df_recode[Data_df_recode['Année'] == '150.000'].index)

Data_df_recode['annee'] = Data_df_recode['Année'].apply(lambda x: int(x[-4:]) if "." in x else int(x))

Data_df_recode = Data_df_recode.drop(Data_df_recode[Data_df_recode['annee'] < 1960].index)
Data_df_recode = Data_df_recode.drop(Data_df_recode[Data_df_recode['annee'] >2023].index)

Data_df_recode=Data_df_recode.drop('Année', axis=1)

Data_df_recode.rename(columns = {'annee':'Annee'}, inplace = True)
Data_df_recode['Annee'] = Data_df_recode['Annee'].astype(str).astype(int)

Data_df_recode = Data_df_recode.drop(Data_df_recode[Data_df_recode['Prix'] == Data_df_recode['Kilomètrage']].index)
Data_df_recode['Kilomètrage'] = Data_df_recode['Kilomètrage'].str.replace('.', '', regex=False)
Data_df_recode['Kilomètrage'] = Data_df_recode['Kilomètrage'].str.replace(' km', '', regex=False)
Data_df_recode = Data_df_recode.drop(Data_df_recode[Data_df_recode['Kilomètrage'] == '111111111111111E+016'].index)
Data_df_recode['Kilomètrage'] = Data_df_recode['Kilomètrage'].astype(str).astype(int)
Data_df_recode.loc[Data_df_recode['Kilomètrage'] < 1000, 'Kilomètrage'] *= 1000

Data_df_recode['Prix'].fillna(0, inplace=True)
Data_df_recode = Data_df_recode.drop(Data_df_recode[Data_df_recode['Prix'] == '9.400 km'].index)
Data_df_recode = Data_df_recode.drop(Data_df_recode[Data_df_recode['Prix'] == '123'].index)
Data_df_recode = Data_df_recode.drop(Data_df_recode[Data_df_recode['Prix'] == '999'].index)
Data_df_recode['Prix'] = Data_df_recode['Prix'].astype(str).astype(float)
Data_df_recode = Data_df_recode.drop(Data_df_recode[Data_df_recode['Prix'] == 0.0].index)
Data_df_recode = Data_df_recode.drop(Data_df_recode[Data_df_recode['Prix'] == 1.0].index)
Data_df_recode.loc[Data_df_recode['Prix'] < 1000, 'Prix'] *= 1000
Data_df_recode['Prix'] = Data_df_recode['Prix'].astype(float).astype(int)


object_columns = ['Couleur', 'Etat', 'Boite', 'Cylindré', 'Marque', 'Carrosserie', 'Carburant', 'Gamme']
label_mapping = {}

for column in object_columns:
    if Data_df_recode[column].dtype == 'object':
        label_encoder = LabelEncoder()
        Data_df_recode[column] = label_encoder.fit_transform(Data_df_recode[column])
        label_mapping[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

X,y= Data_df_recode.drop('Prix', axis=1), Data_df_recode['Prix']

regr = linear_model.LinearRegression()
regr.fit(X, y)

joblib.dump(regr, "regr.pkl")
joblib.dump(label_mapping, 'label_mapping.pkl')

