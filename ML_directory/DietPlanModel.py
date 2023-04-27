#importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import random

#importing the recipes file
df=pd.read_csv('ML_directory/formatedRecipes.csv')
dfForRD=pd.read_csv('ML_directory/formatedRecipes.csv')

#transforming the name
le=LabelEncoder()
df["nameEncoded"]=le.fit_transform(df["name"])
dfForRD["nameEncoded"]=le.fit_transform(dfForRD["name"])

def getCarbs(calories):
  # Define the independent variables (kCal,carbs,sugar) and dependent variable (nameEncoded)
  X = df.loc[:,["kCal"]]
  Y = df.loc[:,"carbs"] # Example output data
  # Create an instance of the LinearRegression class and fit the model to the data
  model = LinearRegression().fit(X.values, Y.values)
  # Predict the output for a new set of independent variables
  predicted= model.predict([[calories]])
  print("Predicted carbs for given calories are: ", predicted)
  # import math
  # ceil = df[df['carbs'] == (math.ceil(predicted))]
  # print(ceil)
  # floor = df[df['carbs'] == (math.floor(predicted))]
  # print(floor)
  return predicted

def getRecipe(calories,carbs,filteredRecipes):
  print("************************************************")
  print("************************************************")
  print(filteredRecipes)
  XRD=filteredRecipes[["kCal","carbs"]]
  YRD=filteredRecipes["nameEncoded"]
  #predict using model
  model=RandomForestClassifier(n_estimators=100)
  #training the model
  model.fit(XRD.values,YRD.values)
  #predicting
  predict=model.predict([[calories,carbs]])
  print(predict)
  print("************************************************")
  recipeDetails = dfForRD[dfForRD['nameEncoded'] == predict[0]]
  print("************************************************")
  print(recipeDetails)

 
  #remove the predicted recipe from random forest dataset
  #dfForRD=dfForRD.loc[dfForRD['nameEncoded'] != predict[0] ]
  dfForRD.drop(dfForRD[dfForRD['nameEncoded'] == predict[0]].index,inplace=True)
  filteredRecipes.drop(filteredRecipes[filteredRecipes['nameEncoded'] == predict[0]].index,inplace=True)
  #change 2D array to 1D
  print(recipeDetails.to_numpy()[0])
  return recipeDetails.to_numpy()[0]

def dietPlan(calories,alergies):
  #allergies has list of ingredients user has allergy with
    #remove all recipes with alergies from df and dfForRD
    dfForRD['name']=dfForRD['name'].str.lower()
    mask = [any(word in x.lower() for word in alergies) for x in dfForRD['name']]
    print(mask)
    # new_df = dfForRD[~mask]
    mask = np.array(mask)
    df_filtered = dfForRD.loc[~mask]
    print(df_filtered.shape)
  #get calories for each meal
    caloriesForBreakfast=calories*25/100
    caloriesForSnack1=calories*10/100
    caloriesForLunch=calories*30/100
    caloriesForSnack2=calories*10/100
    caloriesForDinner=calories*25/100
    # print(caloriesForBreakfast,caloriesForSnack1,caloriesForLunch,caloriesForSnack2,caloriesForDinner)
    #dietPlan contains the arrays of all the meals
    dietPlan=[]
    i=1
    while(i<=5):
      recipe=[]
      if(i==1):
        carbs=getCarbs(caloriesForBreakfast)
        recipe=getRecipe(caloriesForBreakfast,carbs[0],df_filtered)

      elif(i==2):
        carbs=getCarbs(caloriesForSnack1)
        recipe=getRecipe(caloriesForSnack1,carbs[0],df_filtered)
      elif(i==3):
        carbs=getCarbs(caloriesForLunch)
        recipe=getRecipe(caloriesForLunch,carbs[0],df_filtered)
      elif(i==4):
        carbs=getCarbs(caloriesForSnack2)
        recipe=getRecipe(caloriesForSnack2,carbs[0],df_filtered)
      elif(i==5):
        carbs=getCarbs(caloriesForDinner)
        recipe=getRecipe(caloriesForDinner,carbs[0],df_filtered)

      dietPlan.append((recipe))
      # print("diet plan is-----------------------")
      # print(dietPlan)
      i+=1
    return dietPlan
