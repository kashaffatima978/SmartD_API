#importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random

#importing the exercise file
df=pd.read_csv('ML_directory/fitness_exercises.csv')
activity_df=pd.read_csv("ML_directory/exercise_dataset.csv")
print(df)


#add a column calories burnt based on randomly generated MET later to be replaced
#"Calories Burned = (MET x Body Weight in kg x Duration in hours)"
# 0.1 means 10 minutes should be cut short 10/8=0.1
data = np.random.randint(4, 16, size=len(df))*50*0.1
df['caloriesBurned'] = data
#transforming or encoding the equipment, bodyPart and target of the dataset
le=LabelEncoder()
df["equipmentEncoded"]=le.fit_transform(df["equipment"])
df["bodyPartEncoded"]=le.fit_transform(df["bodyPart"])
df["targetEncoded"]=le.fit_transform(df["target"])


activity_df['exercise'] = le.fit_transform(activity_df['Activity, Exercise or Sport (1 hour)'])
print(df)

#applying k means clustering
km=KMeans(n_clusters=19,n_init=5)
y_predicted=km.fit_predict(df[["bodyPartEncoded","targetEncoded"]])

#adding a column in dataset naming cluster which names each cluster
df["cluster"]=y_predicted

#saperating the clusters made
df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]
df4=df[df.cluster==3]
df5=df[df.cluster==4]
df6=df[df.cluster==5]
df7=df[df.cluster==6]
df8=df[df.cluster==7]
df9=df[df.cluster==8]
df10=df[df.cluster==9]
df11=df[df.cluster==10]
df12=df[df.cluster==11]
df13=df[df.cluster==12]
df14=df[df.cluster==13]
df15=df[df.cluster==14]
df16=df[df.cluster==15]
df17=df[df.cluster==16]
df18=df[df.cluster==17]
df19=df[df.cluster==18]

#returns the array of dataframes related to the sent category
def getDataFrame(category):
  dataframesExisting=[df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19]
  dataframes=[]
  for i in dataframesExisting:
    if(i.to_numpy()[0][0]==category):
       dataframes.append(i)
  #print("The dataframes included are "+str(dataframes)+ str(len(dataframes)) )
  return dataframes


def predictParticularExercise(caloriestoBeBurnt, dataframe):
    # applying random forest
    # preprocessing and splitting X and Y
    # print(df1)

    # splitting
    from sklearn.model_selection import train_test_split
    # getting the independent and dependent variables]
    X = dataframe[["caloriesBurned", "equipmentEncoded"]]
    Y = dataframe["name"]

    # predict using model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100)
    # training the model
    model.fit(X.values, Y.values)
    # predicting
    # get the encoded body weight
    bodyWeightEncoded = (getDataFrame("neck")[0].to_numpy())[0][7]
    # print("body wight is encoded to")
    # print(bodyWeightEncoded)
    predict = model.predict([[caloriestoBeBurnt, bodyWeightEncoded]])
    print(predict)
    a = ((dataframe.loc[(dataframe['name'].isin(predict))])["caloriesBurned"])
    b = dataframe[dataframe['name'] == predict[0]]
    # print("b is")
    # print(b.to_numpy()[0])
    # return ((predict[0],a.to_numpy()[0]))
    return b.to_numpy()[0]


# add exercises from the frame to plan
def appendExercise(frames, caloriesBurntForEachCategory):
    exercises = []
    caloriesAlreadyBurnt = 0
    for index, k in enumerate(frames):
        result = predictParticularExercise(caloriesBurntForEachCategory, k)
        # print("result got is")
        # print(result)
        # if the newly predicted calories increases the calories to be burned limit break the loop
        if (caloriesAlreadyBurnt + result[6] > caloriesBurntForEachCategory and index != 0):
            # print("In break if of back")
            break

        caloriesAlreadyBurnt += result[6]
        # print("Calories already burnt for back are now "+str(caloriesAlreadyBurnt))
        # add to exercises the name and calories burned in the exercise
        exercises.append(result)
    return exercises


def makeExercisePlan(routine, caloriesToBeBurned):
    # routine is an array [back,arms,shoulders,waist,legs,chest,cardio,neck] containing boolean values
    activated = sum(routine)
    timeForEachCategory = (10 / activated)  # gives minutes
    caloriesBurntForEachCategory = caloriesToBeBurned / activated
    exercises = []
    print(timeForEachCategory)
    print(caloriesBurntForEachCategory)

    # frames store all clusters for the category
    frames = []
    if (routine[0] == True):
        # for back
        frames = getDataFrame("back")
        exercises.extend(appendExercise(frames, caloriesBurntForEachCategory))

    if (routine[1] == True):
        # for arm
        frames = getDataFrame("upper arms")
        frames1 = getDataFrame("lower arms")
        for index, i in enumerate(frames):
            if (index % 2 != 0 and index < len(frames) and index < len(frames1)):
                temp = frames[index]
                frames[index] = frames1[index]
                frames1[index] = temp
        frames.extend(frames1)
        exercises.extend(appendExercise(frames, caloriesBurntForEachCategory))

    if (routine[2] == True):
        # for shoulders
        frames = getDataFrame("shoulders")
        exercises.extend(appendExercise(frames, caloriesBurntForEachCategory))

    if (routine[3] == True):
        # for waist
        frames = getDataFrame("waist")
        exercises.extend(appendExercise(frames, caloriesBurntForEachCategory))

    if (routine[4] == True):
        # for legs
        frames = getDataFrame("upper legs")
        frames1 = getDataFrame("lower legs")
        for index, i in enumerate(frames):
            if (index % 2 != 0 and index < len(frames) and index < len(frames1)):
                temp = frames[index]
                frames[index] = frames1[index]
                frames1[index] = temp
        frames.extend(frames1)
        exercises.extend(appendExercise(frames, caloriesBurntForEachCategory))

    if (routine[5] == True):
        # for chest
        frames = getDataFrame("chest")
        exercises.extend(appendExercise(frames, caloriesBurntForEachCategory))

    if (routine[6] == True):
        # for cardio
        frames = getDataFrame("cardio")
        exercises.extend(appendExercise(frames, caloriesBurntForEachCategory))

    if (routine[7] == True):
        # for neck
        frames = getDataFrame("neck")
        exercises.extend(appendExercise(frames, caloriesBurntForEachCategory))

    # print(exercises)
    return exercises

# makeExercisePlan([True,True,True,True,True,True,True,True],220)

def getActivity(weight,caloriestoBeBurnt):
  # method which gets the weight of the user and chnages the dataframe
  #150 min per week gives 150/3=50 mins a day
  activity_df['caloriesBurnedIn1Hour'] =weight*activity_df["Calories per kg"]
  activity_df['caloriesBurnedIn50Minutes'] =(weight*activity_df["Calories per kg"]*50)/60
  new_df=activity_df.iloc[:,[0,6,7,8]]
  #print(new_df)

  # applying random forest
  #preprocessing and splitting X and Y
  X=new_df[["caloriesBurnedIn50Minutes"]]
  Y=new_df["exercise"]
  #predict using model
  from sklearn.ensemble import RandomForestClassifier
  model=RandomForestClassifier(n_estimators=100)
  #training the model
  model.fit(X.values,Y.values)
  #predicting
  predict=model.predict([[caloriestoBeBurnt]])
  print(predict)
  #print(new_df.loc[new_df['exercise'] == predict[0]])
  return (new_df.loc[new_df['exercise'] == predict[0]]).to_numpy()[0]
#result=predictActivity(27.441188)

def getExercise(day,calories,routine,weight):
  #routine is an array [back,arms,shoulders,waist,legs,chest,cardio,neck] containing boolean values
  # 1,3,5 have activity 2,4 have exercise plan and 0 is free
  if(day==1 or day==3 or day==5):
    print("activity day")
    activity=getActivity(weight,calories)
    return(activity)
  elif(day==2 or day==4 ):
    print("exercise plan day")
    plan=makeExercisePlan(routine,calories)
    return(plan)
  else:
    return([["free"]])



