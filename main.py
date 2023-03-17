# This is a sample Python script.
from typing import List

import cv2
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import uvicorn
from fastapi import Body, FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
from pandas.io import json
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
from ML_directory.DietPlanModel import dietPlan
from ML_directory.ExercisePlanModel import getExercise
import json

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("./RetinopathyModel/RetinopathyModel/")

CLASS_NAMES = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']
@app.get('/')
def index():
    return {'message': 'hello to index page'}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    print('done yay')
    return image

@app.post('/predictRetinopathy')
async def predictRP(file: UploadFile = File(...)):
    print("hello i am here ")
    contents = await file.read()
    with open('./Images/file.jpg', 'wb') as f:
        f.write(contents)

    img = cv2.imread("./Images/file.jpg")
    print(img)
    # image = read_file_as_image(img)
    img_batch = np.expand_dims(img, 0)
    prediction = MODEL.predict(img_batch)
    print(prediction[0])
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return{
        'class': predicted_class,
        'confidence': float(confidence)
    }

def changeFloatToStringInArray(array: list):
    ind =0
    for i in array:
        array[ind] = str(i)
        ind = ind +1
    return array

@app.post('/dietPlan')
def DietPlan (calories: int = 0):
    res = dietPlan(calories)
    breakfast = changeFloatToStringInArray(list(res[0]))
    snack1 = changeFloatToStringInArray(list(res[1]))
    lunch = changeFloatToStringInArray(list(res[2]))
    snack2 = changeFloatToStringInArray(list(res[3]))
    dinner = changeFloatToStringInArray(list(res[4]))
    return {
        "breakfast": breakfast,
        "snack1": snack1,
        "lunch": lunch,
        "snack2": snack2,
        "dinner": dinner,
    }
class Item(BaseModel):
    day: int
    routine: list | None = None

def to_nested_list(lst):
    if isinstance(lst, list):
        return [to_nested_list(i) for i in lst]
    elif isinstance(lst, np.ndarray):
        return to_nested_list(lst.tolist())
    else:
        return lst

@app.post('/exercisePlan')
def ExercisePlan (day: int = Body(embed=True), weight: float = Body(embed=True), calories: int = Body(embed=True), routine: list[bool]=  Body(embed=True)):
    print(day)
    print(routine)
    print(calories)
    print(weight)
    res = getExercise(day, calories, routine, weight)
    print(type(res))
    pyList = to_nested_list(res)
    # pyList = res.tolist()
    print(pyList)
    arr_json = json.dumps(pyList)
    print(arr_json)
    return{
        'res': arr_json
    }


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    uvicorn.run(app, host='192.168.1.10', port=8000)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
