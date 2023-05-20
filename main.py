# This is a sample Python script.
import io
from typing import List
import re
import cv2
import uvicorn
from fastapi import Body, FastAPI, File, UploadFile
from fastapi import Body, FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
from pandas.io import json
from starlette.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
from ML_directory.DietPlanModel import dietPlan
from ML_directory.ExercisePlanModel import getExercise
import json
from bs4 import BeautifulSoup
import requests
from Food.VolumeEstimation import preprocess_image , estimate


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
def DietPlan (calories: int = Body(embed=True),alergies: list[str]=  Body(embed=True)):
    print(calories)
    print(alergies)
    res = dietPlan(calories,alergies)
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

def to_nested_list(lst):
    if isinstance(lst, list):
        return [to_nested_list(i) for i in lst]
    elif isinstance(lst, np.ndarray):
        return to_nested_list(lst.tolist())
    else:
        return lst

@app.post('/exercisePlan')
def ExercisePlan (day: int = Body(embed=True), weight: float = Body(embed=True), calories: int = Body(embed=True), routine: list[bool]=  Body(embed=True)):
    print("Hello i am in execise function ")
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

# async def run_spider(name: str):
#     url = 'https://www.diabetes.org.uk/guide-to-diabetes/recipes/'+name
#     configure_logging({'LOG_FORMAT': '%(levelname)s: %(message)s'})
#     runner = CrawlerRunner()
#     d = runner.crawl(OneRecipe, start_urls = [url])
#     d.addBoth(lambda _: reactor.stop())
#     reactor.run()


# async def run_spider(name: str):
#     url = 'https://www.diabetes.org.uk/guide-to-diabetes/recipes/'+name
#     configure_logging({'LOG_FORMAT': '%(levelname)s: %(message)s'})
#     runner = CrawlerRunner()
#     d = runner.crawl(OneRecipe, start_urls = [url])
#     d.addBoth(lambda _: reactor.stop())
#     reactor.run()


@app.post('/getRecipe')
async def sendRecipe(name: str= Body(embed=True)):
    name = name.replace(" ", "+")
    MainPageUrl = "https://www.diabetes.org.uk/guide-to-diabetes/recipes/recipe-search-results?search="+name
    name = name.replace("+", " ")
    mainPage = requests.get(MainPageUrl)
    htmlContent = mainPage.content
    soup = BeautifulSoup(htmlContent, "html.parser")
    try:
        recipeArticle = soup.find('figure', class_= "card__img").find('a')
    except:
        recipeArticle = soup.find('div', class_ = 'card__title').find('a')
    url = "https://www.diabetes.org.uk"+recipeArticle['href']
    print(recipeArticle['href'])
    response = requests.get(url)
    htmlContent = response.content
    soup = BeautifulSoup(htmlContent, "html.parser")
    print(name)

    #ingredients
    ingredientList = []
    ingredients = soup.find_all("div", class_= "recipeIngredient")
    for ing in ingredients:
        ingredientList.append(ing.text)
    print(ingredientList)

    #method
    method = []
    stepDiv = soup.find_all("div", class_ = "large-checkbox__content")
    for st in stepDiv:
        step = st.find("p")
        method.append(step.text)
    print(method)

    #prepTime
    serve = []
    try:
        preptime = soup.find("div", class_="instruction instruction_prep prepTime")
        preptime = (preptime.text).strip()
    except:
        preptime = ' '
    try:
        cooktime = soup.find("div", class_="instruction instruction_time prepTime")
        cooktime = (cooktime.text).strip()
    except:
        cooktime = " "
    print(cooktime)
    print(preptime)
    return{
        "name": name,
        "ingredient": ingredientList,
        "method": method,
        "preptime": preptime,
        "cooktime": cooktime,
    }

@app.get('/getBlog')
def getBlog():
    blogs =[]
    page = 0
    while(page<=10):
        website1 = "https://www.diabetes.org.uk/about_us/news/search?page="+str(page)
        content = requests.get(website1)
        htmlContent = content.content
        soup = BeautifulSoup(htmlContent, "html.parser")
        articles = soup.find_all('article')
        for article in articles:
            divs = article.find("div", class_= "image__media-image")
            image = divs.find('img', {"itemprop" : "image"})
            body = article.find("div", class_ = 'news-card__body')
            anchor = body.find('a')
            title = anchor.text
            url = anchor['href']
            content = (article.find("div", class_= "news-card__summary")).text
            oneblog = {
                'title': title,
                'image': image['src'],
                'blogUrl': url,
                'summary': content
            }
            print(oneblog)
            blogs.append(oneblog)
        page = page+1
    return {'blogs': blogs}


@app.get('/getVideos')
def getVideos():
    query = "diabetes+awareness"
    query2 = "diabetes Technology"

    # getting news url input
    diabtes_url = "https://www.youtube.com/results?search_query=" + query2 + "&sp=CAASBBABIAE%253D"
    diabtes_url2 = "https://www.youtube.com/results?search_query=" + query + "&sp=CAASBBABIAE%253D"
    print(diabtes_url)

    html_text = requests.get(diabtes_url).text
    soup = BeautifulSoup(html_text, "lxml")
    divs = soup.find_all("script")
    test_str = divs[33].text
    # getting ids
    ids = []
    regex = r"(?<=\"videoId\":\")[\w]+(?=\")"
    matches = re.finditer(regex, test_str, re.MULTILINE)
    for matchNum, match in enumerate(matches, start=1):
        id = match.group()
        if id not in ids:
            ids.append(id)

    html_text = requests.get(diabtes_url2).text
    soup = BeautifulSoup(html_text, "lxml")
    divs = soup.find_all("script")
    test_str = divs[33].text
    matches = re.finditer(regex, test_str, re.MULTILINE)
    for matchNum, match in enumerate(matches, start=1):
        id = match.group()
        if id not in ids:
            ids.append(id)

    return{'ids': ids}

VolumeModel = tf.keras.models.load_model("./Food/foodClassificationModel")
foodClasses = ['apple_pie',
 'baby_back_ribs',
 'baklava',
 'beef_carpaccio',
 'beef_tartare',
 'beet_salad',
 'beignets',
 'bibimbap',
 'bread_pudding',
 'breakfast_burrito',
 'bruschetta',
 'caesar_salad',
 'cannoli',
 'caprese_salad',
 'carrot_cake',
 'ceviche',
 'cheesecake',
 'cheese_plate',
 'chicken_curry',
 'chicken_quesadilla',
 'chicken_wings',
 'chocolate_cake',
 'chocolate_mousse',
 'churros',
 'clam_chowder',
 'club_sandwich',
 'crab_cakes',
 'creme_brulee',
 'croque_madame',
 'cup_cakes',
 'deviled_eggs',
 'donuts',
 'dumplings',
 'edamame',
 'eggs_benedict',
 'escargots',
 'falafel',
 'filet_mignon',
 'fish_and_chips',
 'foie_gras',
 'french_fries',
 'french_onion_soup',
 'french_toast',
 'fried_calamari',
 'fried_rice',
 'frozen_yogurt',
 'garlic_bread',
 'gnocchi',
 'greek_salad',
 'grilled_cheese_sandwich',
 'grilled_salmon',
 'guacamole',
 'gyoza',
 'hamburger',
 'hot_and_sour_soup',
 'hot_dog',
 'huevos_rancheros',
 'hummus',
 'ice_cream',
 'lasagna',
 'lobster_bisque',
 'lobster_roll_sandwich',
 'macaroni_and_cheese',
 'macarons',
 'miso_soup',
 'mussels',
 'nachos',
 'omelette',
 'onion_rings',
 'oysters',
 'pad_thai',
 'paella',
 'pancakes',
 'panna_cotta',
 'peking_duck',
 'pho',
 'pizza',
 'pork_chop',
 'poutine',
 'prime_rib',
 'pulled_pork_sandwich',
 'ramen',
 'ravioli',
 'red_velvet_cake',
 'risotto',
 'samosa',
 'sashimi',
 'scallops',
 'seaweed_salad',
 'shrimp_and_grits',
 'spaghetti_bolognese',
 'spaghetti_carbonara',
 'spring_rolls',
 'steak',
 'strawberry_shortcake',
 'sushi',
 'tacos',
 'takoyaki',
 'tiramisu',
 'tuna_tartare',
 'waffles']

def estimateVolume(topPath, sidePath):
    cnts, cntsArea = preprocess_image(topPath)
    length, width = estimate(cnts,cntsArea, topPath)
    cnts1, cntsArea1 = preprocess_image(sidePath)
    height, width = estimate(cnts1,cntsArea1, sidePath)
    Volume = (length) * (width) * (height)
    Volume = Volume * 16.3871
    print('Volume is ', Volume)
    return Volume

@app.post('/getVolume')
async def getVolume(top: bytes = File(...),side: bytes = File(...)):
    print("hello i am here ")

    dataBytesIO = io.BytesIO(top)
    topImg = Image.open(dataBytesIO)
    topImg.save('./Images/top.jpg')

    dataBytesIO = io.BytesIO(side)
    sideImg = Image.open(dataBytesIO)
    sideImg.save('./Images/side.jpg')

    topimg = cv2.imread("./Images/top.jpg")
    print(topimg)
    sideimg = cv2.imread("./Images/side.jpg")
    print(sideimg)
    # Resize the image to 224x224 pixels and add a batch dimension
    resized_img = cv2.resize(topimg, (224, 224), 3)
    resized_img = np.expand_dims(resized_img, axis=0)
    prediction = VolumeModel.predict(resized_img)
    max_index = np.argmax(prediction)
    foodName = foodClasses[max_index]
    foodName = foodName.replace("_", " " )

    Volume = estimateVolume("./Images/top.jpg", "./Images/side.jpg" )
    query = str(Volume)+' cm3 '+foodName
    print("***********************************************",query)
    api_url = 'https://api.api-ninjas.com/v1/nutrition?query={}'.format(query)
    print(api_url)
    response = requests.get(api_url, headers={'X-Api-Key': 'avnKxya254Me1ijNIJ6uIw==JA9ffe9fNotaycST'})
    if response.status_code == requests.codes.ok:
        information = response.text
    else:
        information = 'Cant find your required food'
    return {'info': information}



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    uvicorn.run(app, host='192.168.1.9', port=8000)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
