import  scrapy
import random

class RecipeSpider(scrapy.Spider):
    name = "recipes"
    n1 = (random.randint(0 , 10))
    n2 = (random.randint(11, 20))
    n3 = (random.randint(21, 30))
    n4 = (random.randint(31, 45))
    start_urls = [
        'https://www.diabetes.org.uk/guide-to-diabetes/recipes?page='+str(n1),
        'https://www.diabetes.org.uk/guide-to-diabetes/recipes?page=' + str(n2),
        'https://www.diabetes.org.uk/guide-to-diabetes/recipes?page=' + str(n3),
        'https://www.diabetes.org.uk/guide-to-diabetes/recipes?page=' + str(n4),
    ]
    def parse(self, response):
        recipes = response.css('article.recipes-card')
        for recipe in recipes:
            name = recipe.css('span::text').get()
            calories = recipe.css('div.nutrition-tag__content::text').getall()
            cookTime = recipe.css('div.recipes__total-cook-time div::attr(content)').get()
            imgUrl = recipe.css('img::attr(src)').get()
            yield {
                'name': name,
                'kCal': calories[0],
                'carbs': calories[1],
                'sugar': calories[2],
                'cookTime': cookTime,
                'img': imgUrl
            }



