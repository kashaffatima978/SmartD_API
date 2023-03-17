import  scrapy
import random

class RecipeSpider(scrapy.Spider):
    name = "recipes"
    page_number = 1
    start_urls = [
        'https://www.diabetes.org.uk/guide-to-diabetes/recipes?page=0'
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
            next_page = 'https://www.diabetes.org.uk/guide-to-diabetes/recipes?page='+str(RecipeSpider.page_number)
            if RecipeSpider.page_number <= 45:
                RecipeSpider.page_number += 1
                yield response.follow(next_page, callback= self.parse)



