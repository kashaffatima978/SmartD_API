import scrapy
from scrapy import Spider, Request


class OneRecipe(scrapy.Spider):
    name = 'OneRecipe'
    def parse(self, response):
        imgUrl = response.css('picture img::attr(src)').get()
        ingredients = response.css('div.wrapper div.recipeIngredient::text').getall()
        method = response.css('div.recipe-step div.large-checkbox__content p::text').getall()
        serve = response.css('div.instruction_serves span::text').get()
        item = {
            'img': imgUrl,
            'ingredient': ingredients,
            'method': method,
            'serve': serve,
        }
        yield item

