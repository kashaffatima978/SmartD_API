import scrapy

class OneRecipe(scrapy.Spider):
    name = 'OneRecipe'
    start_urls = [
        # "https://www.diabetes.org.uk/guide-to-diabetes/recipes/chocolate-and-banana-mousse"
        "https://www.diabetes.org.uk/guide-to-diabetes/recipes/green-thai-fish-curry"
    ]

    def parse(self, response):
        imgUrl = response.css('picture img::attr(src)').get()
        ingredients = response.css('div.wrapper div.recipeIngredient::text').getall()
        method = response.css('div.recipe-step div.large-checkbox__content p::text').getall()
        serve = response.css('div.instruction_serves span::text').get()
        yield {
            'img': imgUrl,
            'ingredient': ingredients,
            'method': method,
            'serve': serve,
        }