import scrapy

class Mets2(scrapy.Spider):
    name = 'Mets2'
    start_urls = [
        "https://community.plu.edu/~chasega/met.html#2"
    ]

    def parse(self, response):
        for i in range(5, 389-5):
            description = response.css(f'tr:nth-child({i}) td:nth-child(5)::text').get()
            Activity = response.css(f'tr:nth-child({i}) td:nth-child(4)::text').get()
            METS = response.css(f'tr:nth-child({i}) td:nth-child(3)::text').get()
            yield {

                "Activity": Activity,
                "description": description,
                "METS": METS
            }



