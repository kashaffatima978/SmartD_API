import scrapy

class Mets(scrapy.Spider):
    name = 'Mets'
    start_urls = [
        "https://golf.procon.org/met-values-for-800-activities/"
    ]

    def parse(self, response):
        mets = response.css('tbody.row-hover tr')
        for met in mets:
            activity= met.css('td.column-1::text').get()
            description=met.css('td.column-2::text').get()
            METS =met.css('td.column-3::text').get()
            yield {
                "activity": activity,
                "description": description,
                "METS": METS
            }
