# import the required packages

# a HTML "searcher"
import bs4
from bs4 import BeautifulSoup as soup

# a URL "opener"
from urllib.request import urlopen as uReq

# easy way to get A-Z
import string

# creating the file to deposit information:
filename = "shortinterest.csv"
f = open(filename, "w")

headers = "Company,Symbol,Percent_Float,DaystoCover\n"

f.write(headers)

# page index for looping through all pages on WSJ
letters = list(string.ascii_uppercase)

letters.insert(0,"0_9")
pages = letters

# looping through each URL
for page in pages:
    # getting the html
    my_url = "http://www.wsj.com/mdc/public/page/2_3062-shtnyse_" + str(page) + "-listing.html"
    uClient = uReq(my_url)
    page_html = uClient.read()
    uClient.close()

    # parsing the HTML using BS4
    page_soup = soup(page_html, "html.parser")

    # grabbing the table and all its TableRows
    table = page_soup.find("div", {"class": "mdcWide"})
    trs = table.table.findAll("tr")

    # print a quick status update to the console
    print("Printing Page: " + page)

    # loop through all "tr" tags and grab name, ticker, %float, and days to cover
    for i in range(1, len(trs)):
        #loop this one to find name and ticker
        links = trs[i].findAll("a")
        name = links[0].text
        ticker = links[1].text

        # also loop this one to find numerical info
        numbers = trs[i].findAll("td", {"class":"num"})
        percent_float = numbers[2].text
        daystocover = numbers[3].text

        # write record to csv if it contains useful info
        if percent_float != "..." and daystocover != "...":
            f.write(name.replace(",", "|") + ",")
            f.write(ticker + ",")
            f.write(percent_float + ",")
            f.write(daystocover + "\n")


f.close()
