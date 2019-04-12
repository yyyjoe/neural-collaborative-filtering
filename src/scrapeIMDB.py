import csv
import urllib
from bs4 import BeautifulSoup
import requests
import re
from progressbar import *


movieIds=[]
links={}
count=0

with open('data/ml-1m/movies.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        count+=1
        if(count>1):
            movieIds.append(row[0].split(',')[0])
count=0
with open('data/ml-1m/links.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        count+=1
        if(count>1):
            row[0].split(',')[0]
            links[row[0].split(',')[0]] = row[0].split(',')[1]

imdbIds={}

for movieid in movieIds:
    movieid=str(movieid)
    if movieid not in links:
        #print(movieid)
        continue
    imdbIds[movieid]=links[movieid]
print(len(movieIds))
print(len(imdbIds))

def scrapingPoster(idToimdb):
    # Display Download Progress
    downloadNum=len(idToimdb)
    widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ',
               ' ', ETA(), ' ', SimpleProgress()]
    p = ProgressBar(widgets=widgets, maxval=downloadNum).start()
    p.start()

    # Start Scraping
    story_list=[]
    i=0
    k=0
    for movieId,imdbId in idToimdb.items():
        k+=1
        if(k<=3378):
            continue
        # Read HTML
        url="https://www.imdb.com/title/tt" + imdbId
        try:
            html=urllib.request.urlopen(url)
        except urllib.request.URLError as err:
            print(movieId,err)
            continue
            
        soup = BeautifulSoup(html, features='html.parser')
        
        # Download Images
        all_href = soup.find_all('img', {"alt": re.compile('Poster$')})
        all_href = [l['src'] for l in all_href]
        if len(all_href)<1:
            print(movieId+" " + imdbId + "   ERROR")
            continue
        try:
            r = requests.get(all_href[0], stream=True)
            r.raise_for_status()
        except requests.exceptions.HTTPError as err:
            print(movieId,err)
            continue
        else:
            with open('.data/ml-1m/posterImages/' + movieId + '.jpg', 'wb') as f:
                for chunk in r.iter_content(chunk_size=32):
                    f.write(chunk)
        p.update(k)

    
    p.finish()

    
scrapingPoster(imdbIds)
