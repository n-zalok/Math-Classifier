from airflow.sdk import task, dag, Param
from airflow.models.baseoperator import chain
import requests
from bs4 import BeautifulSoup
import pickle
import os
from tqdm import tqdm
import math

# Mimic a browser visit
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
}
session = requests.Session()

# Depth parameter controls which pickle file to use
@dag(
    dag_id='fetch_math_articles',
    params={'depth': Param(2, type='integer')},
    catchup=False
)


def articles_dag():

    @task
    def get_articles_sets(**context):
        depth = context['params']['depth']

        # Load categories from pickle file
        with open(f'categories_depth_{depth}.pkl', 'rb') as file:
            categories = pickle.load(file)
        num_cats = len(categories)
        print(f'{num_cats} categories loaded successfully')

        # Get articles names' in each category
        percent = 1
        articles_sets = []
        for i, cat in enumerate(categories):
            # Get pages in category
            page = session.get(f'https://en.wikipedia.org/w/index.php?title=Special:Export&addcat&catname={cat['category']}&curonly=1', headers=headers)
            src = page.content
            soup = BeautifulSoup(src, 'lxml')

            # Progress tracking
            if (((i+1)/num_cats)*100) >= percent:
                tqdm.write(f"{percent}% completed")
                percent  = math.ceil(((i+1)/num_cats)*100)
            else:
                pass
            
            # Extract article names and store them
            try:
                articles_set = soup.find('textarea', {'id': 'ooui-php-2'}).text.strip()
                cat['articles_set'] = articles_set.replace(' ', '\n')
                articles_sets.append(cat)
            except:
                print(f'Error collecting pages of category: {cat['category']}')
                continue
        
        # Push articles sets to XCom for downstream tasks
        context['ti'].xcom_push(key='articles_sets', value=articles_sets)
        print(f'{len(articles_sets)} articles sets created successfully')
    

    @task
    # Download articles in each set and save as XML files
    def download_articles_sets(**context):
        # Pull articles sets from XCom
        depth = context['params']['depth']
        articles_sets = context['ti'].xcom_pull(key='articles_sets', task_ids='get_articles_sets')
        num_sets = len(articles_sets)

        # Download and save articles based on their categories and layers
        percent = 1
        for i, set in enumerate(articles_sets):
            # Create directory structure if it doesn't exist
            # Each category xml file is stored in data_depth/og_cat/layer/
            path = f'./data_{depth}/{set['og_cat']}/{set['layer']}'
            os.makedirs(path, exist_ok=True)

            # Progress tracking
            if (((i+1)/num_sets)*100) >= percent:
                tqdm.write(f"{percent}% completed")
                percent  = math.ceil(((i+1)/num_sets)*100)
            else:
                pass
                
            # Prepare data for POST request to export articles
            data = {
                    "catname": set["category"],
                    "curonly": "1",
                    "pages": set["articles_set"],
                    "wpDownload": "1",
                    "title": "Special:Export",
                    "wpEditToken": "+\\"
                }
            
            try:
                # Send POST request to export articles and save response as XML
                resp = session.post("https://en.wikipedia.org/wiki/Special:Export", data=data, headers=headers)
                with open(f"{path}/{set['category']}.xml", "w", encoding="utf-8") as file:
                    file.write(resp.text)
            except:
                print(f'Error downloading pages of category: {set['category']}')
                continue
        
        print("articles sets downloaded successfully")


    chain(
        get_articles_sets(),
        download_articles_sets()
        )


articles_dag()