from airflow.sdk import task, dag, Param
from airflow.models.baseoperator import chain
import requests
from bs4 import BeautifulSoup
import pickle
import os
from tqdm import tqdm
import math

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
}
session = requests.Session()

    
@dag(
    dag_id='fetch_math_articles',
    params={'depth': Param(2, type='integer')},
    catchup=False
)


def articles_dag():

    @task
    def get_articles_sets(**context):
        depth = context['params']['depth']

        with open(f'categories_depth_{depth}.pkl', 'rb') as file:
            categories = pickle.load(file)
        num_cats = len(categories)
        print(f'{num_cats} categories loaded successfully')

        percent = 1
        articles_sets = []
        for i, cat in enumerate(categories):
            page = session.get(f'https://en.wikipedia.org/w/index.php?title=Special:Export&addcat&catname={cat['category']}&curonly=1', headers=headers)
            src = page.content
            soup = BeautifulSoup(src, 'lxml')

            if (((i+1)/num_cats)*100) >= percent:
                tqdm.write(f"{percent}% completed")
                percent  = math.ceil(((i+1)/num_cats)*100)
            else:
                pass

            try:
                articles_set = soup.find('textarea', {'id': 'ooui-php-2'}).text.strip()
                cat['articles_set'] = articles_set.replace(' ', '\n')
                articles_sets.append(cat)
            except:
                print(f'Error collecting pages of category: {cat['category']}')
                continue

        context['ti'].xcom_push(key='articles_sets', value=articles_sets)
        print(f'{len(articles_sets)} articles sets created successfully')
    

    @task
    def download_articles_sets(**context):
        depth = context['params']['depth']
        articles_sets = context['ti'].xcom_pull(key='articles_sets', task_ids='get_articles_sets')
        num_sets = len(articles_sets)

        percent = 1
        for i, set in enumerate(articles_sets):
            path = f'./data_{depth}/{set['og_cat']}/{set['layer']}'
            os.makedirs(path, exist_ok=True)

            if (((i+1)/num_sets)*100) >= percent:
                tqdm.write(f"{percent}% completed")
                percent  = math.ceil(((i+1)/num_sets)*100)
            else:
                pass
        
            data = {
                    "catname": set["category"],
                    "curonly": "1",
                    "pages": set["articles_set"],
                    "wpDownload": "1",
                    "title": "Special:Export",
                    "wpEditToken": "+\\"
                }
            try:
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