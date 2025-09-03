from airflow.sdk import task, dag, Param
from airflow.models.baseoperator import chain
import requests
from bs4 import BeautifulSoup
import pickle

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
}
session = requests.Session()


def get_subcats(cat):
    page = session.get(f'https://en.wikipedia.org/wiki/Category:{cat['category']}', headers=headers)
    src = page.content
    soup = BeautifulSoup(src, 'lxml')

    try:
        subcats_html = soup.find('div', {'id': 'mw-subcategories'}).find_all('a')
    except AttributeError:
        return []

    subcats = []
    for subcat in subcats_html:
        subcats.append({'category': subcat.text.strip().replace('\xa0', ' '),
                           'og_cat': cat['og_cat'],
                           'layer': cat['layer']+1})

    return subcats
    
@dag(
    dag_id='fetch_math_categories',
    params={'depth': Param(2, type='integer')},
    catchup=False
)


def categories_dag():

    @task
    def get_top_categories(**context):
        categories = []
        
        main_page = session.get('https://en.wikipedia.org/wiki/Portal:Mathematics', headers=headers)
        print("HTTP status:", main_page.status_code)

        src = main_page.content
        soup = BeautifulSoup(src, 'lxml')

        cats_html = soup.find_all('div', {'class': 'box-header-body'})[7].find('p').find_all('a')
        for cat in cats_html:
            categories.append({'category': cat.text.strip().replace('\xa0', ' '),
                            'og_cat': cat.text.strip().replace('\xa0', ' '),
                            'layer': 1})

        context['ti'].xcom_push(key='categories', value=categories)
        print('Layer 1 parsed')

    @task
    def get_subcategories(**context):
        categories = context['ti'].xcom_pull(key='categories', task_ids='get_top_categories')
        depth = context['params']['depth']

        last_cats_len = 0
        layer = 1

        while True:
            subcats = []
            if last_cats_len == len(categories) or layer == depth:
                break
            else:
                print(f"Looking into {len(categories) - last_cats_len} new categories")
                last_cats_len = len(categories)

                for cat in categories:
                    if cat['layer'] == layer:
                        subcats.append(get_subcats(cat))
                    else:
                        pass
                
                for subcat in subcats:
                    for sub in subcat:
                        if sub['category'] not in [cat['category'] for cat in categories]:
                            categories.append(sub)
                        else:
                            pass
                
                print(f'Layer {layer+1} parsed')
                layer += 1
        
        context['ti'].xcom_push(key='categories', value=categories)

    @task
    def save_categories(**context):
        depth = context['params']['depth']
        categories = context['ti'].xcom_pull(key='categories', task_ids='get_subcategories')
        print(f"Collected {len(categories)} Subcategories")

        with open(f'categories_depth_{depth}.pkl', 'wb') as file:
            pickle.dump(categories, file)


    chain(
        get_top_categories(),
        get_subcategories(),
        save_categories()
        )


categories_dag()