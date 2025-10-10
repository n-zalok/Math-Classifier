from airflow.sdk import task, dag, Param
from airflow.models.baseoperator import chain
import requests
from bs4 import BeautifulSoup
import pickle

# Mimic a browser visit
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
}
session = requests.Session()

# Get subcategories of a given category
def get_subcats(cat):
    # Get the category page
    page = session.get(f'https://en.wikipedia.org/wiki/Category:{cat['category']}', headers=headers)
    src = page.content
    soup = BeautifulSoup(src, 'lxml')

    # Find div of subcategories
    try:
        subcats_html = soup.find('div', {'id': 'mw-subcategories'}).find_all('a')
    except AttributeError:
        return []

    # Extract info from html
    subcats = []
    for subcat in subcats_html:
        subcats.append({'category': subcat.text.strip().replace('\xa0', ' '),
                           'og_cat': cat['og_cat'],
                           'layer': cat['layer']+1})

    return subcats

# Depth parameter controls how deep the DAG should go in category hierarchy
@dag(
    dag_id='fetch_math_categories',
    params={'depth': Param(2, type='integer')},
    catchup=False
)


def categories_dag():

    @task
    # Get top-level math categories from the main math portal page
    def get_top_categories(**context):
        categories = []
        
        # Get the main math portal page
        main_page = session.get('https://en.wikipedia.org/wiki/Portal:Mathematics', headers=headers)
        src = main_page.content
        soup = BeautifulSoup(src, 'lxml')

        # Find div of top-level categories 
        cats_html = soup.find_all('div', {'class': 'box-header-body'})[7].find('p').find_all('a')

        # Extract info from html
        for cat in cats_html:
            categories.append({'category': cat.text.strip().replace('\xa0', ' '),
                            'og_cat': cat.text.strip().replace('\xa0', ' '),
                            'layer': 1})

        # Push initial categories to XCom
        context['ti'].xcom_push(key='categories', value=categories)
        print('Layer 1 parsed')

    @task
    # Recursively get subcategories up to the specified depth
    def get_subcategories(**context):
        # Pull top categories and depth from XCom
        categories = context['ti'].xcom_pull(key='categories', task_ids='get_top_categories')
        depth = context['params']['depth']

        # Number of categories in the last iteration
        last_cats_len = 0
        # Current layer being processed
        layer = 1

        while True:
            subcats = []

            # If no new categories were found or the desired depth is reached, stop
            if last_cats_len == len(categories) or layer == depth:
                break
            else:
                print(f"Looking into {len(categories) - last_cats_len} new categories")
                last_cats_len = len(categories)

                for cat in categories:
                    # Only process categories at the current layer
                    if cat['layer'] == layer:
                        subcats.append(get_subcats(cat))
                    else:
                        pass
                
                for subcat in subcats:
                    for sub in subcat:
                        # Avoid duplicates
                        if sub not in categories:
                            categories.append(sub)
                        else:
                            pass
                
                print(f'Layer {layer+1} parsed')
                layer += 1
        
        # Push all collected categories to XCom
        context['ti'].xcom_push(key='categories', value=categories)

    @task
    # Save the collected categories to a pickle file
    def save_categories(**context):
        # Pull categories and depth from XCom
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