from airflow.sdk import task, dag, Param
from airflow.models.baseoperator import chain
import os
import xml.etree.ElementTree as ET
import mwparserfromhell
from bs4 import BeautifulSoup
import pickle
import json
from tqdm import tqdm
import math

    
@dag(
    dag_id='clean_math_articles',
    params={'depth': Param(2, type='integer')},
    catchup=False
)


def clean_dag():

    @task
    def clean_articles(**context):
        depth = context['params']['depth']
        attachments = ["See also", "Notes", "References", "External links", "Further reading",
                       "Sources", "Citations", "Categories", "Bibliography"]

        with open(f'categories_depth_{depth}.pkl', 'rb') as file:
            categories = pickle.load(file)
        num_cats = len(categories)

        total = 0
        success = 0
        
        percent = 1
        for category in os.scandir(f'./data_{depth}'):

            articles = []
            for layer in os.scandir(category):
                for file in os.scandir(layer):
                    total += 1

                    if (((total)/num_cats)*100) >= percent:
                        tqdm.write(f"{percent}% completed")
                        percent  = math.ceil(((total)/num_cats)*100)
                    else:
                        pass


                    try:
                        tree = ET.parse(file)
                        root = tree.getroot()

                        ns = {"ns": "http://www.mediawiki.org/xml/export-0.11/"}  # XML namespace

                        for page in root.findall("ns:page", ns):
                            title = page.find("ns:title", ns).text
                            if title.startswith("Category:"):
                                continue
                            else:
                                pass

                            text_elem = page.find("ns:revision/ns:text", ns)
                            if text_elem is None or text_elem.text is None:
                                continue
                            else:
                                pass

                            wiki_markup = text_elem.text
                            wikicode = mwparserfromhell.parse(wiki_markup)

                            sections = {}
                            current_section = "Introduction"
                            sections[current_section] = []

                            for node in wikicode.nodes:
                                if node.__class__.__name__ == "Heading":
                                    current_section = str(node.title).strip()

                                    soup = BeautifulSoup(current_section, "html.parser")
                                    current_section = soup.get_text(" ", strip=True)

                                    sections[current_section] = []
                                else:
                                    # Skip images/thumbnails entirely
                                    node_str = str(node)
                                    if node_str.startswith("[[File:") or node_str.startswith("[[Image:"):
                                        continue
                                    else:
                                        soup = BeautifulSoup(node_str, "html.parser")
                                        node_str = soup.get_text(" ", strip=True)

                                        sections[current_section].append(node_str)

                            # Clean markup in each section
                            clean_sections = {}
                            for sec, content in sections.items():
                                if sec not in attachments:
                                    clean_sections[sec] = mwparserfromhell.parse("".join(content)).strip_code()
                                else:
                                    pass
                            
                            text = ''
                            for section in clean_sections.values():
                                if section:
                                    text += section
                                    text += ' '
                                else:
                                    pass
                            
                            if len(text.split()) < 200:
                                articles.append({
                                    "title": title,
                                    "sub_title": None,
                                    "text": text,
                                    "category": category.name,
                                    "layer": layer.name
                                })
                            else:
                                for sub_title, txt in clean_sections.items():
                                    if txt:
                                        articles.append({
                                            "title": title,
                                            "sub_title": sub_title,
                                            "text": txt,
                                            "category": category.name,
                                            "layer": layer.name
                                        })
                                    else:
                                        pass
                        
                        success += 1
                        
                    except:
                        pass

            # Save result as JSON
            save_path = f'cleaned_data_{depth}'
            os.makedirs(save_path, exist_ok=True)

            with open(f'{save_path}/{category.name}.json', "w", encoding="utf-8") as f:
                json.dump(articles, f, indent=2, ensure_ascii=False)
            print(f'Cleaned and Saved category: {category.name}')
        
        print(f'Cleaned {(success/total):.2%} of categories')
        print(f"Saved cleaned articles at {save_path}")
                        
                    
    chain(
        clean_articles()
        )


clean_dag()