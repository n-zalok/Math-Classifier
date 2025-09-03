import xml.etree.ElementTree as ET
import mwparserfromhell
import json
import sys
import os

directory_path = './airflow/data_2' 

counter = 0
for entry in os.scandir(directory_path):
    print(entry.name)
sys.exit(0)

# Load XML file
tree = ET.parse("template.xml")
root = tree.getroot()

ns = {"ns": "http://www.mediawiki.org/xml/export-0.11/"}  # XML namespace

articles = []

for page in root.findall("ns:page", ns):
    title = page.find("ns:title", ns).text
    text_elem = page.find("ns:revision/ns:text", ns)
    if text_elem is None or text_elem.text is None:
        continue

    wiki_markup = text_elem.text
    wikicode = mwparserfromhell.parse(wiki_markup)

    sections = {}
    current_section = "Introduction"
    sections[current_section] = []

    for node in wikicode.nodes:
        if node.__class__.__name__ == "Heading":
            current_section = str(node.title).strip()
            sections[current_section] = []
        else:
            # Skip images/thumbnails entirely
            node_str = str(node)
            if node_str.startswith("[[File:") or node_str.startswith("[[Image:"):
                continue
            sections[current_section].append(node_str)

    # Clean markup in each section
    clean_sections = {}
    for sec, content in sections.items():
        clean_sections[sec] = mwparserfromhell.parse("".join(content)).strip_code()

    articles.append({
        "title": title,
        "sections": clean_sections
    })

# Save result as JSON
with open("clean_articles_template.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, indent=2, ensure_ascii=False)