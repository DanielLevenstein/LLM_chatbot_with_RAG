import glob
import json
import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
from collections import deque
feature_level_page_skip = False
BASE_DOMAIN = "docs.aws.amazon.com"
SCHEME = "https"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (DepthCrawler-RAG)"
}

def normalize_url(url):
    parsed = urlparse(url)
    parsed = parsed._replace(fragment="", query="")
    return urlunparse(parsed)


def is_valid(url):
    return urlparse(url).netloc.endswith(BASE_DOMAIN)


def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    main = soup.find("main") or soup.body
    if not main:
        return ""

    text = main.get_text(separator="\n", strip=True)
    lines = [l.strip() for l in text.split("\n")]
    return "\n".join([l for l in lines if l])


def extract_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        url = normalize_url(urljoin(base_url, a["href"]))
        if is_valid(url):
            links.append(url)

    return links


def save(path, text, url):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Source\n{url}\n\n")
        f.write(text)

def crawl(start_url, feature_name, max_depth=2, max_pages=100, delay=0.5):
    visited = set()
    start_url = normalize_url(start_url)
    queued = {start_url}
    queue = deque([(start_url, 0)])
    os.makedirs("data/", exist_ok=True)
    print(f"Crawling {start_url} for {feature_name}:")
    page_id = 0
    downloaded = 0

    with requests.Session() as session:
        session.headers.update(HEADERS)
        while queue and page_id < max_pages:
            url, depth = queue.popleft()

            if url in visited:
                continue

            if depth > max_depth:
                continue
            existing_files = glob.glob(f"data/{feature_name}*")
            if feature_level_page_skip and existing_files:
                print(f"Skipping {url} as files for {feature_name} already exist.")
                continue
            try:
                with session.get(url, timeout=15) as resp:
                    if resp.status_code != 200:
                        print(f"ERROR: {resp.status_code}: {url}")
                        continue

                    html = resp.text
                url_parts = urlparse(url)
                # print(f"Saving {url_parts}")
                url_path = url_parts.path.replace("/", "_")
                path = f"data/{feature_name}{url_path}.md"
                skip_download = False
                if os.path.exists(path):
                    print(f"Skipping {url} as file already exists.")
                    skip_download = True
                    # continue
                else:
                    print(f"[depth={depth}] {url}")
                text = extract_text(html)
                visited.add(url)

                if len(text) > 300:
                    if not skip_download:
                        print(f"Saving {url} to {path}")
                        save(path, text, url)
                        downloaded += 1
                    page_id += 1

                if depth <= max_depth:
                    for link in extract_links(html, url):
                        if link not in queued and link not in visited:
                            queued.add(link)
                            queue.append((link, depth + 1))

                time.sleep(delay)

            except Exception as e:
                print(f"Error {url}: {e}")
    print(f"Downloading {downloaded} of {page_id} pages for {feature_name}")

def write_to_json(features, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=4)

def read_from_json(filepath):
    try:
        print(f"Reading {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        if filepath.endswith("current.json"):
            return {"cli": 1}
        else:
            return {"cli": 0}


def run():
    current_features = read_from_json("config/features_current.json")
    write_to_json(current_features, "config/features_current.json")
    previous_features = read_from_json("config/features_downloaded.json")
    for f in current_features:
        if f not in previous_features:
            previous_features[f] = 0
        if previous_features[f] < current_features[f]:
            crawl(f"{SCHEME}://{BASE_DOMAIN}/{f}", feature_name=f , max_depth=current_features.get(f), max_pages=80)
            previous_features[f] = current_features[f]
            write_to_json(previous_features, "config/features_downloaded.json")
        else :
            print(f"Skipping {f} as it has not been updated.")
    print(f"Finished AWS documentation for the following features:", current_features.keys())
    print("If this is your first time running this script please update 'config/features_current.json' with the features you want to crawl.")
if __name__ == "__main__":
    run()
