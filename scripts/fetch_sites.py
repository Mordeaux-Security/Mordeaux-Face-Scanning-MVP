import sys
import os
import re
import requests

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive',
}

def fetch_wikifeet():
    url = 'https://wikifeet.com'
    print(f'Fetching WikiFeet: {url}')
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        print(f'WikiFeet status: {resp.status_code}')
        html = resp.text
        out_path = 'wikifeet_fetch.html'
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f'Saved HTML to {out_path} ({len(html):,} chars)')
        # Quick analysis: count img/picture/source
        img_count = len(re.findall(r'<img\b', html, re.IGNORECASE))
        picture_count = len(re.findall(r'<picture\b', html, re.IGNORECASE))
        source_count = len(re.findall(r'<source\b', html, re.IGNORECASE))
        print(f'Counts -> img: {img_count}, picture: {picture_count}, source: {source_count}')
        # Show first 5 img tags
        imgs = re.findall(r'<img[^>]*>', html, re.IGNORECASE)
        for i, tag in enumerate(imgs[:5]):
            print(f'IMG[{i+1}]: {tag[:200]}')
    except Exception as e:
        print(f'WikiFeet fetch error: {e}')

def fetch_candidshiny_root():
    url = 'https://candidshiny.com'
    print(f'Fetching CandidShiny (no redirects): {url}')
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20, allow_redirects=False)
        print(f'CandidShiny status: {resp.status_code}')
        loc = resp.headers.get('Location') or resp.headers.get('location')
        if loc:
            print(f'Redirect Location: {loc}')
        if resp.status_code == 200 and resp.text:
            html = resp.text
            out_path = 'candidshiny_fetch.html'
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f'Saved HTML to {out_path} ({len(html):,} chars)')
            img_count = len(re.findall(r'<img\b', html, re.IGNORECASE))
            print(f'img tags (no-redirect fetch): {img_count}')
        else:
            print('No body saved due to redirect or non-200 status.')
    except Exception as e:
        print(f'CandidShiny fetch error: {e}')

def probe_candidshiny_redirects():
    paths = ['/', '/categories', '/new/', '/best/']
    base = 'https://candidshiny.com'
    print('Probing CandidShiny subpaths (no redirects):')
    for p in paths:
        url = base + p if p != '/' else base
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20, allow_redirects=False)
            loc = resp.headers.get('Location') or resp.headers.get('location')
            print(f'  {url} -> {resp.status_code}  Location: {loc}')
        except Exception as e:
            print(f'  {url} error: {e}')

if __name__ == '__main__':
    fetch_wikifeet()
    print('-' * 60)
    fetch_candidshiny_root()
    print('-' * 60)
    probe_candidshiny_redirects()
