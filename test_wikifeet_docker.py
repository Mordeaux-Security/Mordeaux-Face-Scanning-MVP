#!/usr/bin/env python3
import asyncio
import sys
sys.path.append('/app')
from new_crawler.http_utils import get_http_utils
import re

async def test():
    http_utils = get_http_utils()
    
    # Initialize browser pool
    await http_utils._browser_pool.initialize_pool(1)
    
    html = await http_utils._browser_pool.render_page('https://wikifeet.com')
    print(f'HTML length: {len(html)}')
    
    # Save HTML
    with open('/tmp/wikifeet.html', 'w') as f:
        f.write(html)
    print('HTML saved to /tmp/wikifeet.html')
    
    # Find img tags
    img_tags = re.findall(r'<img[^>]*>', html, re.IGNORECASE)
    print(f'Found {len(img_tags)} img tags')
    
    # Show first 3
    for i, tag in enumerate(img_tags[:3]):
        print(f'{i+1}: {tag}')
    
    # Find picture tags
    picture_tags = re.findall(r'<picture[^>]*>.*?</picture>', html, re.IGNORECASE | re.DOTALL)
    print(f'Found {len(picture_tags)} picture tags')
    
    # Find data attributes
    data_attrs = re.findall(r'data-[a-zA-Z-]+="[^"]*"', html, re.IGNORECASE)
    unique_data_attrs = set()
    for attr in data_attrs:
        attr_name = attr.split('=')[0].lower()
        unique_data_attrs.add(attr_name)
    
    print('Unique data attributes:')
    for attr in sorted(unique_data_attrs):
        print(f'  {attr}')

if __name__ == "__main__":
    asyncio.run(test())
