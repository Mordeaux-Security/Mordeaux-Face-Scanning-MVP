#!/usr/bin/env python3
import asyncio
import sys
sys.path.append('/app')
from new_crawler.http_utils import get_http_utils

async def main():
    http_utils = get_http_utils()
    
    # Initialize browser pool
    await http_utils._browser_pool.initialize_pool(1)
    
    try:
        print("Rendering wikifeet.com...")
        html = await http_utils._browser_pool.render_page('https://wikifeet.com')
        
        print(f"HTML length: {len(html)} chars")
        
        # Save HTML
        with open('/app/wikifeethtml.txt', 'w', encoding='utf-8') as f:
            f.write(html)
        
        print("HTML saved to /app/wikifeethtml.txt")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
