import asyncio
import aiohttp
import json
import time
import schedule
from bs4 import BeautifulSoup
from typing import List, Dict

AVITO_URL = 'https://www.avito.ru/moskva/kvartiry/sdam/na_dlitelnyy_srok'

async def fetch(session: aiohttp.ClientSession, url: str) -> str:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/122.0.0.0 Safari/537.36',
        'Accept-Language': 'ru-RU,ru;q=0.9'
    }
    async with session.get(url, headers=headers) as response:
        return await response.text()

async def parse_avito(session: aiohttp.ClientSession, url: str) -> List[Dict]:
    html = await fetch(session, url)
    soup = BeautifulSoup(html, 'html.parser')
    listings = []

    items = soup.find_all('div', {'data-marker': 'item'})
    print(f"🔍 Найдено {len(items)} объявлений на Avito")

    for item in items:
        title_tag = item.select_one('h3')
        price_tag = item.select_one('[data-marker="item-price"]')
        link_tag = item.select_one('a')
        address_tag = item.select_one('[data-marker="item-address"]')

        title = title_tag.get_text(strip=True) if title_tag else 'N/A'
        price = price_tag.get_text(strip=True) if price_tag else 'N/A'
        url = 'https://avito.ru' + link_tag['href'] if link_tag and link_tag.get('href') else 'N/A'
        address = address_tag.get_text(strip=True) if address_tag else 'N/A'

        listings.append({
            'title': title,
            'price': price,
            'address': address,
            'url': url
        })

    return listings

def save_to_json(data: List[Dict], filename: str):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 Сохранено {len(data)} объявлений в {filename}")

async def run_scraper():
    async with aiohttp.ClientSession() as session:
        avito_data = await parse_avito(session, AVITO_URL)
        save_to_json(avito_data, 'avito_listings.json')

def job():
    print(f"\n🔄 Запуск скрапинга: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    asyncio.run(run_scraper())

def start_scheduler():
    schedule.every(30).minutes.do(job)
    job()
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    start_scheduler()


