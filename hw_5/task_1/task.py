import aiohttp
import aiofiles
import asyncio
from pathlib import Path
import uuid

async def download_image(session, url, folder, index):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                image_data = await response.read()
                filename = f"{uuid.uuid4().hex}.jpg"
                filepath = Path(folder) / filename
                async with aiofiles.open(filepath, 'wb') as f:
                    await f.write(image_data)
                print(f"[{index}] Downloaded: {filename}")
            else:
                print(f"[{index}] Failed with status: {response.status}")
    except Exception as e:
        print(f"[{index}] Exception: {e}")

async def main(num_images, folder):
    Path(folder).mkdir(parents=True, exist_ok=True)

    url = "https://thispersondoesnotexist.com/"
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_images):
            task = asyncio.create_task(download_image(session, url, folder, i + 1))
            tasks.append(task)
            await asyncio.sleep(1.5)
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    num_images = int(input("Enter the number of images to download: "))
    folder = input("Enter the destination folder: ")
    asyncio.run(main(num_images, folder))
