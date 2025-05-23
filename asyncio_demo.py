import asyncio
import time
from time import sleep

async def fetch_data(url):
    
    print(f"Fetching data from {url}")
    await asyncio.sleep(3)
    print(f"Data fetched from {url}")
    return "Data"


async def main():
    start_time = time.time()
    url = "https://www.google.com"
    task1 = asyncio.create_task(fetch_data(url))
    print(1)
    task2 = asyncio.create_task(fetch_data(url))
    print(2)
    task3 = asyncio.create_task(fetch_data(url))
    print(3)

    await


    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


if __name__ == "__main__":
    asyncio.run(main())
