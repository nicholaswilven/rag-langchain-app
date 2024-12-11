import time
import re
import logging

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

logger = logging.getLogger(__name__)

def get_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=chrome_options)     
    return driver 

def get_coordinates(location, max_retries = 3):
    retries = 0
    while retries < max_retries:
        try:
            driver = get_driver()
            pattern = r'@(-?\d+\.\d+),(-?\d+\.\d+)'
            driver.get(f'https://www.google.com/maps/place/{location.replace(" ","+")}')
            time.sleep(4)
            current_url = driver.current_url
            matches = re.search(pattern, current_url)
            if matches:
                latitude = matches.group(1)
                longitude = matches.group(2)
                result = [latitude, longitude]
                retries = max_retries
            else:
                raise ValueError
        except:
            retries += 1
            if retries == max_retries:
                logger.error(f"Scrape coordinates for {location} failed after retrying {max_retries} times")
        finally:
            driver.quit()                
            time.sleep(1)
    return result
