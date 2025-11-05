import re
import requests
import random

# --- A list of common User-Agents for rotation ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    # "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    # "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
    # "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    # "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36 Edg/117.0.2045.60",
]


def __create_session() -> requests.Session:
    """
    Creates a requests session with standard headers and a rotated User-Agent.
    
    A User-Agent is selected at random from the global USER_AGENTS list
    to help mimic different browsers and reduce the chance of being
    blocked by basic anti-scraping measures.

    Returns:
        requests.Session: An initialized requests session object.
    """
    session = requests.Session()
    
    # --- This implements the rotation ---
    random_user_agent = random.choice(USER_AGENTS)
    
    session.headers.update(
        {
            "User-Agent": random_user_agent, # Use the randomly selected agent
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }
    )
    return session

    
def __get_raw_html(url: str, session: requests.Session) -> str:
    """
    Fetch raw HTML using a provided session.
    
    Args:
        url: The URL to fetch.
        session: An existing requests.Session object.

    Returns:
        The raw HTML content as a string.
        
    Raises:
        Exception: If the request fails.
    """
    try:
        # Use the passed-in session instead of creating a new one
        response = requests.get(url, timeout=15) # Reduced timeout for faster fails
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        
        # response.text handles decoding (e.g., from UTF-8) for us
        return response.text
    except requests.RequestException as e:
        # Provide a more specific error message
        raise Exception(f"Failed to fetch HTML from {url}: {e}")
    

PAIRED_UNWANTED_TAGS = ['script', 'style', 'noscript', 'iframe', 'footer', 'header', 'nav', 'aside', 'form', 'head']
UNPAIRED_UNWANTED_TAGS = ['br', 'hr']

def __proccess_raw_html(raw_html: str) -> str:
    """
    Processes raw HTML to remove unwanted tags and their content.
    
    - For paired tags (e.g., 'script'), it removes the tag and everything between
      the opening and closing tags.
    - For unpaired tags (e.g., 'br'), it just removes the tag itself.
    - Handles extra whitespace inside tags (e.g., < script > or </ script >).
    """
    
    processed_html = raw_html

    # Remove paired tags and their content
    for tag in PAIRED_UNWANTED_TAGS:
        processed_html = re.sub(rf'<\s*{tag}\b[^>]*>.*?<\s*/{tag}\s*>', '', processed_html, flags=re.DOTALL | re.IGNORECASE)

    # Remove unpaired tags
    for tag in UNPAIRED_UNWANTED_TAGS:
        processed_html = re.sub(rf'<\s*{tag}\b[^>]*?>', '', processed_html, flags=re.IGNORECASE)

    # Remove comments
    processed_html = re.sub(r'<!--.*?-->', '', processed_html, flags=re.DOTALL)

    return processed_html





# --- Example Usage ---
if __name__ == "__main__":
    # Test with a sample HTML string
    session = __create_session()

    test_html = __get_raw_html('https://www.artonpaper.ch/', session)


    processed_html = __proccess_raw_html(test_html)
    print(processed_html)
        
   
