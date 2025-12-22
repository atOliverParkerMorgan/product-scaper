"""
ProductScraper: Main interface for web scraping and machine learning-based element detection.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import requests
import yaml
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from create_data import select_data
from train_model.predict_data import group_prediction_to_products, predict_category_selectors
from train_model.process_data import html_to_dataframe
from train_model.train_model import train_model
from utils.console import log_error, log_info, log_warning

PRODUCT_SCRAPER_SAVE_DIR = Path("product_scraper_data")


class ProductScraper:
    def __init__(
        self,
        categories: List[str],
        websites_urls: Optional[List[str]] = None,
        selectors: Optional[Dict[str, Dict[str, List[str]]]] = None,
        training_data: Optional[pd.DataFrame] = None,
        model: Optional[Dict[str, Any]] = None,
        pipeline: Optional[Any] = None,
    ):
        self.website_html_cache = {}
        self.website_cache_metadata = {}

        self._categories: Set[str] = set(categories)
        self._websites_urls: Set[str] = set(websites_urls) if websites_urls else set()

        # url -> {category: [selectors]}
        self.selectors: Dict[str, Dict[str, List[str]]] = selectors if selectors is not None else {}

        self.url_in_training_data = set()
        self.training_data = training_data

        self.model = model
        self.pipeline = pipeline

        # Iterator state
        self._iterator_index = 0
        self._iter_list = []

    @property
    def categories(self) -> List[str]:
        return sorted(list(self._categories))

    @property
    def websites_urls(self) -> List[str]:
        return sorted(list(self._websites_urls))

    def __iter__(self) -> "ProductScraper":
        self._iterator_index = 0
        self._iter_list = self.websites_urls
        return self

    def __next__(self) -> Tuple[str, Dict[str, Any]]:
        if self._iterator_index >= len(self._iter_list):
            self._iter_list = []
            raise StopIteration

        url = self._iter_list[self._iterator_index]
        self._iterator_index += 1
        predictions = self.get_selectors(url)
        return (url, predictions)

    def __len__(self) -> int:
        return len(self._websites_urls)

    def get_html(self, website_url: str, use_browser: bool = True) -> str:
        """
        Fetch HTML content. Uses Playwright first, falls back to requests.
        """
        if website_url in self.website_html_cache:
            return self.website_html_cache[website_url]

        if use_browser:
            try:
                with sync_playwright() as p:
                    # Launch headless for performance
                    browser = p.chromium.launch(headless=True)
                    try:
                        page = browser.new_page()
                        # networkidle ensures scripts have likely finished loading
                        page.goto(website_url, wait_until="networkidle", timeout=30000)
                        html_content = page.content()
                    finally:
                        browser.close()

                    self.website_html_cache[website_url] = html_content
                    return html_content

            except PlaywrightError as e:
                log_warning(f"Playwright error for {website_url}: {e}. Falling back to requests.")
                use_browser = False
            except Exception as e:
                log_warning(f"Unexpected error for {website_url}: {e}. Falling back to requests.")
                use_browser = False

        # Fallback to requests
        try:
            response = requests.get(website_url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (Bot)"})
            response.raise_for_status()
            html_content = response.text
            self.website_html_cache[website_url] = html_content
            return html_content

        except requests.RequestException as e:
            log_error(f"Request error for {website_url}: {e}")
            raise

    def set_pipeline(self, pipeline: Any) -> None:
        """Set a custom sklearn pipeline."""
        self.pipeline = pipeline

    def add_website(self, website_url: str) -> None:
        """Add a new website URL to the configured set."""
        if website_url not in self._websites_urls:
            self._websites_urls.add(website_url)
        else:
            log_warning(f"Website URL {website_url} already in configured list")

    def remove_website(self, website_url: str) -> None:
        """Remove a website URL and clean up all associated data."""
        if website_url in self._websites_urls:
            self._websites_urls.remove(website_url)
            self.selectors.pop(website_url, None)
            self.url_in_training_data.discard(website_url)
            self.website_html_cache.pop(website_url, None)

            if self.training_data is not None and not self.training_data.empty and "SourceURL" in self.training_data.columns:
                self.training_data = self.training_data[self.training_data["SourceURL"] != website_url]
                self.training_data.reset_index(drop=True, inplace=True)
        else:
            log_warning(f"Website URL {website_url} not found")

    def set_website_selectors_from_yaml(self, website_url: str, yaml_path: str) -> None:
        """Load and set element selectors for a specific website URL from a YAML file."""
        try:
            with open(yaml_path, "r") as f:
                selectors = yaml.safe_load(f)
            self.selectors[website_url] = selectors
            # Ensure URL is in our set
            self.add_website(website_url)
        except Exception as e:
            log_error(f"Failed to load selectors from {yaml_path}: {e}")

    def set_website_selectors(self, website_url: str, selectors: Dict[str, List[str]]) -> None:
        """Set element selectors for a specific website URL manually."""
        self.selectors[website_url] = selectors
        self.add_website(website_url)

    def create_selectors(self, website_url: str, save: bool = True) -> Dict[str, List[str]]:
        """
        Interactively select elements for training data on a single URL.
        """
        # Ensure URL is tracked
        self.add_website(website_url)

        if website_url in self.selectors:
            return self.selectors[website_url]

        try:
            data = select_data(self, website_url)
        except Exception as e:
            log_error(f"Failed to create selectors for {website_url}: {e}. Skipping.")
            return {}

        if len(data) == 0:
            log_warning(f"No selectors were created for {website_url}.")
            return {}

        self.selectors[website_url] = data

        self.create_training_data([website_url])
        self.train_model()

        if save:
            self.save()
            self.save_selectors()
            self.save_training_data()

        return data

    def create_all_selectors(self) -> Dict[str, Dict[str, List[str]]]:
        """Interactively collect selectors for all configured websites."""
        # Iterate over a list copy to allow modification if needed
        for url in self.websites_urls:
            self.create_selectors(url)
        return self.selectors

    def add_websites(self, website_urls: List[str]) -> None:
        """Add multiple website URLs."""
        for url in website_urls:
            self.add_website(url)

    def add_category(self, category: str) -> None:
        """Add a new data category to extract."""
        if category not in self._categories:
            self._categories.add(category)
        else:
            log_warning(f"Category '{category}' already exists")

    def add_categories(self, categories: List[str]) -> None:
        """Add new data categories to extract."""
        for category in categories:
            if category not in self._categories:
                self._categories.add(category)
            else:
                log_warning(f"Category '{category}' already exists")

    def create_training_data(self, websites_to_use: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """Convert collected selectors into training dataframe."""
        all_data = []

        # If urls is None, use all tracked websites
        target_urls = websites_to_use if websites_to_use is not None else self.websites_urls

        for url in target_urls:
            if url in self.url_in_training_data:
                continue

            if url not in self._websites_urls:
                log_warning(f"URL {url} not in configured websites. Skipping.")
                continue

            try:
                html_content = self.get_html(url)
            except requests.RequestException:
                log_warning(f"Skipping {url} due to network error.")
                continue

            if url not in self.selectors:
                try:
                    self.create_selectors(url)
                except Exception as e:
                    log_warning(f"Failed to create selectors for {url}: {e}")
                    continue

            selectors = self.selectors.get(url, {})
            if not selectors:
                log_warning(f"No selectors found for {url}, skipping.")
                continue

            try:
                df = html_to_dataframe(html_content, selectors, url=url)
                if not df.empty:
                    all_data.append(df)
                    self.url_in_training_data.add(url)
                    log_info(f"Extracted samples from {url}")
                else:
                    log_warning(f"No data extracted from {url}")
            except Exception as e:
                log_warning(f"Error processing {url}: {e}")
                continue

        if all_data:
            if self.training_data is not None and not self.training_data.empty:
                all_data.insert(0, self.training_data)
            self.training_data = pd.concat(all_data, ignore_index=True)
        elif self.training_data is None:
            log_warning("No data was successfully extracted from any URL")

        return self.training_data

    def train_model(self, create_data: bool = False, test_size: float = 0.2, min_samples: int = 5) -> None:
        """
        Train the machine learning model with proper Test/Train splitting.
        """
        if create_data:
            log_info("Creating training dataframe from selectors...")
            self.create_training_data()

        if self.training_data is None or self.training_data.empty:
            log_error("No training data available. Please collect selectors or load data.")
            return

        if len(self.training_data) < min_samples:
            log_warning(f"Insufficient training data: {len(self.training_data)} samples.")
            return

        self.model = train_model(self.training_data, self.pipeline, test_size=test_size)

    def predict_category(self, website_url: str, category: str) -> List[Dict[str, Any]]:
        if self.model is None:
            raise ValueError("Model is not trained.")
        if category not in self._categories:
            raise ValueError(f"Category '{category}' is not configured.")

        html = self.get_html(website_url)
        existing = self.selectors.get(website_url, None)
        return predict_category_selectors(self.model, html, category, existing_selectors=existing)

    def predict(self, website_urls: List[str]) -> List[Dict[str, List[Dict[str, Any]]]]:
        """Predict element selectors for all configured categories."""
        if not website_urls:
            raise ValueError("Please provide a list of website URLs.")

        all_products: List[Dict[str, List[Dict[str, Any]]]] = []
        for website_url in website_urls:
            raw_predictions = self.get_selectors(website_url)
            html_content = self.get_html(website_url)

            # Access via property to get list
            products = group_prediction_to_products(html_content, raw_predictions, self.categories)
            all_products.append({website_url: products})

        return all_products

    def get_selectors(self, website_url: str) -> Dict[str, Any]:
        """Get or predict selectors for a URL."""
        if website_url not in self.selectors:
            result = {}
            for category in self.categories:
                result[category] = self.predict_category(website_url, category)
            return result
        return self.selectors.get(website_url, {})

    # --- Storage Methods (Unchanged) ---
    def save_model(self, path: str = "model.pkl") -> None:
        PRODUCT_SCRAPER_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            with open(PRODUCT_SCRAPER_SAVE_DIR / path, "wb") as f:
                pickle.dump(self.model, f)
        else:
            raise ValueError("Model is not trained.")

    def load_model(self, path: str = "model.pkl") -> None:
        try:
            with open(PRODUCT_SCRAPER_SAVE_DIR / path, "rb") as f:
                self.model = pickle.load(f)
        except Exception as e:
            log_error(f"Failed to load model from {path}: {e}")

    def save_training_data(self, path: str = "training_data.csv") -> None:
        PRODUCT_SCRAPER_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        if self.training_data is not None:
            self.training_data.to_csv(PRODUCT_SCRAPER_SAVE_DIR / path, index=False)
        else:
            raise ValueError("Dataframe is empty.")

    def load_dataframe(self, path: str = "training_data.csv") -> None:
        try:
            self.training_data = pd.read_csv(PRODUCT_SCRAPER_SAVE_DIR / path)
        except Exception as e:
            log_error(f"Failed to load training data from {path}: {e}")

    def save_selectors(self, path: str = "selectors.yaml") -> None:
        PRODUCT_SCRAPER_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        with open(PRODUCT_SCRAPER_SAVE_DIR / path, "w") as f:
            yaml.dump(self.selectors, f, default_flow_style=False, allow_unicode=True)

    def load_selectors(self, path: str = "selectors.yaml") -> None:
        try:
            with open(PRODUCT_SCRAPER_SAVE_DIR / path, "r") as f:
                self.selectors = yaml.safe_load(f)
        except Exception as e:
            log_error(f"Failed to load selectors from {path}: {e}")

        # Ensure loaded URLs are tracked in the set
        for url in self.selectors.keys():
            self.add_website(url)

    def save(self, path: str = "product_scraper.pkl") -> None:
        PRODUCT_SCRAPER_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        with open(PRODUCT_SCRAPER_SAVE_DIR / path, "wb") as f:
            pickle.dump(self, f)
        self.save_selectors()
        self.save_training_data()

    @staticmethod
    def load(path: str = "product_scraper.pkl") -> "ProductScraper":
        with open(PRODUCT_SCRAPER_SAVE_DIR / path, "rb") as f:
            return pickle.load(f)
