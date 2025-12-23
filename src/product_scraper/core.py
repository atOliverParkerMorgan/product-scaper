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

from product_scraper.create_data import select_data
from product_scraper.train_model.predict_data import (
    group_prediction_to_products,
    predict_category_selectors,
)
from product_scraper.train_model.process_data import html_to_dataframe
from product_scraper.train_model.train_model import train_model
from product_scraper.utils.console import log_error, log_info, log_warning

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
        """
        Initialize the ProductScraper instance.

        Args:
            categories (List[str]): List of product categories to scrape (e.g., 'price', 'title').
            websites_urls (Optional[List[str]]): List of initial website URLs to track.
            selectors (Optional[Dict[str, Dict[str, List[str]]]]): Dictionary of existing selectors mapping URL -> Category -> XPaths.
            training_data (Optional[pd.DataFrame]): Pre-loaded training data DataFrame.
            model (Optional[Dict[str, Any]]): Pre-trained model dictionary.
            pipeline (Optional[Any]): Sklearn pipeline for model training/prediction.

        Returns:
            None
        """
        self.website_html_cache = {}
        self.website_cache_metadata = {}

        self._categories: Set[str] = set(categories)
        self._websites_urls: Set[str] = set(websites_urls) if websites_urls else set()

        # url -> {category: [selectors]}
        self.selectors: Dict[str, Dict[str, List[str]]] = (
            selectors if selectors is not None else {}
        )
        self.predicted_selectors: Dict[str, List[Dict[str, Any]]] = {}

        self.url_in_training_data = set()
        self.training_data = training_data

        self.model = model
        self.pipeline = pipeline

        # Iterator state
        self._iterator_index = 0
        self._iter_list = []

    @property
    def categories(self) -> List[str]:
        """
        Get the sorted list of configured categories.

        Args:
            None

        Returns:
            List[str]: Sorted list of category names.
        """
        return sorted(list(self._categories))

    @property
    def websites_urls(self) -> List[str]:
        """
        Get the sorted list of tracked website URLs.

        Args:
            None

        Returns:
            List[str]: Sorted list of website URLs.
        """
        return sorted(list(self._websites_urls))

    def __iter__(self) -> "ProductScraper":
        """
        Initialize the iterator for iterating over websites.

        Args:
            None

        Returns:
            ProductScraper: The scraper instance itself.
        """
        self._iterator_index = 0
        self._iter_list = self.websites_urls
        return self

    def __next__(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get the next website URL and its predicted selectors.

        Args:
            None

        Returns:
            Tuple[str, Dict[str, Any]]: A tuple containing the URL and a dictionary of predictions.

        Raises:
            StopIteration: When there are no more websites to iterate over.
        """
        if self._iterator_index >= len(self._iter_list):
            self._iter_list = []
            raise StopIteration

        url = self._iter_list[self._iterator_index]
        self._iterator_index += 1
        predictions = self.get_selectors(url)
        return (url, predictions)

    def __len__(self) -> int:
        """
        Get the number of tracked websites.

        Args:
            None

        Returns:
            int: The count of websites.
        """
        return len(self._websites_urls)

    def get_html(self, website_url: str, use_browser: bool = True) -> str:
        """
        Fetch HTML content from a URL, using Playwright or requests.

        Args:
            website_url (str): The URL to fetch.
            use_browser (bool): Whether to use Playwright (True) or requests (False).

        Returns:
            str: The HTML content of the page.
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
                log_warning(
                    f"Playwright error for {website_url}: {e}. Falling back to requests."
                )
                use_browser = False
            except Exception as e:
                log_warning(
                    f"Unexpected error for {website_url}: {e}. Falling back to requests."
                )
                use_browser = False

        # Fallback to requests
        try:
            response = requests.get(
                website_url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (Bot)"}
            )
            response.raise_for_status()
            html_content = response.text
            self.website_html_cache[website_url] = html_content
            return html_content

        except requests.RequestException as e:
            log_error(f"Request error for {website_url}: {e}")
            raise

    def set_pipeline(self, pipeline: Any) -> None:
        """
        Set a custom Scikit-Learn pipeline.

        Args:
            pipeline (Any): The sklearn pipeline object.

        Returns:
            None
        """
        self.pipeline = pipeline

    def add_website(self, website_url: str) -> None:
        """
        Add a new website URL to the configured set.

        Args:
            website_url (str): The URL to add.

        Returns:
            None
        """
        if website_url not in self._websites_urls:
            self._websites_urls.add(website_url)
        else:
            log_warning(f"Website URL {website_url} already in configured list")

    def remove_website(self, website_url: str) -> None:
        """
        Remove a website URL and clean up all associated data (selectors, cache, training rows).

        Args:
            website_url (str): The URL to remove.

        Returns:
            None
        """
        if website_url in self._websites_urls:
            self._websites_urls.remove(website_url)
            self.selectors.pop(website_url, None)
            self.url_in_training_data.discard(website_url)
            self.website_html_cache.pop(website_url, None)

            if (
                self.training_data is not None
                and not self.training_data.empty
                and "SourceURL" in self.training_data.columns
            ):
                self.training_data = self.training_data[
                    self.training_data["SourceURL"] != website_url
                ]
                self.training_data.reset_index(drop=True, inplace=True)
        else:
            log_warning(f"Website URL {website_url} not found")

    def set_website_selectors_from_yaml(self, website_url: str, yaml_path: str) -> None:
        """
        Load and set element selectors for a specific website URL from a YAML file.

        Args:
            website_url (str): The URL to associate the selectors with.
            yaml_path (str): Path to the YAML file containing selectors.

        Returns:
            None
        """
        try:
            with open(yaml_path, "r") as f:
                selectors = yaml.safe_load(f)
            self.selectors[website_url] = selectors
            # Ensure URL is in our set
            self.add_website(website_url)
        except Exception as e:
            log_error(f"Failed to load selectors from {yaml_path}: {e}")

    def set_website_selectors(
        self, website_url: str, selectors: Dict[str, List[str]]
    ) -> None:
        """
        Set element selectors for a specific website URL manually.

        Args:
            website_url (str): The URL to associate the selectors with.
            selectors (Dict[str, List[str]]): Dictionary mapping category names to lists of XPaths.

        Returns:
            None
        """
        self.selectors[website_url] = selectors
        self.add_website(website_url)

    def create_selectors(
        self, website_url: str, save: bool = True
    ) -> Dict[str, List[str]]:
        """
        Interactively select elements for training data on a single URL via the UI.

        Args:
            website_url (str): The URL to open for selection.
            save (bool): Whether to save state to disk after selection.

        Returns:
            Dict[str, List[str]]: The dictionary of created selectors.
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

        if save:
            self.save()
            self.save_selectors()
            self.save_training_data()

        return data

    def create_all_selectors(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Interactively collect selectors for all configured websites.

        Args:
            None

        Returns:
            Dict[str, Dict[str, List[str]]]: Global dictionary of selectors for all websites.
        """
        # Iterate over a list copy to allow modification if needed
        for url in self.websites_urls:
            self.create_selectors(url)
        return self.selectors

    def add_websites(self, website_urls: List[str]) -> None:
        """
        Add multiple website URLs to the scraper.

        Args:
            website_urls (List[str]): List of URLs to add.

        Returns:
            None
        """
        for url in website_urls:
            self.add_website(url)

    def add_category(self, category: str) -> None:
        """
        Add a new data category to extract.

        Args:
            category (str): The name of the category (e.g., 'price').

        Returns:
            None
        """
        if category not in self._categories:
            self._categories.add(category)
        else:
            log_warning(f"Category '{category}' already exists")

    def add_categories(self, categories: List[str]) -> None:
        """
        Add new data categories to extract.

        Args:
            categories (List[str]): List of category names.

        Returns:
            None
        """
        for category in categories:
            if category not in self._categories:
                self._categories.add(category)
            else:
                log_warning(f"Category '{category}' already exists")

    def create_training_data(
        self, websites_to_use: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Convert collected selectors into a training DataFrame by extracting features from pages.

        Args:
            websites_to_use (Optional[List[str]]): Specific list of URLs to process. If None, uses all.

        Returns:
            Optional[pd.DataFrame]: The updated training DataFrame.
        """
        all_data = []

        # If urls is None, use all tracked websites
        target_urls = (
            websites_to_use if websites_to_use is not None else self.websites_urls
        )

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
                    # This saves our progress so far and prepares the full dataset for training
                    if all_data:
                        new_batch = pd.concat(all_data, ignore_index=True)
                        if (
                            self.training_data is not None
                            and not self.training_data.empty
                        ):
                            self.training_data = pd.concat(
                                [self.training_data, new_batch], ignore_index=True
                            )
                        else:
                            self.training_data = new_batch
                        all_data = []

                    if self.training_data is not None and not self.training_data.empty:
                        self.train_model()

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

    def train_model(
        self, create_data: bool = False, test_size: float = 0.2, min_samples: int = 5
    ) -> None:
        """
        Train the machine learning model with proper Test/Train splitting.

        Args:
            create_data (bool): Whether to generate training data from selectors first.
            test_size (float): Proportion of data to use for testing.
            min_samples (int): Minimum number of samples required to proceed with training.

        Returns:
            None
        """
        if create_data:
            log_info("Creating training dataframe from selectors...")
            self.create_training_data()

        if self.training_data is None or self.training_data.empty:
            log_error(
                "No training data available. Please collect selectors or load data."
            )
            return

        if len(self.training_data) < min_samples:
            log_warning(
                f"Insufficient training data: {len(self.training_data)} samples."
            )
            return

        self.model = train_model(self.training_data, self.pipeline, test_size=test_size)

    def predict_category(self, website_url: str, category: str) -> List[Dict[str, Any]]:
        """
        Predict selectors for a specific category on a specific URL.

        Args:
            website_url (str): The target URL.
            category (str): The category to predict.

        Returns:
            List[Dict[str, Any]]: List of predicted element details (xpath, probability, etc.).
        """
        if self.model is None:
            raise ValueError("Model is not trained.")
        if category not in self._categories:
            raise ValueError(f"Category '{category}' is not configured.")

        html = self.get_html(website_url)
        existing = self.selectors.get(website_url, None)
        result = predict_category_selectors(
            self.model, html, category, existing_selectors=existing
        )

        # cache predicted selectors to use in predicting other elements
        if website_url not in self.predicted_selectors or not isinstance(
            self.predicted_selectors[website_url], list
        ):
            self.predicted_selectors[website_url] = []
        self.predicted_selectors[website_url].append({category: result})

        return result

    def predict(self, website_urls: List[str]) -> List[Dict[str, List[Dict[str, Any]]]]:
        """
        Predict element selectors for all configured categories on multiple URLs.

        Args:
            website_urls (List[str]): List of URLs to predict on.

        Returns:
            List[Dict[str, List[Dict[str, Any]]]]: List of dictionaries containing predictions grouped by URL.
        """
        if not website_urls:
            raise ValueError("Please provide a list of website URLs.")

        all_products: List[Dict[str, List[Dict[str, Any]]]] = []
        for website_url in website_urls:
            raw_predictions = self.get_selectors(website_url)
            html_content = self.get_html(website_url)

            # Access via property to get list
            products = group_prediction_to_products(
                html_content, raw_predictions, self.categories
            )
            all_products.append({website_url: products})

        return all_products

    def get_selectors(self, website_url: str) -> Dict[str, Any]:
        """
        Get existing selectors or predict new ones for a URL.

        Args:
            website_url (str): The target URL.

        Returns:
            Dict[str, Any]: Dictionary mapping categories to lists of selectors/predictions.
        """
        if website_url not in self.selectors:
            result = {}
            for category in self.categories:
                result[category] = self.predict_category(website_url, category)
            return result
        return self.selectors.get(website_url, {})

    # --- Storage Methods (Unchanged) ---
    def save_model(self, path: str = "model.pkl") -> None:
        """
        Save the trained model to disk.

        Args:
            path (str): Filename for the model pickle.

        Returns:
            None
        """
        PRODUCT_SCRAPER_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            with open(PRODUCT_SCRAPER_SAVE_DIR / path, "wb") as f:
                pickle.dump(self.model, f)
        else:
            raise ValueError("Model is not trained.")

    def load_model(self, path: str = "model.pkl") -> None:
        """
        Load a trained model from disk.

        Args:
            path (str): Filename of the model pickle.

        Returns:
            None
        """
        try:
            with open(PRODUCT_SCRAPER_SAVE_DIR / path, "rb") as f:
                self.model = pickle.load(f)
        except Exception as e:
            log_error(f"Failed to load model from {path}: {e}")

    def save_training_data(self, path: str = "training_data.csv") -> None:
        """
        Save the training data DataFrame to disk.

        Args:
            path (str): Filename for the CSV.

        Returns:
            None
        """
        PRODUCT_SCRAPER_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        if self.training_data is not None:
            self.training_data.to_csv(PRODUCT_SCRAPER_SAVE_DIR / path, index=False)
        else:
            raise ValueError("Dataframe is empty.")

    def load_dataframe(self, path: str = "training_data.csv") -> None:
        """
        Load the training data DataFrame from disk.

        Args:
            path (str): Filename of the CSV.

        Returns:
            None
        """
        try:
            self.training_data = pd.read_csv(PRODUCT_SCRAPER_SAVE_DIR / path)
        except Exception as e:
            log_error(f"Failed to load training data from {path}: {e}")

    def save_selectors(self, path: str = "selectors.yaml") -> None:
        """
        Save current selectors to a YAML file.

        Args:
            path (str): Filename for the YAML.

        Returns:
            None
        """
        PRODUCT_SCRAPER_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        with open(PRODUCT_SCRAPER_SAVE_DIR / path, "w") as f:
            yaml.dump(self.selectors, f, default_flow_style=False, allow_unicode=True)

    def load_selectors(self, path: str = "selectors.yaml") -> None:
        """
        Load selectors from a YAML file.

        Args:
            path (str): Filename of the YAML.

        Returns:
            None
        """
        try:
            with open(PRODUCT_SCRAPER_SAVE_DIR / path, "r") as f:
                self.selectors = yaml.safe_load(f)
        except Exception as e:
            log_error(f"Failed to load selectors from {path}: {e}")

        # Ensure loaded URLs are tracked in the set
        for url in self.selectors.keys():
            self.add_website(url)

    def save(self, path: str = "product_scraper.pkl") -> None:
        """
        Save the entire ProductScraper instance state.

        Args:
            path (str): Filename for the pickle.

        Returns:
            None
        """
        PRODUCT_SCRAPER_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        with open(PRODUCT_SCRAPER_SAVE_DIR / path, "wb") as f:
            pickle.dump(self, f)
        self.save_selectors()
        self.save_training_data()

    @staticmethod
    def load(path: str = "product_scraper.pkl") -> "ProductScraper":
        """
        Load a ProductScraper instance from disk.

        Args:
            path (str): Filename of the pickle.

        Returns:
            ProductScraper: The loaded instance.
        """
        with open(PRODUCT_SCRAPER_SAVE_DIR / path, "rb") as f:
            return pickle.load(f)
