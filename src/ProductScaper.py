
"""
ProductScraper: Main interface for web scraping and machine learning-based element detection.

This module provides the ProductScraper class, which allows for collecting training data,
training models, predicting selectors, and managing scraping state for product websites.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import yaml
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from create_data import select_data
from train_model.evaluate_model import evaluate_model
from train_model.predict_data import group_prediction_to_products, predict_selectors
from train_model.process_data import html_to_dataframe
from train_model.train_model import train_model
from utils.console import CONSOLE, log_debug, log_error, log_info, log_warning

# Configuration for save directory
PRODUCT_SCRAPER_SAVE_DIR = Path('product_scraper_data')

class ProductScraper:
    """
    Main interface for web scraping with machine learning-based element detection.
    
    This class provides methods to:
    - Collect training data by manually selecting elements from web pages.
    - Train a model to automatically detect product information.
    - Predict element selectors on new pages.
    
    The class is iterable and will yield (url, predictions_dict) for each configured website.
    Automatically saves state on destruction.
    """

    def __init__(self, categories: List[str], websites_urls: List[str], selectors: Optional[Dict[str, Dict[str, List[str]]]] = None, training_data: Optional[pd.DataFrame] = None, model: Optional[Dict[str, Any]] = None, pipeline: Optional[Any] = None):
        """
        Initialize the ProductScraper.
        
        Args:
            categories: List of data categories to extract (e.g., ['title', 'price', 'image']).
            websites_urls: List of URLs to train on.
            selectors: Existing dictionary of selectors.
            training_data: Existing training data.
            model: Trained model dictionary.
            pipeline: Sklearn pipeline for custom preprocessing.
        """
        self.website_html_cache = {}
        self.website_cache_metadata = {}  # Store ETag and Last-Modified headers

        self.categories: List[str] = categories
        self.websites_urls: List[str] = websites_urls

        # url -> {category: [selectors]}
        self.selectors: Dict[str, Dict[str, List[str]]] = selectors if selectors is not None else {}

        self.url_in_training_data = set()
        self.training_data = training_data
        self.model = model
        self.pipeline = pipeline

        self._iterator_index = 0  # For iterator support

    def __del__(self) -> None:
        """
        Automatically save state when object is destroyed.
        """
        try:
            if self.model is not None:
                log_debug("Auto-saving ProductScraper state...")
                self.save()
        except Exception as e:
            # Catching broadly here to prevent errors during interpreter shutdown
            log_warning(f"Could not auto-save: {e}")

    def __iter__(self) -> 'ProductScraper':
        """
        Make ProductScraper iterable over websites.
        """
        self._iterator_index = 0
        return self

    def __next__(self) -> tuple:
        """
        Iterate over websites, yielding (url, predictions) for each category.
        """
        if self._iterator_index >= len(self.websites_urls):
            raise StopIteration
        url = self.websites_urls[self._iterator_index]
        self._iterator_index += 1
        predictions = self.get_selectors(url)
        return (url, predictions)

    def __len__(self) -> int:
        """
        Return number of websites.
        """
        return len(self.websites_urls)

    def get_html(self, website_url: str, use_browser: bool = True) -> str:
        """
        Fetch HTML content for a URL with caching and cache validation.
        Uses Playwright by default to render JavaScript, with fallback to requests.
        
        Args:
            website_url: URL to fetch.
            use_browser: If True, use Playwright to render JavaScript. If False, use requests.
            
        Returns:
            HTML content as string.
            
        Raises:
            requests.RequestException: If unable to fetch the URL.
            PlaywrightError: If browser automation fails.
        """
        # Return cached content if available
        if website_url in self.website_html_cache:
            log_debug(f"Using cached HTML for {website_url}")
            return self.website_html_cache[website_url]

        # Use Playwright to get JavaScript-rendered HTML (same as selector creation)
        if use_browser:
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto(website_url, wait_until='networkidle', timeout=30000)
                    html_content = page.content()
                    browser.close()

                    # Cache the content
                    self.website_html_cache[website_url] = html_content
                    log_debug(f"Fetched and cached JavaScript-rendered HTML for {website_url}")
                    return html_content

            except PlaywrightError as e:
                log_warning(f"Playwright error for {website_url}: {e}. Falling back to requests.")
                use_browser = False
            except Exception as e:
                log_warning(f"Unexpected error with Playwright for {website_url}: {e}. Falling back to requests.")
                use_browser = False

        # Fallback to requests (no JavaScript rendering)
        if not use_browser:
            try:
                response = requests.get(website_url, timeout=10)
                response.raise_for_status()
                html_content = response.text

                # Cache the content
                self.website_html_cache[website_url] = html_content
                log_debug(f"Fetched and cached static HTML for {website_url}")
                return html_content

            except requests.exceptions.ConnectionError as e:
                log_error(f"Connection error for {website_url}: {str(e).split(':')[0]}")
                raise
            except requests.exceptions.Timeout:
                log_error(f"Timeout error for {website_url}")
                raise
            except requests.exceptions.HTTPError as e:
                log_error(f"HTTP error for {website_url}: {e.response.status_code}")
                raise
            except requests.RequestException as e:
                log_error(f"Request error for {website_url}: {e}")
                raise

    def set_pipeline(self, pipeline: Any) -> None:
        """
        Set a custom sklearn pipeline for preprocessing and model training.
        
        Args:
            pipeline: sklearn Pipeline object.
        """
        self.pipeline = pipeline


    def add_website(self, website_url: str) -> None:
        """
        Add a new website URL to the configured list.
        
        Args:
            website_url: URL to add.
        """
        if website_url not in self.websites_urls:
            self.websites_urls.append(website_url)
        else:
            log_warning(f"Website URL {website_url} already in configured list")


    def remove_website(self, website_url: str) -> None:
        """
        Remove a website URL and clean up all associated data (selectors, cache, training rows).
        
        Args:
            website_url: URL to remove.
        """
        if website_url in self.websites_urls:
            # 1. Remove from configuration list
            self.websites_urls.remove(website_url)

            # 2. Remove associated selectors
            if website_url in self.selectors:
                del self.selectors[website_url]

            # 3. Remove from tracking set
            if website_url in self.url_in_training_data:
                self.url_in_training_data.remove(website_url)

            # 4. Clean up HTML caches
            if website_url in self.website_html_cache:
                del self.website_html_cache[website_url]
            if website_url in self.website_cache_metadata:
                del self.website_cache_metadata[website_url]

            # 5. Remove rows from training data
            if self.training_data is not None and not self.training_data.empty:
                if 'SourceURL' in self.training_data.columns:
                    initial_count = len(self.training_data)
                    self.training_data = self.training_data[self.training_data['SourceURL'] != website_url]

                    # Reset index to maintain data integrity
                    self.training_data.reset_index(drop=True, inplace=True)

                    removed_count = initial_count - len(self.training_data)
                    log_info(f"Removed {website_url} and {removed_count} associated training samples.")
                else:
                    log_warning("SourceURL column missing from training data; could not filter rows.")
        else:
            log_warning(f"Website URL {website_url} not found in configured list")

    def set_website_selectors_from_yaml(self, website_url: str, yaml_path: str) -> None:
        """
        Load and set element selectors for a specific website URL from a YAML file.
        
        Args:
            website_url: URL to set selectors for.
            yaml_path: Path to YAML file containing selectors.
        """
        try:
            with open(yaml_path, 'r') as f:
                selectors = yaml.safe_load(f)
            self.selectors[website_url] = selectors
        except Exception as e:
            log_error(f"Failed to load selectors from {yaml_path}: {e}")

    def set_website_selectors(self, website_url: str, selectors: Dict[str, List[str]]) -> None:
        """
        Set element selectors for a specific website URL manually.
        
        Args:
            website_url: URL to set selectors for.
            selectors: Dictionary mapping categories to lists of selectors.
        """
        self.selectors[website_url] = selectors

    def create_selectors(self, website_url: str, save: bool = True) -> Dict[str, List[str]]:
        """
        Interactively select elements for training data on a single URL.
        Opens a browser window for manual element selection.
        
        Args:
            website_url: URL to select elements from.
            save: Whether to save after creating selectors.
            
        Returns:
            Dictionary mapping categories to element selectors.
        """
        if website_url in self.selectors:
            return self.selectors[website_url]

        # Interactive selection - pass self (ProductScraper instance) and url
        try:
            data = select_data(self, website_url)
        except Exception as e:
            log_error(f"Failed to create selectors for {website_url}: {e}. Skipping.")
            return {}

        self.selectors[website_url] = data

        # Immediately update dataframe and retrain model with new data
        self.create_training_data([website_url])
        self.train_model()

        if save:
            self.save()
            self.save_selectors()
            self.save_training_data()

        return data

    def create_all_selectors(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Interactively collect selectors for all configured websites.
        Opens browser windows sequentially for each URL.
        
        Returns:
            Dictionary of all selectors {url: {category: [selectors]}}.
        """
        for url in self.websites_urls:
            self.create_selectors(url)
        return self.selectors

    def create_training_data(self, urls: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        Convert collected selectors into training dataframe.
        Automatically creates selectors for URLs that don't have them yet.
        
        Args:
            urls (list, optional): List of URLs to process. If None, processes all configured URLs.
        
        Returns:
            pd.DataFrame: DataFrame with features extracted from HTML elements.
        """
        all_data = []

        if urls is None:
            urls = self.websites_urls

        for url in urls:
            if url in self.url_in_training_data:
                continue

            if url not in self.websites_urls:
                log_warning(f"URL {url} not in configured websites. Skipping.")
                continue

            try:
                html_content = self.get_html(url)
            except requests.RequestException:
                log_warning(f"Skipping {url} due to network error.")
                continue

            # Ensure selectors exist for this URL
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

    def train_model(self, create_data: bool = False, min_samples: int = 5) -> None:
        """
        Train the machine learning model on collected data.
        Automatically creates dataframe if not already created.
        
        Args:
            create_data (bool): If True, creates training dataframe if not present.
            min_samples (int): Minimum number of samples required for training (default: 5).
        """
        if create_data and (self.training_data is None or self.training_data.empty):
            log_info("Creating training dataframe from selectors...")
            self.create_training_data()

        if self.training_data is None or self.training_data.empty:
            log_error("No training data available. Please:")
            CONSOLE.print("  1. Call create_all_selectors() to collect data interactively, or")
            CONSOLE.print("  2. Load existing selectors with load_selectors(), or")
            CONSOLE.print("  3. Load existing training data with load_dataframe()")
            return

        # Check minimum sample requirement
        sample_count = len(self.training_data)
        if sample_count < min_samples:
            log_warning(f"Insufficient training data: {sample_count} samples (minimum: {min_samples})")
            log_info(f"Please collect more selectors from websites. Current: {len(self.selectors)}/{len(self.websites_urls)} sites")
            CONSOLE.print("\n[bold]Training Data Info:[/bold]")
            self.training_data.info()
            CONSOLE.print("\n[bold]Training Data Sample:[/bold]")
            # Reduced head count for readability
            CONSOLE.print(self.training_data.head(10))
            CONSOLE.print(self.training_data.describe(include='all'))
            return

        log_info(f"Training model on {sample_count} samples...")
        self.model = train_model(self.training_data, self.pipeline)

    def evaluate(self) -> Any:
        """
        Evaluate the trained model on the training data and display results.
        Returns performance metrics.
        """
        if self.model is None:
            raise ValueError("Model is not trained. Please train the model before evaluation.")

        if self.training_data is None:
            raise ValueError("Training data is not available. Please train the model or load training data.")

        # Extract model components
        pipeline = self.model['pipeline']
        label_encoder = self.model['label_encoder']

        # Prepare features and labels
        X = self.training_data.drop(columns=['Category'])
        y = self.training_data['Category']

        # evaluate_model handles the display of results
        metrics = evaluate_model(
            model=pipeline,
            X=X,
            y=y,
            label_encoder=label_encoder,
            split_name="Training",
            display_results=True
        )

        return metrics

    def predict_category(self, website_url: str, category: str) -> List[Dict[str, Any]]:
        """
        Predict element selectors for a specific category on a page.
        
        Args:
            website_url (str): URL to predict selectors for.
            category (str): Category to predict (e.g., 'title', 'price').
            
        Returns:
            list: List of predicted elements with xpath and preview information.
            
        Raises:
            ValueError: If model hasn't been trained or category is invalid.
        """
        if self.model is None:
            raise ValueError("Model is not trained. Please train the model before prediction.")
        if category not in self.categories:
            raise ValueError(f"Category '{category}' is not in configured categories: {self.categories}")

        return predict_selectors(self.model, self.get_html(website_url), category)

    def predict_product(self, website_url: str) -> List[Dict[str, Any]]:
        """
        Predict element selectors for all configured categories on a page and group them into products.
        
        Args:
            website_url (str): URL to predict selectors for.
        Returns:
            list: List of dictionaries, where each dictionary represents a product 
                  containing the predicted elements for each category.
        """

        raw_predictions = self.get_selectors(website_url)
        html_content = self.get_html(website_url)

        products = group_prediction_to_products(html_content, raw_predictions, self.categories)

        log_info(f"Found {len(products)} products on {website_url}")
        return products



    def get_selectors(self, website_url: str) -> Dict[str, Any]:
        """
        Get the currently stored selectors for a specific website URL.
        
        Args:
            website_url (str): URL to get selectors for.
        Returns:
            dict: Dictionary of selectors for the URL.
        """
        if website_url not in self.selectors:
            log_warning(f"No selectors found for {website_url}. Predicting instead.")
            result = {}
            for category in self.categories:
                result[category] = self.predict_category(website_url, category)
            return result
        return self.selectors.get(website_url, {})

    def save_model(self, path: str = 'model.pkl') -> None:
        """
        Save the trained model to disk.
        
        Args:
            path (str): Filename to save model data.
        """
        PRODUCT_SCRAPER_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            with open(PRODUCT_SCRAPER_SAVE_DIR / path, 'wb') as f:
                pickle.dump(self.model, f)
        else:
            raise ValueError("Model is not trained. Cannot save.")

    def load_model(self, path: str = 'model.pkl') -> None:
        """
        Load a trained model from disk.
        
        Args:
            path (str): Filename to load model data from.
        """
        try:
            with open(PRODUCT_SCRAPER_SAVE_DIR / path, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            log_error(f"Failed to load model from {path}: {e}")

    def save_training_data(self, path: str = 'training_data.csv') -> None:
        """
        Save the training dataframe to a CSV file.
        
        Args:
            path (str): Filename to save training data.
        """
        PRODUCT_SCRAPER_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        if self.training_data is not None:
            self.training_data.to_csv(PRODUCT_SCRAPER_SAVE_DIR / path, index=False)
        else:
            raise ValueError("Dataframe is empty. Cannot save.")

    def load_dataframe(self, path: str = 'training_data.csv') -> None:
        """
        Load training dataframe from a CSV file.
        
        Args:
            path (str): Filename to load training data from.
        """
        try:
            self.training_data = pd.read_csv(PRODUCT_SCRAPER_SAVE_DIR / path)
        except Exception as e:
            log_error(f"Failed to load training data from {path}: {e}")

    def save_selectors(self, path: str = 'selectors.yaml') -> None:
        """
        Save collected selectors to disk.
        
        Args:
            path (str): Filename to save selectors.
        """
        PRODUCT_SCRAPER_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        with open(PRODUCT_SCRAPER_SAVE_DIR / path, 'w') as f:
            yaml.dump(self.selectors, f, default_flow_style=False, allow_unicode=True)

    def load_selectors(self, path: str = 'selectors.yaml') -> None:
        """
        Load previously collected selectors from disk.
        
        Args:
            path (str): Filename to load selectors from.
        """
        try:
            with open(PRODUCT_SCRAPER_SAVE_DIR / path, 'r') as f:
                self.selectors = yaml.safe_load(f)
        except Exception as e:
            log_error(f"Failed to load selectors from {path}: {e}")

    def save(self, path: str = 'product_scraper.pkl') -> None:
        """
        Save the entire ProductScraper instance including model and data.
        
        Args:
            path (str): Filename to save the instance.
        """
        PRODUCT_SCRAPER_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        with open(PRODUCT_SCRAPER_SAVE_DIR / path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str = 'product_scraper.pkl') -> 'ProductScraper':
        """
        Load a previously saved ProductScraper instance.
        
        Args:
            path (str): Filename to load from.
            
        Returns:
            ProductScraper: Loaded instance.
        """

        with open(PRODUCT_SCRAPER_SAVE_DIR / path, 'rb') as f:
            return pickle.load(f)
