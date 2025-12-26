"""
ProductScraper: Main interface for web scraping and machine learning-based element detection.
"""
# @generated "partially" Gemini 3: Fix linting errors and add docstrings + refactored for linting .

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

# Default directory if none is provided
DEFAULT_SAVE_DIR = Path("product_scraper_data")


class ProductScraper:
    """
    Main interface for web scraping and machine learning-based element detection.
    """

    def __init__(
        self,
        categories: List[str],
        websites_urls: List[str],
        selectors: Optional[Dict[str, Dict[str, List[str]]]] = None,
        training_data: Optional[pd.DataFrame] = None,
        save_dir: Union[str, Path] = DEFAULT_SAVE_DIR,
    ):
        """
        Initialize the ProductScraper instance.

        Args:
            categories (List[str]): List of product categories to scrape (e.g., 'price', 'title').
            websites_urls (Optional[List[str]]): List of initial website URLs to track.
            selectors (Optional[Dict[str, Dict[str, List[str]]]]): Dictionary of existing
                                                                    selectors mapping URL -> Category -> XPaths.
            training_data (Optional[pd.DataFrame]): Pre-loaded training data DataFrame.
            model (Optional[Dict[str, Any]]): Pre-trained model dictionary.
            pipeline (Optional[Any]): Sklearn pipeline for model training/prediction.
            save_dir (Union[str, Path]): Directory path to save/load models and data.
                                         Defaults to "product_scraper_data".

        Returns:
            None
        """
        self.website_html_cache = {}
        self.website_cache_metadata = {}

        # Set the save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._categories: List[str] = categories
        self._websites_urls: List[str] = websites_urls

        # url -> {category: [selectors]}
        self.selectors: Dict[str, Dict[str, List[str]]] = (
            selectors if selectors is not None else {}
        )
        self.predicted_selectors: Dict[str, List[Dict[str, Any]]] = {}

        self.url_in_training_data = set()
        self.training_data = training_data

        self.model = None
        self.pipeline = None

        # Iterator state
        self._iterator_index = 0
        self._iter_list = []

    @property
    def categories(self) -> List[str]:
        """
        Get the sorted list of configured categories.
        """
        return self._categories

    @property
    def websites_urls(self) -> List[str]:
        """
        Get the sorted list of tracked website URLs.
        """
        return self._websites_urls

    def __iter__(self) -> "ProductScraper":
        """
        Initialize the iterator for iterating over websites.
        """
        self._iterator_index = 0
        self._iter_list = self._websites_urls
        return self

    def __next__(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get the next website URL and its predicted selectors.
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
        """
        return len(self._websites_urls)

    def get_html(self, website_url: str, use_browser: bool = True) -> str:
        """
        Fetch HTML content from a URL, using Playwright or requests.
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
            except Exception as e:  # pylint: disable=broad-exception-caught
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
        """
        self.pipeline = pipeline

    def add_website(self, website_url: str) -> None:
        """
        Add a new website URL to the configured list if not present.
        """
        if website_url not in self._websites_urls:
            self._websites_urls.append(website_url)

    def remove_website(self, website_url: str) -> None:
        """
        Remove a website URL and clean up all associated data (selectors, cache, training rows).
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
        """
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                selectors = yaml.safe_load(f)
            self.selectors[website_url] = selectors
            # Ensure URL is in our list
            self.add_website(website_url)
        except (OSError, yaml.YAMLError) as e:
            log_error(f"Failed to load selectors from {yaml_path}: {e}")

    def set_website_selectors(
        self, website_url: str, selectors: Dict[str, List[str]]
    ) -> None:
        """
        Set element selectors for a specific website URL manually.
        """
        self.selectors[website_url] = selectors
        self.add_website(website_url)

    def create_selectors(
        self, website_url: str, save: bool = True
    ) -> Dict[str, List[str]]:
        """
        Interactively select elements for training data on a single URL via the UI.
        """
        # Ensure URL is tracked
        self.add_website(website_url)

        has_all_selectors = website_url in self.selectors and set(
            self._categories
        ).issubset(self.selectors[website_url].keys())

        if has_all_selectors and website_url in self.selectors:
            return self.selectors[website_url]

        try:
            data = select_data(self, website_url)
        # catch full exception to avoid breaking the loop on unexpected errors like invalid: html loading site, img problems, etc.
        except Exception as e:  # pylint: disable=broad-exception-caught
            log_error(f"Failed to create selectors for {website_url}: {e}. Skipping.")
            return {}

        if len(data) == 0:
            log_warning(f"No selectors were created for {website_url}.")
            return {}

        self.selectors[website_url] = data

        if save:
            self.save()

        return data

    def create_all_selectors(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Interactively collect selectors for all configured websites.
        """
        # Iterate over a copy to allow safe modification if needed
        for url in list(self._websites_urls):
            self.create_selectors(url)
        return self.selectors

    def add_websites(self, website_urls: List[str]) -> None:
        """
        Add multiple website URLs to the scraper.
        """
        for url in website_urls:
            self.add_website(url)

    def add_category(self, category: str) -> None:
        """
        Add a new data category to extract.
        """
        if category not in self._categories:
            self._categories.append(category)
        else:
            log_warning(f"Category '{category}' already exists")

    def add_categories(self, categories: List[str]) -> None:
        """
        Add new data categories to extract.
        """
        for category in categories:
            self.add_category(category)

    def __validate_training_url(self, url: str) -> bool:
        """Internal helper to validate if a URL should be processed for training."""
        if url in self.url_in_training_data:
            return False

        if url not in self._websites_urls:
            log_warning(f"URL {url} not in configured websites. Skipping.")
            return False

        return True

    def __fetch_training_html(self, url: str) -> Optional[str]:
        """Internal helper to safely fetch HTML for training."""
        try:
            return self.get_html(url)
        except requests.RequestException:
            log_warning(f"Skipping {url} due to network error.")
            return None

    def __extract_data_from_html(
        self, url: str, html_content: str, all_data: List[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """Internal helper to handle selector logic and extract DataFrame."""

        has_all_selectors = url in self.selectors and set(self._categories).issubset(
            self.selectors[url].keys()
        )

        if url not in self.selectors or not has_all_selectors:
            try:
                if all_data:
                    new_batch = pd.concat(all_data, ignore_index=True)
                    if self.training_data is not None and not self.training_data.empty:
                        self.training_data = pd.concat(
                            [self.training_data, new_batch], ignore_index=True
                        )
                    else:
                        self.training_data = new_batch
                    all_data.clear()

                if self.training_data is not None and not self.training_data.empty:
                    self.train_model()

                self.create_selectors(url)
            except (ValueError, KeyError, TypeError) as e:
                log_warning(f"Failed to create selectors for {url}: {e}")
                return None

        selectors = self.selectors.get(url, {})
        if not selectors:
            log_warning(f"No selectors found for {url}, skipping.")
            return None

        try:
            df = html_to_dataframe(html_content, selectors, url=url)
        except (ValueError, KeyError, TypeError) as e:
            log_warning(f"Error processing {url}: {e}")
            return None

        if not df.empty:
            self.url_in_training_data.add(url)
            log_info(f"Extracted samples from {url}")
            return df

        log_warning(f"No data extracted from {url}")
        return None

    def __process_training_url(
        self, url: str, all_data: List[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """
        Internal helper to process a single URL for training data creation.
        Returns a DataFrame if data was extracted, None otherwise.
        """
        if not self.__validate_training_url(url):
            return None

        html_content = self.__fetch_training_html(url)
        if html_content is None:
            return None

        return self.__extract_data_from_html(url, html_content, all_data)

    def create_training_data(
        self, websites_to_use: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Convert collected selectors into a training DataFrame by extracting features from pages.
        """
        all_data = []

        # If urls is None, use all tracked websites
        target_urls = (
            websites_to_use if websites_to_use is not None else self._websites_urls
        )

        for url in target_urls:
            df = self.__process_training_url(url, all_data)
            if df is not None:
                all_data.append(df)

        if all_data:
            if self.training_data is not None and not self.training_data.empty:
                all_data.insert(0, self.training_data)
            self.training_data = pd.concat(all_data, ignore_index=True)
        elif self.training_data is None:
            log_warning("No data was successfully extracted from any URL")

        return self.training_data

    def train_model(
        self,
        create_data: bool = False,
        test_size: float = 0.2,
        min_samples: int = 5,
        show_model_figure: bool = False,
    ) -> None:
        """
        Train the machine learning model with proper Test/Train splitting.
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

        self.model = train_model(
            self.training_data,
            self.pipeline,
            test_size=test_size,
            show_model_figure=show_model_figure,
        )

    def predict_category(self, website_url: str, category: str) -> List[Dict[str, Any]]:
        """
        Predict selectors for a specific category on a specific URL.
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

    def predict(
        self, website_urls: List[str], only_xpaths: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Predict element selectors for all configured categories on multiple URLs.

        This method orchestrates the prediction pipeline:
        1. Fetches HTML for each URL.
        2. Uses the trained model to identify candidate elements for each category.
        3. Groups these candidates into coherent 'Product' entities based on proximity.

        Args:
            website_urls (List[str]): A list of URLs to scrape.
            only_xpaths (bool): If True, returns a simplified dictionary containing only
                                the XPaths (strings) for each element.
                                If False, returns the full candidate dictionaries including
                                metadata like text preview, tag, IDs, etc.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary mapping URLs to lists of found products.

            Example (only_xpaths=True):
            {
                "https://example.com": [
                    {
                        "price": ["/html/body/div[1]/span"],
                        "title": ["/html/body/div[1]/h1"],
                        "image": ["/html/body/div[1]/img"]
                    },
                    ...
                ],
                "https://another-site.com": [...]
            }
        """
        if not website_urls:
            raise ValueError("Please provide a list of website URLs.")

        all_products: Dict[str, List[Dict[str, Any]]] = {}

        for website_url in website_urls:
            # 1. Get raw candidates
            raw_predictions = self.get_selectors(website_url)

            # 2. Group candidates into products
            products = group_prediction_to_products(raw_predictions, self.categories)

            # 3. Store directly in dictionary
            all_products[website_url] = products

        if only_xpaths:
            # Simplify output structure to remove heavy metadata (text, IDs, etc)
            simplified_results = {}

            for url, product_list in all_products.items():
                simplified_products_list = []

                for product in product_list:
                    # product is {category: [candidate_dict, ...]}
                    simplified_product = {}
                    for category, elements in product.items():
                        # Extract just the XPath string from each candidate dict
                        if isinstance(elements, list):
                            simplified_product[category] = [
                                item["xpath"]
                                for item in elements
                                if isinstance(item, dict) and "xpath" in item
                            ]

                    simplified_products_list.append(simplified_product)

                simplified_results[url] = simplified_products_list

            return simplified_results

        return all_products

    def get_selectors(self, website_url: str) -> Dict[str, Any]:
        """
        Get existing selectors or predict new ones for a URL.
        """
        if website_url not in self.selectors:
            result = {}
            for category in self._categories:
                result[category] = self.predict_category(website_url, category)
            return result
        return self.selectors.get(website_url, {})

    # --- Storage Methods ---

    def save_model(self, filename: str = "model.pkl") -> None:
        """
        Save the trained model to disk in the configured save_dir.
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            with open(self.save_dir / filename, "wb") as f:
                pickle.dump(self.model, f)
        else:
            raise ValueError("Model is not trained.")

    def load_model(self, filename: str = "model.pkl") -> None:
        """
        Load a trained model from disk in the configured save_dir.
        """
        try:
            with open(self.save_dir / filename, "rb") as f:
                self.model = pickle.load(f)
        except (OSError, pickle.PickleError) as e:
            log_error(f"Failed to load model from {self.save_dir / filename}: {e}")

    def save_training_data(self, filename: str = "training_data.csv") -> None:
        """
        Save the training data DataFrame to disk in the configured save_dir.
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.training_data is not None:
            self.training_data.to_csv(self.save_dir / filename, index=False)

    def load_dataframe(self, filename: str = "training_data.csv") -> None:
        """
        Load the training data DataFrame from disk in the configured save_dir.
        """
        try:
            self.training_data = pd.read_csv(self.save_dir / filename)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            log_error(
                f"Failed to load training data from {self.save_dir / filename}: {e}"
            )

    def save_selectors(self, filename: str = "selectors.yaml") -> None:
        """
        Save current selectors to a YAML file in the configured save_dir.
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)
        with open(self.save_dir / filename, "w", encoding="utf-8") as f:
            yaml.dump(self.selectors, f, default_flow_style=False, allow_unicode=True)

    def load_selectors(self, filename: str = "selectors.yaml") -> None:
        """
        Load selectors from a YAML file in the configured save_dir.
        """
        try:
            with open(self.save_dir / filename, "r", encoding="utf-8") as f:
                self.selectors = yaml.safe_load(f)
        except (OSError, yaml.YAMLError) as e:
            log_error(f"Failed to load selectors from {self.save_dir / filename}: {e}")

        # Ensure loaded URLs are tracked in the list
        if self.selectors:
            for url in self.selectors.keys():
                self.add_website(url)

    def save(self, filename: str = "product_scraper.pkl") -> None:
        """
        Save the entire ProductScraper instance state in the configured save_dir.
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)
        with open(self.save_dir / filename, "wb") as f:
            pickle.dump(self, f)

        # Save auxiliary files
        self.save_selectors()
        self.save_training_data()

    @staticmethod
    def load(
        save_dir: Union[str, Path] = DEFAULT_SAVE_DIR,
        filename: str = "product_scraper.pkl",
    ) -> "ProductScraper":
        """
        Load a ProductScraper instance from disk.

        Args:
            save_dir: The directory where the file is located.
            filename: The specific pickle filename.
        """
        path = Path(save_dir) / filename
        with open(path, "rb") as f:
            scraper = pickle.load(f)

        # Update save_dir of loaded instance to match where we loaded it from,
        # in case the folder was moved.
        scraper.save_dir = Path(save_dir)
        return scraper
