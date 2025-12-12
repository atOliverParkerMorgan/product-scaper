from pathlib import Path
import pickle
import requests
import pandas as pd
import yaml

from create_data import select_data
from train_model.process_data import html_to_dataframe
from train_model.train_model import train_model
from train_model.predict_data import predict_selectors
from train_model.evaluate_model import evaluate_model
from utils.console import CONSOLE, log_info, log_warning, log_error, log_debug

# Configuration for save directory
PRODUCT_SCRAPER_SAVE_DIR = Path('product_scraper_data')
PRODUCT_SCRAPER_SAVE_DIR.mkdir(parents=True, exist_ok=True)

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
    
    def __init__(self, categories, websites_urls, selectors=None, training_data=None, model=None, pipeline=None):
        """
        Initialize the ProductScraper.
        
        Args:
            categories (list): List of data categories to extract (e.g., ['title', 'price', 'image']).
            websites_urls (list): List of URLs to train on.
            selectors (dict, optional): Existing dictionary of selectors.
            training_data (pd.DataFrame, optional): Existing training data.
            model (dict, optional): Trained model dictionary.
            pipeline (sklearn.Pipeline, optional): Sklearn pipeline for custom preprocessing.
        """
        self.website_html_cache = {}
        self.website_cache_metadata = {}  # Store ETag and Last-Modified headers
        
        self.categories = categories
        self.websites_urls = websites_urls
        self.selectors = selectors if selectors is not None else {}

        self.url_in_training_data = set()
        self.training_data = training_data
        self.model = model
        self.pipeline = pipeline
        
        self._iterator_index = 0  # For iterator support
    
    def __del__(self):
        """Automatically save state when object is destroyed."""
        try:
            if self.model is not None:
                log_debug("Auto-saving ProductScraper state...")
                self.save()
        except Exception as e:
            # Catching broadly here to prevent errors during interpreter shutdown
            log_warning(f"Could not auto-save: {e}")
    
    def __iter__(self):
        """Make ProductScraper iterable over websites."""
        self._iterator_index = 0
        return self
    
    def __next__(self):
        """Iterate over websites, yielding (url, predictions) for each category."""
        if self._iterator_index >= len(self.websites_urls):
            raise StopIteration
        
        url = self.websites_urls[self._iterator_index]
        self._iterator_index += 1
        
        # Predict all categories for this URL
        predictions = {}
        for category in self.categories:
            try:
                predictions[category] = self.predict(url, category)
            except Exception as e:
                log_warning(f"Failed to predict '{category}' for {url}: {e}")
                predictions[category] = []
        
        return (url, predictions)
    
    def __len__(self):
        """Return number of websites."""
        return len(self.websites_urls)

    def get_html(self, website_url):
        """
        Fetch HTML content for a URL with caching and cache validation.
        Uses HTTP HEAD request with ETag/Last-Modified headers to check if content changed.
        
        Args:
            website_url (str): URL to fetch.
            
        Returns:
            str: HTML content.
            
        Raises:
            requests.RequestException: If unable to fetch the URL.
        """
        # If we have cached content, check if it's still valid
        if website_url in self.website_html_cache:
            try:
                # Make a HEAD request to check if content has changed
                head_response = requests.head(website_url, timeout=5, allow_redirects=True)
                cached_metadata = self.website_cache_metadata.get(website_url, {})
                
                # Check ETag (unique content identifier)
                current_etag = head_response.headers.get('ETag')
                cached_etag = cached_metadata.get('etag')
                
                # Check Last-Modified timestamp
                current_modified = head_response.headers.get('Last-Modified')
                cached_modified = cached_metadata.get('last_modified')
                
                # If either matches, cache is still valid
                if (current_etag and current_etag == cached_etag) or \
                   (current_modified and current_modified == cached_modified):
                    return self.website_html_cache[website_url]
                
            except requests.RequestException:
                # If HEAD request fails, use cached version anyway
                log_debug(f"Using cached version for {website_url}")
                return self.website_html_cache[website_url]
        
        # Fetch fresh content with error handling
        try:
            response = requests.get(website_url, timeout=10)
            response.raise_for_status()
            html_content = response.text
            
            # Cache the content and metadata
            self.website_html_cache[website_url] = html_content
            self.website_cache_metadata[website_url] = {
                'etag': response.headers.get('ETag'),
                'last_modified': response.headers.get('Last-Modified')
            }
            
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
    
    def set_pipeline(self, pipeline):
        """
        Set a custom sklearn pipeline for preprocessing and model training.
        
        Args:
            pipeline: sklearn Pipeline object.
        """
        self.pipeline = pipeline

    
    def add_website(self, website_url):
        """
        Add a new website URL to the configured list.
        
        Args:
            website_url (str): URL to add.
        """
        if website_url not in self.websites_urls:
            self.websites_urls.append(website_url)
        else:
            log_warning(f"Website URL {website_url} already in configured list")
            

    def remove_website(self, website_url):
        """
        Remove a website URL and clean up all associated data (selectors, cache, training rows).
        
        Args:
            website_url (str): URL to remove.
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

    def set_website_selectors_from_yaml(self, website_url, yaml_path):
        """
        Load and set element selectors for a specific website URL from a YAML file.
        
        Args:
            website_url (str): URL to set selectors for.
            yaml_path (str): Path to YAML file containing selectors.
        """
        try:
            with open(yaml_path, 'r') as f:
                selectors = yaml.safe_load(f)
            self.selectors[website_url] = selectors
        except Exception as e:
            log_error(f"Failed to load selectors from {yaml_path}: {e}")

    def set_website_selectors(self, website_url, selectors):
        """
        Set element selectors for a specific website URL manually.
        
        Args:
            website_url (str): URL to set selectors for.
            selectors (dict): Dictionary mapping categories to lists of selectors.
        """
        self.selectors[website_url] = selectors

    def create_selectors(self, website_url):
        """
        Interactively select elements for training data on a single URL.
        Opens a browser window for manual element selection.
        
        Args:
            website_url (str): URL to select elements from.
            
        Returns:
            dict: Dictionary mapping categories to element selectors.
        """
        if website_url in self.selectors:
            return self.selectors[website_url]
        
        # Interactive selection
        data = select_data(website_url, self.categories, self.model)
        self.selectors[website_url] = data

        # Immediately update dataframe and retrain model with new data
        self.create_dataframe([website_url])
        self.train_model()

        return data
    
    def create_all_selectors(self):
        """
        Interactively collect selectors for all configured websites.
        Opens browser windows sequentially for each URL.
        
        Returns:
            dict: Dictionary of all selectors {url: {category: [selectors]}}.
        """
        for url in self.websites_urls:
            self.create_selectors(url)
        return self.selectors

    def create_dataframe(self, urls=None):
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

            # Ensure we have selectors for this URL
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
            
            log_debug(f"Processing {url} with selectors: {selectors}")
                
            try:
                df = html_to_dataframe(html_content, selectors)
                df['SourceURL'] = url 
                if not df.empty:
                    all_data.append(df)
                    self.url_in_training_data.add(url)
                    log_info(f"Extracted {len(df)} samples from {url}")
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

    def train_model(self, create_data=False, min_samples=5):
        """
        Train the machine learning model on collected data.
        Automatically creates dataframe if not already created.
        
        Args:
            create_data (bool): If True, creates training dataframe if not present.
            min_samples (int): Minimum number of samples required for training (default: 5).
        """
        if create_data and (self.training_data is None or self.training_data.empty):
            log_info("Creating training dataframe from selectors...")
            self.create_dataframe()

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
        self.save_model()

    def evaluate(self):
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

    def predict(self, website_url, category):
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
    
    def save_model(self, path='model.pkl'):
        """
        Save the trained model to disk.
        
        Args:
            path (str): Filename to save model data.
        """
        if self.model is not None:
            with open(PRODUCT_SCRAPER_SAVE_DIR / path, 'wb') as f:
                pickle.dump(self.model, f)
        else:
            raise ValueError("Model is not trained. Cannot save.")
        
    def load_model(self, path='model.pkl'):
        """
        Load a trained model from disk.
        
        Args:
            path (str): Filename to load model data from.
        """
        with open(PRODUCT_SCRAPER_SAVE_DIR / path, 'rb') as f:
            self.model = pickle.load(f)
    
    def save_dataframe(self, path='training_data.csv'):
        """
        Save the training dataframe to a CSV file.
        
        Args:
            path (str): Filename to save training data.
        """
        if self.training_data is not None:
            self.training_data.to_csv(PRODUCT_SCRAPER_SAVE_DIR / path, index=False)
        else:
            raise ValueError("Dataframe is empty. Cannot save.")
        
    def load_dataframe(self, path='training_data.csv'):
        """
        Load training dataframe from a CSV file.
        
        Args:
            path (str): Filename to load training data from.
        """
        self.training_data = pd.read_csv(PRODUCT_SCRAPER_SAVE_DIR / path)

    def save_selectors(self, path='selectors.pkl'):
        """
        Save collected selectors to disk.
        
        Args:
            path (str): Filename to save selectors.
        """
        with open(PRODUCT_SCRAPER_SAVE_DIR / path, 'wb') as f:
            pickle.dump(self.selectors, f)

    def load_selectors(self, path='selectors.pkl'):
        """
        Load previously collected selectors from disk.
        
        Args:
            path (str): Filename to load selectors from.
        """
        with open(PRODUCT_SCRAPER_SAVE_DIR / path, 'rb') as f:
            self.selectors = pickle.load(f)

    def save(self, path='product_scraper.pkl'):
        """
        Save the entire ProductScraper instance including model and data.
        
        Args:
            path (str): Filename to save the instance.
        """
        with open(PRODUCT_SCRAPER_SAVE_DIR / path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path='product_scraper.pkl'):
        """
        Load a previously saved ProductScraper instance.
        
        Args:
            path (str): Filename to load from.
            
        Returns:
            ProductScraper: Loaded instance.
        """
        with open(PRODUCT_SCRAPER_SAVE_DIR / path, 'rb') as f:
            return pickle.load(f)