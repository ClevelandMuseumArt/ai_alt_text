import os
import csv
import sys
import math
import time
import argparse
import requests
import logging
from datetime import datetime
from cma_piction import PictionSession


class LoadCollectionData:
    def __init__(self, testing=False) -> None:
        self.TESTING = testing
        self.CO_API = "https://oai-api-02.clevelandart.org/api/collection/artworks"
        self.LIMIT = 1000
        self.SKIP = 0
        self.TOTAL_ART = math.inf
        self.TODAY_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.DATA_FOLDER = "image_data"
        self.DATA_CSV = f"{self.TODAY_TS}_co_api_data.csv"
        self.CSV_COLS = ["athena_id", "accession_number", "image_src", "UMO_ID"]
        self.CWD = os.getcwd()
        self.CO_PARAMS = {"has_image": 1, "limit": self.LIMIT, "fields": "image_assets"}

        if self.TESTING:
            self.CO_PARAMS["randomize"] = 1
            self.TOTAL_ART = 500
            self.CO_PARAMS["limit"] = self.TOTAL_ART

        self.DATA_PATH = os.path.join(self.CWD, self.DATA_FOLDER)
        self.ERROR_LOG_PATH = os.path.join(self.DATA_PATH, f"{self.TODAY_TS}_errors.log")

        self.piction = PictionSession({
            'username': 'DMZ GLOBAL EXTRA2',
            'password': 'yR0zbGa5u9ZQS2F6',
            'root_url': 'https://piction.clevelandart.org'
        })

        # Set up logger
        self.logger = logging.getLogger(__name__)

    def set_up(self):
        """Initialize data directory and CSV file."""
        try:
            full_data_path = os.path.join(self.DATA_PATH, self.DATA_CSV)

            # Remove existing file if present
            if os.path.exists(full_data_path):
                self.logger.warning(f"Removing existing data file: {full_data_path}")
                os.remove(full_data_path)

            # Create directory if it doesn't exist
            if not os.path.exists(self.DATA_PATH):
                os.makedirs(self.DATA_PATH)
                self.logger.info(f"Created data directory: {self.DATA_PATH}")

            # Write CSV header
            self.write_data_row(self.CSV_COLS)
            self.logger.info(f"Initialized CSV file: {full_data_path}")
            return True

        except Exception as e:
            self.logger.error(f"Setup failed: {e}", exc_info=True)
            self._log_error(f"Setup error: {e}")
            return False

    def get_img_url(self, img_data):
        """Extract image URL from image data, trying multiple size options."""
        size_priorities = ["web", "web_large", "print", "medium", "small"]
        image_assets = img_data.get('image_assets')
        primary_image = image_assets.get('primary_image')
        for size in size_priorities:
            if primary_image.get(size) and primary_image[size].get("url"):
                self.logger.debug(f"Found image URL in '{size}' size")
                return primary_image[size]["url"]

        self.logger.warning("No image URL found in any size option")
        return None

    def has_manually_written_alt_text(self, img_data, acc_nbr):
        """Retrieve if image has manually written alt text."""
        try:
            is_human_reviewed = img_data.get('human_reviewed')
            if is_human_reviewed:
                self.logger.debug(f"{acc_nbr} is human_reviewed: {is_human_reviewed}")
                return is_human_reviewed
            else:
                img_data = self.piction.get_image_data_object(acc_nbr)
                has_manual_alt_text = img_data.get_metadata('ALT_TEXT_MEETS_THRESHOLD', 0)
                self.logger.debug(f"piction ALT_TEXT_MEETS_THRESHOLD metadata for {acc_nbr}: {has_manual_alt_text}")
                return has_manual_alt_text == 'OVERWRITTEN'
        except Exception as e:
            self.logger.error(f"Failed to get metadata ID for {acc_nbr}: {e}")
            return None


    def get_piction_umo_id(self, img_data, acc_nbr):
        """Retrieve UMO ID from CO-API or Piction system."""
        try:
            image_assets = img_data.get('image_assets')
            primary_image = image_assets.get('primary_image')
            if primary_image.get('umo_id'):
                umo_id = primary_image.get('umo_id')
            else:
                img_data = self.piction.get_image_data_object(acc_nbr)
                umo_id = img_data.get_umo_id()
            self.logger.debug(f"Retrieved UMO ID for {acc_nbr}: {umo_id}")
            return umo_id
        except Exception as e:
            self.logger.error(f"Failed to get UMO ID for {acc_nbr}: {e}")
            return None

    def write_data_row(self, row):
        """Write a single row to the CSV file."""
        csv_path = os.path.join(self.DATA_PATH, self.DATA_CSV)
        try:
            with open(csv_path, "a", newline='') as dcsv:
                writer = csv.writer(dcsv)
                writer.writerow(row)
            self.logger.debug(f"Wrote row to CSV: {row[0] if row else 'header'}")
        except Exception as e:
            self.logger.error(f"Failed to write CSV row: {e}")
            self._log_error(f"CSV write error for row {row}: {e}")

    def create_data_row(self, artwork_data):
        """Process artwork data and write to CSV."""
        try:
            acc_nbr = artwork_data.get('accession_number')
            athena_id = artwork_data.get('id') or artwork_data.get('athena_id')

            if not athena_id:
                self.logger.warning("Artwork missing athena_id, skipping")
                return


            img_url = self.get_img_url(artwork_data)

            if acc_nbr is not None:
                piction_umo = self.get_piction_umo_id(artwork_data, acc_nbr)
                if piction_umo is None:
                  self.logger.warning(f"Artwork {athena_id} missing piction umo id")
                  return

                has_manual_alt_text = self.has_manually_written_alt_text(artwork_data, acc_nbr)

                if has_manual_alt_text:
                  self.logger.warning(f"Skipping artwork {athena_id} ({acc_nbr}) has manually written alt text")
                  return

                self.write_data_row([athena_id, acc_nbr, img_url, piction_umo])
                self.logger.info(f"Processed artwork {athena_id} ({acc_nbr})")
            else:
                self.logger.warning(f"Artwork {athena_id} missing accession number")

        except Exception as e:
            self.logger.error(f"Failed to create data row: {e}", exc_info=True)
            self._log_error(f"Data row creation error: {e}")

    def retrieve_co_data(self, uri):
        """Retrieve artwork data from Cleveland Museum API."""
        try:
            self.logger.debug(f"Requesting data from: {uri}")
            r = requests.get(uri, params=self.CO_PARAMS, timeout=30)
            r.raise_for_status()

            data = r.json()

            if data.get("data") is None:
                self.logger.warning(f"No data returned from {uri}")
                return

            if isinstance(data['data'], list):
                if not self.TESTING:
                    self.TOTAL_ART = data.get('info', {}).get('total', 0)
                    self.logger.info(f"Total artworks available: {self.TOTAL_ART}")

                artworks = data.get("data", [])
                self.logger.info(f"Processing {len(artworks)} artworks from batch")

                for artwork in artworks:
                    self.create_data_row(artwork)
            else:
                self.create_data_row(data.get('data'))

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for {uri}: {e}")
            athena_id = self._extract_athena_id(uri)
            self._log_error(f"Request error {athena_id}: {uri} -> {e}")

        except Exception as e:
            self.logger.error(f"Unexpected error retrieving data from {uri}: {e}", exc_info=True)
            athena_id = self._extract_athena_id(uri)
            self._log_error(f"Unexpected error {athena_id}: {uri} -> {e}")

    def download_all(self):
        """Download all artworks with images from the collection."""
        self.logger.info("Starting download of all artworks")

        while self.SKIP < self.TOTAL_ART:
            remaining = self.TOTAL_ART - self.SKIP
            self.logger.info(f"Progress: {self.SKIP}/{self.TOTAL_ART} ({remaining} remaining)")

            if self.SKIP:
                self.CO_PARAMS['skip'] = self.SKIP

            self.retrieve_co_data(self.CO_API)
            self.SKIP += self.LIMIT

        self.logger.info("Completed downloading all artworks")

    def download_curated(self, art_ids):
        """Download specific artworks by ID."""
        self.logger.info(f"Starting curated download of {len(art_ids)} artworks")

        try:
            for idx, art_id in enumerate(art_ids, 1):
                self.logger.info(f"Downloading artwork {idx}/{len(art_ids)}: {art_id}")
                co_url = f"{self.CO_API}/{art_id}"
                self.retrieve_co_data(co_url)

        except Exception as e:
            self.logger.error(f"Unexpected error in curated download: {e}", exc_info=True)
            self._log_error(f"Curated download error: {e}")

    def create_dataset(self, art_ids=None):
        """Create the image training dataset."""
        self.logger.info("Creating image training dataset")
        start = time.perf_counter()

        try:
            if art_ids is None:
                self.download_all()
            elif isinstance(art_ids, list):
                self.download_curated(art_ids)
            else:
                raise ValueError("Artwork IDs must be formatted as a list")

            end = time.perf_counter()
            elapsed = end - start

            csv_path = os.path.join(self.DATA_PATH, self.DATA_CSV)

            self.logger.info(f"Dataset created successfully")
            self.logger.info(f"Mode: {'TEST' if self.TESTING else 'PRODUCTION'}")
            self.logger.info(f"Total artworks: {self.TOTAL_ART}")
            self.logger.info(f"Time elapsed: {elapsed:.2f} seconds")
            self.logger.info(f"Dataset CSV: {csv_path}")

            return csv_path

        except Exception as e:
            self.logger.error(f"Failed to create dataset: {e}", exc_info=True)
            raise

    def _extract_athena_id(self, uri):
        """Extract Athena ID from URI for error logging."""
        parts = uri.split('/artworks/')
        if len(parts) > 1 and parts[-1].isdigit():
            return f"ATHENA_ID:{parts[-1]}"
        return ''

    def _log_error(self, message):
        """Write error to separate error log file."""
        try:
            with open(self.ERROR_LOG_PATH, 'a') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            self.logger.error(f"Failed to write to error log: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Load Cleveland Museum of Art collection data"
    )
    parser.add_argument(
        '--test',
        type=int,
        help="Run in test mode (1) or production mode (0)",
        required=False,
        default=0,
        choices=[0, 1]
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file (default: logs to console only)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        '--art-ids',
        type=str,
        default=None,
        help='Comma-separated list of Athena IDs or accession numbers to download'
    )
    args = parser.parse_args()

    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers = [logging.StreamHandler()]
    if args.log_file:
        log_filename = f"{args.log_file}_{timestamp}.log"
        handlers.append(logging.FileHandler(log_filename))

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=log_format,
        handlers=handlers
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Starting Collection Data Loader")
    logger.info(f"Log level: {args.log_level}")
    logger.info(f"Test mode: {'ENABLED' if args.test == 1 else 'DISABLED'}")
    logger.info("=" * 60)

    # Initialize data loader
    test_mode = args.test == 1
    training_data = LoadCollectionData(testing=test_mode)

    # Set up data directory
    setup_status = training_data.set_up()
    if not setup_status:
        logger.critical("Setup failed. Exiting.")
        sys.exit(-1)

    # Parse art IDs if provided
    art_ids = None
    if args.art_ids:
        art_ids = [id.strip() for id in args.art_ids.split(',')]
        logger.info(f"Curated download mode: {len(art_ids)} IDs provided")

    # Create dataset
    try:
        training_data_path = training_data.create_dataset(art_ids)
        logger.info(f"Dataset successfully created at: {training_data_path}")
        return training_data_path
    except Exception as e:
        logger.critical(f"Dataset creation failed: {e}", exc_info=True)
        sys.exit(-1)


if __name__ == "__main__":
    main()
