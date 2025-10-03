#!/usr/bin/env python3
"""
Script to fetch NASA Earthdata using the CMR API.
Based on the provided instructions for accessing Earthdata Search programmatically.

This script demonstrates how to search for and download Earthdata products like IMERG precipitation data.
"""

import requests
import json
import os
import subprocess
import h5py
import numpy as np
from datetime import datetime, timedelta

# Earthdata authentication token
EARTHDATA_TOKEN = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImRhYXZpZG1vcmVubyIsImV4cCI6MTc2NDY3NzkxNywiaWF0IjoxNzU5NDkzOTE3LCJpc3MiOiJodHRwczovL3Vycy5lYXJ0aGRhdGEubmFzYS5nb3YiLCJpZGVudGl0eV9wcm92aWRlciI6ImVkbF9vcHMiLCJhY3IiOiJlZGwiLCJhc3N1cmFuY2VfbGV2ZWwiOjN9.IbaWyqqro8pSaJy85RNmekWE_nfM4WxOpam6UBrR-0MMGeBXeuM1gnc-rSAi-oDm20tKDF9zsous_gLZ0O0q3qG0m58spPqpPZsctMx1XxoKB0Sk4iC7PoZFDvB1rl6ujoRJTewAWZvlwpe3Xm9xHL2mVbFw4PwdbAO4rHPXZDQmJ3_rSY-_CPh2Pzuc8UO5d9lgN9LYUtkRYVIXtYO-Scmx5Nz3B6uSlLBP95jGkndhrMwgforfl1E8-hIR6-CWduRiWtsd8faxnlRNoppoliT_m0gr2hROU8yUeKUJbM320dbjvpqL8lgLXasrHpyfJWRLCzER5jdMir01RJCZmg"

class EarthdataFetcher:
    def __init__(self):
        self.cmr_base_url = "https://cmr.earthdata.nasa.gov/search/granules.json"

    def search_granules(self, short_name, version, temporal_range, bounding_box=None, day_night_flag=None):
        """
        Search for granules using CMR API.

        Args:
            short_name (str): Product short name (e.g., 'GPM_3IMERGHH')
            version (str): Product version (e.g., '07')
            temporal_range (tuple): (start_date, end_date) in ISO format
            bounding_box (list): [west, south, east, north] coordinates
            day_night_flag (str): 'day', 'night', or 'both'

        Returns:
            list: List of granule metadata
        """
        params = {
            "short_name": short_name,
            "version": version,
            "temporal": f"{temporal_range[0]}Z,{temporal_range[1]}Z"
        }

        if bounding_box:
            params["bounding_box"] = f"{bounding_box[0]},{bounding_box[1]},{bounding_box[2]},{bounding_box[3]}"

        if day_night_flag:
            params["day_night_flag"] = day_night_flag

        print(f"Searching for {short_name} v{version} from {temporal_range[0]} to {temporal_range[1]}")
        if bounding_box:
            print(f"Bounding box: {bounding_box}")

        response = requests.get(self.cmr_base_url, params=params)
        response.raise_for_status()

        data = response.json()
        granules = data.get('feed', {}).get('entry', [])

        print(f"Found {len(granules)} granules")
        return granules

    def download_granule(self, granule, download_dir="downloads"):
        """
        Download a single granule using requests with Earthdata token.

        Args:
            granule (dict): Granule metadata from CMR
            download_dir (str): Directory to save downloads
        """
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # Get download URL
        links = granule.get('links', [])
        download_url = None
        for link in links:
            if link.get('rel') == 'http://esipfed.org/ns/fedsearch/1.1/data#':
                download_url = link.get('href')
                break

        if not download_url:
            print(f"No download URL found for granule {granule.get('id')}")
            return False

        filename = download_url.split('/')[-1]
        filepath = os.path.join(download_dir, filename)

        print(f"Downloading {filename}...")

        # Use requests with Bearer token in header
        headers = {'Authorization': f'Bearer {EARTHDATA_TOKEN}'}
        try:
            response = requests.get(download_url, headers=headers, stream=True, timeout=300)
            if response.status_code == 403:
                # Parse resolution URL from response
                import re
                match = re.search(r'resolution_url=(https://[^&]+)', response.text)
                if match:
                    print(f"Authorization required. Please visit {match.group(1)} to approve the application, then try again.")
                else:
                    print("Download error: 403 Forbidden")
                return False
            elif response.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Saved to {filepath}")
                return True
            else:
                print(f"Download error: {response.status_code}")
                return False
        except Exception as e:
            print(f"Download error: {e}")
            return False

    def fetch_precipitation_data(self, start_date, end_date, bbox=None, download_dir="downloads/precipitation"):
        """
        Fetch IMERG precipitation data.

        Args:
            start_date (str): Start date in YYYY-MM-DDTHH:MM:SS format
            end_date (str): End date in YYYY-MM-DDTHH:MM:SS format
            bbox (list): Bounding box [west, south, east, north]
            download_dir (str): Download directory
        """
        print("Fetching IMERG Final Run precipitation data...")

        granules = self.search_granules(
            short_name="GPM_3IMERGHH",
            version="07",
            temporal_range=(start_date, end_date),
            bounding_box=bbox
        )

        downloaded = 0
        for granule in granules:
            if self.download_granule(granule, download_dir):
                downloaded += 1

        print(f"Downloaded {downloaded} out of {len(granules)} granules")

    def process_precipitation_hdf5(self, download_dir="downloads/precipitation", output_dir="processed_data/precipitation"):
        """
        Process downloaded IMERG HDF5 files to extract precipitationCal variable.

        Args:
            download_dir (str): Directory with HDF5 files
            output_dir (str): Directory to save processed data
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        hdf5_files = [f for f in os.listdir(download_dir) if f.endswith('.HDF5')]
        print(f"Processing {len(hdf5_files)} HDF5 files...")

        for filename in hdf5_files:
            filepath = os.path.join(download_dir, filename)
            output_file = os.path.join(output_dir, filename.replace('.HDF5', '.csv'))

            try:
                # Load the HDF5 file and extract precipitation data
                with h5py.File(filepath, 'r') as f:
                    precip_data = f['Grid/precipitation'][0]  # Take the first time slice

                # Save to output directory as CSV
                np.savetxt(output_file, precip_data, delimiter=',')

                print(f"Processed {filename}: shape {precip_data.shape}, "
                      f"min {precip_data.min():.4f}, max {precip_data.max():.4f}, "
                      f"mean {precip_data.mean():.4f}")
                print(f"Saved to {output_file}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        print(f"Processing complete. Data saved to {output_dir}")

    def fetch_sea_level_data(self, start_date, end_date, bbox=None, download_dir="downloads/sea_level"):
        """
        Fetch sea surface height anomaly data.

        Args:
            start_date (str): Start date in YYYY-MM-DDTHH:MM:SS format
            end_date (str): End date in YYYY-MM-DDTHH:MM:SS format
            bbox (list): Bounding box [west, south, east, north]
            download_dir (str): Download directory
        """
        print("Fetching Sea Surface Height Anomaly data...")

        granules = self.search_granules(
            short_name="SEA_SURFACE_HEIGHT_ALT_GRIDS_L4_2SATS_5DAY_66N_66S",
            version="3",
            temporal_range=(start_date, end_date),
            bounding_box=bbox
        )

        downloaded = 0
        for granule in granules:
            if self.download_granule(granule, download_dir):
                downloaded += 1

        print(f"Downloaded {downloaded} out of {len(granules)} granules")

    def fetch_aerosol_data(self, start_date, end_date, bbox=None, download_dir="downloads/aerosols"):
        """
        Fetch aerosol optical depth data.

        Args:
            start_date (str): Start date in YYYY-MM-DDTHH:MM:SS format
            end_date (str): End date in YYYY-MM-DDTHH:MM:SS format
            bbox (list): Bounding box [west, south, east, north]
            download_dir (str): Download directory
        """
        print("Fetching MCD19A2 aerosol data...")

        granules = self.search_granules(
            short_name="MCD19A2.061",
            version="061",
            temporal_range=(start_date, end_date),
            bounding_box=bbox
        )

        downloaded = 0
        for granule in granules:
            if self.download_granule(granule, download_dir):
                downloaded += 1

        print(f"Downloaded {downloaded} out of {len(granules)} granules")

def main():
    """
    Example usage of the EarthdataFetcher.
    """
    fetcher = EarthdataFetcher()

    # Example: Fetch precipitation data for Spain in January 2023
    spain_bbox = [-9.5, 36, 3.5, 44]  # [west, south, east, north]

    start_date = "2023-01-01T00:00:00"
    end_date = "2023-01-01T23:59:59"

    print("NASA Earthdata Fetcher")
    print("======================")
    print("This script uses requests with Earthdata token for authentication.")
    print("It can also process downloaded HDF5 files to extract variables.\n")

    # Fetch precipitation data
    fetcher.fetch_precipitation_data(start_date, end_date, spain_bbox)

    # Process the downloaded HDF5 files
    fetcher.process_precipitation_hdf5()

    # Fetch sea level data (global, no bbox needed for this example)
    # fetcher.fetch_sea_level_data(start_date, end_date)

    # Fetch aerosol data
    # fetcher.fetch_aerosol_data(start_date, end_date, spain_bbox)

if __name__ == "__main__":
    main()