import os
import requests
import zipfile
import io

def download_and_unzip(url, save_dir):
    """Download and unzip a file from a URL."""
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get the file name from the URL
    file_name = url.split('/')[-1]
    save_path = os.path.join(save_dir, file_name)

    # Download the file
    print("Downloading", file_name)
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print("Download completed")

    # Unzip the downloaded file
    print("Extracting files")
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(save_dir)
    print("Extraction completed")

# Example usage
url = "http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip"
save_directory = "./data"
download_and_unzip(url, save_directory)
