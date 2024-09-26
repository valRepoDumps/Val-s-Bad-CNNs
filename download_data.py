"""
Download data from a given URL and save it to file.
"""

def download_data(url: str, file_path: str, remove_source: bool = True, zip: bool = True):
    """
    Download data from a given url, unzip it, and save it to file.

    Args:
    url: the link to the file.
    file_path: the path to save the file.
    remove_source: whether to remove the source file. (should be true in case of zip file.)
    zip: whether the file is a zip file. (True will extrac the zip file.)
    """
    import requests
    import os
    import zipfile
    from pathlib import Path

    # Download the file

    path = Path(file_path)

    if path.is_dir():
        print(f"Image path directory exist, skipping download.")

    else:
        path.mkdir(parents = True, exist_ok = True) #create path

        response = requests.get(url) #get the file content
        file_name = Path(url).name #get the name of the zipfile

        with open(path/file_name, 'wb') as f:
            print(f"[INFO] Downloading {file_name} from {url}")
            f.write(response.content) #write the content


        if zip:
            with zipfile.ZipFile(path/file_name, 'r') as zip_ref:
                print(f"[INFO] Unzipping {file_name}")
                zip_ref.extractall(path) #unzip the file

            if remove_source:
                os.remove(path/file_name) #remove the source file
                print(f"[INFO] Removed {file_name}")
