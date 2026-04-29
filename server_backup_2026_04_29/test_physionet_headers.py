import requests
import os

url = "https://physionet.org/content/vindr-spinexr/get-zip/1.0.0/"
auth = ("umairejaz04", "Umair@825")
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

try:
    print(f"Testing connection to {url}...")
    # Just request headers first
    response = requests.head(url, auth=auth, headers=headers, allow_redirects=True)
    print(f"Response status: {response.status_code}")
    print(f"Headers: {response.headers}")
except Exception as e:
    print(f"Error: {e}")
