import os
import requests
import datetime
import zipfile
from time import sleep

BASE_URL = "http://data.gdeltproject.org/events"
OUT_DIR = "/Users/jeremyserillon/Downloads/test"
START_DATE = datetime.date(2016, 1, 1)
END_DATE   = datetime.date(2024, 12, 31)

os.makedirs(OUT_DIR, exist_ok=True)

def download(url, dest, retries=3, timeout=30):
    """Stream download with basic retries."""
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code == 200:
                    with open(dest, "wb") as f:
                        for chunk in r.iter_content(1024 * 1024):
                            if chunk:
                                f.write(chunk)
                    return True
                elif r.status_code == 404:
                    return False  # file doesn't exist
                else:
                    print(f"⚠️ HTTP {r.status_code} for {url}")
        except Exception as e:
            print(f"❌ Attempt {attempt}/{retries} failed for {url}: {e}")
        sleep(2 * attempt)
    return False

def extract_and_delete(zip_path, out_dir):
    """Extract CSV from the zip and delete the zip if successful."""
    if not os.path.exists(zip_path):
        return False
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            # Usually there is exactly one CSV inside
            members = [m for m in z.namelist() if m.lower().endswith(".csv")]
            if not members:
                print(f"⚠️ No CSV in {zip_path}")
                return False
            z.extractall(out_dir, members)
        os.remove(zip_path)
        return True
    except zipfile.BadZipFile:
        print(f"⚠️ Corrupt ZIP: {zip_path}. Deleting and will redownload next run.")
        try:
            os.remove(zip_path)
        except OSError:
            pass
        return False

date = START_DATE
one_day = datetime.timedelta(days=1)

while date <= END_DATE:
    ymd = date.strftime("%Y%m%d")
    zip_name = f"{ymd}.export.CSV.zip"
    csv_name = f"{ymd}.export.CSV"          # name inside the zip
    zip_path = os.path.join(OUT_DIR, zip_name)
    csv_path = os.path.join(OUT_DIR, csv_name)

    # Skip if CSV already extracted
    if os.path.exists(csv_path):
        print(f"⏩ CSV exists: {csv_name}")
    else:
        # If we don't have the zip, try to download it
        if not os.path.exists(zip_path):
            url = f"{BASE_URL}/{zip_name}"
            ok = download(url, zip_path)
            if ok is False:
                # 404 or couldn't download after retries
                print(f"🚫 Missing or unavailable: {zip_name}")
                date += one_day
                sleep(0.2)
                continue

        # Try extracting and deleting the zip
        if extract_and_delete(zip_path, OUT_DIR):
            print(f"✅ Extracted & cleaned: {csv_name}")
        else:
            # If extraction failed, CSV still missing: mark and move on
            if not os.path.exists(csv_path):
                print(f"❌ Failed to extract: {zip_name}")

    date += one_day
    sleep(0.2)  # be polite to the server
