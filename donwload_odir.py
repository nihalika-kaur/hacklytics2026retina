import os
import subprocess

def download_odir(dest_dir="data/odir5k"):
    os.makedirs(dest_dir, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", "andrewmvd/odir5k", "-p", dest_dir, "--unzip"]
    subprocess.check_call(cmd)
    print("Downloaded to", dest_dir)

if __name__ == "__main__":
    download_odir()