import os
import subprocess
import urllib.request
import zipfile


def ensure_deno():

    deno_dir = os.path.expanduser("~/.deno/bin")
    deno_path = os.path.join(deno_dir, "deno")

    # Already installed
    if os.path.exists(deno_path):
        return deno_path

    os.makedirs(deno_dir, exist_ok=True)

    print("Installing Deno...")

    url = "https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip"

    zip_path = "/tmp/deno.zip"

    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extract("deno", deno_dir)

    os.chmod(deno_path, 0o755)

    subprocess.run(
        [deno_path, "--version"],
        check=True
    )

    print("Deno installed successfully.")

    return deno_path