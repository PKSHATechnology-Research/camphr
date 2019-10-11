from pathlib import Path
import os
import re
import subprocess
import fire
import requests

ENDPOINT = "https://api.github.com"


def auth_header(token):
    return {"Authorization": f"token {token}"}


def get_upload_url(tag, repo, token) -> str:
    res = requests.get(
        ENDPOINT + f"/repos/{repo}/releases", headers=auth_header(token)
    ).json()
    for d in res:
        try:
            if d["tag_name"] == tag:
                return d["upload_url"]
        except:
            print(res)
            exit(1)
    raise ValueError(f"tag {tag} not found")


def make_upload_url(url: str, file: str):
    url = url.rstrip("{?name,label}")
    file = Path(file).name
    return url + f"?name={file}"


def make_header(file: str, token: str):
    header = auth_header(token)
    header["Content-Type"] = "application/octet-stream"
    return header


def get_repo() -> str:
    remote_versbose = subprocess.check_output(["git", "remote", "-v"]).decode()
    url = None
    for line in remote_versbose.split("\n"):
        items = line.split()
        if not items:
            continue
        if items[0] == "origin":
            url = items[1]
    assert url
    repo = re.findall(r"github.com[:/](.*?/.*)", url)[0]
    repo = repo.rstrip(".git")
    return repo


def get_token() -> str:
    token = os.getenv("GITHUB_TOKEN", "")
    if not token:
        token = input("Input github token: ")
    return token


def main(file, tag, token: str = ""):
    token = get_token()
    repo = get_repo()
    print(f"repo: {repo}")
    url = get_upload_url(tag, repo, token)
    url = make_upload_url(url, file)
    header = make_header(file, token)
    print(header)
    print(url)
    with open(file, "rb") as f:
        data = f.read()
    res = requests.post(url, headers=header, data=data)
    print(res.json())


if __name__ == "__main__":
    fire.Fire(main)
