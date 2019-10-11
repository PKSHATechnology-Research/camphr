from pathlib import Path
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
        if d["tag_name"] == tag:
            return d["upload_url"]
    raise ValueError(f"tag {tag} not found")


def make_upload_url(url: str, file: str):
    url = url.rstrip("{?name,label}")
    file = Path(file).name
    return url + f"?name={file}"


def make_header(file: str, token: str):
    header = auth_header(token)
    header["Content-Type"] = "application/octet-stream"
    return header


def main(file, tag, repo, token):
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
