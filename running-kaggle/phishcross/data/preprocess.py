import re
from pathlib import Path

from bs4 import BeautifulSoup


def read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return ""


def clean_html_text(html: str, max_chars: int = 4000) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + " " + text[-half:]



class UrlVocab:
    def __init__(self):
        chars = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            "-._~:/?#[]@!$&'()*+,;=%"
        )
        self.pad_idx = 0
        self.unk_idx = 1
        self.stoi = {ch: i + 2 for i, ch in enumerate(chars)}

    @property
    def size(self):
        return len(self.stoi) + 2

    def encode(self, url: str, max_len: int):
        ids = [self.stoi.get(ch, self.unk_idx) for ch in (url or "")[:max_len]]
        if len(ids) < max_len:
            ids.extend([self.pad_idx] * (max_len - len(ids)))
        return ids

