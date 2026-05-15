# scraping from azlyrics.com because they just use plain html
import json
import os
import random
import re
import time
from threading import Lock
from typing import Any, List
from urllib.parse import quote, urljoin
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from dataclasses import asdict, dataclass
from datetime import datetime


@dataclass
class SongToScrape:
    title: str
    url: str


@dataclass
class Lyrics:
    title: str
    lyrics: str


@dataclass
class SongInfo:
    release_date: datetime | None
    album: str | None


@dataclass
class Song:
    title: str
    lyrics: str
    album: str | None
    release_date: datetime


load_dotenv()

_AZLYRICS_MIN_INTERVAL_SEC = 5.0
_AZLYRICS_JITTER_SEC = 2.0  # extra random delay; gaps are ~min..min+jitter
_GENIUS_MIN_INTERVAL_SEC = 2.0
_GENIUS_JITTER_SEC = 2.0
SKIP_URLS = [
    "https://www.azlyrics.com/lyrics/drake/whatyouneed.html",
    "https://www.azlyrics.com/lyrics/drake/zone.html",
]


class _RateLimiter:
    """Enforce a minimum delay between successive outbound calls, plus optional jitter."""

    def __init__(self, min_interval_sec: float, jitter_sec: float = 0.0) -> None:
        self._min = min_interval_sec
        self._jitter = max(0.0, jitter_sec)
        self._prev_end = 0.0
        self._lock = Lock()

    def throttle(self) -> None:
        with self._lock:
            now = time.monotonic()
            base_wait = self._min - (now - self._prev_end)
            extra = random.uniform(0.0, self._jitter) if self._jitter else 0.0
            wait = base_wait + extra
            if wait > 0:
                time.sleep(wait)
            self._prev_end = time.monotonic()


_azlyrics_limiter = _RateLimiter(_AZLYRICS_MIN_INTERVAL_SEC, _AZLYRICS_JITTER_SEC)
_genius_limiter = _RateLimiter(_GENIUS_MIN_INTERVAL_SEC, _GENIUS_JITTER_SEC)

_AZLYRICS_INDEX = "https://www.azlyrics.com/d/drake.html"

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


def _azlyrics_get(
    session: requests.Session, url: str, **kwargs: Any
) -> requests.Response:
    """Follow redirects without requests' 30-hop cap; fail fast on cycles (proxy/cookie loops)."""
    max_hops = int(os.getenv("AZLYRICS_MAX_REDIRECTS", "12"))
    timeout = float(os.getenv("AZLYRICS_REQUEST_TIMEOUT", "60"))
    visited: set[str] = set()
    current = url
    for _ in range(max_hops):
        if current in visited:
            raise RuntimeError(
                "AZLyrics redirect loop (revisited the same URL). "
                "Try: unset AZLYRICS_COOKIE, toggle OXYLABS_PROXY_SCHEME (http vs https), "
                "or run without proxy to confirm."
            )
        visited.add(current)
        r = session.get(current, allow_redirects=False, timeout=timeout, **kwargs)
        if r.status_code in _REDIRECT_STATUSES and "Location" in r.headers:
            current = urljoin(current, r.headers["Location"])
            continue
        return r
    raise RuntimeError(
        f"AZLyrics exceeded {max_hops} redirects (last target {current!r}). "
        "Proxy or site may be misconfigured."
    )


def _azlyrics_is_challenge_page(html: str) -> bool:
    h = html.lower()
    return "request for access" in h or ("az_unblock" in html and "g-recaptcha" in html)


def _azlyrics_raise_if_blocked(response: requests.Response, what: str) -> None:
    if response.status_code != 200:
        response.raise_for_status()
    if _azlyrics_is_challenge_page(response.text):
        raise RuntimeError(
            f"AZLyrics returned a bot-check page ({what}), not real content. "
            "reCAPTCHA cannot be solved by this script. Try: "
            "unset or refresh AZLYRICS_COOKIE / consent, slow down requests, "
            "or use a lyrics source that allows programmatic access."
        )


# Match desktop Chrome (same shape as DevTools for www.azlyrics.com).
# TLS + IP still differ from a real browser; optional AZLYRICS_COOKIE / AZLYRICS_COOKIE_CONSENT help.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/148.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Upgrade-Insecure-Requests": "1",
    "Sec-CH-UA": '"Not/A)Brand";v="99", "Chromium";v="148"',
    "Sec-CH-UA-Mobile": "?0",
    "Sec-CH-UA-Platform": '"macOS"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Priority": "u=0, i",
}

_azlyrics_http: requests.Session | None = None


def _oxylabs_proxies() -> dict[str, str] | None:
    """Oxylabs datacenter: tunnel user user-<account>-country-<CC>.

    Proxy URL uses http://… (CONNECT to Oxylabs); target sites stay HTTPS.
    Set OXYLABS_PROXY_SCHEME=https if your product requires it.
    """
    account = os.getenv("OXYLABS_USERNAME")
    password = os.getenv("OXYLABS_PASSWORD")
    if not account or not password:
        return None
    country = os.getenv("OXYLABS_COUNTRY", "US")
    scheme = os.getenv("OXYLABS_PROXY_SCHEME", "http").strip().rstrip(":")
    host = os.getenv("OXYLABS_PROXY_HOST", "dc.oxylabs.io:8000")
    tunnel_user = f"user-{account}-country-{country}"
    auth = f"{quote(tunnel_user, safe='')}:{quote(password, safe='')}"
    proxy_url = f"{scheme}://{auth}@{host}"
    return {"http": proxy_url, "https": proxy_url}


def _azlyrics_build_cookie_header() -> str | None:
    """Merge AZLYRICS_COOKIE with optional CookieConsent (Cookiebot-style) from env."""
    parts: list[str] = []
    if main := os.getenv("AZLYRICS_COOKIE", "").strip():
        parts.append(main.rstrip(";").strip())
    if consent := os.getenv("AZLYRICS_COOKIE_CONSENT", "").strip():
        c = consent.rstrip(";").strip()
        if not c.lower().startswith("cookieconsent="):
            c = f"CookieConsent={c}"
        parts.append(c)
    if not parts:
        return None
    return "; ".join(parts)


def _get_azlyrics_session() -> requests.Session:
    """Reuse one session so Set-Cookie from the index page is sent on lyric requests."""
    global _azlyrics_http
    if _azlyrics_http is None:
        _azlyrics_http = requests.Session()
        _azlyrics_http.trust_env = False  # avoid env HTTP(S)_PROXY stacking on Oxylabs
        _azlyrics_http.headers.update(_BROWSER_HEADERS)
        if (cookie_hdr := _azlyrics_build_cookie_header()) is not None:
            _azlyrics_http.headers["Cookie"] = cookie_hdr
        if (px := _oxylabs_proxies()) is not None:
            _azlyrics_http.proxies.update(px)
        # No separate "warm-up" GET: hitting / first can 30x redirect-loop behind some
        # proxies/cookies; cookies come from the first real page (artist index or lyrics).
    return _azlyrics_http


def get_all_songs(artist_name: str) -> List[SongToScrape]:
    url = _AZLYRICS_INDEX
    _azlyrics_limiter.throttle()
    response = _azlyrics_get(
        _get_azlyrics_session(),
        url,
        headers={
            "Sec-Fetch-Site": "none",
        },
    )
    _azlyrics_raise_if_blocked(response, "artist index")
    soup = BeautifulSoup(response.text, "html.parser")
    songs = soup.find_all("div", class_="listalbum-item")
    songs_to_scrape = []
    for song in songs:
        a_tag = song.find("a")
        title = song.text
        if a_tag is None:
            continue
        url = urljoin("https://www.azlyrics.com/", str(a_tag["href"]))
        if url in SKIP_URLS:
            print(f"Skipping song: {title} - {url}")
            continue
        songs_to_scrape.append(SongToScrape(title=title, url=url))
    return songs_to_scrape


def get_lyrics(song: SongToScrape) -> str:
    _azlyrics_limiter.throttle()
    try:
        response = _azlyrics_get(
            _get_azlyrics_session(),
            urljoin("https://www.azlyrics.com/", song.url),
            headers={
                "Referer": _AZLYRICS_INDEX,
                "Sec-Fetch-Site": "same-origin",
            },
        )
    except Exception as e:
        print(f"Error getting lyrics for song: {song.title} - {e}")
        raise
    _azlyrics_raise_if_blocked(response, f"lyrics page ({song.title!r})")
    soup = BeautifulSoup(response.text, "html.parser")
    selector = "html > body > div:nth-of-type(2) > div:nth-of-type(2) > div:nth-of-type(2) > div:nth-of-type(5)"
    lyrics_div = soup.select_one(selector)
    if lyrics_div is None:
        raise ValueError("Lyrics not found for song: " + song.title)
    lyrics_raw = lyrics_div.text
    segments = re.split(r"(?=\[.*?\])", lyrics_raw)
    lyrics: str = ""
    for segment in segments:
        if not segment:
            continue
        if segment.startswith("["):
            tag_text = segment.split("[", 1)[1]
            artist = None
            if ":" in tag_text:
                artist = tag_text.split(":", 1)[0].strip().lower()

            if artist and artist != "drake":
                continue
            _, _, after_tag = segment.partition("]")
            lyrics += after_tag.strip() + "\n"
        else:
            lyrics += segment.strip() + "\n"
    return lyrics


def get_song_id(title: str) -> str | None:
    search_url = "https://api.genius.com/search"
    headers = {"Authorization": f"Bearer {os.getenv('GENIUS_ACCESS_TOKEN')}"}
    params = {"q": "Drake " + title}

    _genius_limiter.throttle()
    response = requests.get(search_url, headers=headers, params=params)

    if response.status_code == 200:
        hits = response.json()["response"]["hits"]

        if hits:
            # Get the ID of the first (top) result
            top_hit = hits[0]["result"]
            print(f"Found: {top_hit['full_title']} (ID: {top_hit['id']})")
            return top_hit["id"]
        else:
            print("No results found.")
            return None
    else:
        print(f"Search failed: {response.status_code}")
        return None


def get_song_info(song_id: str) -> SongInfo | None:
    url = f"https://api.genius.com/songs/{song_id}"
    headers = {"Authorization": f"Bearer {os.getenv('GENIUS_ACCESS_TOKEN')}"}
    _genius_limiter.throttle()
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
    else:
        print(f"Search failed: {response.status_code}")
        return None

    if (
        "drake" not in str(data["response"]["song"]["artist_names"]).lower()
        and "ft. drake" not in str(data["response"]["song"]["artist_names"]).lower()
    ):
        print(f"Song not by Drake: {data['response']['song']['artist_names']}")
        return None

    release_date = data["response"]["song"]["release_date"]

    try:
        album = data["response"]["song"]["album"]["name"]
    except TypeError:
        album = None

    if release_date is not None:
        release_date = datetime.strptime(release_date, "%Y-%m-%d")
    return SongInfo(
        release_date=release_date,
        album=album,
    )


def save_lyrics():
    songs_to_scrape = get_all_songs("drake")
    print(f"Found {len(songs_to_scrape)} songs to scrape")
    for song_to_scrape in reversed(songs_to_scrape):
        # get lyrics
        lyrics = get_lyrics(song_to_scrape)
        # get song id
        song_id = get_song_id(song_to_scrape.title)
        if song_id is None:
            print("Song not found")
            continue
        # get song info
        song_info = get_song_info(song_id)
        if song_info is None:
            print(f"Song info not found for song: {song_to_scrape.title}")
            continue
        if song_info.release_date is None:
            print(f"Release date not found for song: {song_to_scrape.title}")
            continue
        # save lyrics
        song = Song(
            title=song_to_scrape.title,
            lyrics=lyrics,
            album=song_info.album,
            release_date=song_info.release_date,
        )
        out_path = f"songs/album/{song_info.album}/{song_to_scrape.title}.txt"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        payload = asdict(song)
        payload["release_date"] = (
            song.release_date.isoformat() if song.release_date is not None else None
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    save_lyrics()
