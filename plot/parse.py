import json
from dataclasses import dataclass
from typing import List
import re

@dataclass
class Song:
    title: str
    lyrics: str
    album: str | None
    year: int
    month: int
    day: int


def process_file(filename: str) -> List[Song]:
    with open(filename, "r") as f:
        data = json.load(f)

    songs = []
    for song in data["songs"]:
        artists = song["primary_artist_names"].map(lambda x: x["name"])
        lines = re.split(r'(\[.*?\])', song["lyrics"])
        lyrics = ""
        for line in lyrics:
            if line is None:
                continue
            if line.startswith("["):
                artists = line.split(":")[1].strip()
            lyrics += line + "\n"
        songs.append(Song(
            title=song["title"],
            lyrics=song["lyrics"],
            album=song["album"]["name"],
            year=song["release_date_components"]["year"],
            month=song["release_date_components"]["month"],
            day=song["release_date_components"]["day"]
        ))
    return songs
