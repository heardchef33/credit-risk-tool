from src.config.core import ROOT, config

with open(ROOT / "VERSION") as version_file: 
    __version__ = version_file.read().strip()