# AIP Downloader
## Create a new bundle
```bash
python main.py new
```

## Download ICAO Maptiles
### Important notes
This requires an old package of `setuptools<82`.
Added to requirements.txt
> Do not use with --country=germany. this causes an error which is related to the bbox api.

```bash
download-tiles icao500.mbtiles \
  --tiles-url "https://ais.dfs.de/static-maps/icao500/tiles/{z}/{x}/{y}.png" \
  --zoom-levels 6-12 \
  --bbox "5.866,47.270,15.042,55.099"
```

Command takes quite some time (like 5-10 Minutes)
