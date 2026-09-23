#!/usr/bin/env bash
# 720p H.264 (8-bit 4:2:0, browser-playable) copies of the sample videos for labelling
# and the website. Dev-only: uses the system ffmpeg. Originals stay untouched.
#
#   tools/make_proxies.sh [samples_dir]
set -euo pipefail

SRC="${1:-samples}"
OUT="$SRC/proxy"
mkdir -p "$OUT"
for video in "$SRC"/*.MP4 "$SRC"/*.mp4; do
  [ -e "$video" ] || continue
  name="$(basename "${video%.*}")_720p.mp4"
  if [ -s "$OUT/$name" ]; then
    echo "skip $name (exists)"
    continue
  fi
  ffmpeg -hide_banner -loglevel error -stats -i "$video" \
    -vf scale=1280:-2 -pix_fmt yuv420p -c:v libx264 -preset veryfast -crf 23 -an \
    -movflags +faststart "$OUT/$name.part.mp4"
  mv "$OUT/$name.part.mp4" "$OUT/$name"
  echo "wrote $OUT/$name"
done
