#!/bin/bash

if [ $# -eq 0 ]; then
      echo "Usage: $0 frame1 frame2 ..."
      exit 1
fi
output_file="$1"
shift

if [ $# -eq 0 ]; then
      echo "Usage: $0 <output_file> <frame1> <frame2> ..."
      exit 1
fi

if [[ "$output_file" != *.mp4 ]]; then
      echo "Error: output file must end with .mp4"
      exit 1
fi

tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT



num=0
while [ $# -gt 0 ]; do
      num=$((++num))
      ln -s "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")" "$tmpdir/$(printf '%09d.png' $num)"
      shift
done

ffmpeg -framerate 20 -i "$tmpdir/%09d.png" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -crf 28 -preset medium -pix_fmt yuv420p "$output_file"