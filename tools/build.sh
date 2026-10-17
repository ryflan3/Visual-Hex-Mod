#!/usr/bin/env bash
# Re-render the tileset images, pack them into game.atlas/game*.png, and refresh the previews.
#   tools/build.sh              # everything
#   tools/build.sh Grassland    # only re-render images whose path contains "Grassland", then pack
set -euo pipefail
cd "$(dirname "$0")/.."

python3 tools/render_tiles.py "$@"
python3 tools/render_items.py "$@"

GDX_VERSION=1.14.2   # the libGDX version Unciv is built with
GDX_DIR="${GDX_DIR:-.gdx}"
mkdir -p "$GDX_DIR"
for jar in gdx gdx-tools; do
  if [ ! -f "$GDX_DIR/$jar.jar" ]; then
    curl -sSfL -o "$GDX_DIR/$jar.jar" \
      "https://repo1.maven.org/maven2/com/badlogicgames/gdx/$jar/$GDX_VERSION/$jar-$GDX_VERSION.jar"
  fi
done
java -cp "$GDX_DIR/gdx.jar:$GDX_DIR/gdx-tools.jar" tools/PackAtlas.java .

python3 tools/preview_map.py --out preview/preview.jpg
python3 tools/preview_map.py --width 160 --cols 9 --rows 6 --seed 9 --out preview/close-up.jpg
python3 tools/preview_map.py --fog --out preview/fog.jpg
python3 tools/preview_map.py --catalog
