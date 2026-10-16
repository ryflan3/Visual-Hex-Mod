#!/usr/bin/env python3
"""Lay the tileset's images out on a sample map the way Unciv does, and save a PNG.

    python3 tools/preview_map.py                      # writes preview/preview.jpg
    python3 tools/preview_map.py --width 160 --out preview/zoomed.jpg
    python3 tools/preview_map.py --improvements PATH  # also draw a few improvement images
                                                      # (e.g. HexaRealm's Tiles folder from the Unciv repo)

It follows the game's rules for terrain images:
* each image is drawn at the hex width, keeping its aspect ratio, anchored at the hex bottom;
* tiles are drawn top of the screen first, so lower tiles cover the overhang of the ones above;
* per tile: base terrain (a numbered version picked from the tile position), then features,
  then every edge image whose origin filter matches this tile and destination filter matches
  the neighbour on that side, then rivers and improvements.
"""

import argparse
import glob
import math
import os
import random

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TILESET = os.path.join(ROOT, "Images", "TileSets", "Visual Hex Satellite")
S3 = math.sqrt(3)
LAND_TYPE = {"Grassland": "Land", "Plains": "Land", "Desert": "Land", "Tundra": "Land", "Snow": "Land",
             "Mountain": "Land", "Ocean": "Water", "Coast": "Water", "Lakes": "Water"}
DIRECTIONS = ["Top", "TopRight", "BottomRight", "Bottom", "BottomLeft", "TopLeft"]


def neighbor(c, r, d):
    odd = c & 1
    return {"Top": (c, r - 1), "Bottom": (c, r + 1),
            "TopRight": (c + 1, r) if odd else (c + 1, r - 1),
            "BottomRight": (c + 1, r + 1) if odd else (c + 1, r),
            "TopLeft": (c - 1, r) if odd else (c - 1, r - 1),
            "BottomLeft": (c - 1, r + 1) if odd else (c - 1, r)}[d]


def value_noise(seed, scale):
    rnd = random.Random(seed)
    grid = {}

    def at(x, y):
        if (x, y) not in grid:
            grid[(x, y)] = rnd.random()
        return grid[(x, y)]

    def f(x, y):
        x, y = x / scale, y / scale
        x0, y0 = math.floor(x), math.floor(y)
        fx, fy = x - x0, y - y0
        sx, sy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
        a = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * sx
        b = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * sx
        return a + (b - a) * sy
    return f


def sample_map(cols, rows, seed):
    """A small continent with every terrain the tileset draws."""
    elev, wet, rough = value_noise(seed, 3.2), value_noise(seed + 1, 2.6), value_noise(seed + 2, 1.7)
    tiles = {}
    for r in range(rows):
        for c in range(cols):
            x, y = 1.5 * c, S3 * (r + 0.5 * (c & 1))
            cx, cy = 1.5 * (cols - 1) / 2, S3 * rows / 2
            fall = math.hypot((x - cx) / (cx + 1), (y - cy) / (cy + 1))
            e = elev(x, y) * 0.9 + 0.55 - fall * 0.9
            lat = r / max(1, rows - 1)
            if e < 0.42:
                base = "Ocean"
            elif e > 0.86 and rough(x, y) > 0.45:
                base = "Mountain"
            elif lat < 0.1:
                base = "Snow"
            elif lat < 0.22:
                base = "Tundra"
            else:
                m = wet(x, y)
                base = "Desert" if m < 0.36 and lat > 0.5 else "Plains" if m < 0.52 else "Grassland"
            tiles[(c, r)] = {"base": base, "features": []}
    for (c, r), t in tiles.items():   # coast next to land
        if t["base"] == "Ocean" and any(tiles.get(neighbor(c, r, d), {"base": "Ocean"})["base"] not in ("Ocean", "Coast")
                                        for d in DIRECTIONS):
            t["base"] = "Coast"
    rnd = random.Random(seed + 3)
    for (c, r), t in tiles.items():
        if LAND_TYPE[t["base"]] != "Land" or t["base"] == "Mountain":
            continue
        x, y = 1.5 * c, S3 * (r + 0.5 * (c & 1))
        if rough(x, y) > 0.62:
            t["features"].append("Hill")
        if t["base"] in ("Grassland", "Plains", "Tundra") and wet(x + 9, y) > 0.55:
            t["features"].append("Jungle" if t["base"] == "Plains" and r / rows > 0.6 else "Forest")
        elif t["base"] == "Grassland" and rnd.random() < 0.06:
            t["features"].append("Marsh")
        elif t["base"] == "Desert" and rnd.random() < 0.25:
            t["features"].append("Flood plains" if rnd.random() < 0.6 else "Oasis")
    # sprinkle the rarer water features so they show up in the preview
    for (c, r), t in tiles.items():
        if t["base"] == "Ocean" and r <= 1 and rnd.random() < 0.6:
            t["features"].append("Ice")
        elif t["base"] == "Ocean" and rnd.random() < 0.04:
            t["features"].append("Atoll")
    return tiles


def load(path_no_ext, cache):
    if path_no_ext not in cache:
        p = path_no_ext + ".png"
        cache[path_no_ext] = Image.open(p).convert("RGBA") if os.path.exists(p) else None
    return cache[path_no_ext]


def variant(name, c, r, cache):
    found = [os.path.join(TILESET, "Tiles", name)]
    i = 2
    while load(os.path.join(TILESET, "Tiles", name + str(i)), cache) is not None:
        found.append(os.path.join(TILESET, "Tiles", name + str(i)))
        i += 1
    return found[(c * 7919 + r * 104729 + len(name) * 31) % len(found)]


def edge_images():
    out = {}
    for p in glob.glob(os.path.join(TILESET, "Edges", "**", "*.png"), recursive=True):
        parts = os.path.splitext(os.path.basename(p))[0].split("-")
        if len(parts) != 4:
            continue
        out.setdefault(parts[3], []).append((parts[1], parts[2], p[:-4]))
    return out


def matches(tile, filt):
    return filt == tile["base"] or filt in tile["features"] or LAND_TYPE[tile["base"]] == filt


def render(tiles, cols, rows, width, improvements, rivers):
    cache = {}
    edges = edge_images()
    hex_h = width * S3 / 2
    top_pad = width * 0.45
    img_w = int(width * (0.75 * (cols - 1) + 1)) + 2
    img_h = int(hex_h * (rows + 0.5) + top_pad) + 2
    canvas = Image.new("RGBA", (img_w, img_h), (13, 20, 28, 255))

    def draw(path, c, r):
        im = load(path, cache)
        if im is None:
            return False
        cx = width / 2 + c * 1.5 * width / 2
        cy = top_pad + hex_h / 2 + (r + 0.5 * (c & 1)) * hex_h
        h = math.ceil(im.height * width / im.width)   # the game scales with floats; round up so no gaps appear here
        scaled = im.resize((width, h), Image.LANCZOS)
        canvas.alpha_composite(scaled, (round(cx - width / 2), round(cy + hex_h / 2) - h))
        return True

    order = sorted(tiles, key=lambda cr: (cr[1] + 0.5 * (cr[0] & 1), cr[0]))
    for c, r in order:
        t = tiles[(c, r)]
        draw(variant(t["base"], c, r, cache), c, r)
        for f in t["features"]:
            draw(variant(f, c, r, cache), c, r)
        for d in DIRECTIONS:
            n = tiles.get(neighbor(c, r, d))
            if n is None:
                continue
            for origin, dest, path in edges.get(d, []):
                if matches(t, origin) and matches(n, dest):
                    draw(path, c, r)
        for side in rivers.get((c, r), []):
            draw(os.path.join(TILESET, "Tiles", "River-" + side), c, r)
        if improvements and (c, r) in improvements["at"]:
            draw(os.path.join(improvements["dir"], improvements["at"][(c, r)]), c, r)
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cols", type=int, default=16)
    ap.add_argument("--rows", type=int, default=10)
    ap.add_argument("--width", type=int, default=96, help="hex width on screen in pixels")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--improvements", help="folder with improvement images, e.g. HexaRealm/Tiles")
    ap.add_argument("--out", default=os.path.join(ROOT, "preview", "preview.jpg"))
    args = ap.parse_args()
    tiles = sample_map(args.cols, args.rows, args.seed)
    land = [cr for cr, t in tiles.items() if t["base"] in ("Grassland", "Plains") and not t["features"]]
    improvements = None
    if args.improvements:
        rnd = random.Random(1)
        picks = rnd.sample(land, min(4, len(land)))
        improvements = {"dir": args.improvements, "at": {picks[0]: "City center"}}
        for cr in picks[1:]:
            improvements["at"][cr] = "Farm"
    rivers = {}
    for (c, r), t in tiles.items():   # a short river along the bottom sides of a few land tiles
        if LAND_TYPE[t["base"]] == "Land" and c in (5, 6) and 3 <= r <= 6:
            rivers[(c, r)] = ["BottomRight"] if c == 5 else ["Bottom"]
    out = render(tiles, args.cols, args.rows, args.width, improvements, rivers)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.out.lower().endswith((".jpg", ".jpeg")):
        out.convert("RGB").save(args.out, quality=88)
    else:
        out.save(args.out)
    print("Saved", args.out, out.size)


if __name__ == "__main__":
    main()
