#!/usr/bin/env python3
"""Lay the tileset's images out on a sample map the way Unciv does, and save an image.

    python3 tools/preview_map.py                         # preview/preview.jpg
    python3 tools/preview_map.py --width 160 --cols 9 --rows 6 --seed 9 --out preview/close-up.jpg
    python3 tools/preview_map.py --catalog               # preview/catalog-*.jpg, every item on its usual terrain
    python3 tools/preview_map.py --bare                  # terrain only, no cities, resources or improvements
    python3 tools/preview_map.py --fallback PATH         # draw missing images from another tileset's Tiles folder

It follows the game's rules for tile images:
* each image is drawn at the hex width, keeping its aspect ratio, anchored at the hex bottom;
* tiles are drawn top of the screen first, so lower tiles cover the overhang of the ones above;
* per tile: base terrain (a numbered version picked from the tile position) and features, or only the
  natural wonder image; then every edge image whose origin filter matches this tile and whose
  destination filter matches the neighbour on that side; then rivers; then the resource and the
  improvement, or the combined "Resource+Improvement" image when the tileset has one.
"""

import argparse
import glob
import math
import os
import random

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TILESET = os.path.join(ROOT, "Images", "TileSets", "Visual Hex Satellite")
S3 = math.sqrt(3)
LAND_TYPE = {"Grassland": "Land", "Plains": "Land", "Desert": "Land", "Tundra": "Land", "Snow": "Land",
             "Mountain": "Land", "Ocean": "Water", "Coast": "Water", "Lakes": "Water"}
DIRECTIONS = ["Top", "TopRight", "BottomRight", "Bottom", "BottomLeft", "TopLeft"]

# Which improvement works each resource, from the base game's rules.
IMPROVES = {}
for imp, res in {"Pasture": "Cattle Sheep Horses", "Camp": "Deer Furs Bison Ivory Truffles",
                 "Plantation": "Bananas Citrus Cocoa Cotton Dyes Incense Silk Spices Sugar Wine",
                 "Mine": "Iron Coal Aluminum Uranium Gems Silver Copper Salt", "Quarry": "Stone Marble",
                 "Fishing Boats": "Fish Whales Pearls Crab", "Farm": "Wheat"}.items():
    for r in res.split():
        IMPROVES[r] = imp
IMPROVES["Gold Ore"] = "Mine"
# Resources that can appear on a terrain or feature (a representative subset of the rules).
FOUND_ON = {"Grassland": ["Cattle", "Cotton", "Wine", "Sugar", "Stone", "Horses", "Marble", "Silver", "Copper"],
            "Plains": ["Wheat", "Horses", "Bison", "Ivory", "Incense", "Wine", "Marble"],
            "Desert": ["Gold Ore", "Salt", "Oil", "Incense", "Stone", "Aluminum", "Marble"],
            "Tundra": ["Deer", "Silver", "Uranium", "Iron", "Oil", "Salt"],
            "Snow": ["Oil", "Iron", "Aluminum"],
            "Hill": ["Sheep", "Iron", "Coal", "Copper", "Gems", "Gold Ore", "Aluminum"],
            "Forest": ["Deer", "Furs", "Silk", "Dyes", "Truffles", "Spices"],
            "Jungle": ["Bananas", "Citrus", "Cocoa", "Spices", "Gems"],
            "Marsh": ["Sugar", "Spices", "Truffles"],
            "Coast": ["Fish", "Whales", "Pearls", "Crab", "Oil"]}


def neighbor(c, r, d):
    odd = c & 1
    return {"Top": (c, r - 1), "Bottom": (c, r + 1),
            "TopRight": (c + 1, r) if odd else (c + 1, r - 1),
            "BottomRight": (c + 1, r + 1) if odd else (c + 1, r),
            "TopLeft": (c - 1, r) if odd else (c - 1, r - 1),
            "BottomLeft": (c - 1, r + 1) if odd else (c - 1, r)}[d]


def hex_dist(a, b):
    def cube(c, r):
        x = c
        z = r - (c - (c & 1)) // 2
        return x, -x - z, z
    p, q = cube(*a), cube(*b)
    return max(abs(p[0] - q[0]), abs(p[1] - q[1]), abs(p[2] - q[2]))


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
    for (c, r), t in tiles.items():
        if t["base"] == "Ocean" and r <= 1 and rnd.random() < 0.6:
            t["features"].append("Ice")
        elif t["base"] == "Ocean" and rnd.random() < 0.04:
            t["features"].append("Atoll")
    return tiles


def populate(tiles, seed):
    """Add cities, resources, improvements and natural wonders the way a mid-game map might look."""
    rnd = random.Random(seed + 10)
    open_land = sorted(cr for cr, t in tiles.items() if t["base"] in ("Grassland", "Plains") and not t["features"])
    cities = []
    for cr in rnd.sample(open_land, len(open_land)):
        if all(hex_dist(cr, o) >= 4 for o in cities):
            cities.append(cr)
        if len(cities) == 3:
            break
    eras = ["Medieval era", "Industrial era", "Information era"]
    for cr, era in zip(cities, eras):
        tiles[cr]["improvement"] = "City center-" + era
    land = [cr for cr, t in tiles.items() if LAND_TYPE[t["base"]] == "Land" and t["base"] != "Mountain"
            and "improvement" not in t and all(hex_dist(cr, o) >= 3 for o in cities)]
    for cr, w in zip(rnd.sample(land, min(2, len(land))), ["Mount Fuji", "Uluru"]):
        tiles[cr].update(base="Mountain", features=[], wonder=w)
    for cr, t in sorted(tiles.items()):
        if "improvement" in t or "wonder" in t or t["base"] in ("Ocean", "Lakes", "Mountain"):
            continue
        keys = [t["base"]] + t["features"]
        options = [res for k in keys for res in FOUND_ON.get(k, [])]
        if options and rnd.random() < 0.3:
            t["resource"] = rnd.choice(options)
        near = any(hex_dist(cr, o) <= 2 for o in cities)
        if not near or rnd.random() < 0.25:
            continue
        res = t.get("resource")
        if res:
            if res == "Oil":
                t["improvement"] = "Offshore Platform" if LAND_TYPE[t["base"]] == "Water" else "Oil well"
            else:
                t["improvement"] = IMPROVES[res]
        elif LAND_TYPE[t["base"]] == "Water":
            continue
        elif "Hill" in t["features"]:
            t["improvement"] = "Mine"
        elif "Forest" in t["features"]:
            t["improvement"] = "Lumber mill"
        elif not t["features"]:
            t["improvement"] = "Farm"
        else:
            t["improvement"] = "Trading post"
    return tiles


class Painter:
    def __init__(self, fallback=None):
        self.cache = {}
        self.fallback = fallback
        self.edges = {}
        for p in glob.glob(os.path.join(TILESET, "Edges", "**", "*.png"), recursive=True):
            parts = os.path.splitext(os.path.basename(p))[0].split("-")
            if len(parts) == 4:
                self.edges.setdefault(parts[3], []).append((parts[1], parts[2], p[:-4]))

    def load(self, path_no_ext):
        if path_no_ext not in self.cache:
            p = path_no_ext + ".png"
            self.cache[path_no_ext] = Image.open(p).convert("RGBA") if os.path.exists(p) else None
        return self.cache[path_no_ext]

    def tile_path(self, name):
        """Our image if it exists, otherwise the fallback tileset's (like Unciv's fallbackTileSet)."""
        own = os.path.join(TILESET, "Tiles", name)
        if self.load(own) is not None or not self.fallback:
            return own
        return os.path.join(self.fallback, name)

    def variant(self, name, c, r):
        base = self.tile_path(name)
        found = [base]
        i = 2
        while self.load(base + str(i)) is not None:
            found.append(base + str(i))
            i += 1
        return found[(c * 7919 + r * 104729 + len(name) * 31) % len(found)]

    def exists(self, name):
        return self.load(os.path.join(TILESET, "Tiles", name)) is not None


def matches(tile, filt):
    return filt == tile["base"] or filt in tile["features"] or LAND_TYPE[tile["base"]] == filt


def tile_layers(painter, tiles, c, r, rivers):
    t = tiles[(c, r)]
    out = []
    if t.get("wonder"):
        out.append(painter.tile_path(t["wonder"]))
    else:
        out.append(painter.variant(t["base"], c, r))
        out += [painter.variant(f, c, r) for f in t["features"]]
    for d in DIRECTIONS:
        n = tiles.get(neighbor(c, r, d))
        if n is None:
            continue
        for origin, dest, path in painter.edges.get(d, []):
            if matches(t, origin) and matches(n, dest):
                out.append(path)
    out += [os.path.join(TILESET, "Tiles", "River-" + side) for side in rivers.get((c, r), [])]
    res, imp = t.get("resource"), t.get("improvement")
    if res and imp and painter.exists(res + "+" + imp):
        out.append(painter.tile_path(res + "+" + imp))
    else:
        out += [painter.tile_path(x) for x in (res, imp) if x]
    return out


def render(tiles, cols, rows, width, painter=None, rivers=None):
    painter = painter or Painter()
    rivers = rivers or {}
    hex_h = width * S3 / 2
    top_pad = width * 0.45
    img_w = int(width * (0.75 * (cols - 1) + 1)) + 2
    img_h = int(hex_h * (rows + 0.5) + top_pad) + 2
    canvas = Image.new("RGBA", (img_w, img_h), (13, 20, 28, 255))
    order = sorted(tiles, key=lambda cr: (cr[1] + 0.5 * (cr[0] & 1), cr[0]))
    for c, r in order:
        cx = width / 2 + c * 1.5 * width / 2
        cy = top_pad + hex_h / 2 + (r + 0.5 * (c & 1)) * hex_h
        for path in tile_layers(painter, tiles, c, r, rivers):
            im = painter.load(path)
            if im is None:
                continue
            h = math.ceil(im.height * width / im.width)   # the game scales with floats; round up so no gaps appear
            canvas.alpha_composite(im.resize((width, h), Image.LANCZOS),
                                   (round(cx - width / 2), round(cy + hex_h / 2) - h))
    return canvas


def save(img, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if out.lower().endswith((".jpg", ".jpeg")):
        img.convert("RGB").save(out, quality=88)
    else:
        img.save(out)
    print("Saved", out, img.size)


def catalog(items, out, cols, width=150):
    """One labelled tile per item, each on a terrain it normally appears on."""
    painter = Painter()
    cells = []
    for label, tile in items:
        im = render({(0, 0): tile}, 1, 1, width, painter)
        cell = Image.new("RGBA", (im.width, im.height + 18), (20, 26, 34, 255))
        cell.paste(im, (0, 0))
        ImageDraw.Draw(cell).text((5, im.height + 3), label, fill=(240, 220, 150, 255))
        cells.append(cell)
    cw, ch = cells[0].size
    sheet = Image.new("RGBA", (cw * cols, ch * ((len(cells) + cols - 1) // cols)), (13, 20, 28, 255))
    for i, cell in enumerate(cells):
        sheet.paste(cell, ((i % cols) * cw, (i // cols) * ch))
    save(sheet, out)


def write_catalogs(folder):
    def t(base, *features, **kw):
        return dict(base=base, features=list(features), **kw)
    usual = {"Cattle": t("Grassland"), "Sheep": t("Grassland", "Hill"), "Horses": t("Plains"), "Deer": t("Tundra", "Forest"),
             "Bison": t("Plains"), "Furs": t("Tundra", "Forest"), "Ivory": t("Plains"), "Truffles": t("Grassland", "Forest"),
             "Wheat": t("Plains"), "Cotton": t("Grassland"), "Wine": t("Plains"), "Sugar": t("Grassland"),
             "Bananas": t("Grassland", "Jungle"), "Citrus": t("Plains", "Jungle"), "Cocoa": t("Grassland", "Jungle"),
             "Dyes": t("Grassland", "Forest"), "Incense": t("Desert"), "Silk": t("Grassland", "Forest"),
             "Spices": t("Plains", "Jungle"), "Stone": t("Grassland"), "Marble": t("Plains"), "Iron": t("Plains", "Hill"),
             "Coal": t("Grassland", "Hill"), "Copper": t("Grassland", "Hill"), "Gold Ore": t("Desert"), "Silver": t("Tundra"),
             "Gems": t("Grassland", "Jungle"), "Aluminum": t("Desert"), "Salt": t("Desert"), "Uranium": t("Tundra"),
             "Oil": t("Desert"), "Fish": t("Coast"), "Whales": t("Coast"), "Pearls": t("Coast"), "Crab": t("Coast")}
    catalog([(name, dict(tile, resource=name)) for name, tile in usual.items()], os.path.join(folder, "catalog-resources.jpg"), 7)
    imps = [("Farm", t("Plains", improvement="Farm")), ("Wheat + Farm", t("Plains", resource="Wheat", improvement="Farm")),
            ("Cattle + Pasture", t("Grassland", resource="Cattle", improvement="Pasture")),
            ("Deer + Camp", t("Tundra", "Forest", resource="Deer", improvement="Camp")),
            ("Citrus + Plantation", t("Plains", "Jungle", resource="Citrus", improvement="Plantation")),
            ("Mine", t("Grassland", "Hill", improvement="Mine")), ("Iron + Mine", t("Plains", "Hill", resource="Iron", improvement="Mine")),
            ("Marble + Quarry", t("Plains", resource="Marble", improvement="Quarry")),
            ("Fish + Fishing Boats", t("Coast", resource="Fish", improvement="Fishing Boats")),
            ("Oil + Oil well", t("Desert", resource="Oil", improvement="Oil well")),
            ("Oil + Offshore Platform", t("Coast", resource="Oil", improvement="Offshore Platform"))]
    imps += [(n, t(b, *f, improvement=n)) for n, b, f in [
        ("Lumber mill", "Grassland", ["Forest"]), ("Trading post", "Plains", []), ("Fort", "Grassland", []), ("Citadel", "Plains", []),
        ("Academy", "Grassland", []), ("Landmark", "Grassland", []), ("Manufactory", "Plains", []), ("Customs house", "Plains", []),
        ("Holy site", "Grassland", []), ("Moai", "Grassland", []), ("Terrace farm", "Grassland", ["Hill"]), ("Polder", "Grassland", ["Marsh"]),
        ("Ancient ruins", "Grassland", ["Forest"]), ("City ruins", "Plains", []), ("Barbarian encampment", "Plains", ["Forest"])]]
    catalog(imps, os.path.join(folder, "catalog-improvements.jpg"), 7)
    eras = ["Ancient era", "Classical era", "Medieval era", "Renaissance era", "Industrial era", "Modern era", "Information era"]
    catalog([(e.replace(" era", ""), t("Grassland", improvement="City center-" + e)) for e in eras],
            os.path.join(folder, "catalog-cities.jpg"), 7)
    wonders = ["Mount Fuji", "Mount Kilimanjaro", "Mount Kailash", "Mount Sinai", "Sri Pada", "Cerro de Potosi", "Uluru", "Grand Mesa",
               "Barringer Crater", "Krakatoa", "Lake Victoria", "Great Barrier Reef", "Old Faithful", "El Dorado", "Fountain of Youth",
               "King Solomon's Mines", "Rock of Gibraltar"]
    catalog([(w, t("Mountain", wonder=w)) for w in wonders], os.path.join(folder, "catalog-wonders.jpg"), 6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cols", type=int, default=16)
    ap.add_argument("--rows", type=int, default=10)
    ap.add_argument("--width", type=int, default=96, help="hex width on screen in pixels")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--bare", action="store_true", help="terrain only")
    ap.add_argument("--fallback", help="another tileset's Tiles folder to use for missing images")
    ap.add_argument("--catalog", action="store_true", help="write labelled catalog sheets instead of a map")
    ap.add_argument("--out", default=os.path.join(ROOT, "preview", "preview.jpg"))
    args = ap.parse_args()
    if args.catalog:
        write_catalogs(os.path.dirname(os.path.abspath(args.out)))
        return
    tiles = sample_map(args.cols, args.rows, args.seed)
    if not args.bare:
        tiles = populate(tiles, args.seed)
    rivers = {}
    for (c, r), t in tiles.items():   # a short river along the bottom sides of a few land tiles
        if LAND_TYPE[t["base"]] == "Land" and not t.get("wonder") and c in (5, 6) and 3 <= r <= 6:
            rivers[(c, r)] = ["BottomRight"] if c == 5 else ["Bottom"]
    save(render(tiles, args.cols, args.rows, args.width, Painter(args.fallback), rivers), args.out)


if __name__ == "__main__":
    main()
