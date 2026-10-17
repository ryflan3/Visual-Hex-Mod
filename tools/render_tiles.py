#!/usr/bin/env python3
"""Render every image in the "Visual Hex Satellite" Unciv tileset.

Run from anywhere:

    python3 tools/render_tiles.py              # render everything
    python3 tools/render_tiles.py Grassland    # only images whose path contains "Grassland"

How the images fit Unciv
------------------------
* Unciv draws each tile image at the width of a hex, keeps the image's aspect ratio and
  anchors it at the bottom. The bottom ``W x 0.866W`` of every image is the hex itself
  (flat-top). Anything taller rises above the tile, and tiles lower on screen are drawn
  over the ones above, so mountains and tree crowns can overlap the tile behind them.
* Heights are drawn with an oblique projection: the ground keeps its exact hex shape and
  a point ``h`` units high is drawn ``K_RISE * h`` units higher on screen.
* Base terrain textures are periodic on the hex lattice, so two neighbouring tiles of the
  same terrain continue each other without a seam. Alternate versions (Grassland2, ...)
  only differ away from the edges for the same reason.
* Edge images reuse those periodic textures: "Blend-Plains-Grassland-Top" paints the exact
  Grassland texture that continues across the shared edge, fading out toward the centre.
"""

import math
import os
import sys
from itertools import combinations

import numpy as np
from PIL import Image

TILESET = "Visual Hex Satellite"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Images", "TileSets", TILESET)

# 224x194 and 112x97 have exactly the same aspect ratio, within 0.01% of a true hex. Tiles and the
# edge images drawn over them must match exactly, or a sliver of the base terrain shows at the top.
TILE_W = 224   # terrain and overlay images
EDGE_W = 112   # edge images are shown at the same size on screen, so they can be smaller files
SS = 2         # supersampling factor
S3 = math.sqrt(3.0)
K_RISE = 0.9   # screen rise per unit of height
LIGHT = np.array([-0.70, -0.06, 0.71])
LIGHT = LIGHT / np.linalg.norm(LIGHT)   # light from the left (west); v points south

# Hex sides as corner pairs. Corner i sits at angle 60*i degrees; v grows downward (south).
DIRS = {
    "BottomRight": (0, 1), "Bottom": (1, 2), "BottomLeft": (2, 3),
    "TopLeft": (3, 4), "Top": (4, 5), "TopRight": (5, 0),
}
LAND5 = ["Grassland", "Plains", "Desert", "Tundra", "Snow"]
# When two land terrains meet, the higher one spreads a little into the lower one.
PRIORITY = {"Snow": 5, "Grassland": 4, "Plains": 3, "Tundra": 2, "Desert": 1}

FILTER = None
WRITTEN = []


# ----------------------------------------------------------------------------- helpers
def C(*rgb):
    return np.array(rgb, dtype=np.float64)


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def mix(c1, c2, t):
    t = np.asarray(t, dtype=np.float64)
    if t.ndim:
        t = t[..., None]
    return c1 + (c2 - c1) * t


def ramp(t, stops):
    out = np.empty(np.shape(t) + (3,))
    pos = [p for p, _ in stops]
    for ch in range(3):
        out[..., ch] = np.interp(t, pos, [c[ch] for _, c in stops])
    return out


def hex_edge_dist(u, v):
    """Distance inside a flat-top hex of circumradius 1 to its nearest side (negative outside)."""
    au, av = np.abs(u), np.abs(v)
    return np.minimum(S3 / 2 - av, (S3 - S3 * au - av) / 2)


def corner(i):
    a = math.pi / 3 * (i % 6)
    return math.cos(a), math.sin(a)


def side_dist(u, v, side):
    """Distance to one side of the hex, measured to the segment (round ends at the corners)."""
    i, j = DIRS[side]
    p, q = corner(i), corner(j)
    dx, dy = q[0] - p[0], q[1] - p[1]
    t = np.clip(((u - p[0]) * dx + (v - p[1]) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
    return np.hypot(u - (p[0] + t * dx), v - (p[1] + t * dy))


class Canvas:
    """Sample grid for one image, in hex units: circumradius 1, centre (0, 0), v points south."""

    def __init__(self, width, overhang=0, pad=2, height=None):
        self.width = width
        self.foot = round(width * S3 / 2)
        self.overhang = overhang
        self.height = height if height is not None else self.foot + overhang
        self.pad = pad
        self.spx = width * SS / 2.0
        w, h = width * SS, self.height * SS
        xs = (np.arange(-pad, w + pad) + 0.5) / self.spx - 1.0
        ys = (np.arange(-pad, h + pad) + 0.5 - overhang * SS) / self.spx - self.foot / width
        self.u, self.v = np.meshgrid(xs, ys)
        self.a = self.u / 1.5                     # coordinates in the hex lattice basis:
        self.b = self.v / S3 - self.a / 2         # t1 = (1.5, sqrt3/2), t2 = (0, sqrt3)
        self.edge = hex_edge_dist(self.u, self.v)
        self.shape = self.u.shape

    def px(self, u, v):
        """Continuous padded pixel position of a point in hex units."""
        return (u + 1.0) * self.spx + self.pad, (v + self.foot / self.width) * self.spx + self.overhang * SS + self.pad

    def hex_alpha(self, bleed=1.5):
        """Opaque inside the hex, bleeding a pixel or so outward to hide hairline seams."""
        return np.clip(self.edge * self.spx + bleed * SS, 0.0, 1.0)

    def empty(self):
        return np.zeros(self.shape + (4,))


# ----------------------------------------------------------------------------- noise
_GRIDS = {}


def _grid(freq, seed):
    key = (freq, seed)
    if key not in _GRIDS:
        _GRIDS[key] = np.random.default_rng(seed).random((freq, freq))
    return _GRIDS[key]


def pnoise(a, b, freq, seed):
    """Value noise with period 1 in both lattice coordinates, so it repeats on every hex."""
    g = _grid(freq, seed)
    x, y = a * freq, b * freq
    xi, yi = np.floor(x), np.floor(y)
    xf, yf = x - xi, y - yi
    xi = xi.astype(np.int64) % freq
    yi = yi.astype(np.int64) % freq
    x1, y1 = (xi + 1) % freq, (yi + 1) % freq
    sx, sy = xf * xf * (3 - 2 * xf), yf * yf * (3 - 2 * yf)
    top = g[xi, yi] + (g[x1, yi] - g[xi, yi]) * sx
    bot = g[xi, y1] + (g[x1, y1] - g[xi, y1]) * sx
    return top + (bot - top) * sy


def fbm(a, b, freq, octaves, seed, gain=0.5):
    total, amp, norm, f = 0.0, 1.0, 0.0, freq
    for i in range(octaves):
        total = total + amp * pnoise(a, b, f, seed + 7919 * i)
        norm += amp
        amp *= gain
        f *= 2
    return total / norm


def ridged(a, b, freq, octaves, seed):
    total, amp, norm, f = 0.0, 1.0, 0.0, freq
    for i in range(octaves):
        n = 1.0 - np.abs(2.0 * pnoise(a, b, f, seed + 104729 * i) - 1.0)
        total = total + amp * n * n
        norm += amp
        amp *= 0.5
        f *= 2
    return total / norm


def warp(a, b, amount, freq, seed):
    return (a + amount * (fbm(a, b, freq, 3, seed) - 0.5),
            b + amount * (fbm(a, b, freq, 3, seed + 1) - 0.5))


_LEAF = np.random.default_rng(7).random((256, 256))
_LEAF = (_LEAF + np.roll(_LEAF, 1, 0) + np.roll(_LEAF, 1, 1) + np.roll(_LEAF, (1, 1), (0, 1))) / 4


# ----------------------------------------------------------------------------- shading and compositing
def shade(h, spx, strength=1.0):
    """Lambert shading of a height field (in hex units), normalised so flat ground is 1."""
    gy, gx = np.gradient(h)
    nx, ny = -gx * spx * strength, -gy * spx * strength
    n = np.sqrt(nx * nx + ny * ny + 1.0)
    return (nx * LIGHT[0] + ny * LIGHT[1] + LIGHT[2]) / n / LIGHT[2]


def lit(rgb, s, lo=0.35, hi=1.6, shadow_tint=None):
    s = np.clip(s, lo, hi)
    out = rgb * s[..., None]
    if shadow_tint is not None:
        out = mix(out, out * shadow_tint, np.clip(1.0 - s, 0, 1) * 1.5)
    return out


def premult(rgb, alpha):
    alpha = np.clip(alpha, 0.0, 1.0)
    return np.concatenate([np.clip(rgb, 0, 1.2) * alpha[..., None], alpha[..., None]], axis=-1)


def over(dst, src):
    return src + dst * (1.0 - src[..., 3:4])


def save(cv, P, relpath, trim=True):
    p = cv.pad
    P = P[p:P.shape[0] - p, p:P.shape[1] - p] if p else P
    h, w = P.shape[:2]
    P = P.reshape(h // SS, SS, w // SS, SS, 4).mean(axis=(1, 3))
    a = P[..., 3:4]
    rgb = np.where(a > 1e-6, P[..., :3] / np.maximum(a, 1e-6), 0.0)
    img = np.concatenate([np.clip(rgb, 0, 1), np.clip(a, 0, 1)], axis=-1)
    arr = (img * 255.0 + 0.5).astype(np.uint8)
    rows = np.nonzero(arr[..., 3].max(axis=1))[0]
    if trim and len(rows) and rows[0] > 0:
        arr = arr[rows[0]:]   # tile images are bottom-anchored, so empty rows on top can go
    path = os.path.join(OUT, relpath + ".png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr, "RGBA").save(path, optimize=True)
    WRITTEN.append(relpath)


def wanted(relpath):
    return FILTER is None or any(f in relpath for f in FILTER)


def fine(cv, seed, amount=0.16):
    """Pixel-scale grain so textures stay crisp when zoomed in."""
    return (1.0 - amount / 2 + amount * pnoise(cv.a, cv.b, 72, seed))[..., None]


def interior(cv, inner=0.28, full=0.46):
    """1 away from the edges, 0 near them: variant details live here so edges stay seamless."""
    return smooth(inner, full, cv.edge)


def blob(cv, u0, v0, r, seed, wobble=0.35):
    n = fbm(cv.a, cv.b, 6, 3, seed)
    d = np.hypot(cv.u - u0, (cv.v - v0) * 1.15) / r + (n - 0.5) * wobble * 2
    return smooth(1.0, 0.6, d)


# ----------------------------------------------------------------------------- trees and shrubs
def place_interior(rng, n, min_edge, min_gap=0.0, region=None):
    pts, tries = [], 0
    while len(pts) < n and tries < n * 400:
        tries += 1
        u, v = rng.uniform(-1, 1), rng.uniform(-S3 / 2, S3 / 2)
        if hex_edge_dist(u, v) < min_edge:
            continue
        if region is not None and not region(u, v):
            continue
        if any((u - x) ** 2 + (v - y) ** 2 < min_gap ** 2 for x, y in pts):
            continue
        pts.append((u, v))
    return pts


def place_periodic(rng, n, min_gap):
    """Points that repeat on every hex (lattice-periodic), returned as positions inside one hex."""
    lat = []
    tries = 0
    while len(lat) < n and tries < n * 600:
        tries += 1
        pa, pb = rng.random(), rng.random()
        ok = True
        for qa, qb in lat:
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    da, db = pa - qa + i, pb - qb + j
                    du, dv = 1.5 * da, S3 * (db + da / 2)
                    if du * du + dv * dv < min_gap * min_gap:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            lat.append((pa, pb))
    pts = []
    for pa, pb in lat:
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                a, b = pa + i, pb + j
                u, v = 1.5 * a, S3 * (b + a / 2)
                if hex_edge_dist(u, v) >= 0:
                    pts.append((u, v))
    return pts


FOREST_GREENS = [C(.13, .25, .09), C(.17, .30, .11), C(.21, .34, .13), C(.15, .27, .12), C(.24, .36, .14)]
CONIFER_GREENS = [C(.08, .17, .09), C(.10, .20, .10), C(.12, .22, .12)]
JUNGLE_GREENS = [C(.16, .35, .12), C(.21, .41, .15), C(.26, .46, .17), C(.18, .38, .16), C(.30, .48, .20)]
SHRUB_GREENS = [C(.18, .28, .11), C(.22, .31, .12), C(.25, .30, .13)]


def make_tree(rng, u, v, r_range, ht_range, palette, kind="broad"):
    return {"u": u, "v": v, "r": rng.uniform(*r_range), "ht": rng.uniform(*ht_range),
            "col": palette[rng.integers(len(palette))] * rng.uniform(0.9, 1.1),
            "kind": kind, "ph": rng.uniform(0, 6.3), "ph2": rng.uniform(0, 6.3)}


def splat_trees(cv, P, trees, shadow_alpha=0.5):
    if not trees:
        return P
    sx_off = -LIGHT[0] / LIGHT[2]
    sy_off = -LIGHT[1] / LIGHT[2]
    for t in trees:   # shadows first, on the ground
        cx, cy = cv.px(t["u"] + sx_off * t["ht"] * 0.8, t["v"] + sy_off * t["ht"] * 0.8)
        k = 0.6 if t["kind"] == "palm" else 1.0
        rx, ry = t["r"] * cv.spx * 1.05 * k, t["r"] * cv.spx * 0.95 * k
        x0, x1 = int(max(0, cx - rx - 3)), int(min(P.shape[1], cx + rx + 3))
        y0, y1 = int(max(0, cy - ry - 3)), int(min(P.shape[0], cy + ry + 3))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5
        d = np.hypot((xx - cx) / rx, (yy - cy) / ry)
        a = np.clip((1.0 - d) * rx / 3.0, 0, 1) * shadow_alpha
        src = premult(np.broadcast_to(C(.03, .06, .03), a.shape + (3,)), a)
        P[y0:y1, x0:x1] = over(P[y0:y1, x0:x1], src)
    for t in sorted(trees, key=lambda q: q["v"]):   # crowns back to front
        cx, cy = cv.px(t["u"], t["v"] - K_RISE * t["ht"])
        rad = t["r"] * cv.spx
        x0, x1 = int(max(0, cx - rad * 1.35 - 2)), int(min(P.shape[1], cx + rad * 1.35 + 2))
        y0, y1 = int(max(0, cy - rad * 1.35 - 2)), int(min(P.shape[0], cy + rad * 1.35 + 2))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5
        dx, dy = (xx - cx) / rad, (yy - cy) / rad
        d, th = np.hypot(dx, dy), np.arctan2(dy, dx)
        if t["kind"] == "conifer":
            er = 1.0 + 0.22 * np.cos(8 * th + t["ph"])
        elif t["kind"] == "palm":
            er = 0.55 + 0.55 * np.abs(np.cos(3 * th + t["ph"])) ** 3
        else:
            er = 1.0 + 0.10 * np.sin(5 * th + t["ph"]) + 0.07 * np.sin(9 * th + t["ph2"])
        a = np.clip((er - d) * rad / SS, 0, 1)
        q = np.clip(d / er, 0, 1)
        nz = np.sqrt(np.clip(1 - q * q, 0, 1))
        lam = np.clip(dx / er * LIGHT[0] + dy / er * LIGHT[1] + nz * LIGHT[2], 0, 1)
        leaf = 0.8 + 0.4 * _LEAF[yy.astype(int) % 256, xx.astype(int) % 256]
        rgb = t["col"] * ((0.32 + 0.9 * lam) * leaf)[..., None]
        P[y0:y1, x0:x1] = over(P[y0:y1, x0:x1], premult(rgb, a))
    return P


# ----------------------------------------------------------------------------- base terrain textures
# Each returns (rgb, height, trees). Variant 1 is the plain periodic texture; other variants add
# details only in the interior, so every version of a terrain shares the same edges.

def tex_grassland(cv, variant=1):
    a, b = warp(cv.a, cv.b, 0.10, 2, 11)
    n1, n2 = fbm(a, b, 2, 4, 101), fbm(a, b, 6, 3, 102)
    n3 = fbm(cv.a, cv.b, 24, 2, 103)
    rgb = ramp(n1, [(0.30, C(.19, .32, .12)), (0.50, C(.27, .41, .16)), (0.70, C(.36, .47, .20))])
    rgb = mix(rgb, C(.46, .50, .25), 0.55 * smooth(0.60, 0.76, n2))
    rgb = mix(rgb, C(.14, .25, .10), 0.45 * smooth(0.40, 0.26, n2))
    rgb = rgb * (0.86 + 0.28 * n3)[..., None] * fine(cv, 106)
    h = 0.06 * n1 + 0.03 * n2
    trees = []
    rng = np.random.default_rng(1000 + variant)
    if variant == 2:   # a dry meadow with scattered shrubs
        m = interior(cv) * blob(cv, -0.15, 0.1, 0.5, 104)
        rgb = mix(rgb, C(.47, .51, .26), 0.6 * m)
        trees = [make_tree(rng, u, v, (0.03, 0.045), (0.03, 0.05), SHRUB_GREENS)
                 for u, v in place_interior(rng, 9, 0.34, 0.12)]
    elif variant == 3:   # a small copse beside a damp hollow
        m = interior(cv) * blob(cv, 0.2, 0.15, 0.32, 105)
        rgb = mix(rgb, C(.12, .22, .09), 0.55 * m)
        trees = [make_tree(rng, u, v, (0.05, 0.075), (0.08, 0.13), FOREST_GREENS)
                 for u, v in place_interior(rng, 11, 0.36, 0.09, lambda u, v: (u + 0.25) ** 2 + (v + 0.05) ** 2 < 0.13)]
    return rgb, h, trees


def tex_plains(cv, variant=1):
    a, b = warp(cv.a, cv.b, 0.12, 2, 21)
    n1, n2 = fbm(a, b, 2, 4, 201), fbm(a, b, 8, 3, 202)
    n3 = fbm(cv.a, cv.b, 32, 2, 203)
    rgb = ramp(n1, [(0.30, C(.50, .46, .25)), (0.50, C(.60, .54, .30)), (0.70, C(.68, .60, .36))])
    rgb = mix(rgb, C(.40, .44, .22), 0.55 * smooth(0.60, 0.78, n2))
    rgb = rgb * (0.9 + 0.2 * n3)[..., None] * fine(cv, 205)
    h = 0.04 * n1 + 0.02 * n2
    trees = []
    rng = np.random.default_rng(2000 + variant)
    if variant == 2:   # greener scrub patches
        m = interior(cv) * smooth(0.5, 0.62, fbm(cv.a, cv.b, 5, 3, 206))
        rgb = mix(rgb, C(.38, .43, .21), 0.55 * m)
        trees = [make_tree(rng, u, v, (0.03, 0.045), (0.03, 0.05), SHRUB_GREENS)
                 for u, v in place_interior(rng, 6, 0.36, 0.15)]
    elif variant == 3:   # bare soil and shrubs
        m = interior(cv) * blob(cv, 0.1, -0.1, 0.4, 204)
        rgb = mix(rgb, C(.58, .46, .32), 0.5 * m)
        trees = [make_tree(rng, u, v, (0.03, 0.05), (0.03, 0.06), SHRUB_GREENS)
                 for u, v in place_interior(rng, 12, 0.32, 0.1)]
    return rgb, h, trees


def tex_desert(cv, variant=1):
    a, b = warp(cv.a, cv.b, 0.16, 2, 31)
    n1 = fbm(a, b, 2, 4, 301)
    d1 = np.sin(2 * np.pi * (3 * a + 1 * b) + 3.0 * fbm(a, b, 2, 3, 302))
    d2 = np.sin(2 * np.pi * (1 * a - 4 * b) + 3.0 * fbm(a, b, 2, 3, 303))
    h = 0.025 * (1 - np.abs(d1)) ** 1.6 + 0.008 * d2 + 0.03 * n1
    n3 = fbm(cv.a, cv.b, 32, 2, 304)
    rgb = ramp(n1, [(0.30, C(.74, .61, .42)), (0.50, C(.81, .69, .49)), (0.70, C(.86, .75, .55))])
    rgb = rgb * (0.94 + 0.12 * n3)[..., None] * fine(cv, 307, 0.1)
    if variant == 2:   # dark rocky outcrops
        m = interior(cv) * smooth(0.55, 0.65, fbm(cv.a, cv.b, 5, 4, 305))
        rgb = mix(rgb, C(.48, .34, .24), 0.8 * m)
        h = h + 0.08 * m * fbm(cv.a, cv.b, 12, 3, 306)
    elif variant == 3:   # a dry wadi
        t = np.abs(cv.v + 0.25 * np.sin(cv.u * 4.0) - 0.05)
        m = interior(cv, 0.2, 0.4) * smooth(0.06, 0.02, t)
        rgb = mix(rgb, C(.66, .55, .40), 0.7 * m)
        h = h - 0.03 * m
    return rgb, h, []


def tex_tundra(cv, variant=1):
    a, b = warp(cv.a, cv.b, 0.12, 2, 41)
    n1, n2 = fbm(a, b, 2, 4, 401), fbm(a, b, 7, 3, 402)
    n3 = fbm(cv.a, cv.b, 28, 2, 403)
    rgb = ramp(n1, [(0.30, C(.36, .36, .28)), (0.50, C(.45, .44, .33)), (0.70, C(.54, .52, .40))])
    rgb = mix(rgb, C(.33, .38, .24), 0.5 * smooth(0.58, 0.74, n2))
    rgb = mix(rgb, C(.62, .60, .50), 0.35 * smooth(0.42, 0.28, n2))
    rgb = rgb * (0.88 + 0.24 * n3)[..., None] * fine(cv, 408)
    ponds = smooth(0.76, 0.79, fbm(cv.a, cv.b, 16, 2, 404))
    rgb = mix(rgb, C(.20, .26, .27), 0.6 * ponds)
    h = 0.05 * n1 + 0.02 * n2
    if variant == 2:   # thaw lakes
        m = interior(cv) * smooth(0.56, 0.62, fbm(cv.a, cv.b, 4, 3, 405))
        rgb = mix(rgb, C(.14, .22, .25), 0.9 * m)
    elif variant == 3:   # frost-shattered rock
        m = interior(cv) * smooth(0.55, 0.68, fbm(cv.a, cv.b, 5, 4, 406))
        rgb = mix(rgb, C(.55, .54, .51), 0.7 * m)
        h = h + 0.05 * m * fbm(cv.a, cv.b, 14, 2, 407)
    return rgb, h, []


def tex_snow(cv, variant=1):
    a, b = warp(cv.a, cv.b, 0.15, 2, 51)
    n1 = fbm(a, b, 2, 4, 501)
    rip = np.sin(2 * np.pi * (5 * a + 2 * b) + 4.0 * fbm(a, b, 3, 3, 502))
    h = 0.05 * n1 + 0.006 * rip
    rgb = ramp(n1, [(0.3, C(.86, .89, .93)), (0.7, C(.95, .96, .98))]) * fine(cv, 504, 0.05)
    if variant == 2:   # exposed rock
        m = interior(cv) * smooth(0.6, 0.66, fbm(cv.a, cv.b, 6, 4, 503))
        rgb = mix(rgb, C(.36, .35, .36), 0.85 * m)
        h = h + 0.04 * m
    return rgb, h, []


def tex_ocean(cv, variant=1):
    a, b = warp(cv.a, cv.b, 0.2, 2, 61)
    n1 = fbm(a, b, 2, 4, 601)
    waves = np.sin(2 * np.pi * (9 * a + 14 * b) + 5.0 * fbm(a, b, 3, 3, 602))
    rgb = ramp(n1, [(0.3, C(.05, .15, .25)), (0.7, C(.08, .20, .31))])
    rgb = rgb * (1.0 + 0.03 * waves)[..., None]
    if variant == 2:   # a slightly lighter swell
        m = interior(cv) * smooth(0.45, 0.65, fbm(cv.a, cv.b, 3, 3, 603))
        rgb = mix(rgb, C(.08, .21, .32), 0.35 * m)
    elif variant == 3:   # a few whitecaps
        caps = smooth(0.84, 0.88, fbm(cv.a, cv.b, 20, 2, 604)) * interior(cv, 0.15, 0.3)
        rgb = mix(rgb, C(.55, .65, .72), 0.35 * caps)
    return rgb, np.zeros(cv.shape), []


def tex_coast(cv, variant=1):
    a, b = warp(cv.a, cv.b, 0.2, 2, 71)
    n1, n2 = fbm(a, b, 2, 4, 701), fbm(a, b, 6, 3, 702)
    waves = np.sin(2 * np.pi * (8 * a + 13 * b) + 5.0 * fbm(a, b, 3, 3, 703))
    rgb = ramp(n1, [(0.3, C(.09, .31, .41)), (0.7, C(.14, .39, .48))])
    rgb = mix(rgb, C(.20, .46, .50), 0.35 * smooth(0.55, 0.75, n2))
    rgb = rgb * (1.0 + 0.025 * waves)[..., None]
    if variant == 2:   # darker seagrass beds
        m = interior(cv) * smooth(0.52, 0.66, fbm(cv.a, cv.b, 5, 3, 704))
        rgb = mix(rgb, C(.08, .27, .33), 0.4 * m)
    return rgb, np.zeros(cv.shape), []


def tex_lakes(cv, variant=1):
    a, b = warp(cv.a, cv.b, 0.2, 2, 81)
    n1 = fbm(a, b, 2, 4, 801)
    rgb = ramp(n1, [(0.3, C(.07, .22, .26)), (0.7, C(.10, .28, .31))])
    return rgb, np.zeros(cv.shape), []


def tex_rock(cv, variant=1):
    """Rocky ground under mountains. Its edges are what Scree edge images continue."""
    a, b = warp(cv.a, cv.b, 0.12, 2, 91)
    n1, n2 = fbm(a, b, 2, 4, 901), fbm(a, b, 9, 3, 902)
    n3 = fbm(cv.a, cv.b, 30, 2, 903)
    rgb = ramp(n1, [(0.3, C(.36, .34, .30)), (0.5, C(.45, .42, .37)), (0.7, C(.53, .50, .45))])
    rgb = mix(rgb, C(.33, .36, .25), 0.45 * smooth(0.6, 0.75, n2))
    rgb = rgb * (0.85 + 0.3 * n3)[..., None] * fine(cv, 904, 0.2)
    h = 0.05 * n1 + 0.04 * n2
    return rgb, h, []


TEXTURES = {"Grassland": tex_grassland, "Plains": tex_plains, "Desert": tex_desert, "Tundra": tex_tundra,
            "Snow": tex_snow, "Ocean": tex_ocean, "Coast": tex_coast, "Lakes": tex_lakes, "Rock": tex_rock}
SHADOW_TINT = {"Snow": C(.72, .80, .95)}
VARIANTS = {"Grassland": 3, "Plains": 3, "Desert": 3, "Tundra": 3, "Snow": 2, "Ocean": 3, "Coast": 2, "Lakes": 1}


def textured(name, cv, variant=1):
    rgb, h, trees = TEXTURES[name](cv, variant)
    return lit(rgb, shade(h, cv.spx), shadow_tint=SHADOW_TINT.get(name)), trees


def tile_name(name, variant):
    return "Tiles/" + name + ("" if variant == 1 else str(variant))


def render_base(name, variant):
    rel = tile_name(name, variant)
    if not wanted(rel):
        return
    cv = Canvas(TILE_W)
    rgb, trees = textured(name, cv, variant)
    P = premult(rgb, cv.hex_alpha())
    P = splat_trees(cv, P, trees)
    save(cv, P, rel)


# ----------------------------------------------------------------------------- mountains
MOUNTAIN_PEAKS = {
    1: [(0.0, 0.08, 1.05, 0.72)],
    2: [(-0.32, 0.16, 0.85, 0.55), (0.30, 0.02, 1.0, 0.58)],
    3: [(-0.40, 0.22, 0.75, 0.48), (0.05, 0.04, 1.0, 0.55), (0.45, 0.24, 0.7, 0.45)],
}
MOUNTAIN_OVERHANG = 40


def render_mountain(variant):
    rel = tile_name("Mountain", variant)
    if not wanted(rel):
        return
    g = Canvas(TILE_W)                                   # ground samples (the footprint)
    out = Canvas(TILE_W, overhang=MOUNTAIN_OVERHANG)     # the taller output image
    envelope = np.zeros(g.shape)
    for u0, v0, hh, r in MOUNTAIN_PEAKS[variant]:
        rr = np.hypot(g.u - u0, (g.v - v0) * 1.1) / r
        envelope = np.maximum(envelope, hh * np.clip(1 - rr, 0, 1) ** 0.95)
    aw, bw = warp(g.a, g.b, 0.25, 3, 940 + variant)
    crags = ridged(aw, bw, 4, 5, 950 + variant)          # sharp ridges and gullies
    hm = envelope * (0.6 + 0.55 * crags) + 0.04 * fbm(g.a, g.b, 16, 2, 955) * envelope
    hm = hm * smooth(0.02, 0.4, g.edge)
    base_rgb, base_h, _ = tex_rock(g)
    h = hm + base_h
    gy, gx = np.gradient(h)
    slope = np.hypot(gx, gy) * g.spx
    n = fbm(g.a, g.b, 10, 3, 960 + variant)
    rock = ramp(n, [(0.3, C(.46, .43, .39)), (0.5, C(.55, .51, .46)), (0.7, C(.64, .60, .54))])
    rock = mix(rock, C(.50, .44, .36), 0.35 * smooth(0.5, 0.7, fbm(g.a, g.b, 6, 3, 965)))
    rgb = mix(base_rgb, rock, smooth(0.04, 0.3, hm))
    rgb = mix(rgb, C(.36, .40, .27), 0.35 * smooth(0.25, 0.05, hm) * smooth(0.02, 0.1, hm))   # alpine meadow
    rgb = rgb * (0.78 + 0.22 * crags)[..., None]                                               # darker gullies
    snow = smooth(0.0, 0.1, hm - (0.72 + 0.16 * (n - 0.5))) * smooth(6.0, 4.0, slope)
    rgb = mix(rgb, C(.94, .96, .99), snow)
    s = 0.5 + 0.55 * np.clip(shade(h, g.spx, 0.8), 0.0, 1.6)   # extra sky light: the camera sees steep faces
    rgb = lit(rgb, s, lo=0.45, hi=1.4, shadow_tint=C(.86, .9, 1.0))

    valid = g.hex_alpha() > 0.5
    P = out.empty()
    off = MOUNTAIN_OVERHANG * SS
    rows = g.shape[0]
    cols = np.arange(g.shape[1])
    for r in range(rows):                                # painter's algorithm, north to south
        top = r + off - K_RISE * h[r] * g.spx
        ylo = np.clip(np.floor(top).astype(int), 0, r + off)
        y0 = int(ylo[valid[r]].min()) if valid[r].any() else r + off
        ys = np.arange(y0, r + off + 1)[:, None]
        fill = (ys >= ylo[None, :]) & valid[r][None, :]
        if not fill.any():
            continue
        block = P[y0:r + off + 1]
        colr = np.concatenate([rgb[r], np.ones((len(cols), 1))], axis=1)
        block[fill] = np.broadcast_to(colr[None, :, :], fill.shape + (4,))[fill]
    save(out, P, rel)


# ----------------------------------------------------------------------------- terrain features
FEATURE_OVERHANG = 36


def render_hill(variant):
    rel = tile_name("Hill", variant)
    if not wanted(rel):
        return
    cv = Canvas(TILE_W)
    mounds = {1: [(-0.4, 0.2, 0.32), (0.3, -0.12, 0.36), (0.05, 0.45, 0.28), (-0.1, -0.4, 0.26), (0.5, 0.35, 0.24)],
              2: [(-0.25, -0.2, 0.36), (0.35, 0.28, 0.32), (-0.48, 0.38, 0.24), (0.45, -0.35, 0.22), (0.0, 0.1, 0.22)]}[variant]
    hh = np.zeros(cv.shape)
    for u0, v0, r in mounds:
        d = np.hypot(cv.u - u0, (cv.v - v0) * 1.2) / r
        hh = hh + 0.2 * np.exp(-d * d * 1.8)
    aw, bw = warp(cv.a, cv.b, 0.2, 3, 1110 + variant)
    hh = hh * (0.65 + 0.5 * fbm(aw, bw, 3, 4, 1100 + variant)) + 0.006 * ridged(aw, bw, 8, 3, 1120)
    hh = hh * smooth(0.0, 0.3, cv.edge)
    s = shade(hh, cv.spx, 1.2)
    light = np.clip((s - 1.0) * 0.9, 0, 0.22)
    dark = np.clip((1.0 - s) * 1.1, 0, 0.5)
    inside = (cv.hex_alpha() > 0)
    P = premult(np.broadcast_to(C(.45, .38, .26), cv.shape + (3,)), smooth(0.04, 0.12, hh) * 0.12 * inside)
    P = over(P, premult(np.broadcast_to(C(.05, .06, .03), cv.shape + (3,)), dark * inside))
    P = over(P, premult(np.ones(cv.shape + (3,)) * C(1.0, .94, .72), light))
    save(cv, P, rel)


def render_forest(kind, variant):
    rel = tile_name(kind, variant)
    if not wanted(rel):
        return
    cv = Canvas(TILE_W, overhang=FEATURE_OVERHANG)
    jungle = kind == "Jungle"
    rng = np.random.default_rng(1200 + (100 if jungle else 0))
    pts = place_periodic(rng, 80 if jungle else 95, 0.12 if jungle else 0.105)
    vrng = np.random.default_rng(1300 + variant + (100 if jungle else 0))
    trees = []
    for u, v in pts:
        e = hex_edge_dist(u, v)
        if variant == 2 and e > 0.3 and (u - 0.15) ** 2 + (v - 0.1) ** 2 < 0.09:
            continue   # a clearing in the middle of the second version
        trng = np.random.default_rng(int(abs(u * 1e4)) * 7 + int(abs(v * 1e4)) * 13 + (5 if jungle else 0))
        if e > 0.3:
            trng = vrng   # interior trees differ per version, edge trees stay identical
        if jungle:
            kind_t = "palm" if trng.random() < 0.07 else "broad"
            trees.append(make_tree(trng, u, v, (0.085, 0.125), (0.12, 0.2), JUNGLE_GREENS, kind_t))
        else:
            conifer = trng.random() < 0.35
            trees.append(make_tree(trng, u, v, (0.055, 0.075) if conifer else (0.065, 0.095), (0.11, 0.18),
                                   CONIFER_GREENS if conifer else FOREST_GREENS, "conifer" if conifer else "broad"))
    floor_n = fbm(cv.a, cv.b, 6, 3, 1400)
    floor = C(.08, .16, .07) if not jungle else C(.09, .19, .08)
    P = premult(np.broadcast_to(floor, cv.shape + (3,)) * (0.85 + 0.3 * floor_n)[..., None],
                cv.hex_alpha() * (0.55 + 0.2 * floor_n) * (cv.v > -S3 / 2 - 0.01))
    P = splat_trees(cv, P, trees, shadow_alpha=0.45)
    save(cv, P, rel)


def render_marsh():
    rel = "Tiles/Marsh"
    if not wanted(rel):
        return
    cv = Canvas(TILE_W)
    a, b = warp(cv.a, cv.b, 0.2, 3, 1500)
    pools = smooth(0.55, 0.6, fbm(a, b, 5, 3, 1501))
    reeds = fbm(cv.a, cv.b, 24, 2, 1502)
    P = premult(C(.30, .35, .19) * (0.8 + 0.4 * reeds)[..., None], cv.hex_alpha() * 0.5)
    P = over(P, premult(np.broadcast_to(C(.10, .19, .19), cv.shape + (3,)), cv.hex_alpha() * pools * 0.9))
    save(cv, P, rel)


def render_flood_plains():
    rel = "Tiles/Flood plains"
    if not wanted(rel):
        return
    cv = Canvas(TILE_W)
    a, b = warp(cv.a, cv.b, 0.25, 2, 1600)
    band = np.sin(2 * np.pi * (2 * a + 1 * b) + 3 * fbm(a, b, 3, 3, 1601)) * 0.5 + 0.5
    veg = smooth(0.4, 0.62, band * 0.45 + fbm(a, b, 8, 3, 1602) * 0.55)
    P = premult(np.broadcast_to(C(.52, .42, .28), cv.shape + (3,)), cv.hex_alpha() * 0.2)
    green = ramp(fbm(cv.a, cv.b, 20, 2, 1603), [(0.3, C(.22, .34, .13)), (0.7, C(.33, .44, .18))])
    P = over(P, premult(green, cv.hex_alpha() * veg * 0.9))
    save(cv, P, rel)


def render_oasis():
    rel = "Tiles/Oasis"
    if not wanted(rel):
        return
    cv = Canvas(TILE_W, overhang=FEATURE_OVERHANG)
    pond = blob(cv, 0.0, 0.08, 0.26, 1700, 0.2)
    halo = blob(cv, 0.0, 0.08, 0.5, 1701, 0.3)
    P = premult(np.broadcast_to(C(.30, .40, .17), cv.shape + (3,)), halo * 0.85 * (cv.edge > 0))
    P = over(P, premult(np.broadcast_to(C(.10, .30, .34), cv.shape + (3,)), pond))
    rng = np.random.default_rng(1702)
    ring = lambda u, v: 0.06 < (u ** 2 + (v - 0.08) ** 2) < 0.2
    trees = [make_tree(rng, u, v, (0.06, 0.08), (0.12, 0.18), JUNGLE_GREENS, "palm")
             for u, v in place_interior(rng, 12, 0.3, 0.1, ring)]
    P = splat_trees(cv, P, trees)
    save(cv, P, rel)


def render_ice():
    rel = "Tiles/Ice"
    if not wanted(rel):
        return
    cv = Canvas(TILE_W)
    a, b = warp(cv.a, cv.b, 0.2, 3, 1800)
    f = fbm(a, b, 4, 4, 1801)
    cracks = smooth(0.93, 0.98, ridged(a, b, 6, 3, 1802))
    h = 0.03 * fbm(cv.a, cv.b, 8, 3, 1804)
    rgb = ramp(fbm(cv.a, cv.b, 10, 3, 1803), [(0.3, C(.82, .88, .93)), (0.7, C(.94, .96, .99))])
    rgb = mix(rgb, C(.55, .65, .72), cracks * 0.7)
    rgb = lit(rgb, shade(h, cv.spx), shadow_tint=C(.75, .85, 1.0))
    leads = smooth(0.22, 0.25, f)
    save(cv, premult(rgb, cv.hex_alpha() * leads * 0.97), rel)


def render_atoll():
    rel = "Tiles/Atoll"
    if not wanted(rel):
        return
    cv = Canvas(TILE_W)
    n = fbm(cv.a, cv.b, 6, 3, 1900)
    r = np.hypot(cv.u, cv.v * 1.1) + (n - 0.5) * 0.12
    lagoon = smooth(0.5, 0.44, r)
    reef = np.exp(-((r - 0.5) / 0.045) ** 2)
    shallows = smooth(0.75, 0.5, r)
    P = premult(np.broadcast_to(C(.20, .52, .56), cv.shape + (3,)), shallows * 0.7)
    P = over(P, premult(np.broadcast_to(C(.32, .68, .68), cv.shape + (3,)), lagoon * 0.85))
    sand = mix(C(.86, .80, .64), C(.36, .52, .24), smooth(0.6, 0.72, fbm(cv.a, cv.b, 12, 2, 1901)))
    P = over(P, premult(np.broadcast_to(sand, cv.shape + (3,)), reef))
    save(cv, P, rel)


def render_fallout():
    rel = "Tiles/Fallout"
    if not wanted(rel):
        return
    cv = Canvas(TILE_W)
    n1, n2 = fbm(cv.a, cv.b, 3, 4, 2000), fbm(cv.a, cv.b, 7, 3, 2001)
    P = premult(np.broadcast_to(C(.58, .64, .18), cv.shape + (3,)), cv.hex_alpha() * (0.15 + 0.3 * n1))
    P = over(P, premult(np.broadcast_to(C(.10, .09, .07), cv.shape + (3,)), cv.hex_alpha() * smooth(0.6, 0.7, n2) * 0.6))
    save(cv, P, rel)


def render_rivers():
    for side in ("Bottom", "BottomLeft", "BottomRight"):
        rel = "Tiles/River-" + side
        if not wanted(rel):
            continue
        cv = Canvas(TILE_W)
        i, j = DIRS[side]
        p, q = np.array(corner(i)), np.array(corner(j))
        d = q - p
        normal = np.array([-d[1], d[0]])
        if np.dot(normal, -(p + q) / 2) < 0:
            normal = -normal                       # point inward
        normal /= np.linalg.norm(normal)
        L2 = float(np.dot(d, d))
        t = np.clip(((cv.u - p[0]) * d[0] + (cv.v - p[1]) * d[1]) / L2, 0, 1)
        bulge = 0.06 * np.sin(np.pi * t) + 0.025 * np.sin(np.pi * t) * np.sin(3 * np.pi * t + 1.3)
        cx = p[0] + t * d[0] + normal[0] * bulge
        cy = p[1] + t * d[1] + normal[1] * bulge
        dist = np.hypot(cv.u - cx, cv.v - cy)
        inside = cv.hex_alpha(0.5)
        bank = smooth(0.075, 0.05, dist) * inside
        water = smooth(0.045, 0.03, dist) * inside
        P = premult(np.broadcast_to(C(.22, .30, .17), cv.shape + (3,)), bank * 0.75)
        P = over(P, premult(np.broadcast_to(C(.10, .27, .33), cv.shape + (3,)), water))
        save(cv, P, rel)


# ----------------------------------------------------------------------------- edge images
def band(cv, side, width, seed, rough=0.35):
    d = side_dist(cv.u, cv.v, side)
    jitter = (fbm(cv.a, cv.b, 6, 3, seed) - 0.5) * width * rough * 2
    return (1.0 - smooth(0.0, width, d + jitter)) * cv.hex_alpha(), d


def edge_rel(group, name, origin, dest, side):
    return "Edges/%s/%s-%s-%s-%s" % (group, name, origin, dest, side)


def render_edges():
    sides = list(DIRS)
    # Land-to-land blends: the higher-priority terrain spreads into the lower one.
    for x, y in combinations(LAND5, 2):
        lower, higher = sorted((x, y), key=PRIORITY.get)
        for side in sides:
            rel = edge_rel("Blend", "Blend", lower, higher, side)
            if not wanted(rel):
                continue
            cv = Canvas(EDGE_W)
            rgb, _ = textured(higher, cv)
            w, _ = band(cv, side, 0.32, 3000 + PRIORITY[lower] * 10 + PRIORITY[higher])
            save(cv, premult(rgb, w), rel)
    # Shorelines on each land terrain.
    shore = {"Grassland": (C(.52, .48, .37), C(.80, .73, .55)), "Plains": (C(.55, .50, .38), C(.82, .75, .57)),
             "Desert": (C(.64, .55, .39), C(.89, .81, .63)), "Tundra": (C(.38, .38, .36), C(.62, .61, .56)),
             "Snow": (C(.60, .66, .72), C(.88, .91, .95))}
    for terrain, (wet, dry) in shore.items():
        for side in sides:
            rel = edge_rel("Shore", "Shore", terrain, "Water", side)
            if not wanted(rel):
                continue
            cv = Canvas(EDGE_W)
            width = 0.08 if side.startswith("Top") else 0.12
            w, d = band(cv, side, width, 3100 + PRIORITY[terrain], 0.9)
            grain = fbm(cv.a, cv.b, 30, 2, 3150)
            wobble = (fbm(cv.a, cv.b, 9, 2, 3160) - 0.5) * 0.03
            rgb = mix(wet, dry, smooth(0.012, 0.04, d + wobble)) * (0.9 + 0.2 * grain)[..., None]
            save(cv, premult(rgb, w * 0.92), rel)
    # Rocky slopes spreading from mountains onto each land terrain.
    for terrain in LAND5:
        for side in sides:
            rel = edge_rel("Scree", "Scree", terrain, "Mountain", side)
            if not wanted(rel):
                continue
            cv = Canvas(EDGE_W)
            rgb, _ = textured("Rock", cv)
            w, _ = band(cv, side, 0.3, 3200 + PRIORITY[terrain], 0.5)
            patches = 0.55 + 0.45 * smooth(0.35, 0.6, fbm(cv.a, cv.b, 8, 3, 3250))
            save(cv, premult(rgb, w * patches), rel)
    # Dark wet rock where mountains meet water. Only the south-facing sides: on the north sides the
    # mountain itself rises over the edge, and an edge image would paint across it.
    for side in ("Bottom", "BottomLeft", "BottomRight"):
        rel = edge_rel("Cliff", "Cliff", "Mountain", "Water", side)
        if not wanted(rel):
            continue
        cv = Canvas(EDGE_W)
        w, _ = band(cv, side, 0.08, 3300, 0.5)
        grain = fbm(cv.a, cv.b, 24, 2, 3301)
        save(cv, premult(C(.24, .23, .22) * (0.8 + 0.4 * grain)[..., None], w), rel)
    # Shallows and surf on water next to land.
    for side in sides:
        rel = edge_rel("Foam", "Foam", "Water", "Land", side)
        if not wanted(rel):
            continue
        cv = Canvas(EDGE_W)
        w, d = band(cv, side, 0.3, 3400, 0.4)
        sand = mix(C(.20, .50, .53), C(.42, .62, .57), smooth(0.18, 0.03, d))
        P = premult(sand, w * 0.85)
        breaks = smooth(0.4, 0.6, fbm(cv.a, cv.b, 10, 2, 3401))
        surf = np.exp(-((d - 0.035) / 0.012) ** 2) * (0.35 + 0.65 * breaks) * cv.hex_alpha()
        P = over(P, premult(np.broadcast_to(C(.92, .96, .97), cv.shape + (3,)), surf * 0.9))
        save(cv, P, rel)
    # Continental shelf: the coast texture continuing into the ocean.
    for side in sides:
        rel = edge_rel("Shelf", "Shelf", "Ocean", "Coast", side)
        if not wanted(rel):
            continue
        cv = Canvas(EDGE_W)
        rgb, _ = textured("Coast", cv)
        w, _ = band(cv, side, 0.45, 3500, 0.5)
        save(cv, premult(rgb, w), rel)


# ----------------------------------------------------------------------------- overlays
def render_overlays():
    if wanted("Hexagon"):
        cv = Canvas(TILE_W)
        save(cv, premult(np.ones(cv.shape + (3,)), cv.hex_alpha(0.0)), "Hexagon")
    if wanted("CrosshatchHexagon"):   # fog of war: thin cloud wisps, shown by the game at 20%
        cv = Canvas(TILE_W)
        a, b = warp(cv.a, cv.b, 0.3, 2, 4000)
        n = fbm(a, b, 3, 5, 4001)
        save(cv, premult(np.ones(cv.shape + (3,)), cv.hex_alpha() * (0.2 + 0.8 * smooth(0.42, 0.72, n))), "CrosshatchHexagon")
    if wanted("UnexploredTile"):      # never-seen tiles next to known ones: a bank of soft mist
        cv = Canvas(TILE_W)           # no shading or crisp detail, which would read as rocky relief
        a, b = warp(cv.a, cv.b, 0.3, 2, 4000)
        wisp = smooth(0.42, 0.72, fbm(a, b, 3, 5, 4001))      # the fog-of-war wisps, so the two join up
        a, b = warp(cv.a, cv.b, 0.6, 1, 4200)
        swell = smooth(0.25, 0.75, fbm(a, b, 2, 2, 4201))     # broad, gentle brightness swells
        rgb = mix(C(.33, .37, .45), C(.52, .56, .62), 0.65 * swell + 0.35 * wisp)
        save(cv, premult(rgb, cv.hex_alpha()), "UnexploredTile")
    if wanted("Highlight"):           # selection and movement circles, tinted by the game
        w, h = TILE_W, round(TILE_W * 212 / 284)
        cv = Canvas(w, height=h, overhang=h - round(w * S3 / 2))
        r = np.hypot(cv.u, cv.v)
        ring = smooth(0.62, 0.6, r) * smooth(0.48, 0.5, r)
        fill = smooth(0.62, 0.6, r) * 0.22
        save(cv, premult(np.ones(cv.shape + (3,)), np.maximum(ring, fill)), "Highlight")
    if wanted("Crosshair"):
        cv = Canvas(TILE_W)
        r, th = np.hypot(cv.u, cv.v), np.arctan2(cv.v, cv.u)
        ticks = (np.abs(np.sin(2 * th)) < 0.09) & (r > 0.2) & (r < 0.62)
        shape = np.maximum(smooth(0.06, 0.04, np.abs(r - 0.4)), ticks.astype(float))
        outline = np.maximum(smooth(0.09, 0.065, np.abs(r - 0.4)), ((np.abs(np.sin(2 * th)) < 0.16) & (r > 0.17) & (r < 0.66)).astype(float))
        P = premult(np.zeros(cv.shape + (3,)), outline * 0.55)
        P = over(P, premult(np.broadcast_to(C(.95, .22, .2), cv.shape + (3,)), shape))
        P = over(P, premult(np.broadcast_to(C(1, .85, .4), cv.shape + (3,)), smooth(0.05, 0.035, r)))
        save(cv, P, "Crosshair")
    render_borders()
    render_arrows()
    render_roads()


def _raster(width, height):
    yy, xx = np.mgrid[0:height * SS, 0:width * SS] + 0.5
    return xx / SS, yy / SS


def _save_raster(P, relpath, width, height):
    # Borders, arrows and roads are positioned by the game from their full size: never trim them.
    class Flat:
        pad = 0
    save(Flat, P, relpath, trim=False)


def render_borders():
    # Same geometry as the base game's border pieces (81x15), drawn three times larger.
    # A border piece runs along the bottom side of the hex and is rotated by the game for each side.
    # The first word of the name sets the right end, the second word the left end.
    sc, w, h = 3, 81 * 3, 15 * 3
    xx, yy = _raster(w, h)
    row = yy / sc
    inner_l = {"Convex": lambda y: 22 - (12 - y) * 5 / 11, "Concave": lambda y: 18 - (12 - y) * 6 / 11}
    inner_r = {"Convex": lambda y: 59 + (12 - y) * 5 / 11, "Concave": lambda y: 63 + (12 - y) * 6 / 11}
    outer_l, outer_r = {"Convex": 21, "Concave": 19}, {"Convex": 60, "Concave": 62}
    for right, left in (("Convex", "Convex"), ("Concave", "Concave"), ("Convex", "Concave"), ("Concave", "Convex")):
        name = right + left if right != left else right
        x = xx / sc
        inside_x = np.clip(np.minimum(x - inner_l[left](row), inner_r[right](row) - x) * sc / 1.0, 0, 1)
        glow = np.clip((row - 1.0) / 11.5, 0, 1) ** 1.6
        a_in = inside_x * glow * (row >= 1) * (row < 13) * 0.95
        if wanted("Borders/" + name + "Inner"):
            _save_raster(premult(np.ones(a_in.shape + (3,)), a_in), "Borders/" + name + "Inner", w, h)
        inside_o = np.clip(np.minimum(x - outer_l[left], outer_r[right] - x) * sc, 0, 1)
        a_out = inside_o * smooth(12.8, 13.2, row)
        if wanted("Borders/" + name + "Outer"):
            _save_raster(premult(np.ones(a_out.shape + (3,)), a_out), "Borders/" + name + "Outer", w, h)


def render_arrows():
    # Arrows are stretched by the game from one tile centre to another; the band sits in the middle third.
    w, h = 200, 120
    xx, yy = _raster(w, h)
    styles = {"Generic": (C(1, 1, 1), 20, 26), "UnitMoving": (C(.95, .95, .95), 13, 18),
              "UnitMoved": (C(.35, .58, .98), 18, 26), "UnitAttacked": (C(.95, .28, .22), 20, 26),
              "UnitHasAttacked": (C(.95, .28, .22), 10, 14), "UnitWithdrew": (C(.30, .82, .46), 18, 26),
              "UnitTeleported": (C(.78, .58, .98), 16, 20)}
    for name, (col, half, period) in styles.items():
        rel = "Arrows/" + name
        if not wanted(rel):
            continue
        t = (xx % period) - np.abs(yy - 60) * 0.7
        arm = np.clip(np.minimum(t, period * 0.42 - t), 0, 1) * (np.abs(yy - 60) <= half)
        if name == "UnitTeleported":
            arm = arm * (np.sin(xx * 0.5) > -0.2)
        edge = np.clip(np.minimum(t + 2, period * 0.42 - t + 2), 0, 1) * (np.abs(yy - 60) <= half + 2)
        P = premult(np.zeros(xx.shape + (3,)), edge * 0.5)
        P = over(P, premult(np.broadcast_to(col, xx.shape + (3,)), arm))
        _save_raster(P, rel, w, h)


def render_roads():
    if wanted("Road"):
        w, h = 120, 64
        xx, yy = _raster(w, h)
        d = np.abs(yy - 32)
        n = fbm(xx / 60, yy / 60, 8, 2, 4200)
        rgb = mix(C(.48, .40, .30), C(.72, .62, .46), smooth(9, 5, d)) * (0.9 + 0.2 * n)[..., None]
        rgb = mix(rgb, C(.55, .46, .34), np.exp(-((d - 3.5) / 1.0) ** 2) * 0.6)
        _save_raster(premult(rgb, smooth(11, 9.5, d)), "Road", w, h)
    if wanted("Railroad"):
        w, h = 128, 72
        xx, yy = _raster(w, h)
        d = np.abs(yy - 36)
        P = premult(np.broadcast_to(C(.47, .45, .42), xx.shape + (3,)), smooth(13, 11.5, d))
        ties = ((xx % 12) < 5) & (d < 10)
        P = over(P, premult(np.broadcast_to(C(.30, .22, .16), xx.shape + (3,)), ties.astype(float)))
        rails = np.exp(-((d - 5.5) / 1.2) ** 2)
        P = over(P, premult(np.broadcast_to(C(.78, .78, .80), xx.shape + (3,)), rails))
        _save_raster(P, "Railroad", w, h)


# ----------------------------------------------------------------------------- main
def main():
    global FILTER
    FILTER = sys.argv[1:] or None
    for name, count in VARIANTS.items():
        for v in range(1, count + 1):
            render_base(name, v)
    for v in (1, 2, 3):
        render_mountain(v)
    for v in (1, 2):
        render_hill(v)
        render_forest("Forest", v)
        render_forest("Jungle", v)
    render_marsh()
    render_flood_plains()
    render_oasis()
    render_ice()
    render_atoll()
    render_fallout()
    render_rivers()
    render_edges()
    render_overlays()
    groups = {}
    for rel in WRITTEN:
        key = rel.split("/")[0] if "/" in rel else "Overlays"
        groups[key] = groups.get(key, 0) + 1
    print("Wrote %d images to %s: %s" % (len(WRITTEN), OUT, ", ".join("%s %d" % kv for kv in sorted(groups.items()))))


if __name__ == "__main__":
    main()
