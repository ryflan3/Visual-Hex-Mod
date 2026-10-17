#!/usr/bin/env python3
"""Render the unit set: units as flat illustrations with ink outlines, so they stand out on the map.

    python3 tools/render_units.py              # every unit
    python3 tools/render_units.py Tank Archer  # only units whose image path contains one of the words

How Unciv draws unit images
---------------------------
* A unit image is drawn at the hex width, keeping its aspect ratio and anchored at the hex
  bottom, like a tile image. A tile's military and civilian unit share the same box.
* The unit flags sit in the middle of the hex, the military unit's in the upper half and the
  civilian's in the lower half. So military units stand on the right of the hex and civilians on
  the left, as in the base game's HexaRealm units.
* "<Unit>-1" and "<Unit>-2" are drawn over "<Unit>", tinted with the owner's inner and outer
  (main) colour. Each unit's banner, tunic, sails or markings are painted into "-2" in light grey,
  so they come out in the owner's colour. Layers have to be numbered from 1, so "-1" is an empty
  placeholder.
* Unique units without their own image use the image of the unit they replace. Units on water
  use EmbarkedUnit-Military or EmbarkedUnit-Civilian, which have later-era versions.
* Aircraft and missiles never stand on a tile (they are listed on their city's or carrier's
  flag), so they have no images.

How they are drawn
------------------
Each unit is a list of flat shapes in local units: a standing figure is about 100 tall, x points
east, y points down and the feet stand on y = 0. Shapes are SVG-style paths, ellipses, rounded
rectangles and round-ended limbs. Every shape gets a dark ink outline and a shading ramp lit from
the west, like the map, and the whole unit a soft dark edge and a ground shadow. Pillow fills the
shapes at four times the final size and the result is averaged down.
"""

import math
import os
import re
import sys
from contextlib import contextmanager

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import render_tiles as rt

UNIT_W = 160                 # the same resolution as resources
S = 4                        # supersampling
OVERHANG = 80                # room above the hex for tall units, in final pixels
FOOT = round(UNIT_W * rt.S3 / 2)
H = FOOT + OVERHANG
K = UNIT_W / 168             # final pixels per local unit, before a unit is fitted to its side of the hex
INK_W = 1.6                  # ink outline width, local units
MIL = (0.56, 0.36)           # where a military unit stands: the middle of its width and its ground line (hex
CIV = (-0.6, 0.32)           # units, centre at 0, 0, v pointing south); civilians stand on the left
AT_SEA = (0.52, 0.34)
CIV_SEA = (-0.6, 0.3)
OWNER = "owner"              # as a colour: paint this part in the owner's colour


def hexcol(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


INK = hexcol("#1d1a16")
PAINT = (0.86, 0.86, 0.86)   # owner-coloured parts are this light grey in the images
C = {name: hexcol(h) for name, h in {
    "skin": "#d9a27c", "skin2": "#b8805c", "hair": "#3b2617", "grey_hair": "#c9c4ba", "beard": "#4a3020",
    "pants": "#5a4b3c", "cloth": "#7a6a55", "linen": "#e8dfc8", "white": "#f1efe9", "boots": "#3a2a1c",
    "leather": "#7b5333", "wood": "#8a5d35", "darkwood": "#5e3e22", "metal": "#8f979f", "steel": "#c7cdd3",
    "iron": "#6f767d", "bronze": "#c08a3e", "gold": "#e2b443", "black": "#26272b", "navy": "#283048",
    "horse": "#8a5a34", "darkhorse": "#4a3020", "greyhorse": "#b9b5ae", "mane": "#2f2117", "olive": "#646e3d",
    "olivedark": "#464d2b", "khaki": "#9a8a5c", "camo": "#4f5a34", "sand": "#b7a57a", "track": "#2c2c2a",
    "hull": "#3d2a1c", "deck": "#a27c52", "sail": "#f3efe2", "darksail": "#55575b", "rope": "#c9b48a",
    "straw": "#d8b65c", "foam": "#e8f4f8", "glass": "#3e5664", "grey": "#8a9096", "navygrey": "#7d858c",
    "darkgrey": "#4a4f55", "red": "#b33a2a", "stone": "#9c9890", "flame": "#ffb43c", "glow": "#9fe6ff",
    "robe": "#6b4a2e", "black_robe": "#2a2626", "paper": "#f4ead2"}.items()}


# ----------------------------------------------------------------------------- geometry
def bezier(p0, ctrl, p1, n=10):
    pts = []
    for i in range(1, n + 1):
        t = i / n
        q = [p0] + list(ctrl) + [p1]
        while len(q) > 1:
            q = [((1 - t) * a[0] + t * b[0], (1 - t) * a[1] + t * b[1]) for a, b in zip(q[:-1], q[1:])]
        pts.append(q[0])
    return pts


def path_points(d):
    """Points along an SVG-style path with absolute M, L, H, V, Q, C and Z commands."""
    toks = re.findall(r"[MLHVQCZ]|-?\d*\.?\d+", d)
    pts, i, cmd, cur = [], 0, None, (0.0, 0.0)

    def num():
        nonlocal i
        i += 1
        return float(toks[i - 1])

    while i < len(toks):
        if toks[i] in "MLHVQCZ":
            cmd = toks[i]
            i += 1
            if cmd == "Z":
                continue
        if cmd in ("M", "L"):
            cur = (num(), num())
            pts.append(cur)
        elif cmd == "H":
            cur = (num(), cur[1])
            pts.append(cur)
        elif cmd == "V":
            cur = (cur[0], num())
            pts.append(cur)
        elif cmd == "Q":
            c, e = (num(), num()), (num(), num())
            pts += bezier(cur, [c], e)
            cur = e
        elif cmd == "C":
            c1, c2, e = (num(), num()), (num(), num()), (num(), num())
            pts += bezier(cur, [c1, c2], e)
            cur = e
        else:
            raise ValueError("bad path: " + d)
    return pts


def ellipse_points(cx, cy, rx, ry, rot=0.0, n=40, a0=0.0, a1=2 * math.pi):
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    out = []
    for i in range(n):
        t = a0 + (a1 - a0) * i / (n if a1 - a0 >= 2 * math.pi - 1e-6 else n - 1)
        x, y = rx * math.cos(t), ry * math.sin(t)
        out.append((cx + x * c - y * s, cy + x * s + y * c))
    return out


def rect_points(x, y, w, h, rx=0.0, rot=0.0):
    rx = min(rx, w / 2, h / 2)
    if rx <= 0:
        pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    else:
        pts = []
        for cx, cy, a in ((x + w - rx, y + rx, -90), (x + w - rx, y + h - rx, 0), (x + rx, y + h - rx, 90), (x + rx, y + rx, 180)):
            pts += [(cx + rx * math.cos(math.radians(a + 90 * j / 6)), cy + rx * math.sin(math.radians(a + 90 * j / 6))) for j in range(7)]
    if rot:
        cx, cy = x + w / 2, y + h / 2
        c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        pts = [(cx + (px - cx) * c - (py - cy) * s, cy + (px - cx) * s + (py - cy) * c) for px, py in pts]
    return pts


# ----------------------------------------------------------------------------- painting
class Sheet:
    """A unit image and its owner-colour layer, painted together at S times the final size.
    Whatever is drawn over an owner-coloured part also hides it in the colour layer."""

    def __init__(self):
        self.base = np.zeros((H * S, UNIT_W * S, 4), np.float32)
        self.civ = np.zeros_like(self.base)
        self.ground = np.zeros_like(self.base)     # shadows on the ground, under everything
        self.owner = False

    def put(self, x0, y0, cover, rgb):
        h, w = cover.shape
        X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, UNIT_W * S), min(y0 + h, H * S)
        if X1 <= X0 or Y1 <= Y0:
            return
        a = cover[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0, None]
        rgb = np.broadcast_to(np.asarray(rgb, np.float32), (h, w, 3))[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
        src = np.concatenate([rgb * a, a], axis=-1)
        self.base[Y0:Y1, X0:X1] = src + self.base[Y0:Y1, X0:X1] * (1 - a)
        if self.owner:
            self.civ[Y0:Y1, X0:X1] = src + self.civ[Y0:Y1, X0:X1] * (1 - a)
        else:
            self.civ[Y0:Y1, X0:X1] *= 1 - a


def ramp(t):
    """Shading across a shape, 0 at its west side to 1 at its east side: lit, flat, shaded."""
    light = np.clip(1 - t / 0.42, 0, 1) * 0.3
    dark = np.clip((t - 0.6) / 0.4, 0, 1) * 0.32
    return light, dark


class Pen:
    """Draws a unit in local units onto a Sheet, standing at (ax, ay) final pixels, k pixels per
    unit. Without a sheet it only measures how far the unit reaches."""

    def __init__(self, sheet, ax=0.0, ay=0.0, k=1.0):
        self.sheet, self.ax, self.ay, self.k = sheet, ax, ay, k
        self.lo = [math.inf, math.inf]
        self.hi = [-math.inf, -math.inf]
        self._owner = False

    # --- coordinates and colours
    def px(self, x, y):
        return (self.ax + self.k * x) * S, (self.ay + self.k * y) * S

    def _grow(self, pts, pad=0.0):
        for x, y in pts:
            self.lo = [min(self.lo[0], x - pad), min(self.lo[1], y - pad)]
            self.hi = [max(self.hi[0], x + pad), max(self.hi[1], y + pad)]

    @contextmanager
    def owner(self):
        """Parts drawn inside this block come out in the owner's colour."""
        old, self._owner = self._owner, True
        try:
            yield
        finally:
            self._owner = old

    def _col(self, col):
        own = self._owner or col is OWNER
        if self.sheet is not None:
            self.sheet.owner = own
        return PAINT if own else (C[col] if isinstance(col, str) else col)

    def _done(self):
        if self.sheet is not None:
            self.sheet.owner = False

    def _mask(self, P, draw_fn, pad):
        xs, ys = [p[0] for p in P], [p[1] for p in P]
        x0, y0 = int(math.floor(min(xs) - pad)), int(math.floor(min(ys) - pad))
        x1, y1 = int(math.ceil(max(xs) + pad)), int(math.ceil(max(ys) + pad))
        img = Image.new("L", (max(1, x1 - x0), max(1, y1 - y0)), 0)
        draw_fn(ImageDraw.Draw(img), [(x - x0, y - y0) for x, y in P])
        return x0, y0, np.asarray(img, np.float32) / 255

    # --- primitives
    def fill(self, pts, col, ink=True, shade=True, alpha=1.0, ink_w=INK_W):
        """A filled shape through local points, with its ink outline."""
        self._grow(pts, ink_w / 2 if ink else 0)
        if self.sheet is None:
            return
        P = [self.px(x, y) for x, y in pts]
        x0, y0, cover = self._mask(P, lambda d, q: d.polygon(q, fill=255), 2)
        rgb = np.asarray(self._col(col), np.float32)
        if shade:
            xs, ys = [p[0] for p in P], [p[1] for p in P]
            gx = (np.arange(cover.shape[1]) + x0 + 0.5 - min(xs)) / max(1.0, max(xs) - min(xs))
            gy = (np.arange(cover.shape[0]) + y0 + 0.5 - min(ys)) / max(1.0, max(ys) - min(ys))
            t = (gx[None, :] + 0.25 * gy[:, None]) / 1.25
            light, dark = ramp(t)
            rgb = (rgb + (1 - rgb) * light[..., None]) * (1 - dark[..., None])
        self.sheet.put(x0, y0, cover * alpha, rgb)
        self._done()
        if ink:
            w = max(1, round(ink_w * self.k * S))
            x0, y0, cover = self._mask(P, lambda d, q: d.line(q + [q[0]], fill=255, width=w, joint="curve"), w)
            self.sheet.put(x0, y0, cover * alpha, INK)

    def path(self, d, col, **kw):
        self.fill(path_points(d), col, **kw)

    def ellipse(self, cx, cy, rx, ry, col, rot=0.0, **kw):
        self.fill(ellipse_points(cx, cy, rx, ry, rot), col, **kw)

    def circle(self, cx, cy, r, col, **kw):
        self.fill(ellipse_points(cx, cy, r, r), col, **kw)

    def rect(self, x, y, w, h, col, rx=0.0, rot=0.0, **kw):
        self.fill(rect_points(x, y, w, h, rx, rot), col, **kw)

    def poly(self, pts, col, **kw):
        self.fill(pts, col, **kw)

    def stroke(self, pts, w, col, ink=True, alpha=1.0, shade=True):
        """A round-ended line through local points, w units wide, outlined in ink."""
        self._grow(pts, w / 2 + (INK_W if ink else 0))
        if self.sheet is None:
            return
        P = [self.px(x, y) for x, y in pts]

        def line(width):
            r = width / 2

            def draw(d, q):
                d.line(q, fill=255, width=max(1, round(width)), joint="curve")
                for x, y in (q[0], q[-1]):
                    d.ellipse((x - r, y - r, x + r, y + r), fill=255)
            return draw

        wp = w * self.k * S
        if ink:
            wi = wp + 2 * INK_W * 0.8 * self.k * S
            x0, y0, cover = self._mask(P, line(wi), wi)
            self.sheet.put(x0, y0, cover * alpha, INK)
        x0, y0, cover = self._mask(P, line(wp), wp)
        rgb = np.asarray(self._col(col), np.float32)
        if shade and len(P) == 2:
            (xa, ya), (xb, yb) = P
            ln = math.hypot(xb - xa, yb - ya) or 1.0
            nx, ny = (yb - ya) / ln, -(xb - xa) / ln
            if nx < 0 or (nx == 0 and ny < 0):
                nx, ny = -nx, -ny
            gx = np.arange(cover.shape[1]) + x0 + 0.5 - xa
            gy = np.arange(cover.shape[0]) + y0 + 0.5 - ya
            t = np.clip((gx[None, :] * nx + gy[:, None] * ny) / max(1.0, wp) + 0.5, 0, 1)
            light, dark = ramp(t)
            rgb = (rgb + (1 - rgb) * light[..., None]) * (1 - dark[..., None])
        self.sheet.put(x0, y0, cover * alpha, rgb)
        self._done()

    def limb(self, x1, y1, x2, y2, w, col, **kw):
        self.stroke([(x1, y1), (x2, y2)], w, col, **kw)

    def curve(self, d, w, col, **kw):
        self.stroke(path_points(d), w, col, **kw)

    def thin(self, x1, y1, x2, y2, w, col):
        """A string, rope or rigging line: no outline, no shading."""
        self.stroke([(x1, y1), (x2, y2)], w, col, ink=False, shade=False)

    def shadow(self, cx, cy, rx, ry, alpha=0.34):
        """A soft shadow on the ground."""
        self._grow([(cx - rx, cy), (cx + rx, cy)])
        if self.sheet is None:
            return
        P = [self.px(x, y) for x, y in ellipse_points(cx, cy, rx, ry)]
        blur = 2.0 * S
        x0, y0, cover = self._mask(P, lambda d, q: d.polygon(q, fill=255), blur * 3)
        img = Image.fromarray((cover * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur))
        a = np.asarray(img, np.float32) / 255 * alpha
        h, w = a.shape
        X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, UNIT_W * S), min(y0 + h, H * S)
        if X1 > X0 and Y1 > Y0:
            g = self.sheet.ground[Y0:Y1, X0:X1]
            a = a[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0, None]
            g[...] = np.concatenate([np.full(a.shape[:2] + (3,), 0.02, np.float32) * a, a], -1) + g * (1 - a)

    def glow(self, cx, cy, r, col, alpha=0.5):
        """A soft round glow (fire, a halo)."""
        self._grow([(cx - r * 0.6, cy), (cx + r * 0.6, cy)])
        if self.sheet is None:
            return
        P = [self.px(x, y) for x, y in ellipse_points(cx, cy, r, r)]
        blur = r * self.k * S * 0.35
        x0, y0, cover = self._mask(P, lambda d, q: d.polygon(q, fill=255), blur * 3)
        img = Image.fromarray((cover * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur))
        self.sheet.put(x0, y0, np.asarray(img, np.float32) / 255 * alpha, C[col] if isinstance(col, str) else col)


# ============================================================================ people
def figure(p, tunic=OWNER, pants="pants", boots="boots", skirt=True, bare=False, belt="leather", robe=None,
           long_coat=False, wide=1.0):
    """A standing figure facing the viewer, feet on y = 0, head centred at (0, -89)."""
    if robe is not None:
        for x in (-6, 6):
            p.ellipse(x, -2, 6, 3.5, boots)
        p.path("M-14,-74 Q-15,-77 -9,-78 L9,-78 Q15,-77 14,-74 L%g,-2 Q0,1 %g,-2 Z" % (19 * wide, -19 * wide), robe)
    else:
        for x in (-6, 6):
            p.rect(x - 4.5, -47, 9, 40, "skin" if bare else pants, rx=3.5)
            p.rect(x - 5.5, -13, 11, 13, boots, rx=3)
        if skirt:
            p.poly([(-13 * wide, -52), (13 * wide, -52), (17 * wide, -33), (-17 * wide, -33)], tunic)
        if long_coat:
            p.poly([(-13, -52), (13, -52), (18, -24), (-18, -24)], tunic)
        p.path("M%g,-70 Q%g,-76 -9,-77 L9,-77 Q%g,-76 %g,-70 L%g,-47 L%g,-47 Z"
               % (-15 * wide, -16 * wide, 16 * wide, 15 * wide, 13 * wide, -13 * wide), "skin" if bare else tunic)
    if belt:
        p.rect(-13.5 * wide, -51, 27 * wide, 4.5, belt, rx=1.5)
    p.rect(-3.2, -84, 6.4, 8, "skin", rx=2)
    p.circle(0, -89, 9.5, "skin")


def hair(p, col="hair"):
    p.path("M-9.6,-89 Q-10,-100 0,-100 Q10,-100 9.6,-89 Q6,-94 0,-94 Q-6,-94 -9.6,-89 Z", col)


def beard(p, col="beard"):
    p.ellipse(0, -82, 7, 4.5, col)


def arm(p, sx, sy, hx, hy, col, hand="skin", w=6.5):
    p.limb(sx, sy, hx, hy, w, col)
    p.circle(hx, hy, 3.8, hand)


def hat(p, kind, col="iron", plume=None):
    """Helmets and hats, on a head centred at (0, -89)."""
    if kind == "round":
        p.path("M-10.5,-88 Q-11,-104 0,-104 Q11,-104 10.5,-88 Z", col)
    elif kind == "cone":
        p.path("M-10.5,-88 Q-11,-98 -4,-102 L0,-113 L4,-102 Q11,-98 10.5,-88 Z", col)
    elif kind == "crest":
        p.path("M-10.5,-84 L-10.5,-92 Q-10,-104 0,-104 Q10,-104 10.5,-92 L10.5,-84 L6,-86 L6,-91 L-6,-91 L-6,-86 Z", col)
    elif kind == "kettle":
        p.ellipse(0, -94, 15, 3.5, col)
        p.path("M-9.5,-94 Q-9,-105 0,-105 Q9,-105 9.5,-94 Z", col)
    elif kind == "morion":
        p.path("M-16,-92 Q-12,-96 -9,-95 Q-8,-106 0,-107 Q8,-106 9,-95 Q12,-96 16,-92 Q0,-90 -16,-92 Z", col)
        p.path("M-6,-104 Q0,-114 6,-104 Z", col)
    elif kind == "great":
        p.rect(-10, -101, 20, 22, col, rx=5)
        p.rect(-7.5, -93, 15, 2.4, "black", ink=False)
    elif kind == "tricorne":
        p.path("M-16,-93 Q-12,-101 0,-104 Q12,-101 16,-93 Q0,-97 -16,-93 Z", col)
    elif kind == "bicorne":
        p.path("M-18,-95 Q-8,-111 0,-112 Q8,-111 18,-95 Q0,-99 -18,-95 Z", col)
        p.circle(5, -102, 2.2, "gold", ink=False)
    elif kind == "kepi":
        p.path("M-9.5,-93 L-8,-106 Q0,-108 8,-106 L9.5,-93 Z", col)
        p.path("M-10,-94 L10,-94 L13,-91 L-3,-91 Z", "black")
    elif kind == "shako":
        p.path("M-9.5,-94 L-10,-112 Q0,-114 10,-112 L9.5,-94 Z", col)
        p.path("M-10,-95 L10,-95 L13,-92 L-3,-92 Z", "black")
    elif kind == "czapka":
        p.path("M-8.5,-95 L-7,-106 L-12,-112 L12,-112 L7,-106 L8.5,-95 Z", col)
    elif kind == "brodie":
        p.ellipse(0, -95, 15.5, 3.2, col)
        p.path("M-9,-95 Q-8,-104 0,-104 Q8,-104 9,-95 Z", col)
    elif kind == "m1":
        p.path("M-12,-87 Q-12,-106 0,-106 Q12,-106 12,-87 Q0,-90 -12,-87 Z", col)
    elif kind == "beret":
        p.path("M-10,-94 Q-10,-102 2,-102 Q14,-101 12,-94 Q0,-96 -10,-94 Z", col)
    elif kind == "hood":
        p.path("M-10.5,-86 Q-12,-104 0,-104 Q12,-104 10.5,-86 L7,-91 Q0,-95 -7,-91 Z", col)
    elif kind == "turban":
        p.path("M-11,-90 Q-12,-104 0,-106 Q12,-104 11,-90 Q0,-95 -11,-90 Z", col)
        p.circle(0, -96, 2.4, "gold", ink=False)
    elif kind == "straw":
        p.ellipse(0, -96, 16, 4.2, col)
        p.path("M-8,-97 Q-7,-107 0,-107 Q7,-107 8,-97 Z", col)
    elif kind == "slouch":
        p.ellipse(0, -96, 15, 4, col)
        p.path("M-8,-97 Q-7,-106 0,-106 Q7,-106 8,-97 Z", col)
    elif kind == "hardhat":
        p.ellipse(0, -94, 13, 3, col)
        p.path("M-9.5,-94 Q-9,-106 0,-106 Q9,-106 9.5,-94 Z", col)
    elif kind == "cap":
        p.path("M-10,-93 Q-10,-101 0,-101 Q10,-101 10,-93 Z", col)
        p.path("M-10,-94 L10,-94 L12,-91 L-6,-91 Z", "black")
    elif kind == "mitre":
        p.path("M-9,-93 L-8,-106 Q0,-118 8,-106 L9,-93 Z", col)
    if plume is not None:
        p.path("M-1,-104 Q2,-118 12,-117 Q6,-110 3,-103 Z", plume)


# ============================================================================ animals and wheels
def horse(p, col="horse", mane="mane", x=0.0, barding=None, gait=0.0, hooves="black"):
    """A horse in profile facing west, about 90 long and 75 tall at the ears, standing on y = 0."""
    q = Shift(p, x, 0)
    far = "darkhorse" if col != "darkhorse" else "black"
    for top, knee, foot in (((-14, -30), (-17, -15), (-18 - gait, -3)), ((24, -32), (28, -17), (25 + gait, -3))):
        q.stroke([top, knee, foot], 5.5, far)                                               # far legs
    q.path("M31,-44 Q44,-40 42,-18 Q38,-28 30,-36 Z", mane)                                   # tail
    q.path("M-22,-24 Q-31,-32 -29,-44 L-40,-58 L-52,-51 Q-57,-53 -54,-59 L-44,-71 Q-38,-74 -33,-70 "
           "Q-24,-57 -12,-50 L18,-49 Q32,-50 33,-38 Q33,-30 28,-24 Q2,-18 -22,-24 Z", col)     # body, neck and head
    q.poly([(-38, -70), (-36, -78), (-32, -70)], col)                                         # ear
    q.path("M-33,-70 Q-23,-57 -12,-50 L-14,-47 Q-26,-55 -37,-66 Z", mane)                     # mane
    q.circle(-44, -64, 1.4, "black", ink=False)
    if barding is not None:
        q.path("M-27,-46 Q-6,-54 30,-46 L33,-18 Q2,-10 -28,-18 Z", barding)
        q.path("M-29,-44 L-40,-58 L-36,-61 L-25,-48 Z", barding)
    for top, knee, foot in (((-20, -28), (-22, -15), (-21 - gait, -4)), ((18, -30), (23, -16), (20 + gait, -4))):
        q.stroke([top, knee, foot], 6.5, col)                                                 # near legs
        q.rect(foot[0] - 4, -6, 8, 6, hooves, rx=1.5)


def rider(p, coat=OWNER, legs="pants", boots="boots", x=0.0, armour=None):
    """A rider sitting on horse(); returns the shoulder points and head centre for arms and hats."""
    p.limb(4 + x, -52, 0 + x, -30, 7, legs)
    p.rect(-5 + x, -32, 8, 9, boots, rx=2)
    p.path("M%g,-74 Q%g,-80 %g,-81 L%g,-81 Q%g,-80 %g,-74 L%g,-48 L%g,-48 Z"
           % (-9 + x, -10 + x, -4 + x, 10 + x, 16 + x, 15 + x, 12 + x, -8 + x), coat)
    if armour is not None:
        p.path("M%g,-72 L%g,-72 L%g,-52 L%g,-52 Z" % (-6 + x, 12 + x, 10 + x, -5 + x), armour)
    p.rect(-0.2 + x, -88, 6.4, 8, "skin", rx=2)
    p.circle(3 + x, -93, 9, "skin")
    return (-8 + x, -74), (14 + x, -74), (3 + x, -93)


def head_at(p, cx, cy, hat_kind=None, hat_col="iron", hair_col="hair", plume=None):
    """Hair and hat for a head at (cx, cy), using the standing figure's shapes moved there."""
    dx, dy = cx, cy + 89
    q = Shift(p, dx, dy)
    if hair_col:
        hair(q, hair_col)
    if hat_kind:
        hat(q, hat_kind, hat_col, plume)


class Shift:
    """A pen whose drawing is moved by (dx, dy) and optionally scaled about the origin."""

    def __init__(self, pen, dx, dy, s=1.0):
        self.pen, self.dx, self.dy, self.s = pen, dx, dy, s

    def _p(self, pts):
        return [(self.dx + self.s * x, self.dy + self.s * y) for x, y in pts]

    def owner(self):
        return self.pen.owner()

    def fill(self, pts, col, **kw):
        self.pen.fill(self._p(pts), col, **kw)

    def path(self, d, col, **kw):
        self.fill(path_points(d), col, **kw)

    def ellipse(self, cx, cy, rx, ry, col, rot=0.0, **kw):
        self.fill(ellipse_points(cx, cy, rx, ry, rot), col, **kw)

    def circle(self, cx, cy, r, col, **kw):
        self.fill(ellipse_points(cx, cy, r, r), col, **kw)

    def rect(self, x, y, w, h, col, rx=0.0, rot=0.0, **kw):
        self.fill(rect_points(x, y, w, h, rx, rot), col, **kw)

    def poly(self, pts, col, **kw):
        self.fill(pts, col, **kw)

    def stroke(self, pts, w, col, **kw):
        self.pen.stroke(self._p(pts), w * self.s, col, **kw)

    def limb(self, x1, y1, x2, y2, w, col, **kw):
        self.stroke([(x1, y1), (x2, y2)], w, col, **kw)

    def curve(self, d, w, col, **kw):
        self.stroke(path_points(d), w, col, **kw)

    def thin(self, x1, y1, x2, y2, w, col):
        self.stroke([(x1, y1), (x2, y2)], w, col, ink=False, shade=False)

    def shadow(self, cx, cy, rx, ry, alpha=0.34):
        (x, y), = self._p([(cx, cy)])
        self.pen.shadow(x, y, rx * self.s, ry * self.s, alpha)

    def glow(self, cx, cy, r, col, alpha=0.5):
        (x, y), = self._p([(cx, cy)])
        self.pen.glow(x, y, r * self.s, col, alpha)


def wheel(p, cx, cy, r, col="darkwood", spokes=6, hub="iron", tyre=None):
    p.circle(cx, cy, r, tyre or col)
    p.circle(cx, cy, r * 0.74, col, ink=False, shade=False)
    for i in range(spokes):
        a = math.pi * i / spokes
        dx, dy = math.cos(a) * r * 0.72, math.sin(a) * r * 0.72
        p.limb(cx - dx, cy - dy, cx + dx, cy + dy, max(1.2, r * 0.14), col, ink=False)
    p.circle(cx, cy, max(1.5, r * 0.22), hub)


def tyre(p, cx, cy, r, col="black", hub="darkgrey"):
    p.circle(cx, cy, r, col)
    p.circle(cx, cy, r * 0.5, hub)


def tracks(p, x0, x1, y, h, col="track", wheel_col="olivedark", n=6):
    """A caterpillar track in side view, bottom on y."""
    p.path("M%g,%g Q%g,%g %g,%g L%g,%g Q%g,%g %g,%g Q%g,%g %g,%g L%g,%g Q%g,%g %g,%g Z"
           % (x0, y - h / 2, x0, y - h, x0 + h / 2, y - h, x1 - h / 2, y - h, x1, y - h, x1, y - h / 2,
              x1, y, x1 - h / 2, y, x0 + h / 2, y, x0, y, x0, y - h / 2), col)
    for i in range(n):
        cx = x0 + h / 2 + (x1 - x0 - h) * i / (n - 1)
        p.circle(cx, y - h / 2, h * 0.34, wheel_col)


# ============================================================================ units
UNITS = {}


def unit(name, civilian=False, scale=1.0, at=None, wide=False):
    """Register a unit. wide: a low unit (ship, vehicle) that may reach toward the middle of the
    hex, under where the military flag sits."""
    def deco(fn):
        UNITS[name] = dict(fn=fn, civilian=civilian, scale=scale, at=at, wide=wide and not civilian)
        return fn
    return deco


def ground(p, w=24, x=3):
    p.shadow(x, 0, w, 4.5)


# --- what they carry -----------------------------------------------------------------------------
def spear_up(p, x, top, bottom=-2, shaft="wood", tip="bronze", w=3.2):
    p.rect(x - w / 2, top, w, bottom - top, shaft, rx=1.4)
    p.path("M%g,%g L%g,%g L%g,%g Z" % (x, top - 12, x + w / 2 + 2.5, top + 1, x - w / 2 - 2.5, top + 1), tip)


def bow(p, recurve=False, col="darkwood"):
    """A bow drawn toward the west, string to the right hand at (-5, -80), with its arrow."""
    if recurve:
        top, bot = (-40, -107), (-40, -45)
        p.curve("M-40,-107 Q-31,-104 -36,-98 Q-52,-76 -36,-54 Q-31,-48 -40,-45", 3.4, col)
    else:
        top, bot = (-36, -106), (-36, -46)
        p.curve("M-36,-106 Q-52,-76 -36,-46", 3.4, col)
    p.thin(top[0], top[1], -5, -80, 0.9, "linen")
    p.thin(-5, -80, bot[0], bot[1], 0.9, "linen")
    p.thin(-5, -80, -47, -77, 1.4, "linen")
    p.path("M-47,-79.5 L-53,-77 L-47,-74.5 Z", "iron", ink=False)


def quiver(p):
    p.rect(8, -86, 8, 27, "leather", rx=3, rot=20)
    for dx in (0, 4):
        p.thin(13 + dx, -86, 15 + dx, -95, 1.6, "linen")


def rifle(p, a, b, col="darkwood", barrel="iron", bayonet=False, w=4.0):
    """A long gun from butt a to muzzle b."""
    (x0, y0), (x1, y1) = a, b
    mx, my = x0 + (x1 - x0) * 0.62, y0 + (y1 - y0) * 0.62
    p.limb(x0, y0, mx, my, w, col)
    p.limb(mx - (x1 - x0) * 0.05, my - (y1 - y0) * 0.05, x1, y1, w * 0.5, barrel)
    if bayonet:
        ln = math.hypot(x1 - x0, y1 - y0)
        dx, dy = (x1 - x0) / ln, (y1 - y0) / ln
        p.limb(x1, y1, x1 + dx * 11, y1 + dy * 11, 1.3, "steel", ink=False)


def round_shield(p, cx, cy, r, face=OWNER, rim="bronze", boss="bronze"):
    p.circle(cx, cy, r, rim)
    p.circle(cx, cy, r * 0.82, face)
    if boss:
        p.circle(cx, cy, r * 0.22, boss)


def banner(p, x, ground_y, top, w=18, h=12, pole="darkwood"):
    p.rect(x - 1.3, top, 2.6, ground_y - top, pole, rx=1)
    p.circle(x, top - 1.5, 2.4, "gold")
    p.path("M%g,%g L%g,%g L%g,%g L%g,%g Z" % (x + 1.3, top + 1, x + w, top + 3, x + w - 1, top + h + 3, x + 1.3, top + h), OWNER)


def sword_up(p, hx, hy, tx, ty, w=3.4, blade="steel", guard="bronze"):
    """A sword held at (hx, hy) with its point at (tx, ty)."""
    ln = math.hypot(tx - hx, ty - hy)
    dx, dy = (tx - hx) / ln, (ty - hy) / ln
    nx, ny = -dy, dx
    bx, by = hx + dx * 3, hy + dy * 3
    p.poly([(bx + nx * w / 2, by + ny * w / 2), (tx - dx * 5 + nx * w * 0.4, ty - dy * 5 + ny * w * 0.4), (tx, ty),
            (tx - dx * 5 - nx * w * 0.4, ty - dy * 5 - ny * w * 0.4), (bx - nx * w / 2, by - ny * w / 2)], blade)
    p.limb(hx + dx * 2 + nx * 5, hy + dy * 2 + ny * 5, hx + dx * 2 - nx * 5, hy + dy * 2 - ny * 5, 2.4, guard)


# --- ancient and classical foot -----------------------------------------------------------------------
@unit("Scout")
def u_scout(p):
    ground(p)
    p.path("M-15,-74 L15,-74 L22,-24 Q0,-18 -22,-24 Z", OWNER)                       # cloak behind
    figure(p, tunic="leather", pants="cloth", belt="darkwood")
    hat(p, "hood", OWNER)
    p.rect(19, -104, 3.2, 104, "wood", rx=1.5, rot=3)                                  # walking staff
    arm(p, 13, -72, 20, -56, "leather")
    arm(p, -13, -72, -7, -92, "leather")                                                # shading the eyes


@unit("Warrior")
def u_warrior(p):
    ground(p)
    p.limb(-13, -72, -24, -94, 7, "skin")                                               # arm raised with the club
    p.path("M-27,-92 L-21,-92 L-26,-126 Q-31,-131 -35,-125 Z", "darkwood")
    figure(p, tunic=OWNER, bare=True)
    hair(p)
    beard(p)
    p.circle(-24, -94, 4, "skin")
    arm(p, 13, -72, 19, -50, "skin")
    round_shield(p, 22, -50, 15, face="wood", rim="darkwood", boss="leather")


@unit("Archer")
def u_archer(p):
    ground(p)
    quiver(p)
    figure(p, tunic=OWNER)
    hat(p, "hood", "cloth")
    p.limb(-13, -72, -34, -76, 6.5, OWNER)
    bow(p)
    arm(p, 13, -72, -5, -80, OWNER)
    p.circle(-34, -76, 3.8, "skin")


@unit("Composite Bowman")
def u_composite_bowman(p):
    ground(p)
    quiver(p)
    figure(p, tunic=OWNER, pants="cloth")
    p.path("M-13,-73 L13,-73 L12,-50 L-12,-50 Z", "bronze")                              # scale armour
    for y in (-67, -61, -55):
        p.thin(-12, y, 12, y, 1.0, "leather")
    hat(p, "cone", "bronze")
    p.limb(-13, -72, -36, -76, 6.5, OWNER)
    bow(p, recurve=True)
    arm(p, 13, -72, -5, -80, OWNER)
    p.circle(-36, -76, 3.8, "skin")


@unit("Spearman")
def u_spearman(p):
    ground(p)
    spear_up(p, 20, -124)
    figure(p, tunic="linen", pants="skin", boots="leather")
    p.path("M-13,-73 L13,-73 L12,-50 L-12,-50 Z", "bronze")                              # cuirass
    for x in (-6, 6):
        p.rect(x - 4.8, -30, 9.6, 16, "bronze", rx=3)                                    # greaves
    hat(p, "crest", "bronze")
    p.path("M-12,-100 Q-4,-120 13,-106 Q2,-110 -12,-100 Z", OWNER)                       # crest
    arm(p, 13, -72, 20, -58, "skin")
    arm(p, -13, -72, -15, -56, "skin")
    round_shield(p, -8, -56, 20)


@unit("Swordsman")
def u_swordsman(p):
    ground(p)
    figure(p, tunic=OWNER, pants="cloth")
    p.path("M-13,-73 L13,-73 L12,-50 L-12,-50 Z", "iron")                                # lorica
    for y in (-66, -60, -55):
        p.thin(-12, y, 12, y, 1.0, "darkgrey")
    hat(p, "round", "iron", plume=OWNER)
    arm(p, 13, -72, 25, -78, OWNER)
    sword_up(p, 25, -80, 30, -118)
    arm(p, -13, -72, -14, -56, OWNER)
    p.rect(-31, -79, 23, 46, "bronze", rx=4)                                              # scutum
    p.rect(-29, -77, 19, 42, OWNER, rx=3)
    p.circle(-19.5, -56, 4, "bronze")


# --- medieval foot ---------------------------------------------------------------------------------------
@unit("Pikeman")
def u_pikeman(p):
    ground(p)
    spear_up(p, 19, -160, tip="steel", w=3.0)
    figure(p, tunic=OWNER, pants=OWNER, boots="black")
    p.path("M-12,-73 L12,-73 L11,-50 L-11,-50 Z", "steel")                               # breastplate
    hat(p, "morion", "steel")
    arm(p, 13, -72, 19, -58, OWNER)
    arm(p, -13, -72, 18, -86, OWNER)


@unit("Longswordsman")
def u_longswordsman(p):
    ground(p)
    figure(p, tunic="steel", pants="steel", boots="steel", skirt=False, belt=None)
    p.path("M-12,-74 L12,-74 L15,-30 L-15,-30 Z", OWNER)                                 # surcoat
    p.rect(-13.5, -52, 27, 4.5, "leather", rx=1.5)
    hat(p, "great", "steel")
    p.limb(-13, -72, 10, -82, 6.5, "steel")
    sword_up(p, 12, -84, 42, -134, w=4.4)
    arm(p, 13, -72, 13, -82, "steel", hand="steel")


@unit("Crossbowman")
def u_crossbowman(p):
    ground(p)
    figure(p, tunic=OWNER, pants="cloth")
    hat(p, "kettle", "iron")
    p.limb(-13, -72, -24, -64, 6.5, OWNER)
    p.rect(-38, -67.5, 46, 5.5, "wood", rx=2)                                            # crossbow, aimed west
    p.curve("M-32,-86 Q-25,-66 -32,-48", 3.2, "iron")
    p.thin(-32, -86, -15, -66, 0.9, "linen")
    p.thin(-15, -66, -32, -48, 0.9, "linen")
    p.thin(-15, -67, -42, -67, 1.4, "linen")
    p.circle(-24, -64, 3.8, "skin")
    arm(p, 13, -72, 5, -62, OWNER)


# --- gunpowder foot -------------------------------------------------------------------------------------
@unit("Musketman")
def u_musketman(p):
    ground(p)
    figure(p, tunic=OWNER, pants="white", boots="black", belt=None, long_coat=True)
    p.limb(-12, -74, 10, -50, 2.4, "white")                                               # cross belts
    p.limb(12, -74, -10, -50, 2.4, "white")
    hair(p, "grey_hair")
    rifle(p, (17, -50), (25, -122))                                                       # musket on the shoulder
    arm(p, 13, -72, 17, -54, OWNER)
    arm(p, -13, -72, -16, -50, OWNER)
    hat(p, "tricorne", "black")


@unit("Rifleman")
def u_rifleman(p):
    ground(p)
    figure(p, tunic=OWNER, pants="navy", boots="black", long_coat=True)
    hat(p, "kepi", OWNER)
    rifle(p, (-16, -44), (20, -118), bayonet=True)                                        # at port arms
    arm(p, 13, -72, 8, -88, OWNER)
    arm(p, -13, -72, -9, -56, OWNER)


@unit("Great War Infantry")
def u_great_war_infantry(p):
    ground(p)
    p.rect(-12, -80, 24, 27, "khaki", rx=3)                                               # pack
    figure(p, tunic=OWNER, pants=OWNER, long_coat=True)
    for x in (-6, 6):
        p.rect(x - 5, -24, 10, 13, "khaki", rx=2)                                         # puttees
    hat(p, "brodie", "olivedark")
    rifle(p, (-16, -44), (20, -118), bayonet=True)
    arm(p, 13, -72, 8, -88, OWNER)
    arm(p, -13, -72, -9, -56, OWNER)


def modern_soldier(p, uniform, helmet, kit="olivedark", gun="black"):
    ground(p)
    p.rect(-12, -80, 24, 30, kit, rx=3)                                                   # pack
    figure(p, tunic=uniform, pants=uniform, skirt=False, belt="khaki")
    p.rect(-8, -80, 16, 5, OWNER, rx=2)                                                   # scarf
    hat(p, "m1", helmet)
    p.rect(-11.5, -94, 23, 3.6, OWNER, rx=1.5)                                            # helmet band
    p.limb(-16, -50, 16, -98, 4.6, gun)                                                   # rifle held ready
    p.rect(-4, -74, 5, 9, gun, rot=-34)
    arm(p, -13, -72, -9, -58, uniform)
    p.limb(-13.2, -69, -12, -65, 7.6, OWNER)                                              # armband
    arm(p, 13, -72, 8, -84, uniform)


@unit("Infantry")
def u_infantry(p):
    modern_soldier(p, "olive", "olivedark", gun="darkwood")


@unit("Marine")
def u_marine(p):
    modern_soldier(p, "camo", "sand")
    for x, y in ((-6, -66), (5, -58), (-2, -36), (7, -20)):                               # camouflage
        p.ellipse(x, y, 4, 2.6, "olivedark", ink=False)


@unit("Paratrooper")
def u_paratrooper(p):
    p.shadow(10, 0, 18, 3.5, 0.22)
    q = Shift(p, 0, -26)
    for x, sx in ((-46, -13), (-24, -13), (24, 13), (46, 13)):                            # rigging lines
        p.thin(x, -150, sx, -98 - 26, 0.8, "rope")
    figure(q, tunic="olive", pants="olive", skirt=False, belt="khaki")
    hat(q, "m1", "olivedark")
    arm(q, -13, -72, -16, -98, "olive")
    arm(q, 13, -72, 16, -98, "olive")
    p.path("M-50,-148 Q-44,-190 0,-194 Q44,-190 50,-148 Q37,-142 25,-148 Q12,-142 0,-148 Q-12,-142 -25,-148 "
           "Q-37,-142 -50,-148 Z", OWNER)                                                  # the canopy
    for x in (-25, 0, 25):
        p.thin(x, -148, x * 0.6, -190, 0.8, "darkgrey")


# --- mounted ---------------------------------------------------------------------------------------------------
@unit("Horseman")
def u_horseman(p):
    ground(p, 40, 0)
    horse(p)
    ls, rs, hd = rider(p, coat=OWNER, legs="skin", boots="leather")
    head_at(p, *hd, hat_kind="round", hat_col="bronze")
    p.limb(34, -76, -38, -104, 2.6, "wood")                                               # javelin, raised
    p.path("M-38,-106.5 L-46,-107 L-38,-101.5 Z", "bronze")
    arm(p, *rs, 17, -86, "skin")
    arm(p, *ls, -20, -66, "skin")


@unit("Chariot Archer", wide=True)
def u_chariot_archer(p):
    ground(p, 56, -4)
    horse(p, x=-34, col="horse")
    p.limb(-20, -40, 16, -32, 2.6, "darkwood")                                            # draught pole
    q = Shift(p, 32, -20, 0.86)
    figure(q, tunic=OWNER, pants="cloth")
    hat(q, "cone", "bronze")
    q.limb(-13, -72, -34, -76, 6.5, OWNER)
    bow(q)
    arm(q, 13, -72, -5, -80, OWNER)
    q.circle(-34, -76, 3.8, "skin")
    p.path("M14,-46 L48,-46 L46,-18 L18,-18 Z", OWNER)                                    # the car
    p.rect(12, -48, 38, 4, "bronze", rx=2)
    wheel(p, 34, -16, 16)


@unit("Knight")
def u_knight(p):
    ground(p, 40, 0)
    horse(p, col="greyhorse", mane="iron", barding=OWNER)
    ls, rs, hd = rider(p, coat="steel", legs="steel", boots="steel")
    head_at(p, *hd, hair_col=None, hat_kind="great", hat_col="steel", plume=OWNER)
    p.rect(20, -132, 3.4, 104, "wood", rx=1.5)                                            # lance, upright
    p.path("M21.7,-144 L24.6,-131 L18.8,-131 Z", "steel")
    p.poly([(23.4, -129), (42, -124), (23.4, -118)], OWNER)
    arm(p, *rs, 21.7, -66, "steel", hand="steel")
    p.path("M-22,-80 L-3,-80 L-3,-64 Q-5,-50 -12.5,-44 Q-20,-50 -22,-64 Z", OWNER)         # kite shield


@unit("Lancer")
def u_lancer(p):
    ground(p, 40, 0)
    horse(p, col="horse", gait=2)
    ls, rs, hd = rider(p, coat=OWNER, legs="navy", boots="black")
    head_at(p, *hd, hat_kind="czapka", hat_col=OWNER)
    p.limb(34, -60, -60, -104, 2.8, "wood")                                               # lance, couched forward
    p.path("M-60,-107 L-70,-109 L-61,-101 Z", "steel")
    p.poly([(-52, -101), (-36, -98), (-41, -91), (-53, -96)], OWNER)
    arm(p, *rs, 16, -66, OWNER)
    arm(p, *ls, -20, -66, OWNER)


@unit("Cavalry")
def u_cavalry(p):
    ground(p, 40, 0)
    horse(p, col="darkhorse", mane="black", gait=3)
    ls, rs, hd = rider(p, coat=OWNER, legs="navy", boots="black")
    head_at(p, *hd, hat_kind="slouch", hat_col="leather")
    arm(p, *rs, 22, -102, OWNER)
    p.curve("M22,-104 Q16,-122 2,-134", 2.6, "steel")                                     # sabre raised
    p.limb(19, -101, 25, -105, 2.2, "gold")
    arm(p, *ls, -20, -66, OWNER)


# --- siege and guns, side on, facing west -----------------------------------------------------------------
@unit("Catapult", wide=True)
def u_catapult(p):
    ground(p, 46, 2)
    p.rect(-44, -30, 88, 10, "wood", rx=2)
    p.poly([(-8, -30), (8, -30), (3, -70), (-3, -70)], "darkwood")
    p.rect(-11, -74, 22, 6, "darkwood", rx=2)
    p.limb(32, -32, -28, -84, 5, "wood")                                                    # throwing arm
    p.ellipse(-30, -86, 8, 5, "leather")
    p.circle(-30, -92, 6, "stone")
    banner(p, 40, -30, -78)
    wheel(p, -26, -13, 13)
    wheel(p, 26, -13, 13)


@unit("Trebuchet", wide=True)
def u_trebuchet(p):
    ground(p, 46, 2)
    p.rect(-44, -12, 88, 8, "darkwood", rx=2)
    p.poly([(-34, -12), (-26, -12), (2, -92), (-2, -92)], "wood")                          # frame
    p.poly([(34, -12), (26, -12), (-2, -92), (2, -92)], "wood")
    p.limb(-24, -74, 44, -148, 4.4, "wood")                                                 # the long arm, raised
    p.thin(44, -148, 42, -116, 1.0, "rope")
    p.circle(42, -112, 5, "stone")
    p.limb(-24, -74, -24, -64, 2.2, "iron")
    p.rect(-38, -64, 28, 26, OWNER, rx=2)                                                   # counterweight box
    p.rect(-38, -64, 28, 5, "iron", rx=1)
    p.circle(0, -90, 4, "iron")


@unit("Cannon", wide=True)
def u_cannon(p):
    ground(p, 46, 4)
    p.path("M-8,-30 L46,-6 L47,-12 L-2,-40 Z", "wood")                                      # trail
    p.path("M-4,-38 L16,-29 L14,-25 L-6,-34 Z", OWNER)
    p.path("M14,-46 Q20,-40 14,-34 L-46,-37 L-48,-35 L-48,-46 L-46,-44 Z", "bronze")          # barrel
    p.rect(-50, -47.5, 5, 14, "bronze", rx=1.5)
    wheel(p, 0, -19, 19)
    for x, y in ((30, -4), (36, -4), (33, -9)):
        p.circle(x, y, 3.2, "black")


@unit("Artillery", wide=True)
def u_artillery(p):
    ground(p, 48, 4)
    p.limb(4, -22, 50, -4, 6, "olivedark")                                                  # trail
    p.rect(-58, -52, 64, 4.4, "olive", rx=2)                                                 # barrel
    p.rect(-62, -53.5, 6, 7.4, "olivedark", rx=1)
    p.rect(-26, -50, 34, 8, "olivedark", rx=2)                                              # recoil gear
    p.path("M-12,-64 L4,-66 L6,-24 L-10,-22 Z", "olive")                                    # gun shield
    p.path("M-9,-58 L2,-59 L3,-48 L-8,-47 Z", OWNER)
    wheel(p, 0, -20, 20, col="olivedark", spokes=7, tyre="black")


@unit("Gatling Gun", wide=True)
def u_gatling_gun(p):
    ground(p, 46, 6)
    q = Shift(p, 34, 0, 0.9)
    figure(q, tunic=OWNER, pants="navy", boots="black")
    hat(q, "kepi", OWNER)
    p.limb(4, -22, 26, -4, 5, "wood")
    for dy in (-2.4, 0, 2.4):
        p.limb(10, -44 + dy, -40, -48 + dy, 2.2, "bronze")                                  # the barrels
    p.rect(-6, -52, 22, 14, "bronze", rx=3)
    p.ellipse(-34, -48, 3, 6, "bronze")
    p.rect(-2, -60, 8, 8, "bronze", rx=2)                                                     # magazine
    arm(q, -13, -72, -24, -56, OWNER)
    arm(q, 13, -72, -16, -60, OWNER)
    wheel(p, 0, -18, 18)


@unit("Machine Gun", wide=True)
def u_machine_gun(p):
    ground(p, 44, 4)
    q = Shift(p, 14, 18, 1.0)                                                                # the gunner, kneeling behind
    q.path("M-15,-70 Q-16,-76 -9,-77 L9,-77 Q16,-76 15,-70 L13,-40 L-13,-40 Z", OWNER)
    q.rect(-3.2, -84, 6.4, 8, "skin", rx=2)
    q.circle(0, -89, 9.5, "skin")
    hat(q, "brodie", "olivedark")
    p.limb(-6, -40, -16, -4, 2.4, "iron")                                                   # tripod
    p.limb(-6, -40, 4, -4, 2.4, "iron")
    p.rect(-52, -48, 50, 8, "olivedark", rx=3)                                               # water-cooled gun
    p.rect(-60, -46, 9, 3.6, "black", rx=1)
    p.rect(-6, -50, 12, 10, "iron", rx=2)
    arm(q, -13, -72, -18, -60, OWNER)
    arm(q, 13, -72, -6, -64, OWNER)
    for i, x in enumerate(range(-48, 46, 13)):                                               # sandbags in front
        p.rect(x, -14, 14, 10, "sand", rx=4)
    for x in range(-42, 40, 13):
        p.rect(x, -24, 14, 10, "khaki", rx=4)


@unit("Anti-Aircraft Gun", wide=True)
def u_anti_aircraft_gun(p):
    ground(p, 46, 2)
    p.rect(-44, -22, 88, 8, "olivedark", rx=2)                                                # platform
    for x in (-30, 30):
        tyre(p, x, -9, 9)
    p.rect(-14, -32, 28, 10, "olive", rx=3)                                                    # turntable
    for d in (0, 6):
        p.limb(-4 + d, -46 - d * 0.6, -40 + d, -104 - d * 0.6, 3.6, "olivedark")              # twin barrels
        p.rect(-44 + d, -110 - d * 0.6, 7, 7, "olivedark", rx=1.5, rot=-32)
    p.rect(-18, -58, 30, 24, OWNER, rx=3, rot=-32)                                             # cradle
    p.circle(-3, -46, 4, "iron")
    p.rect(12, -46, 12, 14, "olive", rx=2)                                                     # gunner's seat


@unit("Anti-Tank Gun", wide=True)
def u_anti_tank_gun(p):
    ground(p, 44, 6)
    p.limb(6, -16, 44, -4, 4.4, "olivedark")                                                 # trail
    p.rect(-54, -36, 62, 4.4, "olive", rx=2)                                                  # long barrel
    p.rect(-58, -38.5, 7, 9.4, "olivedark", rx=1.5)                                           # muzzle brake
    p.path("M-14,-58 L8,-60 L10,-12 L-16,-10 Z", OWNER)                                      # sloped shield
    p.path("M-14,-58 L8,-60 L8,-55 L-14,-53 Z", "olivedark")
    tyre(p, 0, -13, 13)


@unit("Rocket Artillery", wide=True)
def u_rocket_artillery(p):
    ground(p, 50, 2)
    p.rect(-46, -24, 92, 8, "black", rx=2)                                                   # chassis
    p.path("M-48,-24 L-48,-46 Q-46,-54 -38,-54 L-24,-54 L-24,-24 Z", "olive")                 # cab
    p.path("M-44,-46 L-40,-51 L-28,-51 L-28,-44 Z", "glass")
    p.rect(-44, -38, 14, 9, OWNER, rx=1.5)                                                   # door panel
    p.rect(-24, -32, 70, 8, "olive", rx=1)                                                    # bed
    p.path("M44,-32 L-18,-62 L-14,-72 L48,-42 Z", "olivedark")                                # launcher, raised
    for i in range(4):
        p.circle(-14 + 2.4 * i, -66 + 2.2 * i, 1.7, "black", ink=False)
    for x in (-36, 8, 30):
        tyre(p, x, -10, 10)


@unit("Mobile SAM", wide=True)
def u_mobile_sam(p):
    ground(p, 50, 2)
    tracks(p, -44, 44, 0, 16)
    p.path("M-48,-16 L-40,-30 L44,-30 L46,-16 Z", "olive")
    p.rect(-30, -28, 26, 8, OWNER, rx=1.5)
    p.rect(8, -40, 22, 10, "olivedark", rx=2)
    for dy in (0, 7):                                                                         # missiles
        p.limb(24, -44 - dy, -24, -84 - dy, 5.6, "white")
        p.path("M-24,%g L-32,%g L-26,%g Z" % (-87 - dy, -95 - dy, -81 - dy), "red")
    p.limb(36, -40, 36, -54, 2, "iron")
    p.ellipse(36, -58, 8, 4, "grey", rot=-20)


# --- armour --------------------------------------------------------------------------------------------------------
@unit("Landship", wide=True)
def u_landship(p):
    ground(p, 50, 4)
    p.path("M-52,-30 L-36,0 L36,0 L50,-22 L38,-50 L-30,-50 Z", "track")                       # the rhomboid track
    for t in range(14):                                                                       # tread links
        x = -36 + t * 5.4
        p.thin(x, -50, x - 1, -46, 1.2, "darkgrey")
        p.thin(x, -4, x - 1, 0, 1.2, "darkgrey")
    p.path("M-42,-30 L-32,-8 L32,-8 L42,-22 L33,-42 L-26,-42 Z", "khaki")
    for x in range(-22, 30, 8):
        p.circle(x, -38, 1.0, "darkwood", ink=False)
    p.path("M-16,-34 L12,-34 L16,-26 L12,-14 L-16,-14 L-20,-24 Z", OWNER)                     # sponson
    p.rect(-34, -26, 16, 3.6, "darkgrey", rx=1.5)
    for x in (52, 62):                                                                        # steering tail wheels
        tyre(p, x, -8, 8, col="darkwood", hub="leather")


@unit("Tank", wide=True)
def u_tank(p):
    ground(p, 50, 2)
    tracks(p, -48, 48, 0, 22, wheel_col="olivedark")
    p.path("M-50,-22 L-44,-34 L44,-34 L50,-22 Z", "olive")
    p.path("M-24,-34 L-20,-50 L18,-50 L24,-34 Z", "olive")
    p.rect(-21.5, -45, 42, 5.5, OWNER)                                                        # turret band
    p.rect(-64, -46, 44, 4.6, "olivedark", rx=2)                                               # gun
    p.rect(4, -66, 1.4, 16, "black", ink=False)
    p.poly([(5.4, -66), (16, -63), (5.4, -60)], OWNER)


@unit("Modern Armor", wide=True)
def u_modern_armor(p):
    ground(p, 54, 2)
    tracks(p, -50, 50, 0, 20, wheel_col="iron", n=7)
    p.path("M-56,-18 L-46,-30 L52,-30 L54,-18 Z", "sand")
    p.rect(-46, -24, 96, 8, "sand", rx=1)                                                     # side skirt
    p.path("M-28,-30 L-36,-38 L-30,-46 L22,-46 L28,-30 Z", "sand")                             # angular turret
    p.rect(-24, -42, 44, 5, OWNER)
    p.rect(-84, -42, 50, 4, "khaki", rx=2)                                                     # long gun
    p.rect(-62, -43.5, 10, 7, "khaki", rx=2)
    p.rect(6, -52, 10, 6, "khaki", rx=1.5)


@unit("Mechanized Infantry", wide=True)
def u_mechanized_infantry(p):
    ground(p, 50, 2)
    tracks(p, -46, 46, 0, 18)
    p.path("M-52,-18 L-40,-44 L46,-44 L48,-18 Z", "olive")
    p.rect(8, -36, 30, 12, OWNER, rx=1.5)                                                      # side panel
    p.path("M-18,-44 L-14,-54 L8,-54 L12,-44 Z", "olivedark")                                  # small turret
    p.rect(-48, -52, 34, 2.6, "black", rx=1)                                                   # autocannon
    p.circle(28, -50, 6, "skin")                                                                # commander in the hatch
    head_at(p, 28, -50, hair_col=None, hat_kind="m1", hat_col="olivedark")


@unit("Helicopter Gunship", wide=True)
def u_helicopter_gunship(p):
    p.shadow(4, 0, 40, 4, 0.22)
    p.limb(-20, -34, 20, -34, 2, "black")                                                      # skids
    p.limb(-12, -34, -10, -44, 1.8, "black")
    p.limb(12, -34, 10, -44, 1.8, "black")
    p.path("M14,-56 L70,-58 L72,-52 L14,-48 Z", "camo")                                       # tail boom
    p.path("M64,-58 L72,-80 L78,-80 L74,-54 Z", "camo")
    p.circle(72, -64, 7, "grey", alpha=0.45, ink=False)
    p.rect(42, -59, 12, 9, OWNER, rx=1.5)
    p.path("M-36,-50 Q-38,-66 -20,-70 L18,-70 Q26,-62 22,-46 L-28,-42 Q-36,-44 -36,-50 Z", "camo")   # fuselage
    p.path("M-33,-52 Q-33,-64 -20,-66 L-14,-66 L-16,-50 Z", "glass")
    p.rect(-10, -48, 24, 5, "olivedark", rx=2)                                                  # stub wing
    p.rect(-6, -46, 14, 6, "darkgrey", rx=2.5)                                                  # rockets
    p.rect(-3, -76, 6, 6, "darkgrey", rx=1)                                                     # rotor mast
    p.ellipse(0, -78, 50, 3.2, "grey", alpha=0.3, ink=False)                                   # spinning blades
    p.limb(-48, -78, 48, -78, 1.8, "black", ink=False)


@unit("Giant Death Robot", wide=True, scale=0.82)
def u_giant_death_robot(p):
    ground(p, 36, 4)
    for x in (-14, 14):                                                                         # legs
        p.rect(x - 11, -8, 22, 8, "darkgrey", rx=2)
        p.rect(x - 6, -48, 12, 40, "iron", rx=3)
        p.circle(x, -48, 7, "grey")
    p.rect(-18, -64, 36, 14, "darkgrey", rx=3)
    p.path("M-30,-118 L30,-118 L34,-96 L24,-62 L-24,-62 L-34,-96 Z", "iron")                     # torso
    p.path("M-20,-112 L20,-112 L22,-96 L16,-72 L-16,-72 L-22,-96 Z", OWNER)
    p.circle(0, -92, 7, "darkgrey")
    p.rect(-10, -132, 20, 14, "darkgrey", rx=3)                                                 # head
    p.glow(0, -125, 7, "red", 0.6)
    p.circle(0, -125, 2.6, "red", ink=False)
    for sx in (-1, 1):                                                                          # arms ending in guns
        p.circle(sx * 36, -112, 9, "grey")
        p.limb(sx * 38, -110, sx * 42, -78, 9, "iron")
        p.rect(sx * 42 - 6, -80, 12, 20, "darkgrey", rx=3)
        p.rect(sx * 42 - 2.5, -60, 5, 14, "black", rx=1.5)


# --- ships, side on, bow to the west -------------------------------------------------------------------------
def water(p, x0, x1, w=6):
    p.shadow((x0 + x1) / 2 + 4, 1, (x1 - x0) / 2 + 6, w * 0.8, 0.3)
    p.ellipse((x0 + x1) / 2, 0, (x1 - x0) / 2 + 4, w, "foam", ink=False, shade=False, alpha=0.85)


def hull(p, x0, x1, h, col="hull", stripe=False, rake=10, stern_h=None, keel=None):
    """A hull from bow x0 to stern x1, deck at -h, waterline at 0."""
    sh = h if stern_h is None else stern_h
    p.path("M%g,%g L%g,%g Q%g,0 %g,0 L%g,0 Q%g,%g %g,%g Z"
           % (x0, -h, x0 + rake, -2, x0 + rake + 4, x0 + rake + 12, x1 - 8, x1, -2, x1 + 2, -sh), col)
    if stripe:
        p.path("M%g,%g L%g,%g L%g,%g L%g,%g Z" % (x0 + 3, -h + 3, x1, -sh + 3, x1, -sh + 7, x0 + 5, -h + 7), OWNER)


def mast(p, x, top, deck, col="darkwood", w=2.4):
    p.rect(x - w / 2, top, w, deck - top, col, rx=1)


def square_sail(p, x, top, bottom, w, col=OWNER):
    p.rect(x - w / 2 - 3, top - 2, w + 6, 2.6, "darkwood", rx=1)
    p.path("M%g,%g L%g,%g Q%g,%g %g,%g L%g,%g Q%g,%g %g,%g Z"
           % (x - w / 2, top, x + w / 2, top, x + w / 2 + 3, (top + bottom) / 2, x + w / 2, bottom,
              x - w / 2, bottom, x - w / 2 + 3, (top + bottom) / 2, x - w / 2, top), col)


def lateen(p, x, top, bottom, w, col=OWNER):
    p.limb(x - w * 0.7, bottom + 4, x + w * 0.3, top - 3, 2, "darkwood")
    p.path("M%g,%g L%g,%g Q%g,%g %g,%g Z" % (x - w * 0.62, bottom + 2, x + w * 0.26, top, x + w * 0.5, (top + bottom) / 2,
                                            x + w * 0.42, bottom), col)


def oars(p, x0, x1, n, deck):
    for i in range(n):
        x = x0 + (x1 - x0) * i / (n - 1)
        p.limb(x, deck + 2, x - 7, 6, 1.6, "wood")


def ensign(p, x, deck, h=26, w=14):
    p.rect(x - 1, deck - h, 2, h, "darkwood", rx=0.8)
    p.rect(x + 1, deck - h, w, w * 0.65, OWNER)


def pennant(p, x, y, w=18):
    p.poly([(x, y), (x + w, y + 2.5), (x, y + 5)], OWNER)


def funnel(p, x, deck, h, w, col="black", smoke=3):
    p.rect(x - w / 2, deck - h, w, h, col, rx=1.5)
    for i in range(smoke):
        p.circle(x + 6 + 7 * i, deck - h - 6 - 4 * i, 5 + 2 * i, "grey", alpha=0.6 - 0.15 * i, ink=False)


def turret(p, x, deck, w, col="navygrey", guns=2, length=22, facing=-1):
    p.path("M%g,%g L%g,%g L%g,%g L%g,%g Z" % (x - w / 2, deck, x + w / 2, deck, x + w / 2 - 3, deck - 9, x - w / 2 + 2, deck - 9), col)
    for i in range(guns):
        y = deck - 6 + 2.6 * i - 1.3 * (guns - 1)
        x0 = x - w / 2 + 1 if facing < 0 else x + w / 2 - 1
        p.rect(min(x0, x0 + facing * length), y - 1.2, length, 2.4, "darkgrey", rx=1)


@unit("Trireme", at=AT_SEA, wide=True)
def u_trireme(p):
    water(p, -56, 56)
    oars(p, -30, 34, 8, -12)
    hull(p, -54, 54, 14, col="hull", rake=12, stern_h=24)
    p.path("M-54,-14 L-66,-4 L-54,-6 Z", "bronze")                                          # ram
    p.circle(-44, -9, 2.4, "white", ink=False)
    p.circle(-44.6, -9, 1.2, "black", ink=False)
    mast(p, 0, -86, -14)
    square_sail(p, 0, -82, -32, 40)


@unit("Galley", at=AT_SEA, wide=True)
def u_galley(p):
    water(p, -48, 48)
    oars(p, -24, 28, 6, -11)
    hull(p, -44, 46, 12, col="darkwood", rake=10, stern_h=20)
    p.curve("M-44,-12 Q-52,-26 -44,-34", 3.6, "darkwood")                                   # high prow
    for x in (-24, -8, 8, 24):
        p.circle(x, -14, 5.5, "wood")
    mast(p, 2, -78, -12)
    square_sail(p, 2, -74, -30, 36)


@unit("Galleass", at=AT_SEA, wide=True)
def u_galleass(p):
    water(p, -62, 62)
    oars(p, -36, 40, 10, -14)
    hull(p, -60, 60, 16, col="hull", stripe=True, rake=12, stern_h=24)
    p.rect(-52, -26, 22, 10, "darkwood", rx=1.5)                                            # forecastle with guns
    p.rect(-64, -22, 14, 3.4, "bronze", rx=1)
    for x, top, w in ((-24, -82, 46), (8, -94, 52), (38, -70, 34)):
        mast(p, x, top, -16)
        lateen(p, x, top + 6, -26, w)


@unit("Caravel", at=AT_SEA, wide=True)
def u_caravel(p):
    water(p, -52, 52)
    hull(p, -50, 50, 16, col="wood", rake=12, stern_h=30)
    p.rect(28, -38, 24, 12, "darkwood", rx=1.5)                                              # sterncastle
    p.limb(-50, -18, -64, -28, 2, "darkwood")
    for x, top, w in ((-22, -84, 40), (6, -98, 50), (32, -74, 32)):
        mast(p, x, top, -16)
        lateen(p, x, top + 6, -26, w)
    pennant(p, 7, -100)


@unit("Privateer", at=AT_SEA, wide=True)
def u_privateer(p):
    water(p, -54, 54)
    hull(p, -52, 52, 18, col="black", rake=12, stern_h=24)
    for x in range(-30, 40, 14):
        p.rect(x, -12, 5, 4, "red", ink=False)
    p.limb(-52, -20, -72, -32, 2, "darkwood")
    p.poly([(-70, -32), (-24, -96), (-22, -22)], "darksail")                                 # jib
    for x, top in ((-14, -104), (22, -108)):
        mast(p, x, top, -18)
        square_sail(p, x, top + 8, top + 48, 30, "darksail")
        square_sail(p, x, top + 52, -26, 36, "darksail")
    ensign(p, 50, -24, 30, 16)


@unit("Frigate", at=AT_SEA, wide=True)
def u_frigate(p):
    water(p, -58, 58)
    hull(p, -56, 56, 24, col="hull", stripe=True, rake=12, stern_h=30)
    for x in range(-36, 46, 14):
        p.rect(x, -14, 5, 4, "black", ink=False)
    p.limb(-56, -24, -76, -36, 2.2, "darkwood")                                             # bowsprit
    p.poly([(-56, -26), (-32, -88), (-34, -30)], "sail")                                     # jib
    for x, h in ((-30, 86), (2, 100), (32, 78)):
        mast(p, x, -24 - h, -24)
        square_sail(p, x, -24 - h * 0.88 + 4, -24 - h * 0.52, 23, "sail")
        square_sail(p, x, -24 - h * 0.48, -34, 30, "sail")
    pennant(p, 3.2, -126, 20)
    ensign(p, 55, -30, 24, 16)


@unit("Ironclad", at=AT_SEA, wide=True)
def u_ironclad(p):
    water(p, -56, 56)
    hull(p, -54, 54, 10, col="black", rake=8)
    p.rect(-38, -26, 26, 16, "darkgrey", rx=6)                                               # turret
    p.rect(-38, -22, 26, 5, OWNER)
    p.rect(-52, -20, 16, 3.4, "black", rx=1)
    funnel(p, 10, -10, 26, 9)
    mast(p, 34, -48, -10, col="darkgrey", w=2)
    ensign(p, 52, -10, 24, 14)


@unit("Destroyer", at=AT_SEA, wide=True)
def u_destroyer(p):
    water(p, -62, 62)
    hull(p, -60, 60, 14, col="navygrey", stripe=True, rake=16)
    turret(p, -38, -14, 12, guns=1, length=16)
    p.rect(-24, -32, 20, 18, "navygrey", rx=2)                                               # bridge
    p.rect(-22, -30, 16, 4, "glass", ink=False)
    mast(p, -14, -58, -32, col="darkgrey", w=2)
    funnel(p, 6, -14, 18, 7, col="darkgrey", smoke=2)
    funnel(p, 20, -14, 16, 7, col="darkgrey", smoke=1)
    turret(p, 42, -14, 12, guns=1, length=14, facing=1)
    ensign(p, 58, -14, 22, 13)


@unit("Battleship", at=AT_SEA, wide=True)
def u_battleship(p):
    water(p, -66, 66)
    hull(p, -64, 64, 16, col="navygrey", stripe=True, rake=18)
    turret(p, -44, -16, 16, guns=2, length=26)
    turret(p, -24, -24, 16, guns=2, length=26)
    p.rect(-32, -24, 16, 8, "navygrey", rx=1)
    p.path("M-12,-16 L-8,-62 L4,-62 L8,-16 Z", "navygrey")                                   # tower
    p.rect(-10, -66, 14, 6, "darkgrey", rx=1.5)
    mast(p, -3, -84, -66, col="darkgrey", w=2)
    funnel(p, 18, -16, 22, 9, col="darkgrey", smoke=2)
    turret(p, 44, -16, 16, guns=2, length=24, facing=1)
    ensign(p, 62, -16, 24, 14)


@unit("Submarine", at=AT_SEA, wide=True)
def u_submarine(p):
    water(p, -56, 56, 5)
    p.path("M-56,-4 Q-50,-10 -40,-10 L44,-10 Q54,-8 56,-2 L56,0 L-56,0 Z", "darkgrey")
    p.path("M-8,-10 L-6,-30 L16,-30 L20,-10 Z", "darkgrey")                                    # conning tower
    p.rect(-6, -26, 23, 5, OWNER)
    mast(p, 6, -42, -30, col="black", w=1.6)
    p.rect(-30, -15, 4, 5, "darkgrey", rx=1)                                                   # deck gun
    p.rect(-40, -15, 12, 2, "black", rx=1)


@unit("Nuclear Submarine", at=AT_SEA, wide=True)
def u_nuclear_submarine(p):
    water(p, -66, 66, 5)
    p.path("M-66,0 Q-64,-14 -46,-14 L50,-12 Q60,-10 64,0 Z", "black")
    p.path("M54,-6 L64,-22 L68,-22 L66,0 Z", "black")                                          # rudder
    p.path("M-30,-14 L-28,-40 L-6,-40 L-2,-14 Z", "black")                                     # sail
    p.rect(-29, -34, 25, 6, OWNER)
    p.rect(-40, -32, 48, 2.6, "black", rx=1)                                                   # dive planes


@unit("Carrier", at=AT_SEA, wide=True)
def u_carrier(p):
    water(p, -68, 68)
    hull(p, -66, 66, 16, col="navygrey", rake=14)
    p.rect(-70, -20, 138, 5, "darkgrey", rx=1)                                                 # flight deck
    p.limb(-64, -18, 64, -18, 0.8, "white", ink=False)
    for x in (-52, -36, -20):                                                                  # parked jets
        p.poly([(x - 8, -21), (x + 8, -21), (x + 5, -25), (x - 4, -25)], "grey")
    p.rect(26, -44, 22, 24, "navygrey", rx=2)                                                   # the island
    p.rect(26, -38, 22, 6, OWNER)
    mast(p, 37, -62, -44, col="darkgrey", w=2)
    p.ellipse(37, -63, 6, 2.4, "grey")


@unit("Missile Cruiser", at=AT_SEA, wide=True)
def u_missile_cruiser(p):
    water(p, -64, 64)
    hull(p, -62, 62, 16, col="navygrey", stripe=True, rake=18)
    turret(p, -46, -16, 10, guns=1, length=12)
    p.rect(-34, -20, 18, 4, "darkgrey", rx=1)                                                  # missile cells
    p.path("M-12,-16 L-6,-40 L24,-40 L30,-16 Z", "navygrey")                                    # superstructure
    p.path("M-2,-40 L2,-54 L16,-54 L20,-40 Z", "navygrey")
    p.rect(-8, -36, 30, 4, "glass", ink=False)
    mast(p, 9, -74, -54, col="darkgrey", w=2)
    p.circle(9, -76, 5, "white")                                                                # radar dome
    p.rect(36, -24, 16, 8, "darkgrey", rx=1.5)                                                  # launcher
    p.limb(38, -26, 52, -34, 3, "white")
    ensign(p, 60, -16, 22, 13)


# --- civilians --------------------------------------------------------------------------------------------------
@unit("Worker", civilian=True)
def u_worker(p):
    ground(p)
    p.rect(-30, -66, 3.2, 60, "wood", rx=1.4, rot=-10)                                        # shovel
    p.path("M-26,-8 L-16,-8 Q-17,6 -21,8 Q-26,6 -26,-8 Z", "metal")
    figure(p, tunic=OWNER, pants="cloth")
    hair(p)
    hat(p, "straw", "straw")
    arm(p, -13, -72, -24, -54, OWNER)
    arm(p, 13, -72, -20, -40, OWNER)


@unit("Settler", civilian=True)
def u_settler(p):
    ground(p, 40, -20)
    wheel(p, -34, -14, 14)                                                                     # handcart
    p.rect(-58, -32, 46, 8, "wood", rx=2)
    p.path("M-58,-32 Q-56,-58 -35,-60 Q-14,-58 -12,-32 Z", OWNER)
    p.limb(-12, -30, 6, -50, 2.8, "wood")
    p.ellipse(10, -66, 12, 11, "linen")                                                        # bedroll
    figure(p, tunic=OWNER, pants="cloth")
    hat(p, "slouch", "leather")
    beard(p)
    arm(p, -13, -72, 4, -51, OWNER)
    p.rect(15, -106, 3, 44, "wood", rx=1.3, rot=12)                                            # axe over the shoulder
    p.path("M20,-110 L30,-112 L30,-100 L22,-101 Z", "metal")
    arm(p, 13, -72, 16, -84, OWNER)


@unit("Work Boats", civilian=True, at=CIV_SEA)
def u_work_boats(p):
    water(p, -40, 40)
    q = Shift(p, -14, -12, 0.62)                                                               # the fisherman
    figure(q, tunic="straw", pants="cloth")
    hat(q, "straw", "straw")
    arm(q, 13, -72, 26, -60, "straw")
    arm(q, -13, -72, -16, -54, "straw")
    hull(p, -38, 38, 12, col="wood", stripe=True, rake=8, stern_h=16)
    mast(p, 12, -70, -12)
    lateen(p, 12, -66, -20, 30)
    for i in range(5):                                                                         # net in the water
        p.thin(4 + i * 4, -18, 14 + i * 6, 2, 0.8, "rope")
    p.thin(4, -18, 40, -18, 0.8, "rope")


def robed(p, robe, stole=None, hair_col="hair", beard_col=None, head=None, head_col=None):
    ground(p)
    figure(p, tunic=robe, robe=robe, belt=None)
    if stole is not None:
        for x in (-5, 5):
            p.rect(x - 2.6, -78, 5.2, 50, stole, rx=1.5)
    if hair_col:
        hair(p, hair_col)
    if beard_col:
        p.path("M-7,-84 Q0,-68 7,-84 Q0,-80 -7,-84 Z", beard_col)
    if head:
        hat(p, head, head_col)


@unit("Great Artist", civilian=True)
def u_great_artist(p):
    ground(p, 34, -14)
    for dx in (-36, -16):                                                                      # easel and canvas
        p.limb(dx, 0, -26, -76, 2.2, "wood")
    p.rect(-44, -76, 32, 26, "white", rx=1)
    for x, y, c in ((-36, -68, "glass"), (-24, -64, "flame"), (-32, -58, "olive"), (-18, -70, "red")):
        p.ellipse(x, y, 5, 3.4, c, ink=False)
    figure(p, tunic=OWNER, pants="black")
    hat(p, "beret", "black")
    beard(p, "hair")
    arm(p, -13, -72, -18, -64, OWNER)
    p.limb(-18, -64, -24, -70, 1.4, "darkwood", ink=False)                                   # brush
    arm(p, 13, -72, 18, -56, OWNER)
    p.ellipse(22, -56, 9, 6, "wood", rot=-10)                                                  # palette
    for i, c in enumerate(("red", "glass", "gold")):
        p.circle(18 + 4 * i, -58 + i, 1.6, c, ink=False)


@unit("Great Scientist", civilian=True)
def u_great_scientist(p):
    ground(p, 34, -14)
    for dx in (-44, -24):                                                                      # telescope on its tripod
        p.limb(dx, 0, -34, -56, 1.8, "darkwood")
    p.limb(-14, -58, -56, -86, 6, "bronze")
    p.circle(-14, -58, 3.4, "darkwood")
    robed(p, OWNER, hair_col="grey_hair", beard_col="grey_hair")
    arm(p, -13, -72, -18, -62, OWNER)
    arm(p, 13, -72, 14, -56, OWNER)
    p.rect(10, -62, 13, 16, "red", rx=1.5, rot=8)                                              # book


@unit("Great Merchant", civilian=True)
def u_great_merchant(p):
    ground(p, 36, -10)
    p.ellipse(-30, -12, 12, 12, "linen")                                                       # sacks of goods
    p.ellipse(-20, -8, 9, 9, "straw")
    robed(p, OWNER, stole="gold", beard_col="beard", head="turban", head_col="white")
    p.rect(14, -16, 20, 14, "darkwood", rx=2)                                                  # a chest of gold
    for x in (17, 22, 27, 32):
        p.circle(x, -18, 2.6, "gold")
    arm(p, -13, -72, -18, -56, OWNER)
    arm(p, 13, -72, 18, -60, OWNER)
    p.circle(18, -64, 2.6, "gold")


@unit("Great Engineer", civilian=True)
def u_great_engineer(p):
    ground(p, 36, -10)
    teeth = []
    for i in range(24):                                                                        # a big cog
        a = 2 * math.pi * i / 24
        r = 24 if i % 4 in (0, 1) else 19
        teeth.append((-32 + r * math.cos(a), -34 + r * math.sin(a)))
    p.poly(teeth, "grey")
    p.circle(-32, -34, 7, "darkgrey")
    figure(p, tunic=OWNER, pants="navy", boots="black")
    hat(p, "hardhat", "gold")
    arm(p, 13, -72, 16, -56, OWNER)
    p.rect(12, -58, 18, 6, "glass", rx=3, rot=-20)                                             # rolled plans
    arm(p, -13, -72, -18, -54, OWNER)
    p.limb(-18, -54, -22, -36, 3, "steel")                                                     # wrench
    p.circle(-22, -34, 3.4, "steel")


@unit("Great Prophet", civilian=True)
def u_great_prophet(p):
    p.glow(0, -92, 20, "gold", 0.55)
    p.circle(0, -92, 15, "gold", alpha=0.35, ink=False)
    p.rect(19, -112, 3.2, 112, "wood", rx=1.5)                                                 # staff
    p.curve("M20.6,-112 Q22,-122 30,-118", 3, "wood")
    robed(p, "white", stole=OWNER, hair_col="grey_hair", beard_col="grey_hair")
    arm(p, 13, -72, 20, -66, "white")
    arm(p, -13, -72, -24, -84, "white")


@unit("Great General", civilian=True)
def u_great_general(p):
    ground(p)
    p.path("M-15,-74 L15,-74 L24,-14 Q0,-8 -24,-14 Z", OWNER)                                  # cape behind
    figure(p, tunic="navy", pants="white", boots="black", long_coat=True, belt="gold")
    for x in (-12, 12):
        p.ellipse(x, -74, 5, 2.6, "gold")                                                       # epaulettes
    hair(p, "grey_hair")
    hat(p, "bicorne", "black", plume=OWNER)
    arm(p, -13, -72, -14, -50, "navy")
    arm(p, 13, -72, 22, -100, "navy")
    p.curve("M22,-102 Q16,-120 4,-132", 2.6, "steel")
    p.limb(19, -99, 25, -103, 2.2, "gold")


@unit("Great Admiral", civilian=True, at=CIV_SEA)
def u_great_admiral(p):
    water(p, -46, 46)
    q = Shift(p, -6, -14, 0.8)
    figure(q, tunic="navy", pants="white", boots="black", long_coat=True, belt="gold")
    for x in (-12, 12):
        q.ellipse(x, -74, 5, 2.6, "gold")
    hair(q, "grey_hair")
    hat(q, "bicorne", "black")
    arm(q, -13, -72, -10, -86, "navy")
    q.limb(-8, -88, -36, -92, 5, "bronze")                                                      # telescope
    arm(q, 13, -72, 16, -52, "navy")
    hull(p, -44, 44, 14, col="white", stripe=True, rake=8, stern_h=18)
    ensign(p, 42, -18, 26, 14)


@unit("Missionary", civilian=True)
def u_missionary(p):
    robed(p, "robe", stole=OWNER, hair_col="hair")
    p.rect(-13, -52, 26, 3, "rope", rx=1.2)
    p.rect(-6, -66, 13, 16, "black", rx=1.5)                                                    # book with a cross
    p.limb(0.5, -62, 0.5, -54, 1.4, "gold", ink=False)
    p.limb(-2.5, -59, 3.5, -59, 1.4, "gold", ink=False)
    arm(p, -13, -72, -6, -58, "robe")
    arm(p, 13, -72, 7, -58, "robe")


@unit("Inquisitor", civilian=True)
def u_inquisitor(p):
    robed(p, "black_robe", stole=OWNER, hair_col=None, head="mitre", head_col=OWNER)
    arm(p, 13, -72, 22, -84, "black_robe")
    p.rect(20, -96, 4, 32, "darkwood", rx=1.5)                                                  # torch
    p.glow(22, -104, 12, "flame", 0.6)
    p.path("M17,-98 Q16,-110 22,-118 Q28,-110 27,-98 Z", "flame", ink=False)
    arm(p, -13, -72, -16, -50, "black_robe")


# --- spaceship parts, on a transporter ---------------------------------------------------------------------------------
def transporter(p):
    ground(p, 48, 2)
    p.rect(-46, -24, 92, 10, "grey", rx=2)
    p.rect(-46, -20, 92, 4, OWNER)
    for x in (-32, 0, 32):
        tyre(p, x, -9, 9)
    return -24


@unit("SS Booster", civilian=True)
def u_ss_booster(p):
    d = transporter(p)
    p.rect(-40, d - 26, 76, 26, "white", rx=4)
    p.path("M-40,%g L-56,%g L-40,%g Z" % (d - 26, d - 13, d), "white")
    for x in (-24, 12):
        p.rect(x, d - 26, 5, 26, "black")
    p.path("M36,%g L46,%g L46,%g L36,%g Z" % (d - 26, d - 32, d + 4, d), "red")


@unit("SS Cockpit", civilian=True)
def u_ss_cockpit(p):
    d = transporter(p)
    p.path("M30,%g L30,%g Q-6,%g -40,%g Q-6,%g 30,%g Z" % (d, d - 34, d - 34, d - 14, d + 4, d), "white")
    p.rect(26, d - 34, 8, 34, "darkgrey", rx=2)
    for x in (-6, 6):
        p.circle(x, d - 22, 3.4, "glass")
    p.rect(14, d - 32, 6, 30, OWNER)


@unit("SS Engine", civilian=True)
def u_ss_engine(p):
    d = transporter(p)
    p.rect(8, d - 34, 30, 34, "grey", rx=3)
    p.rect(8, d - 26, 30, 6, OWNER)
    for y in (d - 26, d - 10):
        p.path("M8,%g L-36,%g L-36,%g L8,%g Z" % (y - 5, y - 12, y + 12, y + 5), "darkgrey")    # nozzles
        p.ellipse(-36, y, 4, 12, "black")


@unit("SS Stasis Chamber", civilian=True)
def u_ss_stasis_chamber(p):
    d = transporter(p)
    p.rect(-40, d - 32, 80, 32, "grey", rx=14)
    p.glow(0, d - 16, 26, "glow", 0.5)
    p.rect(-26, d - 26, 52, 20, "glow", rx=9, alpha=0.9)
    p.ellipse(0, d - 16, 16, 4, "white", ink=False, alpha=0.8)
    for x in (-34, 30):
        p.rect(x, d - 32, 5, 32, OWNER)


# --- embarked units -------------------------------------------------------------------------------------------------------
def troop(p, x, deck, kind):
    """A soldier standing in a boat, seen from the waist up."""
    q = Shift(p, x, deck + 30, 0.62)
    q.path("M-15,-70 Q-16,-76 -9,-77 L9,-77 Q16,-76 15,-70 L13,-48 L-13,-48 Z", OWNER)
    q.rect(-3.2, -84, 6.4, 8, "skin", rx=2)
    q.circle(0, -89, 9.5, "skin")
    if kind == "ancient":
        hat(q, "round", "bronze")
        spear_up(q, 17, -130, -48)
    elif kind == "musket":
        hat(q, "tricorne", "black")
        rifle(q, (16, -50), (22, -120))
    else:
        hat(q, "m1", "olivedark")


def cargo(p, xs, deck):
    for i, x in enumerate(xs):
        if i % 2:
            p.rect(x - 6, deck - 12, 12, 12, "wood", rx=1.5)
        else:
            p.ellipse(x, deck - 7, 8, 7, "linen")


def longboat(p, load):
    water(p, -48, 48)
    oars(p, -22, 26, 5, -10)
    load(p, -10)
    hull(p, -46, 46, 10, col="wood", rake=10, stern_h=18)
    p.curve("M-46,-10 Q-52,-22 -46,-28", 3, "wood")
    mast(p, 6, -74, -10)
    square_sail(p, 6, -70, -40, 32)


@unit("EmbarkedUnit-Military", at=AT_SEA, wide=True)
def u_embarked_military(p):
    longboat(p, lambda p, d: [troop(p, x, d, "ancient") for x in (-28, -14, 22)])


@unit("EmbarkedUnit-Civilian", civilian=True, at=CIV_SEA)
def u_embarked_civilian(p):
    longboat(p, lambda p, d: cargo(p, (-30, -18, 22, 32), d))


def transport_ship(p, load):
    water(p, -52, 52)
    hull(p, -50, 50, 18, col="wood", rake=12, stern_h=28)
    p.rect(30, -38, 20, 10, "darkwood", rx=1.5)
    for x, top in ((-18, -86), (12, -92)):
        mast(p, x, top, -18)
        square_sail(p, x, top + 6, -44, 26)
    load(p, -18)
    ensign(p, 48, -38, 22, 13)


@unit("EmbarkedUnit-Military-Renaissance era", at=AT_SEA, wide=True)
def u_embarked_military_renaissance(p):
    transport_ship(p, lambda p, d: [troop(p, x, d, "musket") for x in (-36, -2, 28)])


@unit("EmbarkedUnit-Civilian-Renaissance era", civilian=True, at=CIV_SEA)
def u_embarked_civilian_renaissance(p):
    transport_ship(p, lambda p, d: cargo(p, (-36, -4, 26), d))


def landing_craft(p, load):
    water(p, -50, 50)
    load(p, -16)
    p.path("M-50,-20 L-44,0 L46,0 L50,-20 Z", "navygrey")
    p.path("M-50,-20 L-58,-30 L-52,-32 L-44,-20 Z", "navygrey")                                   # bow ramp
    p.rect(-40, -14, 70, 5, OWNER)
    p.rect(34, -32, 14, 12, "navygrey", rx=1.5)
    ensign(p, 48, -32, 20, 12)


@unit("EmbarkedUnit-Military-Modern era", at=AT_SEA, wide=True)
def u_embarked_military_modern(p):
    landing_craft(p, lambda p, d: [troop(p, x, d, "modern") for x in (-30, -16, -2, 12, 26)])


@unit("EmbarkedUnit-Civilian-Modern era", civilian=True, at=CIV_SEA)
def u_embarked_civilian_modern(p):
    landing_craft(p, lambda p, d: cargo(p, (-30, -16, -2, 12, 26), d))


# ============================================================================ rendering
def placement(spec):
    """Where to stand a unit and how big to draw it: centred on its side of the hex, clear of the
    flags, shrunk if it is too wide for that side or too tall for the image."""
    civilian = spec["civilian"]
    u_mid, v0 = spec["at"] or (CIV if civilian else MIL)
    probe = Pen(None)
    spec["fn"](probe)
    half = UNIT_W / 2
    k = K * spec["scale"]
    lo, hi = probe.lo[0] * k / half, probe.hi[0] * k / half
    # The flags are discs in the middle of the hex, the military one in the upper half and the
    # civilian one in the lower half. Low military units (ships, vehicles) can pass under the
    # military flag; civilians stand at the same height as theirs, so they keep to the left.
    room = (-0.95, -0.18) if civilian else (-0.14, 0.95) if spec["wide"] else (0.07, 0.95)
    if hi - lo > room[1] - room[0]:
        f = (room[1] - room[0]) / (hi - lo)
        k, lo, hi = k * f, lo * f, hi * f
    u0 = min(max(u_mid - (lo + hi) / 2, room[0] - lo), room[1] - hi)
    ay = OVERHANG + (v0 + rt.S3 / 2) / rt.S3 * FOOT
    if ay + k * probe.lo[1] < 2:
        k = (ay - 2) / -probe.lo[1]
    return (1 + u0) * half, ay, k


def to_img(P):
    a = P[..., 3:4]
    rgb = np.where(a > 1e-6, P[..., :3] / np.maximum(a, 1e-6), 0.0)
    return Image.fromarray((np.concatenate([np.clip(rgb, 0, 1), np.clip(a, 0, 1)], axis=-1) * 255 + 0.5).astype(np.uint8), "RGBA")


def save(sheet, name):
    """Write the unit image with a soft dark edge and its ground shadow, its empty "-1" placeholder
    and its "-2" owner-colour layer, all cropped alike so they line up in the game."""
    def down(P):
        h, w = P.shape[:2]
        return P.reshape(h // S, S, w // S, S, 4).mean(axis=(1, 3))

    base, civ, gnd = down(sheet.base), down(sheet.civ), down(sheet.ground)
    a = Image.fromarray((np.clip(base[..., 3], 0, 1) * 255).astype(np.uint8))
    edge = np.asarray(a.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(0.7)), np.float32) / 255 * 0.6
    under = np.concatenate([np.zeros(edge.shape + (3,), np.float32), edge[..., None]], -1)
    under = under + gnd * (1 - under[..., 3:4])
    base = base + under * (1 - base[..., 3:4])
    rows = np.nonzero(base[..., 3].max(axis=1) > 0.5 / 255)[0]
    top = rows[0] if len(rows) else 0
    rel = "Units/" + name
    write(to_img(base[top:]), rel)
    write(to_img(civ[top:]), rel + "-2")
    write(Image.new("RGBA", (8, 7), (0, 0, 0, 0)), rel + "-1")


def write(img, rel):
    path = os.path.join(rt.OUT, rel + ".png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, optimize=True)
    rt.WRITTEN.append(rel)


def render_unit(name, spec):
    if not rt.wanted("Units/" + name):
        return
    ax, ay, k = placement(spec)
    sheet = Sheet()
    spec["fn"](Pen(sheet, ax, ay, k))
    save(sheet, name)


def main():
    rt.FILTER = sys.argv[1:] or None
    for name, spec in UNITS.items():
        render_unit(name, spec)
    print("Wrote %d unit images (%d units)" % (len(rt.WRITTEN), len(rt.WRITTEN) // 3))


if __name__ == "__main__":
    main()
