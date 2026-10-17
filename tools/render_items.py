#!/usr/bin/env python3
"""Render the tileset's resources, improvements, city centres and natural wonders.

    python3 tools/render_items.py            # everything
    python3 tools/render_items.py Mine Iron  # only images whose path contains one of the words

Uses the same projection, light and helpers as render_tiles.py.

Layout rules, so that everything stays readable in game
--------------------------------------------------------
* Unciv draws a tile's resource first and its improvement on top of it.
* Resources sit in the middle of the tile (within about 0.42 of the centre, where the hex has
  radius 1) on a cleared, contrasting patch with a soft shadow, and are drawn larger than life.
* Improvements that only ever exist on a resource (pasture, camp, plantation, mine, quarry,
  fishing boats, oil well, offshore platform) are drawn as a ring around the middle and leave
  the centre clear, so the resource under them stays visible. Tall parts go on the north side
  or the far east and west, because height is drawn upward on screen.
* Farms usually stand alone, so "Farm" covers the whole tile and "Wheat+Farm" (the name Unciv
  looks for when both are present) keeps the wheat in the middle.
* Natural wonders replace the whole tile, so they are opaque. Their outer ring uses the ground
  of the terrain they turn into, which is what the neighbours' edge images blend toward.
"""

import math
import sys

import numpy as np

import render_tiles as rt
from render_tiles import C, Canvas, fbm, mix, over, premult, ramp, ridged, smooth, warp

K = rt.K_RISE
L = rt.LIGHT
SS = rt.SS
SHX, SHY = -L[0] / L[2], -L[1] / L[2]   # ground shadow offset per unit of height (toward the east)

RES_W = 160    # resources
IMP_W = 192    # improvements
BIG_W = 224    # city centres and natural wonders (full tiles, same size as the terrain)


# ============================================================================ primitives
def footprint(cv):
    return (cv.hex_alpha(0.5) > 0).astype(float)


def _box(cv, P, cx, cy, rx, ry):
    x0, x1 = int(max(0, cx - rx - 3)), int(min(P.shape[1], cx + rx + 3))
    y0, y1 = int(max(0, cy - ry - 3)), int(min(P.shape[0], cy + ry + 3))
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5
    return x0, x1, y0, y1, xx, yy


def _put(P, win, rgb, a):
    x0, x1, y0, y1 = win[:4]
    P[y0:y1, x0:x1] = over(P[y0:y1, x0:x1], premult(rgb, a))


def col_arr(col, shape):
    return np.broadcast_to(np.asarray(col, dtype=float), shape + (3,))


def poly(cv, P, pts, col, alpha=1.0, lift=0.0):
    """Anti-aliased convex polygon, points in hex units."""
    q = [cv.px(u, v - K * lift) for u, v in pts]
    xs, ys = zip(*q)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    win = _box(cv, P, cx, cy, (max(xs) - min(xs)) / 2, (max(ys) - min(ys)) / 2)
    if win is None:
        return
    xx, yy = win[4], win[5]
    n = len(q)
    area = sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))
    sgn = 1.0 if area > 0 else -1.0
    inside = np.full(xx.shape, np.inf)
    for i in range(n):
        ax, ay = q[i]
        bx, by = q[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        ln = max(1e-9, math.hypot(ex, ey))
        inside = np.minimum(inside, -sgn * ((xx - ax) * ey - (yy - ay) * ex) / ln)
    a = np.clip(inside / SS + 0.5, 0, 1) * alpha
    _put(P, win, col_arr(col, a.shape), a)


def seg(cv, P, pts, width, col, alpha=1.0, lift=0.0):
    """Polyline of the given width (hex units)."""
    for (u0, v0), (u1, v1) in zip(pts[:-1], pts[1:]):
        x0, y0 = cv.px(u0, v0 - K * lift)
        x1, y1 = cv.px(u1, v1 - K * lift)
        hw = width * cv.spx / 2
        win = _box(cv, P, (x0 + x1) / 2, (y0 + y1) / 2, abs(x1 - x0) / 2 + hw, abs(y1 - y0) / 2 + hw)
        if win is None:
            continue
        xx, yy = win[4], win[5]
        dx, dy = x1 - x0, y1 - y0
        t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / max(1e-9, dx * dx + dy * dy), 0, 1)
        d = np.hypot(xx - (x0 + t * dx), yy - (y0 + t * dy))
        a = np.clip((hw - d) / SS + 0.5, 0, 1) * alpha
        _put(P, win, col_arr(col, a.shape), a)


def shadow(cv, P, u, v, rx, ry, height, alpha=0.38, ang=0.0):
    blob(cv, P, u + SHX * height * 0.85, v + SHY * height * 0.85, rx * 1.05, ry * 1.05,
         C(.02, .03, .02), ang=ang, alpha=alpha, flat=True)


def blob(cv, P, u, v, rx, ry, col, ang=0.0, lift=0.0, alpha=1.0, gloss=0.0, rough=0.0, flat=False,
         shade_lo=0.35, shadow_h=None):
    """A shaded ellipsoid seen from above, lifted `lift` units off the ground."""
    if shadow_h:
        shadow(cv, P, u, v, rx, ry, shadow_h, ang=ang)
    cx, cy = cv.px(u, v - K * lift)
    rxp, ryp = max(rx * cv.spx, 0.6), max(ry * cv.spx, 0.6)
    R = max(rxp, ryp)
    win = _box(cv, P, cx, cy, R * 1.3, R * 1.3)
    if win is None:
        return
    xx, yy = win[4], win[5]
    dx, dy = xx - cx, yy - cy
    ca, sa = math.cos(ang), math.sin(ang)
    lx, ly = (dx * ca + dy * sa) / rxp, (-dx * sa + dy * ca) / ryp
    d = np.hypot(lx, ly)
    if rough:
        th = np.arctan2(ly, lx)
        d = d * (1.0 + rough * (np.sin(5 * th + rx * 97) * 0.6 + np.sin(9 * th + ry * 53) * 0.4))
    a = np.clip((1.0 - d) * min(rxp, ryp) / SS + 0.5, 0, 1) * alpha
    if flat:
        _put(P, win, col_arr(col, a.shape), a)
        return
    nz = np.sqrt(np.clip(1 - d * d, 0, 1))
    nx, ny = dx / R, dy / R
    lam = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    rgb = np.asarray(col, float) * (shade_lo + (1.05 - shade_lo) * lam)[..., None]
    if gloss:
        spec = np.clip(lam, 0, 1) ** 18 * gloss
        rgb = rgb + spec[..., None]
    _put(P, win, rgb, a)


def box(cv, P, u, v, w, d, h, wall, roof, kind="hip", shadow_a=0.42, rh=None):
    """Axis-aligned building in the tileset's oblique view. kind: hip, gable, flat."""
    wall, roof = np.asarray(wall, float), np.asarray(roof, float)
    rh = 0.0 if kind == "flat" else (0.5 * min(w, d) if rh is None else rh)
    top = h + rh
    hw, hd = w / 2, d / 2
    sx, sy = SHX * top, SHY * top
    poly(cv, P, [(u - hw, v - hd), (u + hw, v - hd), (u + hw + sx, v - hd + sy), (u + hw + sx, v + hd + sy),
                 (u - hw + sx, v + hd + sy), (u - hw, v + hd)], C(.02, .03, .02), shadow_a)
    poly(cv, P, [(u - hw, v + hd), (u + hw, v + hd), (u + hw, v + hd - K * h), (u - hw, v + hd - K * h)], wall * 0.8)
    es, en = v + hd - K * h, v - hd - K * h
    if kind == "flat":
        poly(cv, P, [(u - hw, en), (u + hw, en), (u + hw, es), (u - hw, es)], roof)
        seg(cv, P, [(u - hw, en), (u + hw, en)], 0.006, roof * 1.2)
        return
    ry = v - K * (h + rh)
    if kind == "gable":   # ridge runs east-west
        poly(cv, P, [(u - hw, es), (u + hw, es), (u + hw, ry), (u - hw, ry)], roof * 0.95)
        poly(cv, P, [(u - hw, ry), (u + hw, ry), (u + hw, en), (u - hw, en)], roof * 0.68)
        return
    ins = min(hw, hd) * 0.9
    poly(cv, P, [(u - hw, es), (u + hw, es), (u + hw - ins, ry), (u - hw + ins, ry)], roof * 0.92)
    poly(cv, P, [(u - hw, en), (u - hw + ins, ry), (u + hw - ins, ry), (u + hw, en)], roof * 0.72)
    poly(cv, P, [(u - hw, en), (u - hw, es), (u - hw + ins, ry)], roof * 1.12)
    poly(cv, P, [(u + hw, en), (u + hw - ins, ry), (u + hw, es)], roof * 0.6)


def cyl(cv, P, u, v, r, h, side, top, shadow_a=0.4):
    """Vertical cylinder (tank, tower, chimney)."""
    side, top = np.asarray(side, float), np.asarray(top, float)
    shadow(cv, P, u, v, r, r * 0.9, h, alpha=shadow_a)
    poly(cv, P, [(u - r, v), (u + r, v), (u + r, v - K * h), (u - r, v - K * h)], side * 0.85)
    poly(cv, P, [(u - r, v), (u - 0.3 * r, v), (u - 0.3 * r, v - K * h), (u - r, v - K * h)], side * 1.05)
    blob(cv, P, u, v, r, r * 0.92, side * 0.85, flat=True)
    blob(cv, P, u, v - K * h, r, r * 0.92, top, flat=True)


def cone(cv, P, u, v, r, h, col, shadow_a=0.4):
    """Cone (tent, heap) with a lit west side and a shaded east side."""
    col = np.asarray(col, float)
    shadow(cv, P, u, v, r, r * 0.85, h * 0.6, alpha=shadow_a)
    blob(cv, P, u, v, r, r * 0.85, col * 0.8, flat=True)
    apex = (u, v - K * h)
    poly(cv, P, [(u - r, v), (u, v + r * 0.85), apex], col * 1.08)
    poly(cv, P, [(u, v + r * 0.85), (u + r, v), apex], col * 0.72)
    poly(cv, P, [(u - r, v), apex, (u, v - r * 0.5)], col * 0.95)
    poly(cv, P, [(u, v - r * 0.5), apex, (u + r, v)], col * 0.6)


def smoke(cv, P, u, v, h0, n=5, col=C(.82, .82, .82), size=0.035, drift=0.03, alpha=0.5):
    for k in range(n):
        blob(cv, P, u + drift * k, v, size * (1 + 0.35 * k), size * (1 + 0.35 * k), col,
             lift=h0 + 0.05 * k, alpha=alpha * (1 - k / (n + 1)), shade_lo=0.7)


def flag(cv, P, u, v, h, col):
    seg(cv, P, [(u, v), (u, v - K * h)], 0.007, C(.30, .25, .2))
    poly(cv, P, [(u, v - K * h), (u + 0.05, v - K * h + 0.008), (u + 0.05, v - K * h + 0.035), (u, v - K * h + 0.03)], col)


def patch(cv, P, r, rgb, seed, cu=0.0, cvv=0.0, halo=0.28, wob=0.05, alpha=1.0):
    """A cleared, contrasting ground patch with a soft drop shadow, so resources pop on any terrain."""
    n = fbm(cv.a, cv.b, 6, 3, seed)
    d = np.hypot(cv.u - cu, (cv.v - cvv) * 1.08) + (n - 0.5) * wob * 2
    fp = footprint(cv)
    if halo:
        P = over(P, premult(np.zeros(cv.shape + (3,)), smooth(r + 0.07, r - 0.01, d) * halo * fp))
    a = smooth(r, r - 0.04, d) * fp * alpha
    if np.ndim(rgb) == 1:
        rgb = col_arr(rgb, cv.shape) * (0.85 + 0.3 * fbm(cv.a, cv.b, 12, 3, seed + 1))[..., None] * rt.fine(cv, seed + 2)
    return over(P, premult(rgb, a))


def ground(cv, col, seed, var=0.3, freq=10):
    return col_arr(col, cv.shape) * (1 - var / 2 + var * fbm(cv.a, cv.b, freq, 3, seed))[..., None] * rt.fine(cv, seed + 5)


def ring_mask(cv, r0, r1, seed=0, soft=0.05):
    n = fbm(cv.a, cv.b, 5, 3, 900 + seed)
    d = np.hypot(cv.u, cv.v * 1.08) + (n - 0.5) * 0.06
    return smooth(r0 - soft, r0, d) * smooth(r1, r1 - soft, d) * footprint(cv)


def scatter(rng, n, rmax, gap, rmin=0.0, center=(0.0, 0.0)):
    pts, tries = [], 0
    while len(pts) < n and tries < n * 500:
        tries += 1
        a, rr = rng.uniform(0, 2 * math.pi), math.sqrt(rng.uniform(rmin ** 2 / rmax ** 2, 1)) * rmax
        u, v = center[0] + rr * math.cos(a), center[1] + rr * math.sin(a) * 0.92
        if all((u - x) ** 2 + (v - y) ** 2 >= gap ** 2 for x, y in pts):
            pts.append((u, v))
    return pts


def trees(cv, P, pts, rng, r=(0.05, 0.075), ht=(0.08, 0.12), palette=None, kind="broad"):
    t = [rt.make_tree(rng, u, v, r, ht, palette or rt.FOREST_GREENS, kind) for u, v in pts]
    return rt.splat_trees(cv, P, t)


def fruit_trees(cv, P, pts, rng, crown, fruit, r=0.07, ht=0.1, n_fruit=9, fruit_r=0.013):
    tl = [rt.make_tree(rng, u, v, (r * 0.9, r * 1.1), (ht, ht * 1.2), [crown]) for u, v in pts]
    P = rt.splat_trees(cv, P, tl)
    for t in tl:
        for _ in range(n_fruit):
            a, rr = rng.uniform(0, 2 * math.pi), rng.uniform(0, t["r"] * 0.8)
            blob(cv, P, t["u"] + rr * math.cos(a), t["v"] + rr * math.sin(a), fruit_r, fruit_r, fruit,
                 lift=t["ht"] + 0.01, gloss=0.25)
    return P


def animal(cv, P, u, v, ang, length, width, col, head=None, lift=0.035, patches=None, rng=None):
    """Body seen from above with a head in front; `ang` is the heading."""
    head = col if head is None else head
    shadow(cv, P, u, v, length, width, lift, alpha=0.35, ang=ang)
    blob(cv, P, u, v, length, width, col, ang=ang, lift=lift)
    if patches is not None and rng is not None:
        for _ in range(3):
            pa = rng.uniform(-0.6, 0.6)
            blob(cv, P, u + math.cos(ang) * length * pa, v + math.sin(ang) * length * pa,
                 width * 0.45, width * 0.4, patches, ang=ang, lift=lift + 0.002, shade_lo=0.5)
    hu, hv = u + math.cos(ang) * length * 1.05, v + math.sin(ang) * length * 1.05
    blob(cv, P, hu, hv, width * 0.62, width * 0.5, head, ang=ang, lift=lift + 0.004)
    return hu, hv


def finish(cv, P, rel):
    rt.save(cv, P, rel)


def canvas(width, overhang=70):
    return Canvas(width, overhang=overhang)


class Zoom:
    """A view of a canvas magnified around the hex centre. Resources are drawn through it so
    they come out larger than life, while clipping still uses the real hex."""

    def __init__(self, cv, z):
        self.__dict__.update(cv.__dict__)
        self.base, self.z = cv, z
        self.u, self.v = cv.u / z, cv.v / z
        self.a = self.u / 1.5
        self.b = self.v / rt.S3 - self.a / 2
        self.spx = cv.spx * z

    def px(self, u, v):
        return self.base.px(u * self.z, v * self.z)

    def hex_alpha(self, bleed=1.5):
        return self.base.hex_alpha(bleed)

    def empty(self):
        return self.base.empty()


RES_ZOOM = 1.4


R0 = 0.42   # resources stay inside this radius; ring improvements stay outside it


# ============================================================================ resources
RESOURCES = {}


def resource(name):
    def deco(fn):
        RESOURCES[name] = fn
        return fn
    return deco


def res_rng(name):
    return np.random.default_rng(sum(ord(c) * (i + 1) for i, c in enumerate(name)))


# --- bonus and strategic animals -------------------------------------------------------------
PASTURE = C(.36, .52, .20)


@resource("Cattle")
def r_cattle(cv, P, rng):
    P = patch(cv, P, R0, PASTURE, 11)
    for (u, v), k in zip([(-0.18, -0.12), (0.16, -0.16), (-0.1, 0.16), (0.2, 0.12)], range(4)):
        col = [C(.42, .25, .13), C(.92, .90, .85), C(.30, .18, .10), C(.92, .90, .85)][k]
        spots = C(.10, .09, .09) if k in (1, 3) else None
        animal(cv, P, u, v, rng.uniform(-0.6, 0.6) + (math.pi if k % 2 else 0), 0.1, 0.06, col,
               head=C(.25, .17, .11) if spots is None else C(.85, .83, .8), patches=spots, rng=rng)
    return P


@resource("Sheep")
def r_sheep(cv, P, rng):
    P = patch(cv, P, R0, C(.40, .54, .22), 12)
    for u, v in scatter(rng, 7, 0.28, 0.15):
        a = rng.uniform(0, 6.3)
        shadow(cv, P, u, v, 0.07, 0.056, 0.035)
        blob(cv, P, u, v, 0.072, 0.058, C(.96, .95, .90), ang=a, lift=0.035, rough=0.08)
        blob(cv, P, u + 0.075 * math.cos(a), v + 0.075 * math.sin(a), 0.026, 0.022, C(.15, .13, .12), lift=0.038)
    return P


@resource("Horses")
def r_horses(cv, P, rng):
    P = patch(cv, P, R0, C(.42, .55, .24), 13)
    seg(cv, P, [(-0.3, 0.25), (-0.1, 0.05), (0.2, 0.0), (0.33, -0.2)], 0.02, C(.55, .48, .32), 0.5)
    for (u, v), col in zip([(-0.17, -0.1), (0.13, -0.15), (0.0, 0.16)], [C(.55, .30, .14), C(.12, .10, .09), C(.86, .84, .80)]):
        a = -0.35 + rng.uniform(-0.2, 0.2)
        hu, hv = animal(cv, P, u, v, a, 0.12, 0.045, col, head=col * 0.9)
        seg(cv, P, [(u - 0.1 * math.cos(a), v - 0.1 * math.sin(a)), (u - 0.16 * math.cos(a), v - 0.16 * math.sin(a) + 0.02)],
            0.02, C(.08, .06, .05), lift=0.03)
        seg(cv, P, [(u + 0.02 * math.cos(a), v + 0.02 * math.sin(a)), (hu, hv)], 0.012, C(.08, .06, .05), lift=0.045)
    return P


@resource("Deer")
def r_deer(cv, P, rng):
    P = patch(cv, P, R0, C(.52, .56, .30), 14)
    for (u, v), antlers in zip([(-0.16, -0.08), (0.15, -0.14), (0.02, 0.15)], [True, False, False]):
        a = rng.uniform(-2.8, -0.3)
        hu, hv = animal(cv, P, u, v, a, 0.095, 0.04, C(.66, .44, .24), head=C(.58, .38, .21))
        blob(cv, P, u - 0.07 * math.cos(a), v - 0.07 * math.sin(a), 0.018, 0.018, C(.95, .93, .88), lift=0.04)
        if antlers:
            for s in (-1, 1):
                p0 = (hu, hv)
                p1 = (hu + 0.05 * math.cos(a + s * 1.0), hv + 0.05 * math.sin(a + s * 1.0))
                p2 = (p1[0] + 0.03 * math.cos(a + s * 0.3), p1[1] + 0.03 * math.sin(a + s * 0.3))
                seg(cv, P, [p0, p1, p2], 0.008, C(.38, .28, .18), lift=0.06)
    return P


@resource("Bison")
def r_bison(cv, P, rng):
    P = patch(cv, P, R0, C(.58, .52, .32), 15)
    for u, v in [(-0.17, -0.1), (0.15, -0.14), (-0.05, 0.16), (0.22, 0.12)]:
        a = math.pi + rng.uniform(-0.4, 0.4)
        shadow(cv, P, u, v, 0.11, 0.065, 0.06)
        blob(cv, P, u - 0.035 * math.cos(a), v - 0.035 * math.sin(a), 0.085, 0.055, C(.30, .20, .12), ang=a, lift=0.05)
        blob(cv, P, u + 0.04 * math.cos(a), v + 0.04 * math.sin(a), 0.066, 0.062, C(.20, .13, .08), ang=a, lift=0.06, rough=0.1)
        blob(cv, P, u + 0.1 * math.cos(a), v + 0.1 * math.sin(a), 0.033, 0.03, C(.12, .08, .06), lift=0.055)
    return P


@resource("Furs")
def r_furs(cv, P, rng):
    P = patch(cv, P, R0, C(.86, .88, .92), 16)
    for u, v in [(-0.15, -0.1), (0.15, -0.05), (0.0, 0.16)]:
        a = rng.uniform(0, 6.3)
        tu, tv = u - 0.1 * math.cos(a), v - 0.1 * math.sin(a)
        blob(cv, P, tu, tv, 0.07, 0.035, C(.85, .42, .12), ang=a, lift=0.03, shadow_h=0.03)
        blob(cv, P, tu - 0.055 * math.cos(a), tv - 0.055 * math.sin(a), 0.026, 0.023, C(.97, .96, .94), lift=0.031)
        hu, hv = animal(cv, P, u, v, a, 0.075, 0.038, C(.88, .46, .14), head=C(.85, .44, .14))
        for s in (-1, 1):
            blob(cv, P, hu + 0.015 * math.cos(a + s * 1.6), hv + 0.015 * math.sin(a + s * 1.6), 0.008, 0.008, C(.2, .12, .08), lift=0.05)
    return P


@resource("Ivory")
def r_ivory(cv, P, rng):
    P = patch(cv, P, R0, C(.74, .66, .42), 17)
    for (u, v), s in [((-0.12, -0.08), 1.0), ((0.16, 0.12), 0.85)]:
        a = math.pi * 0.15 if s == 1.0 else math.pi * 1.1
        shadow(cv, P, u, v, 0.13 * s, 0.085 * s, 0.09 * s)
        blob(cv, P, u, v, 0.13 * s, 0.085 * s, C(.40, .40, .42), ang=a, lift=0.07 * s)
        hu, hv = u + math.cos(a) * 0.12 * s, v + math.sin(a) * 0.12 * s
        for side in (-1, 1):
            blob(cv, P, hu + math.cos(a + side * 1.7) * 0.05 * s, hv + math.sin(a + side * 1.7) * 0.05 * s,
                 0.05 * s, 0.035 * s, C(.36, .35, .37), ang=a + side * 1.7, lift=0.075 * s)
        blob(cv, P, hu, hv, 0.05 * s, 0.045 * s, C(.38, .38, .40), lift=0.08 * s)
        tip = (hu + math.cos(a) * 0.1 * s, hv + math.sin(a) * 0.1 * s)
        seg(cv, P, [(hu, hv), tip], 0.016 * s, C(.36, .35, .37), lift=0.05 * s)
        for side in (-1, 1):
            seg(cv, P, [(hu + math.cos(a + side * 0.5) * 0.03 * s, hv + math.sin(a + side * 0.5) * 0.03 * s),
                        (hu + math.cos(a + side * 0.3) * 0.09 * s, hv + math.sin(a + side * 0.3) * 0.09 * s)],
                0.013 * s, C(.97, .95, .88), lift=0.06 * s)
    return P


@resource("Truffles")
def r_truffles(cv, P, rng):
    P = patch(cv, P, R0, C(.32, .25, .16), 18)
    for _ in range(5):
        a, l = rng.uniform(0, 6.3), rng.uniform(0.12, 0.3)
        seg(cv, P, [(0, 0), (l * math.cos(a), l * math.sin(a))], 0.012, C(.42, .32, .20), 0.7)
    for u, v in scatter(rng, 8, 0.3, 0.09, rmin=0.12):
        blob(cv, P, u, v, 0.038, 0.033, C(.17, .12, .09), lift=0.012, rough=0.2, gloss=0.15, shadow_h=0.012)
    animal(cv, P, -0.03, -0.04, 0.4, 0.11, 0.07, C(.93, .66, .62), head=C(.95, .70, .66))
    blob(cv, P, -0.03 + 0.18 * math.cos(0.4), -0.04 + 0.18 * math.sin(0.4), 0.024, 0.021, C(.85, .50, .50), lift=0.05)
    return P


# --- plants -----------------------------------------------------------------------------------
def rows_texture(cv, col_a, col_b, freq, ang, seed):
    x = cv.u * math.cos(ang) + cv.v * math.sin(ang)
    t = 0.5 + 0.5 * np.sin(x * 2 * math.pi * freq)
    rgb = mix(col_arr(col_a, cv.shape), col_arr(col_b, cv.shape), t)
    return rgb * (0.88 + 0.24 * fbm(cv.a, cv.b, 12, 3, seed))[..., None] * rt.fine(cv, seed + 1)


@resource("Wheat")
def r_wheat(cv, P, rng):
    P = patch(cv, P, R0, rows_texture(cv, C(.86, .70, .30), C(.70, .55, .22), 9, 0.5, 21), 21)
    for u, v in scatter(rng, 6, 0.28, 0.13):
        cone(cv, P, u, v, 0.035, 0.07, C(.90, .74, .34))
    return P


@resource("Cotton")
def r_cotton(cv, P, rng):
    P = patch(cv, P, R0, rows_texture(cv, C(.32, .44, .20), C(.50, .40, .28), 8, -0.3, 22), 22)
    for i in range(-6, 7):
        for j in range(-6, 7):
            u = i * 0.06 + 0.03 * (j % 2)
            v = j * 0.055
            if u * u + v * v < 0.33 ** 2 and rng.random() < 0.85:
                blob(cv, P, u + rng.uniform(-0.01, 0.01), v, 0.022, 0.02, C(.98, .98, .96), lift=0.02, rough=0.15)
    return P


@resource("Wine")
def r_wine(cv, P, rng):
    P = patch(cv, P, R0, C(.52, .43, .29), 23)
    for j in range(-5, 6):
        v = j * 0.065
        half = math.sqrt(max(0, 0.36 ** 2 - v * v))
        seg(cv, P, [(-half, v), (half, v)], 0.026, C(.30, .48, .16), lift=0.015)
        u = -half + 0.03
        while u < half - 0.02:
            blob(cv, P, u, v + 0.012, 0.021, 0.024, C(.50, .14, .48), lift=0.025, gloss=0.35)
            u += rng.uniform(0.045, 0.07)
    cyl(cv, P, 0.27, -0.25, 0.04, 0.06, C(.45, .28, .15), C(.38, .22, .12))
    return P


@resource("Sugar")
def r_sugar(cv, P, rng):
    x = cv.u * 0.4 + cv.v
    stalks = 0.5 + 0.5 * np.sin(x * 2 * math.pi * 14 + fbm(cv.a, cv.b, 6, 2, 24) * 6)
    rgb = mix(col_arr(C(.42, .62, .16), cv.shape), col_arr(C(.66, .82, .30), cv.shape), stalks)
    P = patch(cv, P, R0, rgb * rt.fine(cv, 25, 0.2), 24)
    for u, v in [(0.18, -0.22), (0.26, -0.16), (0.12, -0.28)]:
        box(cv, P, u, v, 0.1, 0.03, 0.025, C(.78, .70, .45), C(.86, .78, .5), kind="flat")
    return P


@resource("Bananas")
def r_bananas(cv, P, rng):
    P = patch(cv, P, R0, C(.50, .56, .28), 26)
    for u, v in [(-0.18, -0.14), (0.16, -0.18), (-0.14, 0.15), (0.18, 0.12), (0.0, -0.01)]:
        shadow(cv, P, u, v, 0.1, 0.09, 0.12)
        for k in range(7):
            a = k * 2 * math.pi / 7 + rng.uniform(-0.2, 0.2)
            blob(cv, P, u + 0.06 * math.cos(a), v + 0.055 * math.sin(a), 0.07, 0.026,
                 C(.40, .68, .18) * rng.uniform(0.9, 1.1), ang=a, lift=0.12)
        for k in range(4):
            blob(cv, P, u + rng.uniform(-0.03, 0.03), v + rng.uniform(-0.03, 0.03), 0.03, 0.02,
                 C(1.0, .88, .20), ang=rng.uniform(0, 3), lift=0.13, gloss=0.4)
    return P


@resource("Citrus")
def r_citrus(cv, P, rng):
    P = patch(cv, P, R0, C(.40, .44, .22), 27)
    pts = [(-0.18, -0.13), (0.16, -0.16), (-0.18, 0.15), (0.16, 0.14), (0.0, 0.0)]
    return fruit_trees(cv, P, pts, rng, C(.16, .38, .13), C(1.0, .56, .08), r=0.09, n_fruit=11, fruit_r=0.022)


@resource("Cocoa")
def r_cocoa(cv, P, rng):
    P = patch(cv, P, R0, C(.36, .28, .18), 28)
    poly(cv, P, [(0.0, 0.05), (0.3, 0.05), (0.3, 0.3), (0.0, 0.3)], C(.70, .58, .40))
    for _ in range(40):
        blob(cv, P, rng.uniform(0.03, 0.27), rng.uniform(0.08, 0.27), 0.012, 0.009, C(.36, .20, .12), ang=rng.uniform(0, 3))
    for u, v in [(-0.2, -0.12), (0.12, -0.2), (-0.18, 0.18)]:
        P = trees(cv, P, [(u, v)], rng, r=(0.085, 0.09), ht=(0.1, 0.11), palette=[C(.20, .36, .14)])
        for k in range(6):
            a = rng.uniform(0, 6.3)
            blob(cv, P, u + 0.05 * math.cos(a), v + 0.05 * math.sin(a), 0.036, 0.021, C(.88, .36, .10), ang=a, lift=0.11, gloss=0.25)
    return P


@resource("Dyes")
def r_dyes(cv, P, rng):
    P = patch(cv, P, R0, C(.44, .46, .26), 29)
    for (u, v), col in zip([(-0.15, -0.1), (0.13, -0.13), (0.0, 0.14)], [C(.20, .16, .62), C(.82, .08, .22), C(.96, .66, .08)]):
        cyl(cv, P, u, v, 0.085, 0.05, C(.45, .32, .20), col)
        blob(cv, P, u, v - K * 0.05, 0.07, 0.065, col, flat=True)
        blob(cv, P, u - 0.02, v - K * 0.05 - 0.02, 0.02, 0.015, C(1, 1, 1), alpha=0.35, flat=True)
    for u, v in scatter(rng, 14, 0.36, 0.05, rmin=0.25):
        blob(cv, P, u, v, 0.016, 0.016, [C(.30, .2, .7), C(.85, .1, .3), C(.95, .7, .1)][rng.integers(3)], lift=0.01)
    return P


@resource("Incense")
def r_incense(cv, P, rng):
    P = patch(cv, P, R0, C(.70, .60, .42), 30)
    P = trees(cv, P, [(-0.2, -0.12), (0.18, -0.16), (-0.12, 0.17), (0.2, 0.14)], rng, r=(0.06, 0.07), ht=(0.07, 0.08),
              palette=[C(.40, .44, .30), C(.36, .40, .28)])
    for u, v in [(-0.03, 0.02), (0.04, 0.05), (0.0, -0.05)]:
        blob(cv, P, u, v, 0.055, 0.045, C(.95, .60, .14), lift=0.02, gloss=0.5, rough=0.15, shadow_h=0.02)
    cyl(cv, P, 0.02, -0.14, 0.035, 0.05, C(.62, .46, .22), C(.25, .15, .1))
    smoke(cv, P, 0.02, -0.14, 0.09, n=7, col=C(.95, .95, .98), size=0.04, drift=0.03, alpha=0.75)
    return P


@resource("Silk")
def r_silk(cv, P, rng):
    P = patch(cv, P, R0, C(.38, .44, .22), 31)
    P = trees(cv, P, [(-0.24, -0.16), (0.24, -0.18), (-0.25, 0.14)], rng, r=(0.065, 0.075), ht=(0.07, 0.08),
              palette=[C(.18, .32, .12)])
    for j, col in enumerate([C(.85, .12, .18), C(.95, .72, .18), C(.15, .62, .62), C(.62, .20, .62)]):
        v = -0.12 + j * 0.08
        seg(cv, P, [(-0.12, v), (0.22, v)], 0.006, C(.30, .22, .14), lift=0.06)
        for k in range(4):
            u = -0.09 + k * 0.085
            poly(cv, P, [(u - 0.028, v), (u + 0.028, v), (u + 0.026, v + 0.05), (u - 0.026, v + 0.05)], col, lift=0.06)
    return P


@resource("Spices")
def r_spices(cv, P, rng):
    P = patch(cv, P, R0, C(.55, .45, .30), 32)
    for (u, v), col in zip([(-0.15, -0.12), (0.14, -0.14), (-0.13, 0.12), (0.15, 0.12)],
                           [C(.78, .12, .07), C(.96, .62, .10), C(.52, .28, .13), C(.45, .56, .18)]):
        cone(cv, P, u, v, 0.1, 0.1, col)
    for u, v in [(0.0, 0.0), (0.02, 0.22)]:
        blob(cv, P, u, v, 0.045, 0.035, C(.72, .62, .44), lift=0.03, rough=0.1, shadow_h=0.03)
    return P


# --- minerals ---------------------------------------------------------------------------------
def rocks(cv, P, rng, n, col, size=(0.05, 0.09), rmax=0.3, gloss=0.0, rough=0.12, lift=0.03, var=0.15):
    pts = scatter(rng, n, rmax, size[0] * 1.4)
    for u, v in sorted(pts, key=lambda p: p[1]):
        s = rng.uniform(*size)
        blob(cv, P, u, v, s, s * rng.uniform(0.7, 0.95), np.asarray(col) * rng.uniform(1 - var, 1 + var),
             ang=rng.uniform(0, 3), lift=lift * s / size[1], gloss=gloss, rough=rough, shadow_h=lift * s / size[1])
    return pts


@resource("Stone")
def r_stone(cv, P, rng):
    P = patch(cv, P, R0, C(.55, .53, .48), 41)
    rocks(cv, P, rng, 7, C(.66, .65, .62), size=(0.06, 0.11), lift=0.06)
    return P


@resource("Marble")
def r_marble(cv, P, rng):
    P = patch(cv, P, R0, C(.80, .78, .74), 42)
    for u, v, w, d, h in [(-0.14, -0.1, 0.14, 0.1, 0.07), (0.13, -0.12, 0.12, 0.12, 0.09),
                          (-0.04, 0.13, 0.18, 0.09, 0.06), (0.2, 0.14, 0.08, 0.08, 0.05)]:
        box(cv, P, u, v, w, d, h, C(.93, .92, .90), C(.97, .96, .94), kind="flat")
        seg(cv, P, [(u - w * 0.4, v - d / 2 - K * h + 0.01), (u + w * 0.3, v + d / 2 - K * h - 0.01)], 0.005, C(.55, .55, .58), 0.6)
    return P


@resource("Iron")
def r_iron(cv, P, rng):
    P = patch(cv, P, R0, C(.56, .32, .18), 43)
    rocks(cv, P, rng, 8, C(.58, .26, .12), size=(0.055, 0.1), lift=0.05, var=0.25)
    return P


@resource("Coal")
def r_coal(cv, P, rng):
    P = patch(cv, P, R0, C(.24, .23, .22), 44)
    rocks(cv, P, rng, 11, C(.10, .10, .11), size=(0.04, 0.075), gloss=0.55, lift=0.04, rough=0.2)
    return P


@resource("Copper")
def r_copper(cv, P, rng):
    P = patch(cv, P, R0, C(.50, .42, .32), 45)
    pts = rocks(cv, P, rng, 6, C(.20, .56, .44), size=(0.06, 0.1), lift=0.05, var=0.2)
    for u, v in pts:
        blob(cv, P, u + 0.02, v - 0.03, 0.025, 0.02, C(.86, .48, .22), lift=0.07, gloss=0.6)
    return P


@resource("Gold Ore")
def r_gold(cv, P, rng):
    P = patch(cv, P, R0, C(.52, .44, .32), 46)
    rocks(cv, P, rng, 5, C(.40, .32, .24), size=(0.06, 0.1), lift=0.05)
    for u, v in scatter(rng, 9, 0.28, 0.08):
        blob(cv, P, u, v, 0.045, 0.036, C(1.0, .80, .20), ang=rng.uniform(0, 3), lift=0.08, gloss=0.9, rough=0.15)
    return P


@resource("Silver")
def r_silver(cv, P, rng):
    P = patch(cv, P, R0, C(.36, .36, .38), 47)
    pts = rocks(cv, P, rng, 6, C(.26, .26, .29), size=(0.06, 0.1), lift=0.05)
    for u, v in pts:
        seg(cv, P, [(u - 0.06, v + 0.02), (u, v - 0.015), (u + 0.05, v + 0.01)], 0.018, C(.94, .96, 1.0), lift=0.07)
    for u, v in scatter(rng, 7, 0.28, 0.08):
        blob(cv, P, u, v, 0.032, 0.026, C(.90, .92, .97), lift=0.08, gloss=0.9)
    return P


@resource("Gems")
def r_gems(cv, P, rng):
    P = patch(cv, P, R0, C(.38, .36, .35), 48)
    cols = [C(.88, .08, .18), C(.12, .30, .90), C(.08, .72, .36), C(.62, .20, .82)]
    for (u, v), col in zip([(-0.14, -0.1), (0.13, -0.12), (-0.08, 0.13), (0.16, 0.12), (0.0, 0.0)], cols + [cols[0]]):
        shadow(cv, P, u, v, 0.08, 0.065, 0.1)
        for k in range(5):
            a = -math.pi / 2 + (k - 2) * 0.45
            l = 0.085 + 0.025 * (k % 2)
            tip = (u + l * math.cos(a) * 0.6, v)
            base = 0.03
            poly(cv, P, [(u - base, v), (u + base, v), (tip[0] + 0.017, tip[1] - K * l * 1.2), (tip[0] - 0.017, tip[1] - K * l * 1.2)],
                 col * (0.75 + 0.1 * k))
            poly(cv, P, [(tip[0] - 0.017, tip[1] - K * l * 1.2), (tip[0] + 0.017, tip[1] - K * l * 1.2), (tip[0], tip[1] - K * l * 1.45)],
                 np.minimum(col * 1.5 + 0.2, 1.0))
    return P


@resource("Aluminum")
def r_aluminum(cv, P, rng):
    P = patch(cv, P, R0, C(.62, .62, .64), 49)
    rocks(cv, P, rng, 4, C(.74, .80, .86), size=(0.06, 0.09), lift=0.04, gloss=0.4, rmax=0.32)
    for i, (u, v) in enumerate([(-0.06, -0.02), (0.06, -0.02), (0.0, -0.06)]):
        box(cv, P, u, v + 0.05, 0.11, 0.045, 0.03 + 0.03 * (i == 2), C(.80, .84, .90), C(.92, .95, 1.0), kind="flat")
    return P


@resource("Salt")
def r_salt(cv, P, rng):
    P = patch(cv, P, R0, C(.92, .91, .88), 50)
    for i in range(-2, 2):
        for j in range(-2, 2):
            u, v = i * 0.14 + 0.07, j * 0.13 + 0.065
            if u * u + v * v > 0.3 ** 2:
                continue
            col = [C(.90, .70, .74), C(.72, .86, .88), C(.96, .95, .92)][(i + j) % 3]
            poly(cv, P, [(u - 0.06, v - 0.055), (u + 0.06, v - 0.055), (u + 0.06, v + 0.055), (u - 0.06, v + 0.055)], col)
    for u, v in [(0.2, -0.22), (0.26, -0.15)]:
        cone(cv, P, u, v, 0.05, 0.07, C(.98, .98, .97))
    return P


@resource("Uranium")
def r_uranium(cv, P, rng):
    P = patch(cv, P, R0, C(.32, .33, .28), 51)
    d = np.hypot(cv.u, cv.v)
    P = over(P, premult(col_arr(C(.6, 1.0, .2), cv.shape), smooth(0.4, 0.0, d) * 0.45 * footprint(cv)))
    pts = rocks(cv, P, rng, 5, C(.25, .26, .23), size=(0.06, 0.09), lift=0.04)
    for u, v in pts + scatter(rng, 6, 0.28, 0.08):
        blob(cv, P, u, v - 0.02, 0.034, 0.03, C(.80, 1.0, .25), lift=0.07, gloss=0.8)
    return P


@resource("Oil")
def r_oil(cv, P, rng):
    # Works on land and water: a black, glossy seep with an iridescent rim.
    n = fbm(cv.a, cv.b, 5, 3, 52)
    d = np.hypot(cv.u, cv.v * 1.1) + (n - 0.5) * 0.12
    fp = footprint(cv)
    P = over(P, premult(np.zeros(cv.shape + (3,)), smooth(0.4, 0.3, d) * 0.3 * fp))
    th = np.arctan2(cv.v, cv.u)
    sheen = ramp(0.5 + 0.5 * np.sin(th * 3 + n * 8), [(0, C(.55, .25, .65)), (0.5, C(.15, .55, .6)), (1, C(.85, .7, .25))])
    P = over(P, premult(sheen, smooth(0.34, 0.31, d) * smooth(0.22, 0.27, d) * 0.8 * fp))
    P = over(P, premult(col_arr(C(.04, .04, .05), cv.shape), smooth(0.3, 0.27, d) * fp))
    hl = smooth(0.12, 0.0, np.hypot(cv.u + 0.08, cv.v + 0.08))
    P = over(P, premult(col_arr(C(.6, .6, .65), cv.shape), hl * 0.35 * fp))
    for u, v in scatter(rng, 5, 0.38, 0.08, rmin=0.33):
        blob(cv, P, u, v, 0.022, 0.018, C(.05, .05, .06), gloss=0.5, flat=False)
    return P


# --- water ------------------------------------------------------------------------------------
SHALLOW = C(.18, .52, .56)


@resource("Fish")
def r_fish(cv, P, rng):
    P = patch(cv, P, R0, SHALLOW, 61, halo=0.15)
    for r in (0.12, 0.24, 0.34):
        ring = smooth(0.008, 0.0, np.abs(np.hypot(cv.u, cv.v) - r)) * footprint(cv)
        P = over(P, premult(col_arr(C(.85, .95, .97), cv.shape), ring * 0.35))
    for k in range(14):
        a = k * 2 * math.pi / 14 + rng.uniform(-0.1, 0.1)
        rr = 0.13 + 0.12 * (k % 3) / 2
        u, v = rr * math.cos(a), rr * math.sin(a)
        blob(cv, P, u, v, 0.055, 0.019, C(.80, .88, .94), ang=a + math.pi / 2, gloss=0.5)
        blob(cv, P, u - 0.055 * math.cos(a + math.pi / 2), v - 0.055 * math.sin(a + math.pi / 2), 0.017, 0.017,
             C(.6, .7, .8), ang=a, flat=True)
    return P


@resource("Whales")
def r_whales(cv, P, rng):
    P = patch(cv, P, R0, C(.10, .30, .40), 62, halo=0.15)
    for (u, v), a, s in [((-0.04, -0.1), 0.3, 1.25), ((0.12, 0.15), 0.25, 1.0)]:
        seg(cv, P, [(u - 0.25 * s * math.cos(a), v - 0.25 * s * math.sin(a) + 0.02), (u, v)], 0.03 * s, C(.85, .93, .95), 0.35)
        blob(cv, P, u, v, 0.17 * s, 0.06 * s, C(.20, .25, .32), ang=a, gloss=0.3)
        tu, tv = u - 0.17 * s * math.cos(a), v - 0.17 * s * math.sin(a)
        for side in (-1, 1):
            blob(cv, P, tu + 0.035 * s * math.cos(a + side * 1.9), tv + 0.035 * s * math.sin(a + side * 1.9),
                 0.045 * s, 0.018 * s, C(.18, .22, .28), ang=a + side * 1.2)
        smoke(cv, P, u + 0.12 * s * math.cos(a), v + 0.12 * s * math.sin(a), 0.03, n=4, col=C(.95, .98, 1.0),
              size=0.025 * s, drift=0.0, alpha=0.75)
    return P


@resource("Pearls")
def r_pearls(cv, P, rng):
    P = patch(cv, P, R0, C(.30, .66, .66), 63, halo=0.15)
    for u, v in [(-0.15, -0.1), (0.15, -0.08), (0.0, 0.15)]:
        blob(cv, P, u, v - 0.035, 0.08, 0.05, C(.55, .50, .58), gloss=0.2, shadow_h=0.01)
        blob(cv, P, u, v + 0.02, 0.08, 0.055, C(.88, .84, .90), gloss=0.4)
        blob(cv, P, u, v + 0.02, 0.032, 0.03, C(.99, .98, .97), lift=0.02, gloss=1.0)
    return P


@resource("Crab")
def r_crab(cv, P, rng):
    P = patch(cv, P, R0, C(.72, .66, .48), 64)
    for u, v in [(-0.14, -0.1), (0.14, -0.12), (-0.04, 0.14), (0.2, 0.12)]:
        for side in (-1, 1):
            for k in range(3):
                seg(cv, P, [(u + side * 0.03, v - 0.01 + k * 0.02), (u + side * 0.08, v + 0.005 + k * 0.025)], 0.01, C(.72, .18, .08), lift=0.01)
            blob(cv, P, u + side * 0.06, v - 0.06, 0.03, 0.022, C(.90, .30, .12), lift=0.02, gloss=0.4)
        blob(cv, P, u, v, 0.055, 0.04, C(.88, .28, .10), lift=0.025, gloss=0.4, shadow_h=0.02)
    return P


def render_resource(name):
    rel = "Tiles/" + name
    if not rt.wanted(rel):
        return
    cv = canvas(RES_W, overhang=60)
    P = RESOURCES[name](Zoom(cv, RES_ZOOM), cv.empty(), res_rng(name))
    finish(cv, P, rel)


# ============================================================================ improvements (ring style)
def ring_ground(cv, P, col, seed, r0=R0 + 0.05, r1=0.86, alpha=0.55):
    return over(P, premult(ground(cv, col, seed), ring_mask(cv, r0, r1, seed) * alpha))


def imp_pasture(cv, P, rng):
    P = ring_ground(cv, P, C(.40, .54, .22), 101)
    pts = [(0.47 * math.cos(a), 0.45 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 25)]
    seg(cv, P, pts, 0.012, C(.42, .30, .18), lift=0.012)
    for a in np.linspace(0, 2 * math.pi, 7)[:-1] + 0.3:
        seg(cv, P, [(0.47 * math.cos(a), 0.45 * math.sin(a)), (0.82 * math.cos(a), 0.74 * math.sin(a))], 0.01, C(.42, .30, .18), lift=0.012)
    box(cv, P, -0.5, -0.52, 0.17, 0.11, 0.05, C(.62, .18, .12), C(.36, .32, .30), kind="gable")
    box(cv, P, 0.55, -0.42, 0.09, 0.07, 0.035, C(.80, .76, .68), C(.55, .25, .16))
    for u, v in [(0.62, 0.15), (0.7, 0.05), (-0.66, 0.1)]:
        blob(cv, P, u, v, 0.04, 0.036, C(.86, .70, .30), lift=0.03, rough=0.1, shadow_h=0.03)
    poly(cv, P, [(-0.2, -0.62), (0.05, -0.62), (0.05, -0.58), (-0.2, -0.58)], C(.25, .45, .62))
    return P


def imp_camp(cv, P, rng):
    P = ring_ground(cv, P, C(.46, .42, .30), 102, alpha=0.4)
    for u, v, s in [(-0.55, -0.4, 1.0), (0.0, -0.62, 1.1), (0.55, -0.38, 0.95), (-0.68, 0.18, 0.85)]:
        cone(cv, P, u, v, 0.075 * s, 0.17 * s, C(.80, .70, .52))
        seg(cv, P, [(u, v - K * 0.17 * s), (u - 0.02, v - K * 0.21 * s)], 0.006, C(.3, .2, .1))
        seg(cv, P, [(u, v - K * 0.17 * s), (u + 0.02, v - K * 0.21 * s)], 0.006, C(.3, .2, .1))
    blob(cv, P, 0.66, 0.12, 0.04, 0.035, C(.95, .45, .1), gloss=0.5)
    smoke(cv, P, 0.66, 0.12, 0.05, n=5, col=C(.75, .75, .75), size=0.03, drift=0.02, alpha=0.5)
    for u in (0.3, 0.42):
        seg(cv, P, [(u, -0.62), (u, -0.62 - K * 0.1)], 0.008, C(.35, .25, .15))
    seg(cv, P, [(0.3, -0.62 - K * 0.09), (0.42, -0.62 - K * 0.09)], 0.008, C(.35, .25, .15))
    poly(cv, P, [(0.31, -0.62 - K * 0.09), (0.35, -0.62 - K * 0.09), (0.35, -0.62 - K * 0.03), (0.31, -0.62 - K * 0.03)], C(.55, .35, .20))
    poly(cv, P, [(0.36, -0.62 - K * 0.09), (0.41, -0.62 - K * 0.09), (0.41, -0.62 - K * 0.02), (0.36, -0.62 - K * 0.02)], C(.85, .80, .72))
    return P


def imp_plantation(cv, P, rng):
    P = ring_ground(cv, P, C(.44, .40, .26), 103, alpha=0.6)
    pts = []
    for i in range(-9, 10):
        for j in range(-9, 10):
            u, v = i * 0.1, j * 0.09
            if 0.5 < math.hypot(u, v / 0.95) < 0.84 and rt.hex_edge_dist(u, v) > 0.06:
                pts.append((u, v))
    P = trees(cv, P, pts, rng, r=(0.04, 0.045), ht=(0.05, 0.06), palette=[C(.16, .32, .12), C(.18, .35, .13)])
    box(cv, P, 0.0, -0.66, 0.16, 0.08, 0.045, C(.82, .78, .70), C(.60, .30, .18))
    return P


def imp_mine(cv, P, rng):
    P = ring_ground(cv, P, C(.50, .46, .40), 104, alpha=0.55)
    seg(cv, P, [(-0.75, 0.3), (-0.5, 0.45), (0.0, 0.52), (0.5, 0.45), (0.72, 0.3)], 0.02, C(.30, .28, .26))
    for u, v, s in [(-0.62, 0.15, 1.0), (0.62, 0.18, 0.9)]:
        cone(cv, P, u, v, 0.13 * s, 0.13 * s, C(.42, .40, .38))
    # headframe on the north side so it rises away from the resource
    u, v, h = 0.0, -0.6, 0.38
    for du, dv in [(-0.05, -0.04), (0.05, -0.04), (-0.05, 0.04), (0.05, 0.04)]:
        seg(cv, P, [(u + du, v + dv), (u + du * 0.4, v + dv * 0.4 - K * h)], 0.012, C(.25, .25, .27))
    for k in range(1, 4):
        hh = h * k / 4
        seg(cv, P, [(u - 0.05 * (1 - 0.6 * k / 4), v + 0.04 - K * hh), (u + 0.05 * (1 - 0.6 * k / 4), v + 0.04 - K * hh)], 0.008, C(.25, .25, .27))
    blob(cv, P, u, v - K * h, 0.04, 0.04, C(.3, .3, .32), flat=True)
    blob(cv, P, u, v - K * h, 0.028, 0.028, C(.6, .2, .15), flat=True)
    shadow(cv, P, u, v, 0.05, 0.04, h * 0.6, alpha=0.3)
    box(cv, P, -0.4, -0.55, 0.16, 0.1, 0.06, C(.55, .40, .32), C(.30, .30, .32), kind="gable")
    box(cv, P, 0.38, -0.55, 0.12, 0.09, 0.05, C(.70, .68, .64), C(.40, .40, .42), kind="flat")
    for u2 in (-0.25, -0.12):
        blob(cv, P, u2, 0.5, 0.035, 0.025, C(.35, .25, .18), lift=0.02, shadow_h=0.02)
    return P


def imp_quarry(cv, P, rng):
    th = np.arctan2(cv.v, cv.u)
    arc = smooth(-0.2, 0.15, -np.sin(th))          # northern half
    r = np.hypot(cv.u, cv.v * 1.08)
    steps = np.floor(np.clip((r - 0.45) / 0.09, 0, 4))
    h = -0.04 * steps * arc
    rgb = ramp(fbm(cv.a, cv.b, 9, 3, 105), [(0.3, C(.72, .70, .66)), (0.7, C(.84, .82, .78))])
    rgb = rt.lit(rgb, rt.shade(h, cv.spx, 2.0), lo=0.45)
    m = ring_mask(cv, R0 + 0.06, 0.85, 105) * np.maximum(arc, 0.35)
    P = over(P, premult(rgb, m))
    for u, v, hh in [(0.58, 0.2, 0.05), (0.66, 0.08, 0.04), (0.5, 0.33, 0.06)]:
        box(cv, P, u, v, 0.08, 0.07, hh, C(.88, .87, .84), C(.94, .93, .90), kind="flat")
    seg(cv, P, [(-0.6, -0.2), (-0.6, -0.2 - K * 0.36)], 0.014, C(.75, .55, .15))
    seg(cv, P, [(-0.6, -0.2 - K * 0.36), (-0.3, -0.38 - K * 0.36)], 0.012, C(.75, .55, .15))
    seg(cv, P, [(-0.3, -0.38 - K * 0.36), (-0.3, -0.38 - K * 0.16)], 0.004, C(.2, .2, .2))
    shadow(cv, P, -0.6, -0.2, 0.03, 0.03, 0.2)
    return P


def imp_fishing_boats(cv, P, rng):
    for r in (0.5,):
        for a in np.linspace(0, 2 * math.pi, 22)[:-1]:
            blob(cv, P, r * math.cos(a), r * 0.95 * math.sin(a), 0.012, 0.012, C(.95, .55, .1), gloss=0.3)
        pts = [(r * math.cos(a), r * 0.95 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 40)]
        seg(cv, P, pts, 0.004, C(.9, .9, .85), 0.6)
    for (u, v), a, col in [((-0.55, -0.42), 0.6, C(.85, .20, .15)), ((0.6, -0.36), 2.6, C(.18, .36, .70)), ((0.0, 0.66), -0.05, C(.92, .92, .9))]:
        ca, sa = math.cos(a), math.sin(a)
        wake = [(u - 0.12 * ca - 0.05 * sa, v - 0.12 * sa + 0.05 * ca), (u, v), (u - 0.12 * ca + 0.05 * sa, v - 0.12 * sa - 0.05 * ca)]
        seg(cv, P, wake, 0.012, C(.92, .96, .97), 0.6)
        hull = [(u + 0.1 * ca, v + 0.1 * sa), (u + 0.04 * ca - 0.035 * sa, v + 0.04 * sa + 0.035 * ca),
                (u - 0.08 * ca - 0.03 * sa, v - 0.08 * sa + 0.03 * ca), (u - 0.08 * ca + 0.03 * sa, v - 0.08 * sa - 0.03 * ca),
                (u + 0.04 * ca + 0.035 * sa, v + 0.04 * sa - 0.035 * ca)]
        shadow(cv, P, u, v, 0.09, 0.035, 0.02, ang=a)
        poly(cv, P, hull, col, lift=0.015)
        box(cv, P, u - 0.02 * ca, v - 0.02 * sa, 0.045, 0.035, 0.03, C(.92, .92, .9), C(.95, .95, .95), kind="flat")
    return P


def pumpjack(cv, P, u, v, s=1.5):
    poly(cv, P, [(u - 0.09 * s, v - 0.07 * s), (u + 0.09 * s, v - 0.07 * s), (u + 0.09 * s, v + 0.07 * s),
                 (u - 0.09 * s, v + 0.07 * s)], C(.60, .58, .54))
    shadow(cv, P, u, v, 0.08 * s, 0.02 * s, 0.1 * s, alpha=0.3)
    seg(cv, P, [(u - 0.02 * s, v + 0.02 * s), (u, v - K * 0.12 * s)], 0.016 * s, C(.25, .25, .27))
    seg(cv, P, [(u + 0.02 * s, v + 0.02 * s), (u, v - K * 0.12 * s)], 0.016 * s, C(.25, .25, .27))
    seg(cv, P, [(u - 0.08 * s, v - K * 0.1 * s), (u + 0.08 * s, v - K * 0.14 * s)], 0.02 * s, C(.92, .62, .10))
    blob(cv, P, u + 0.085 * s, v - K * 0.13 * s, 0.026 * s, 0.032 * s, C(.92, .62, .10), flat=True)
    box(cv, P, u - 0.06 * s, v + 0.03 * s, 0.04 * s, 0.03 * s, 0.03 * s, C(.4, .4, .42), C(.3, .3, .32), kind="flat")


def imp_oil_well(cv, P, rng):
    P = ring_ground(cv, P, C(.52, .48, .40), 106, alpha=0.45)
    for u, v in [(-0.52, -0.42), (0.0, -0.64), (-0.66, 0.18)]:
        pumpjack(cv, P, u, v)
    for u, v in [(0.55, -0.35), (0.66, -0.12), (0.55, 0.12)]:
        cyl(cv, P, u, v, 0.08, 0.08, C(.88, .88, .86), C(.95, .95, .94))
    seg(cv, P, [(0.0, -0.5), (0.0, -0.42), (0.45, -0.3)], 0.012, C(.35, .35, .38))
    return P


def imp_offshore_platform(cv, P, rng):
    return _platform(Zoom(cv, 1.25), P)


def _platform(cv, P):
    u, v, h = 0.0, -0.5, 0.16
    for du, dv in [(-0.15, -0.08), (0.15, -0.08), (-0.15, 0.1), (0.15, 0.1)]:
        seg(cv, P, [(u + du, v + dv), (u + du, v + dv - K * h)], 0.022, C(.85, .62, .12))
    shadow(cv, P, u, v, 0.18, 0.12, h, alpha=0.35)
    poly(cv, P, [(u - 0.2, v - 0.11), (u + 0.2, v - 0.11), (u + 0.2, v + 0.12), (u - 0.2, v + 0.12)], C(.55, .55, .56), lift=h)
    blob(cv, P, u + 0.1, v + 0.04, 0.06, 0.055, C(.25, .55, .30), lift=h + 0.002, flat=True)
    blob(cv, P, u + 0.1, v + 0.04, 0.04, 0.035, C(.9, .9, .9), lift=h + 0.003, flat=True)
    box(cv, P, u - 0.1, v + 0.06, 0.12, 0.08, 0.06, C(.92, .92, .9), C(.85, .25, .15), kind="flat")
    seg(cv, P, [(u - 0.16, v - 0.08 - K * h), (u - 0.16, v - 0.08 - K * (h + 0.3))], 0.014, C(.3, .3, .32))
    blob(cv, P, u - 0.16, v - 0.08, 0.03, 0.04, C(1.0, .6, .1), lift=h + 0.32, gloss=0.6)
    seg(cv, P, [(u + 0.05, v - 0.05 - K * h), (u + 0.05, v - 0.05 - K * (h + 0.2)), (u + 0.3, v - 0.2 - K * (h + 0.18))], 0.01, C(.9, .7, .15))
    for (bu, bv), a in [((0.55, -0.15), 2.0), ((-0.55, 0.3), 0.4)]:
        ca, sa = math.cos(a), math.sin(a)
        poly(cv, P, [(bu + 0.08 * ca, bv + 0.08 * sa), (bu - 0.07 * ca - 0.03 * sa, bv - 0.07 * sa + 0.03 * ca),
                     (bu - 0.07 * ca + 0.03 * sa, bv - 0.07 * sa - 0.03 * ca)], C(.9, .9, .88), lift=0.01)
    return P


RING_IMPROVEMENTS = {"Pasture": imp_pasture, "Camp": imp_camp, "Plantation": imp_plantation, "Mine": imp_mine,
                     "Quarry": imp_quarry, "Fishing Boats": imp_fishing_boats, "Oil well": imp_oil_well,
                     "Offshore Platform": imp_offshore_platform}


# ============================================================================ improvements (whole tile)
def farm_fields(cv, P, rng):
    ang = 0.35
    ca, sa = math.cos(ang), math.sin(ang)
    x, y = cv.u * ca - cv.v * sa, cv.u * sa + cv.v * ca
    wx = x * 3.0 + 0.3 * np.sin(y * 4)
    fx, fy = np.floor(wx), np.floor(y * 2.2)
    pick = (fx * 7 + fy * 13) % 5
    crops = [C(.74, .64, .32), C(.40, .54, .20), C(.58, .46, .30), C(.64, .66, .30), C(.34, .48, .18)]
    rgb = np.zeros(cv.shape + (3,))
    for i, c in enumerate(crops):
        rgb[pick == i] = c
    rgb = rgb * (0.88 + 0.12 * np.sin((wx + pick * 0.3) * 2 * math.pi * 8))[..., None] * rt.fine(cv, 111, 0.12)
    edge = np.minimum(np.abs(wx - np.round(wx)) * 0.6, np.abs(y * 2.2 - np.round(y * 2.2)) * 0.9)
    rgb = mix(rgb, C(.16, .25, .11), smooth(0.05, 0.015, edge) * 0.9)
    n = fbm(cv.a, cv.b, 5, 3, 112)
    d = np.hypot(cv.u, cv.v * 1.05) + (n - 0.5) * 0.12
    P = over(P, premult(rgb, smooth(0.9, 0.78, d) * footprint(cv)))
    for u, v, w, dd, h, wall, roof in [(0.36, -0.5, 0.15, 0.08, 0.04, C(.84, .80, .72), C(.58, .25, .16)),
                                        (0.53, -0.42, 0.08, 0.12, 0.045, C(.62, .20, .14), C(.36, .34, .33)),
                                        (0.28, -0.38, 0.07, 0.06, 0.035, C(.82, .78, .70), C(.55, .25, .16))]:
        box(cv, P, u, v, w, dd, h, wall, roof)
    P = trees(cv, P, [(0.18, -0.55), (0.22, -0.62), (0.62, -0.58)], rng, r=(0.04, 0.05), ht=(0.06, 0.08))
    return P


def imp_farm(cv, P, rng):
    return farm_fields(cv, P, rng)


def imp_lumber_mill(cv, P, rng):
    """A timber yard: a thatched saw shed, a sawpit, split logs and a cart."""
    P = patch(cv, P, 0.6, C(.48, .38, .26), 1001, halo=0.3)
    for u, v in scatter(rng, 14, 0.56, 0.08, rmin=0.35):
        blob(cv, P, u, v, 0.02, 0.02, C(.66, .52, .32), lift=0.01)
    box(cv, P, -0.15, -0.2, 0.3, 0.13, 0.06, C(.54, .40, .26), C(.74, .62, .34), kind="gable")
    for k in range(3):
        blob(cv, P, 0.05 + k * 0.03, 0.02, 0.1, 0.018, C(.52, .36, .20), lift=0.02 + 0.03 * k, shadow_h=0.02, ang=0.0)
    poly(cv, P, [(-0.2, 0.18), (0.0, 0.18), (0.0, 0.26), (-0.2, 0.26)], C(.12, .09, .07))               # sawpit
    blob(cv, P, -0.1, 0.22, 0.13, 0.014, C(.62, .44, .26), lift=0.02)
    box(cv, P, 0.28, 0.3, 0.12, 0.07, 0.03, C(.50, .36, .22), C(.50, .36, .22), kind="flat")             # cart
    for du in (-0.04, 0.04):
        blob(cv, P, 0.28 + du, 0.34, 0.025, 0.012, C(.30, .22, .14), flat=True)
    smoke(cv, P, -0.3, -0.24, 0.1, n=4, alpha=0.35)
    return P


def imp_trading_post(cv, P, rng):
    P = patch(cv, P, 0.58, C(.66, .58, .44), 122, halo=0.3)
    for u, v, w, d in [(-0.28, -0.3, 0.16, 0.1), (0.26, -0.32, 0.14, 0.12), (-0.32, 0.2, 0.12, 0.1)]:
        box(cv, P, u, v, w, d, 0.05, C(.82, .76, .64), C(.58, .30, .18))
    for (u, v), (c1, c2) in zip([(-0.06, -0.06), (0.1, -0.04), (0.02, 0.1), (0.2, 0.12)],
                               [(C(.85, .15, .15), C(.95, .95, .9)), (C(.15, .35, .75), C(.95, .95, .9)),
                                (C(.95, .7, .1), C(.85, .3, .1)), (C(.2, .6, .3), C(.95, .95, .9))]):
        for k in range(4):
            poly(cv, P, [(u - 0.06 + k * 0.03, v - 0.04), (u - 0.03 + k * 0.03, v - 0.04), (u - 0.03 + k * 0.03, v + 0.04),
                         (u - 0.06 + k * 0.03, v + 0.04)], c1 if k % 2 == 0 else c2, lift=0.05)
        shadow(cv, P, u, v, 0.07, 0.05, 0.05)
    cyl(cv, P, 0.3, 0.22, 0.035, 0.03, C(.6, .58, .55), C(.15, .3, .4))
    return P


def star_points(n, r_out, r_in, rot=0.0):
    pts = []
    for k in range(2 * n):
        a = rot + k * math.pi / n
        r = r_out if k % 2 == 0 else r_in
        pts.append((r * math.cos(a), r * 0.95 * math.sin(a)))
    return pts


def fill_star(cv, P, pts, col, lift=0.0):
    for i in range(len(pts)):
        poly(cv, P, [(0.0, 0.0), pts[i], pts[(i + 1) % len(pts)]], col, lift=lift)


def imp_fort(cv, P, rng):
    outer = star_points(5, 0.66, 0.42, -math.pi / 2)
    fill_star(cv, P, star_points(5, 0.74, 0.5, -math.pi / 2), C(.20, .26, .16))
    fill_star(cv, P, outer, C(.36, .46, .22))
    fill_star(cv, P, outer, C(.42, .52, .26), lift=0.04)
    fill_star(cv, P, star_points(5, 0.5, 0.32, -math.pi / 2), C(.56, .48, .34), lift=0.04)
    box(cv, P, -0.1, -0.06, 0.18, 0.08, 0.05, C(.75, .70, .62), C(.40, .36, .32), kind="gable")
    box(cv, P, 0.12, 0.08, 0.14, 0.08, 0.045, C(.75, .70, .62), C(.40, .36, .32), kind="gable")
    flag(cv, P, 0.1, -0.15, 0.25, C(.85, .15, .15))
    return P


def imp_citadel(cv, P, rng):
    fill_star(cv, P, star_points(6, 0.86, 0.6, 0.0), C(.22, .38, .48))
    outer = star_points(6, 0.76, 0.52, 0.0)
    fill_star(cv, P, outer, C(.48, .46, .42))
    fill_star(cv, P, outer, C(.62, .60, .56), lift=0.07)
    fill_star(cv, P, star_points(6, 0.6, 0.42, 0.0), C(.52, .48, .40), lift=0.07)
    box(cv, P, 0.0, 0.02, 0.2, 0.18, 0.2, C(.66, .64, .60), C(.42, .40, .40), kind="flat")
    for u, v in [(-0.09, -0.07), (0.09, -0.07), (-0.09, 0.11), (0.09, 0.11)]:
        cyl(cv, P, u, v, 0.03, 0.26, C(.66, .64, .60), C(.45, .43, .42))
    flag(cv, P, 0.0, 0.0, 0.42, C(.85, .15, .15))
    return P


def imp_academy(cv, P, rng):
    P = patch(cv, P, 0.66, C(.66, .62, .54), 131, halo=0.3)
    poly(cv, P, [(-0.2, -0.12), (0.2, -0.12), (0.2, 0.2), (-0.2, 0.2)], C(.34, .56, .24))
    seg(cv, P, [(0.0, -0.12), (0.0, 0.2)], 0.03, C(.80, .76, .68))
    for u, v, w, d in [(-0.38, 0.0, 0.14, 0.4), (0.38, 0.0, 0.14, 0.4), (0.0, 0.36, 0.5, 0.12)]:
        box(cv, P, u, v, w, d, 0.07, C(.80, .70, .55), C(.36, .40, .46))
    box(cv, P, 0.0, -0.36, 0.3, 0.14, 0.08, C(.88, .86, .82), C(.55, .30, .2))
    cyl(cv, P, 0.0, -0.36, 0.08, 0.12, C(.90, .90, .88), C(.9, .9, .9))
    blob(cv, P, 0.0, -0.36, 0.08, 0.075, C(.95, .95, .96), lift=0.13, gloss=0.4)
    seg(cv, P, [(0.0, -0.36 - K * 0.15), (0.06, -0.36 - K * 0.2)], 0.012, C(.3, .3, .35))
    P = trees(cv, P, [(-0.12, -0.02), (0.12, 0.08)], rng, r=(0.045, 0.05), ht=(0.07, 0.08))
    return P


def imp_landmark(cv, P, rng):
    P = patch(cv, P, 0.66, C(.36, .52, .24), 132, halo=0.3)
    d = np.hypot(cv.u, cv.v * 1.08)
    P = over(P, premult(ground(cv, C(.86, .82, .74), 133, 0.1), smooth(0.42, 0.4, d) * footprint(cv)))
    for k in range(10):
        a = k * 2 * math.pi / 10
        cyl(cv, P, 0.36 * math.cos(a), 0.33 * math.sin(a), 0.022, 0.09, C(.92, .90, .86), C(.96, .95, .93))
    box(cv, P, 0.0, 0.0, 0.1, 0.1, 0.03, C(.80, .78, .74), C(.86, .84, .80), kind="flat")
    shadow(cv, P, 0.0, 0.0, 0.03, 0.03, 0.5, alpha=0.4)
    poly(cv, P, [(-0.03, 0.03), (0.03, 0.03), (0.02, 0.03 - K * 0.5), (-0.02, 0.03 - K * 0.5)], C(.86, .84, .80))
    poly(cv, P, [(-0.03, 0.03), (-0.005, 0.03), (-0.005, 0.03 - K * 0.5), (-0.02, 0.03 - K * 0.5)], C(.96, .95, .93))
    poly(cv, P, [(-0.02, 0.03 - K * 0.5), (0.02, 0.03 - K * 0.5), (0.0, 0.03 - K * 0.58)], C(.95, .80, .35))
    return P


def imp_manufactory(cv, P, rng):
    P = patch(cv, P, 0.7, C(.50, .48, .46), 134, halo=0.3)
    for k in range(5):
        u0 = -0.42 + k * 0.12
        poly(cv, P, [(u0, -0.25), (u0 + 0.12, -0.25), (u0 + 0.12, 0.2), (u0, 0.2)], C(.40, .42, .46), lift=0.1)
        poly(cv, P, [(u0, -0.25), (u0 + 0.05, -0.25), (u0 + 0.05, 0.2), (u0, 0.2)], C(.70, .80, .88), lift=0.1)
    poly(cv, P, [(-0.42, 0.2), (0.18, 0.2), (0.18, 0.2 - K * 0.1), (-0.42, 0.2 - K * 0.1)], C(.55, .30, .22))
    shadow(cv, P, -0.12, 0.0, 0.3, 0.22, 0.1, alpha=0.3)
    for u, v in [(0.3, -0.35), (0.42, -0.22)]:
        cyl(cv, P, u, v, 0.035, 0.42, C(.55, .30, .22), C(.2, .2, .2))
        smoke(cv, P, u, v, 0.48, n=6, col=C(.70, .70, .72), size=0.04, drift=0.035, alpha=0.55)
    for u, v in scatter(rng, 6, 0.12, 0.06, center=(0.35, 0.25)):
        box(cv, P, u, v, 0.05, 0.05, 0.04, C(.55, .40, .25), C(.62, .48, .30), kind="flat")
    return P


def imp_customs_house(cv, P, rng):
    P = patch(cv, P, 0.68, C(.62, .58, .52), 135, halo=0.3)
    box(cv, P, -0.15, -0.3, 0.42, 0.18, 0.1, C(.84, .80, .72), C(.20, .42, .62), kind="hip")
    flag(cv, P, -0.15, -0.3, 0.3, C(.95, .75, .15))
    cols = [C(.80, .20, .15), C(.15, .45, .70), C(.95, .70, .15), C(.20, .55, .30)]
    for i in range(3):
        for j in range(3):
            u, v = 0.0 + i * 0.13, 0.0 + j * 0.1
            box(cv, P, u, v, 0.11, 0.07, 0.05 + 0.03 * ((i + j) % 2), cols[(i * 3 + j) % 4], cols[(i * 3 + j) % 4] * 1.15, kind="flat")
    for u, v in [(-0.35, 0.05), (-0.28, 0.12), (-0.35, 0.18)]:
        cyl(cv, P, u, v, 0.035, 0.06, C(.50, .32, .18), C(.42, .28, .16))
    return P


def imp_holy_site(cv, P, rng):
    P = patch(cv, P, 0.66, C(.38, .52, .26), 136, halo=0.3)
    sq = [(-0.4, -0.38), (0.4, -0.38), (0.4, 0.38), (-0.4, 0.38)]
    poly(cv, P, sq, C(.86, .82, .74))
    seg(cv, P, sq + [sq[0]], 0.035, C(.74, .66, .56), lift=0.04)
    poly(cv, P, [(-0.03, 0.38), (0.03, 0.38), (0.03, 0.7), (-0.03, 0.7)], C(.82, .76, .64))
    box(cv, P, 0.0, 0.0, 0.32, 0.32, 0.06, C(.94, .92, .88), C(.92, .90, .86), kind="flat")
    blob(cv, P, 0.0, 0.0, 0.13, 0.12, C(.96, .95, .92), lift=0.1, gloss=0.3)
    seg(cv, P, [(0.0, -K * 0.18), (0.0, -K * 0.34)], 0.022, C(.95, .75, .2))
    blob(cv, P, 0.0, 0.0, 0.022, 0.022, C(1.0, .82, .25), lift=0.35, gloss=0.8)
    P = trees(cv, P, [(-0.3, -0.28), (0.3, -0.28), (-0.3, 0.28), (0.3, 0.28)], rng, r=(0.05, 0.055), ht=(0.07, 0.08))
    return P


def imp_moai(cv, P, rng):
    poly(cv, P, [(-0.55, -0.12), (0.55, -0.12), (0.55, 0.0), (-0.55, 0.0)], C(.42, .40, .38), lift=0.03)
    poly(cv, P, [(-0.55, 0.0), (0.55, 0.0), (0.55, -K * 0.03), (-0.55, -K * 0.03)], C(.32, .30, .28))
    for k in range(6):
        u = -0.45 + k * 0.18
        v = -0.06
        h = 0.3 + 0.04 * (k % 2)
        shadow(cv, P, u, v, 0.05, 0.03, h, alpha=0.45)
        poly(cv, P, [(u - 0.045, v), (u + 0.045, v), (u + 0.04, v - K * h), (u - 0.04, v - K * h)], C(.34, .31, .29), lift=0.03)
        poly(cv, P, [(u - 0.045, v), (u - 0.01, v), (u - 0.01, v - K * h), (u - 0.04, v - K * h)], C(.46, .42, .38), lift=0.03)
        blob(cv, P, u, v, 0.045, 0.03, C(.38, .35, .32), lift=0.03 + h, flat=True)
        poly(cv, P, [(u - 0.03, v - K * h * 0.7), (u + 0.03, v - K * h * 0.7), (u + 0.03, v - K * h * 0.62), (u - 0.03, v - K * h * 0.62)],
             C(.15, .14, .13), lift=0.03)
    return P


def imp_terrace_farm(cv, P, rng):
    r = np.hypot(cv.u, (cv.v - 0.05) * 1.1)
    n = fbm(cv.a, cv.b, 4, 3, 141)
    rr = r + (n - 0.5) * 0.08
    level = np.floor(rr / 0.11)
    h = -0.06 * level
    rgb = ramp((level % 3) / 2, [(0, C(.40, .54, .20)), (0.5, C(.50, .58, .24)), (1, C(.36, .48, .18))])
    edge = np.abs(rr / 0.11 - np.round(rr / 0.11))
    rgb = mix(rgb, C(.50, .40, .28), smooth(0.12, 0.04, edge))
    rgb = rt.lit(rgb, rt.shade(h, cv.spx, 1.6), lo=0.5) * rt.fine(cv, 142)
    P = over(P, premult(rgb, smooth(0.86, 0.76, rr) * footprint(cv)))
    box(cv, P, 0.0, 0.05, 0.1, 0.08, 0.04, C(.80, .76, .66), C(.55, .45, .30))
    return P


def imp_polder(cv, P, rng):
    m = smooth(0.0, 0.06, cv.edge) * footprint(cv)
    x, y = cv.u * 3.2, cv.v * 2.6
    pick = (np.floor(x) * 5 + np.floor(y) * 3) % 3
    rgb = ramp(pick / 2, [(0, C(.36, .54, .20)), (0.5, C(.48, .60, .26)), (1, C(.62, .62, .30))]) * rt.fine(cv, 151)
    canal = smooth(0.08, 0.03, np.minimum(np.abs(x - np.round(x)), np.abs(y - np.round(y))))
    rgb = mix(rgb, C(.20, .40, .50), canal)
    P = over(P, premult(rgb, m))
    seg(cv, P, [(-0.6, 0.62), (0.6, 0.62)], 0.05, C(.40, .52, .26), lift=0.03)
    u, v = 0.3, -0.25
    cyl(cv, P, u, v, 0.05, 0.22, C(.55, .40, .30), C(.35, .25, .20))
    for k in range(4):
        a = 0.6 + k * math.pi / 2
        seg(cv, P, [(u, v - K * 0.22), (u + 0.17 * math.cos(a), v - K * 0.22 + 0.17 * math.sin(a))], 0.022, C(.90, .88, .82), lift=0.0)
    return P


def imp_ancient_ruins(cv, P, rng):
    P = patch(cv, P, 0.52, C(.56, .52, .40), 161, halo=0.25)
    for u, v, w, d in [(-0.2, -0.15, 0.24, 0.18), (0.18, -0.12, 0.16, 0.2), (0.0, 0.18, 0.3, 0.14)]:
        pts = [(u - w / 2, v - d / 2), (u + w / 2, v - d / 2), (u + w / 2, v + d / 2), (u - w / 2, v + d / 2)]
        for i in range(4):
            if rng.random() < 0.8:
                a, b = pts[i], pts[(i + 1) % 4]
                seg(cv, P, [a, (a[0] + (b[0] - a[0]) * rng.uniform(0.5, 1), a[1] + (b[1] - a[1]) * rng.uniform(0.5, 1))],
                    0.03, C(.86, .82, .70), lift=0.035)
    for u, v in [(-0.32, 0.25), (0.3, 0.12)]:
        cyl(cv, P, u, v, 0.03, 0.12, C(.88, .85, .76), C(.92, .90, .84))
    blob(cv, P, 0.25, 0.28, 0.1, 0.025, C(.86, .82, .72), ang=0.4, lift=0.02, shadow_h=0.02)
    P = trees(cv, P, [(-0.05, -0.02), (0.36, -0.3)], rng, r=(0.04, 0.05), ht=(0.06, 0.08))
    return P


def imp_city_ruins(cv, P, rng):
    P = patch(cv, P, 0.7, C(.48, .46, .42), 162, halo=0.25)
    for i in range(-3, 4):
        for j in range(-3, 4):
            u, v = i * 0.16, j * 0.15
            if u * u + v * v > 0.36 or rng.random() < 0.3:
                continue
            w = rng.uniform(0.08, 0.12)
            pts = [(u - w / 2, v - w / 2), (u + w / 2, v - w / 2), (u + w / 2, v + w / 2), (u - w / 2, v + w / 2)]
            for k in range(4):
                if rng.random() < 0.7:
                    seg(cv, P, [pts[k], pts[(k + 1) % 4]], 0.018, C(.62, .60, .56), lift=rng.uniform(0.02, 0.06))
            if rng.random() < 0.5:
                blob(cv, P, u + 0.02, v + 0.02, 0.03, 0.025, C(.55, .52, .48), rough=0.3, lift=0.01)
    return P


def imp_barbarian(cv, P, rng):
    P = patch(cv, P, 0.6, C(.40, .32, .22), 171, halo=0.3)
    for k in range(28):
        a = k * 2 * math.pi / 28
        u, v = 0.55 * math.cos(a), 0.5 * math.sin(a)
        seg(cv, P, [(u, v), (u, v - K * 0.09)], 0.03, C(.32, .22, .13))
    for u, v in [(-0.22, -0.18), (0.2, -0.22), (-0.25, 0.18), (0.25, 0.16)]:
        cone(cv, P, u, v, 0.1, 0.14, C(.22, .16, .12))
    blob(cv, P, 0.0, 0.02, 0.05, 0.045, C(1.0, .45, .1), gloss=0.6)
    smoke(cv, P, 0.0, 0.02, 0.06, n=6, col=C(.45, .45, .45), size=0.035, drift=0.03, alpha=0.6)
    for u, v in [(-0.45, -0.3), (0.45, -0.3)]:
        flag(cv, P, u, v, 0.3, C(.75, .05, .05))
    return P


FULL_IMPROVEMENTS = {"Farm": imp_farm, "Lumber mill": imp_lumber_mill, "Trading post": imp_trading_post,
                     "Fort": imp_fort, "Citadel": imp_citadel, "Academy": imp_academy, "Landmark": imp_landmark,
                     "Manufactory": imp_manufactory, "Customs house": imp_customs_house, "Holy site": imp_holy_site,
                     "Moai": imp_moai, "Terrace farm": imp_terrace_farm, "Polder": imp_polder,
                     "Ancient ruins": imp_ancient_ruins, "City ruins": imp_city_ruins,
                     "Barbarian encampment": imp_barbarian}


# ============================================================================ era improvements
# Unciv shows a city's tiles with "<image>-<era>" when there is one, trying the owner's era first and
# then each earlier one, and the plain image only when none matches. So an era image covers the eras
# after it until the next one, and the base look is repeated under the era it belongs to, or the
# eras after an earlier image would never get it back.
def hut(cv, P, u, v, r=0.055, roof=C(.70, .58, .32), wall=C(.62, .50, .36)):
    """A round hut with a conical thatched roof."""
    shadow(cv, P, u, v, r, r * 0.9, 0.06)
    cyl(cv, P, u, v, r * 0.85, 0.025, wall, wall * 0.9, shadow_a=0.0)
    cone(cv, P, u, v - K * 0.025, r, 0.07, roof, shadow_a=0.0)


def plots(cv, P, seed, cells=7, palette=None, wall=None, rmax=0.84, center_clear=0.0):
    """Small irregular fields, like a Voronoi patchwork, separated by low walls or hedges."""
    rng = np.random.default_rng(seed)
    pts = np.array([(rng.uniform(-1, 1), rng.uniform(-0.9, 0.9)) for _ in range(cells * cells)])
    uu, vv = cv.u[..., None], cv.v[..., None]
    wa, wb = warp(cv.a, cv.b, 0.08, 6, seed)
    d = np.hypot(uu + (wa - cv.a)[..., None] * 0.6 - pts[:, 0], vv + (wb - cv.b)[..., None] * 0.6 - pts[:, 1])
    order = np.sort(d, axis=-1)
    idx = np.argmin(d, axis=-1)
    edge = order[..., 1] - order[..., 0]
    palette = palette or [C(.56, .44, .28), C(.50, .54, .26), C(.66, .58, .34), C(.44, .50, .22), C(.60, .48, .32)]
    rgb = np.zeros(cv.shape + (3,))
    for i in range(len(pts)):
        rgb[idx == i] = palette[i % len(palette)] * rng.uniform(0.9, 1.08)
    ang = (idx % 3) * 0.9
    rows = np.sin((cv.u * np.cos(ang) + cv.v * np.sin(ang)) * 70)
    rgb = rgb * (0.9 + 0.1 * rows)[..., None] * rt.fine(cv, seed + 3, 0.14)
    rgb = mix(rgb, wall if wall is not None else C(.26, .30, .16), smooth(0.03, 0.012, edge) * 0.85)
    n = fbm(cv.a, cv.b, 5, 3, seed + 4)
    dm = np.hypot(cv.u, cv.v * 1.05) + (n - 0.5) * 0.12
    m = smooth(rmax, rmax - 0.12, dm) * footprint(cv)
    if center_clear:
        m = m * smooth(center_clear, center_clear + 0.06, dm)
    return over(P, premult(rgb, m))


def farm_ancient(cv, P, rng):
    P = plots(cv, P, 501, cells=6, wall=C(.42, .36, .26), rmax=0.8)
    seg(cv, P, [(-0.8, 0.35), (-0.3, 0.1), (0.1, -0.2), (0.35, -0.45)], 0.035, C(.62, .52, .38), alpha=0.8)   # a dirt path
    for u, v in [(0.32, -0.55), (0.48, -0.48), (0.42, -0.36)]:
        hut(cv, P, u, v, 0.07)
    seg(cv, P, [(0.25, -0.62), (0.6, -0.62), (0.62, -0.3), (0.26, -0.28), (0.25, -0.62)], 0.012, C(.42, .30, .18), lift=0.02)
    P = trees(cv, P, [(0.12, -0.6), (0.66, -0.2), (-0.62, -0.4)], rng, r=(0.045, 0.06), ht=(0.06, 0.08))
    for u, v, ang in [(-0.2, 0.32, 0.3), (-0.05, 0.38, 0.4)]:
        animal(cv, P, u, v, ang, 0.045, 0.026, C(.52, .38, .24), head=C(.40, .28, .18))
    return P


def farm_industrial(cv, P, rng):
    """Big square fields, hedgerows, a red barn, a farmhouse, a silo and a wind pump."""
    ang = 0.2
    x = cv.u * math.cos(ang) - cv.v * math.sin(ang)
    y = cv.u * math.sin(ang) + cv.v * math.cos(ang)
    fx, fy = np.floor(x * 1.6 + 0.5), np.floor(y * 1.7 + 0.3)
    pick = (fx * 3 + fy * 5) % 4
    crops = [C(.80, .70, .34), C(.42, .58, .22), C(.62, .50, .32), C(.70, .72, .34)]
    rgb = np.zeros(cv.shape + (3,))
    for i, c in enumerate(crops):
        rgb[pick == i] = c
    stripes = np.sin(x * 2 * math.pi * 18 + pick)
    rgb = rgb * (0.9 + 0.1 * stripes)[..., None] * rt.fine(cv, 601, 0.1)
    ex = np.abs(x * 1.6 + 0.5 - np.round(x * 1.6 + 0.5))
    ey = np.abs(y * 1.7 + 0.3 - np.round(y * 1.7 + 0.3))
    hedge = smooth(0.035, 0.015, np.minimum(ex, ey))
    rgb = mix(rgb, C(.18, .30, .12), hedge)
    n = fbm(cv.a, cv.b, 5, 3, 602)
    d = np.hypot(cv.u, cv.v * 1.05) + (n - 0.5) * 0.1
    P = over(P, premult(rgb, smooth(0.9, 0.8, d) * footprint(cv)))
    poly(cv, P, [(0.12, -0.7), (0.72, -0.7), (0.72, -0.24), (0.12, -0.24)], C(.56, .52, .40), alpha=0.9)   # farmyard
    box(cv, P, 0.30, -0.52, 0.2, 0.13, 0.07, C(.66, .16, .12), C(.36, .30, .28), kind="gable")              # barn
    box(cv, P, 0.58, -0.56, 0.11, 0.09, 0.05, C(.92, .90, .84), C(.30, .30, .34))                           # farmhouse
    cyl(cv, P, 0.48, -0.36, 0.045, 0.16, C(.82, .80, .74), C(.55, .54, .52))                                # silo
    blob(cv, P, 0.48, -0.36, 0.045, 0.042, C(.55, .54, .52), lift=0.165)
    u, v, h = 0.2, -0.32, 0.26                                                                               # wind pump
    for du in (-0.025, 0.025):
        seg(cv, P, [(u + du, v), (u, v - K * h)], 0.008, C(.40, .40, .42))
    for k in range(8):
        a2 = k * math.pi / 4
        seg(cv, P, [(u, v - K * h), (u + 0.06 * math.cos(a2), v - K * h + 0.06 * math.sin(a2))], 0.01, C(.85, .85, .82))
    for u2, v2 in [(0.36, -0.30), (0.62, -0.32)]:
        animal(cv, P, u2, v2, 0.2, 0.035, 0.02, C(.30, .26, .22))
    P = trees(cv, P, [(0.08, -0.62), (0.74, -0.42), (-0.5, -0.1), (-0.12, 0.3)], rng, r=(0.04, 0.055), ht=(0.06, 0.08))
    return P


def pivot(cv, P, u, v, r, crop, seed):
    """A centre-pivot irrigated field: a green disc with wheel tracks and the pivot arm."""
    d = np.hypot(cv.u - u, cv.v - v)
    th = np.arctan2(cv.v - v, cv.u - u)
    rings = 0.92 + 0.08 * np.sin(d * 140)
    harvested = smooth(0.0, 0.3, np.sin(th * 1 + seed)) * 0.5
    rgb = mix(col_arr(crop, cv.shape), C(.72, .64, .40), harvested) * rings[..., None] * rt.fine(cv, seed, 0.1)
    P = over(P, premult(rgb, smooth(r, r - 0.012, d) * footprint(cv)))
    a = seed * 1.7
    seg(cv, P, [(u, v), (u + r * 0.97 * math.cos(a), v + r * 0.97 * math.sin(a))], 0.01, C(.85, .86, .88), lift=0.015)
    blob(cv, P, u, v, 0.015, 0.015, C(.6, .6, .62), lift=0.02)
    return P


def farm_modern(cv, P, rng):
    n = fbm(cv.a, cv.b, 5, 3, 701)
    d = np.hypot(cv.u, cv.v * 1.05) + (n - 0.5) * 0.1
    P = over(P, premult(ground(cv, C(.66, .60, .42), 702, 0.15), smooth(0.92, 0.82, d) * footprint(cv)))
    for (u, v, r, crop, s) in [(-0.38, -0.1, 0.36, C(.34, .56, .20), 1), (0.32, 0.36, 0.32, C(.46, .62, .22), 2),
                               (-0.2, 0.52, 0.2, C(.60, .62, .26), 3), (0.62, -0.05, 0.2, C(.40, .58, .22), 4)]:
        P = pivot(cv, P, u, v, r, crop, s)
    poly(cv, P, [(0.08, -0.72), (0.66, -0.72), (0.66, -0.36), (0.08, -0.36)], C(.62, .62, .60))           # concrete yard
    for k in range(4):                                                                                   # grain bins
        u = 0.16 + k * 0.1
        cyl(cv, P, u, -0.6, 0.045, 0.1, C(.80, .82, .84), C(.70, .72, .74))
        cone(cv, P, u, -0.6 - K * 0.1, 0.047, 0.04, C(.82, .84, .86), shadow_a=0.0)
    box(cv, P, 0.36, -0.44, 0.24, 0.09, 0.06, C(.70, .72, .74), C(.40, .48, .56), kind="gable")           # machine shed
    box(cv, P, 0.58, -0.44, 0.08, 0.05, 0.05, C(.84, .82, .78), C(.32, .32, .34))
    box(cv, P, 0.16, -0.42, 0.06, 0.035, 0.03, C(.20, .55, .25), C(.20, .55, .25), kind="flat")           # a tractor
    return P


def mine_ancient(cv, P, rng):
    """Adits dug into the ground with timber frames, a windlass over a shaft, spoil heaps and huts."""
    P = ring_ground(cv, P, C(.54, .46, .36), 801, alpha=0.6)
    for u, v in [(-0.48, -0.42), (0.5, -0.4)]:                              # adits, with timber posts and a lintel
        blob(cv, P, u, v, 0.075, 0.05, C(.08, .06, .05), flat=True)
        for du in (-0.07, 0.07):
            seg(cv, P, [(u + du, v + 0.02), (u + du, v + 0.02 - K * 0.07)], 0.014, C(.44, .30, .18))
        seg(cv, P, [(u - 0.085, v + 0.02 - K * 0.07), (u + 0.085, v + 0.02 - K * 0.07)], 0.016, C(.44, .30, .18))
    for u, v, s in [(-0.66, 0.08, 0.9), (0.66, 0.12, 0.8), (-0.3, 0.6, 0.7), (0.3, 0.62, 0.65)]:
        cone(cv, P, u, v, 0.1 * s, 0.08 * s, C(.56, .48, .38))
    u, v, h = 0.0, -0.62, 0.14                                              # windlass
    blob(cv, P, u, v + 0.03, 0.06, 0.04, C(.06, .05, .04), flat=True)
    for du in (-0.06, 0.06):
        seg(cv, P, [(u + du - 0.02, v + 0.03), (u + du, v - K * h)], 0.012, C(.50, .34, .20))
        seg(cv, P, [(u + du + 0.02, v + 0.03), (u + du, v - K * h)], 0.012, C(.50, .34, .20))
    seg(cv, P, [(u - 0.07, v - K * h), (u + 0.07, v - K * h)], 0.02, C(.42, .28, .16))
    seg(cv, P, [(u, v - K * h), (u, v + 0.02)], 0.004, C(.80, .70, .50))
    for u2, v2 in [(-0.25, -0.66), (0.26, -0.66)]:
        hut(cv, P, u2, v2, 0.05)
    for u2 in (-0.14, 0.14):
        blob(cv, P, u2, 0.55, 0.03, 0.022, C(.62, .48, .28), lift=0.02, shadow_h=0.02)    # baskets of ore
    return P


def mine_modern(cv, P, rng):
    """An open pit: stepped benches on the far side, a haul road, big trucks, a conveyor and a plant."""
    r = np.hypot(cv.u, cv.v * 1.08)
    steps = np.floor(np.clip((r - (R0 + 0.06)) / 0.07, 0, 5))
    h = -0.035 * steps
    base = mix(C(.62, .52, .42), C(.74, .66, .56), smooth(0.3, 0.7, fbm(cv.a, cv.b, 9, 3, 901)))
    lit = rt.lit(base, rt.shade(h, cv.spx, 2.2), lo=0.45)
    bench = np.abs((r - (R0 + 0.06)) / 0.07 - np.round((r - (R0 + 0.06)) / 0.07))
    lit = mix(lit, C(.50, .42, .34), smooth(0.08, 0.02, bench) * 0.3)
    m = ring_mask(cv, R0 + 0.04, 0.88, 901)
    P = over(P, premult(lit, m))
    seg(cv, P, [(0.86, 0.12), (0.74, 0.34), (0.56, 0.5), (0.36, 0.56)], 0.05, C(.58, .50, .42), alpha=0.9)   # haul ramp
    for u, v, ang in [(0.66, 0.36, 2.3), (-0.6, -0.3, -0.6)]:              # haul trucks
        ca, sa = math.cos(ang), math.sin(ang)
        L, W = 0.09, 0.05
        shadow(cv, P, u, v, L, W, 0.06, ang=ang)
        poly(cv, P, [(u + L * ca - W * sa, v + L * sa + W * ca), (u + L * ca + W * sa, v + L * sa - W * ca),
                     (u - L * ca + W * sa, v - L * sa - W * ca), (u - L * ca - W * sa, v - L * sa + W * ca)],
             C(.95, .72, .12), lift=0.045)
        poly(cv, P, [(u - 0.01 * ca - 0.04 * sa, v - 0.01 * sa + 0.04 * ca), (u - 0.01 * ca + 0.04 * sa, v - 0.01 * sa - 0.04 * ca),
                     (u - 0.08 * ca + 0.04 * sa, v - 0.08 * sa - 0.04 * ca), (u - 0.08 * ca - 0.04 * sa, v - 0.08 * sa + 0.04 * ca)],
             C(.58, .48, .40), lift=0.05)
    seg(cv, P, [(-0.15, -0.5), (0.0, -0.62), (0.3, -0.7)], 0.025, C(.35, .35, .38), lift=0.08)          # conveyor
    seg(cv, P, [(-0.15, -0.5), (0.0, -0.62), (0.3, -0.7)], 0.01, C(.55, .45, .35), lift=0.085)
    box(cv, P, 0.42, -0.62, 0.22, 0.12, 0.12, C(.70, .72, .74), C(.42, .46, .50), kind="flat")          # processing plant
    cyl(cv, P, 0.58, -0.5, 0.05, 0.14, C(.84, .84, .82), C(.6, .6, .6))
    cone(cv, P, -0.5, -0.6, 0.1, 0.09, C(.60, .52, .44))                                                # stockpiles
    cone(cv, P, -0.68, -0.22, 0.09, 0.08, C(.28, .26, .26))
    return P


def lumber_industrial(cv, P, rng):
    """A steam sawmill: brick mill with a tall chimney, rows of sawn planks and a rail siding."""
    P = patch(cv, P, 0.66, C(.44, .40, .34), 1101, halo=0.3)
    seg(cv, P, [(-0.66, 0.42), (0.66, 0.42)], 0.03, C(.32, .30, .28))
    seg(cv, P, [(-0.66, 0.41), (0.66, 0.41)], 0.006, C(.75, .75, .78))
    seg(cv, P, [(-0.66, 0.43), (0.66, 0.43)], 0.006, C(.75, .75, .78))
    for k in range(3):
        box(cv, P, -0.4 + k * 0.17, 0.41, 0.14, 0.06, 0.04, C(.30, .26, .24), C(.36, .32, .28), kind="flat")
    box(cv, P, -0.18, -0.24, 0.42, 0.16, 0.1, C(.62, .32, .22), C(.34, .34, .36), kind="gable")
    cyl(cv, P, 0.08, -0.36, 0.03, 0.42, C(.54, .28, .20), C(.15, .15, .15))
    smoke(cv, P, 0.08, -0.36, 0.45, n=7, col=C(.75, .75, .75), size=0.045, drift=0.04, alpha=0.55)
    for row in range(3):                                                                                  # plank stacks
        for col in range(3):
            u, v = 0.18 + col * 0.14, -0.06 + row * 0.13
            box(cv, P, u, v, 0.11, 0.07, 0.04 + 0.01 * ((row + col) % 2), C(.70, .56, .38), C(.86, .74, .54), kind="flat")
    for k in range(5):
        blob(cv, P, -0.5 + k * 0.08, 0.16, 0.1, 0.02, C(.50, .34, .20), lift=0.02, shadow_h=0.02)
    return P


def lumber_modern(cv, P, rng):
    """A log yard: huge decks of logs, a log stacker, metal sheds and a truck."""
    P = patch(cv, P, 0.7, C(.50, .46, .40), 1201, halo=0.3)
    for row in range(3):
        for k in range(2):
            u, v = -0.42 + k * 0.42, -0.02 + row * 0.18
            shadow(cv, P, u, v, 0.18, 0.06, 0.06)
            for j in range(9):
                blob(cv, P, u - 0.16 + j * 0.04, v + 0.01 * (j % 2), 0.025, 0.06, C(.46, .32, .18), lift=0.03 + 0.012 * (j % 3))
                blob(cv, P, u - 0.16 + j * 0.04, v + 0.05, 0.02, 0.012, C(.84, .70, .48), lift=0.03, flat=True)
    box(cv, P, -0.1, -0.44, 0.44, 0.16, 0.1, C(.72, .74, .76), C(.36, .52, .62), kind="gable")
    box(cv, P, 0.36, -0.46, 0.18, 0.12, 0.08, C(.78, .78, .76), C(.56, .58, .60), kind="flat")
    u, v = 0.52, 0.1                                                                                     # log stacker
    shadow(cv, P, u, v, 0.06, 0.04, 0.08)
    box(cv, P, u, v, 0.08, 0.06, 0.06, C(.95, .70, .12), C(.95, .72, .12), kind="flat")
    seg(cv, P, [(u, v - K * 0.06), (u - 0.12, v - K * 0.12)], 0.016, C(.95, .70, .12), lift=0.0)
    box(cv, P, 0.3, 0.46, 0.18, 0.05, 0.04, C(.80, .20, .15), C(.80, .20, .15), kind="flat")             # truck
    for j in range(4):
        blob(cv, P, 0.26 + j * 0.035, 0.45, 0.014, 0.04, C(.46, .32, .18), lift=0.05)
    return P


ERA_IMPROVEMENTS = {   # improvement: [(era, drawing, fires when pillaged)]
    "Farm": [("Ancient era", farm_ancient, [(0.40, -0.48, 0.06, 1.0), (-0.3, 0.1, 0.0, 0.8), (0.15, 0.4, 0.0, 0.7)]),
             ("Medieval era", None, None),
             ("Industrial era", farm_industrial, [(0.30, -0.52, 0.07, 1.0), (0.58, -0.56, 0.05, 0.8), (-0.3, 0.2, 0.0, 0.8)]),
             ("Modern era", farm_modern, [(0.36, -0.44, 0.06, 1.0), (0.2, -0.6, 0.1, 0.7), (-0.38, -0.1, 0.0, 0.8)])],
    "Mine": [("Ancient era", mine_ancient, [(-0.25, -0.66, 0.04, 0.8), (0.26, -0.66, 0.04, 0.8), (0.0, -0.62, 0.08, 0.8)]),
             ("Industrial era", None, None),
             ("Modern era", mine_modern, [(0.42, -0.62, 0.1, 1.0), (-0.6, -0.3, 0.05, 0.8), (0.66, 0.36, 0.05, 0.8)])],
    "Lumber mill": [("Industrial era", lumber_industrial, [(-0.18, -0.24, 0.1, 1.1), (0.32, 0.06, 0.05, 0.9)]),
                    ("Modern era", lumber_modern, [(-0.1, -0.44, 0.1, 1.1), (-0.3, 0.12, 0.06, 0.9), (0.3, 0.3, 0.06, 0.8)])],
}


def wheat_farm(fn):
    """Wheat in the middle of a farm drawn by fn, the farm cleared round it. Unciv shows "Wheat+Farm"
    instead of the two images when both are on a tile. The wheat and the farm are kept as separate
    layers too (under, layer), so that pillaging burns the farm and leaves the wheat."""
    def layer(cv, P, rng):
        F = fn(cv, P, rng)
        n = fbm(cv.a, cv.b, 5, 3, 112)
        d = np.hypot(cv.u, cv.v * 1.05) + (n - 0.5) * 0.12
        return F * smooth(R0 - 0.02, R0 + 0.05, d)[..., None]

    def under(cv):
        return r_wheat(Zoom(cv, RES_ZOOM), cv.empty(), res_rng("Wheat"))

    def draw(cv, P, rng):
        return over(over(P, under(cv)), layer(cv, cv.empty(), rng))
    draw.layer, draw.under = layer, under
    return draw


def round_wheat(fires):
    """The same fires, moved out of the middle where the wheat is."""
    out = []
    for u, v, h, s in fires:
        r = math.hypot(u, v)
        k = max(1.0, 0.62 / max(r, 1e-6))
        out.append((u * k, v * k, h, s))
    return out


# ============================================================================ pillaged improvements
# Unciv tints a pillaged tile red unless the tileset has "<improvement>-Pillaged"; for a city's tile
# it then looks for "<improvement>-Pillaged-<era>" too, like any other image.
def flames(cv, P, u, v, h, rng, s=1.0):
    """A fire: a glow on the ground, licking flames and a column of dark smoke drifting east."""
    blob(cv, P, u, v, 0.16 * s, 0.13 * s, C(1.0, .42, .06), lift=h, alpha=0.3, flat=True)
    for k in range(7):
        du, dv = rng.uniform(-0.045, 0.045) * s, rng.uniform(-0.02, 0.02) * s
        r = rng.uniform(0.028, 0.042) * s
        lift = h + rng.uniform(0.0, 0.05) * s
        blob(cv, P, u + du, v + dv, r, r * 1.35, C(1.0, .45, .08), lift=lift, shade_lo=0.85)
        blob(cv, P, u + du, v + dv + 0.004, r * 0.55, r * 0.75, C(1.0, .85, .40), lift=lift + 0.01 * s, shade_lo=1.0)
    for k in range(9):
        r = (0.05 + 0.017 * k) * s
        blob(cv, P, u + 0.045 * k * s + rng.uniform(-0.01, 0.01), v - 0.012 * k, r, r * 0.95,
             C(.20, .19, .18) * (1 + 0.1 * k), lift=h + 0.08 * s + 0.085 * k * s, alpha=0.82 * (1 - k / 11), shade_lo=0.7)


def pillage(cv, P, rng, fires, clear=0.0, water=False, char=0.92, seed=7000, debris=12):
    """Burn an improvement: char and ash where the fire spread, the rest dulled and sooty, scorched
    ground (or an oil slick, at sea) round each fire, embers in the ashes, broken beams, and fires
    with smoke where it still burns. `clear` keeps the middle of the hex clear, for improvements
    drawn round a resource; char is how black the burnt parts get (stone only blackens a little)."""
    a = P[..., 3:4]
    rgb = np.where(a > 1e-6, P[..., :3] / np.maximum(a, 1e-6), 0.0)
    wa, wb = warp(cv.a, cv.b, 0.5, 2, seed)
    n = fbm(wa, wb, 3, 4, seed + 1)
    burn = smooth(0.42, 0.52, n)
    speck = fbm(cv.a, cv.b, 24, 2, seed + 2)
    lum = rgb.mean(-1, keepdims=True)
    ash = mix(C(.10, .085, .07), C(.24, .22, .20), smooth(0.45, 0.75, speck)) * (0.7 + 0.6 * lum)
    dull = mix(rgb, np.repeat(lum, 3, -1), 0.5) * np.array([0.76, 0.70, 0.64])
    rgb = mix(dull, ash, burn * char)
    front = smooth(0.40, 0.42, n) * smooth(0.45, 0.43, n) * smooth(0.45, 0.6, fbm(cv.a, cv.b, 9, 2, seed + 5))
    rgb = mix(rgb, C(.95, .36, .06), front * 0.5)                    # the edge of the burn still smoulders
    out = premult(rgb, a[..., 0])
    # scorch marks on the ground round each fire, under the drawing
    d_mid = np.hypot(cv.u, cv.v * 1.08)
    scorch = np.zeros(cv.shape)
    for u, v, h, s in fires:
        n = fbm(cv.a, cv.b, 7, 3, seed + int(u * 100) + int(v * 1000))
        d = np.hypot(cv.u - u, (cv.v - v) * 1.15) + (n - 0.5) * 0.12
        scorch = np.maximum(scorch, smooth(0.24 * s, 0.08 * s, d))
    scorch *= footprint(cv) * (smooth(clear, clear + 0.06, d_mid) if clear else 1.0)
    if water:
        ground_rgb = mix(C(.04, .05, .06), C(.20, .18, .30), smooth(0.6, 0.8, speck) * 0.6)
    else:
        ground_rgb = mix(C(.11, .09, .07), C(.24, .21, .18), smooth(0.45, 0.75, speck))
    out = over(premult(ground_rgb, scorch * (0.6 if water else 0.8)), out)
    # embers glowing in the ashes
    for _ in range(0 if water else 40):
        u, v = rng.uniform(-0.85, 0.85), rng.uniform(-0.75, 0.75)
        iy, ix = cv.px(u, v)[1], cv.px(u, v)[0]
        iy, ix = int(min(max(iy, 0), cv.shape[0] - 1)), int(min(max(ix, 0), cv.shape[1] - 1))
        if rt.hex_edge_dist(u, v) < 0.05 or (clear and math.hypot(u, v * 1.08) < clear + 0.04):
            continue
        if burn[iy, ix] > 0.5 and a[iy, ix, 0] > 0.5 or scorch[iy, ix] > 0.5:
            r = rng.uniform(0.006, 0.012)
            blob(cv, out, u, v, r * 2.2, r * 2.2, C(1.0, .4, .05), alpha=0.35, flat=True)
            blob(cv, out, u, v, r, r, C(1.0, .62, .18), flat=True)
    # broken, charred beams
    for u, v, h, s in fires:
        for _ in range(debris):
            ang = rng.uniform(0, math.pi)
            r, t = rng.uniform(0.05, 0.2) * s, rng.uniform(0, 2 * math.pi)
            cu, cvv = u + r * math.cos(t), v + r * math.sin(t) * 0.9
            if rt.hex_edge_dist(cu, cvv) < 0.06 or (clear and math.hypot(cu, cvv * 1.08) < clear + 0.04):
                continue
            ln = rng.uniform(0.03, 0.07)
            seg(cv, out, [(cu - ln * math.cos(ang), cvv - ln * math.sin(ang) * 0.6), (cu + ln * math.cos(ang), cvv + ln * math.sin(ang) * 0.6)],
                0.011, C(.30, .22, .14) if water else C(.12, .09, .07), lift=0.006)
    for u, v, h, s in sorted(fires, key=lambda f: f[1]):
        flames(cv, out, u, v, h, rng, s)
    return out


PILLAGE = {   # improvement: (fires as (u, v, height, size), radius kept clear for the resource, at sea, how black it burns)
    "Farm": ([(0.40, -0.47, 0.05, 1.0), (-0.35, 0.15, 0.0, 0.8), (0.1, 0.45, 0.0, 0.7)], 0.0, False, 0.92),
    "Lumber mill": ([(-0.15, -0.2, 0.07, 1.1), (0.1, 0.04, 0.04, 0.8)], 0.0, False, 0.92),
    "Trading post": ([(-0.28, -0.3, 0.05, 0.9), (0.26, -0.32, 0.05, 0.9), (0.06, 0.06, 0.04, 0.8)], 0.0, False, 0.92),
    "Pasture": ([(-0.5, -0.52, 0.05, 0.9), (0.55, -0.42, 0.04, 0.7)], R0 + 0.02, False, 0.92),
    "Camp": ([(-0.55, -0.4, 0.06, 0.8), (0.55, -0.38, 0.06, 0.8)], R0 + 0.02, False, 0.92),
    "Plantation": ([(0.0, -0.66, 0.05, 0.9), (0.6, 0.25, 0.04, 0.8), (-0.62, 0.1, 0.04, 0.8)], R0 + 0.02, False, 0.92),
    "Mine": ([(-0.4, -0.55, 0.07, 1.0), (0.4, -0.55, 0.05, 0.8)], R0 + 0.02, False, 0.92),
    "Quarry": ([(-0.55, -0.25, 0.04, 0.8), (0.58, 0.2, 0.05, 0.7)], R0 + 0.02, False, 0.92),
    "Fishing Boats": ([(-0.55, -0.42, 0.02, 0.7), (0.6, -0.36, 0.02, 0.7)], R0 + 0.02, True, 0.92),
    "Oil well": ([(0.6, -0.24, 0.08, 1.2), (-0.52, -0.42, 0.05, 0.8)], R0 + 0.02, False, 0.92),
    "Offshore Platform": ([(0.0, -0.62, 0.2, 1.2)], R0 + 0.02, True, 0.92),
    "Fort": ([(-0.1, -0.06, 0.09, 0.9), (0.12, 0.08, 0.09, 0.8)], 0.0, False, 0.92),
    "Citadel": ([(0.0, 0.02, 0.27, 1.0), (-0.4, 0.3, 0.07, 0.8)], 0.0, False, 0.92),
    "Academy": ([(0.0, -0.36, 0.2, 1.0), (0.38, 0.0, 0.07, 0.8), (-0.38, 0.1, 0.07, 0.7)], 0.0, False, 0.92),
    "Landmark": ([(0.0, 0.0, 0.03, 0.9), (0.3, 0.2, 0.05, 0.6)], 0.0, False, 0.5),
    "Manufactory": ([(-0.12, 0.0, 0.1, 1.1), (0.35, 0.25, 0.04, 0.8)], 0.0, False, 0.92),
    "Customs house": ([(-0.15, -0.3, 0.1, 1.0), (0.13, 0.1, 0.08, 0.9)], 0.0, False, 0.92),
    "Holy site": ([(0.0, 0.0, 0.06, 1.0), (-0.3, 0.28, 0.04, 0.7)], 0.0, False, 0.92),
    "Moai": ([(-0.2, 0.0, 0.03, 0.7), (0.3, 0.05, 0.03, 0.7)], 0.0, False, 0.3),
    "Terrace farm": ([(0.0, 0.05, 0.04, 0.9), (-0.4, 0.3, 0.0, 0.8), (0.35, -0.35, 0.0, 0.7)], 0.0, False, 0.92),
    "Polder": ([(0.3, -0.25, 0.2, 1.0), (-0.3, 0.2, 0.0, 0.8)], 0.0, False, 0.92),
}


def render_improvement(name, fn, era=None, pillage_as=None, seed_name=None, overhang=110):
    """Write Tiles/<name>[-<era>] and, with pillage_as (a PILLAGE entry), its burned image
    Tiles/<name>-Pillaged[-<era>]. seed_name picks the random layout: an era image that repeats the
    base look uses the base's, so it comes out identical and the atlas packer stores it once."""
    tail = "-" + era if era else ""
    rel, rel_p = "Tiles/" + name + tail, "Tiles/" + name + "-Pillaged" + tail
    if not (rt.wanted(rel) or pillage_as and rt.wanted(rel_p)):
        return
    seed_name = seed_name or name
    cv = canvas(IMP_W, overhang=overhang)
    P = fn(cv, cv.empty(), res_rng(seed_name))
    if rt.wanted(rel):
        finish(cv, P, rel)
    if pillage_as and rt.wanted(rel_p):
        layer = fn.layer(cv, cv.empty(), res_rng(seed_name)) if hasattr(fn, "layer") else P.copy()
        B = pillage(cv, layer, res_rng(seed_name + "-Pillaged"), *pillage_as, seed=7000 + sum(map(ord, seed_name)))
        finish(cv, over(fn.under(cv), B) if hasattr(fn, "under") else B, rel_p)


def render_improvements():
    for name, fn in {**RING_IMPROVEMENTS, **FULL_IMPROVEMENTS}.items():
        render_improvement(name, fn, pillage_as=PILLAGE.get(name))
    fires, clear, water, char = PILLAGE["Farm"]
    render_improvement("Wheat+Farm", wheat_farm(imp_farm), pillage_as=(round_wheat(fires), R0 + 0.02, water, char), seed_name="Farm")
    for name, eras in ERA_IMPROVEMENTS.items():
        base_fires, clear, water, char = PILLAGE[name]
        for era, fn, fires in eras:
            seed = name if fn is None else name + "-" + era
            fn, fires = fn or {**RING_IMPROVEMENTS, **FULL_IMPROVEMENTS}[name], fires or base_fires
            render_improvement(name, fn, era, (fires, clear, water, char), seed)
            if name == "Farm":
                render_improvement("Wheat+Farm", wheat_farm(fn), era, (round_wheat(fires), R0 + 0.02, water, char), seed)


# ============================================================================ city centres
CITY_ERAS = ["Ancient era", "Classical era", "Medieval era", "Renaissance era", "Industrial era", "Modern era", "Information era"]


def city_ground(cv, rad, seed, col, streets=None):
    n = fbm(cv.a, cv.b, 5, 3, seed)
    d = np.hypot(cv.u, cv.v * 1.05) + (n - 0.5) * 0.1
    m = smooth(rad, rad - 0.12, d) * footprint(cv)
    g = ground(cv, col, seed + 1, 0.2)
    if streets is not None:
        g = mix(g, col_arr(streets[1], cv.shape), streets[0])
    return premult(g, m)


def grid_streets(cv, freq, width=0.12, ang=0.0):
    x = cv.u * math.cos(ang) + cv.v * math.sin(ang)
    y = -cv.u * math.sin(ang) + cv.v * math.cos(ang)
    gx, gy = np.abs(x * freq - np.round(x * freq)), np.abs(y * freq - np.round(y * freq))
    return smooth(width, width * 0.5, np.minimum(gx, gy))


def wall_ring(cv, P, r, h, col, towers=0, tower_r=0.04):
    """A round city wall: its shadow, the northern half and its towers. Returns the southern half's
    points for wall_front(), which draws it after the houses."""
    ring = lambda a0, a1, n: [(r * math.cos(a), r * 0.92 * math.sin(a)) for a in np.linspace(a0, a1, n)]
    seg(cv, P, ring(0, 2 * math.pi, 49), 0.04, np.asarray(col) * 0.55, alpha=0.6)
    seg(cv, P, ring(math.pi, 2 * math.pi, 25), 0.045, col, lift=h)
    for k in range(towers):
        a = k * 2 * math.pi / towers
        if math.sin(a) <= 0:
            cyl(cv, P, r * math.cos(a), r * 0.92 * math.sin(a), tower_r, h * 1.5, col, np.asarray(col) * 0.75)
    return ring(0, math.pi, 25)


def wall_front(cv, P, front, h, col, towers=0, r=0.0, tower_r=0.04):
    seg(cv, P, front, 0.045, col, lift=h)
    for k in range(towers):
        a = k * 2 * math.pi / towers
        if math.sin(a) > 0:
            cyl(cv, P, r * math.cos(a), r * 0.92 * math.sin(a), tower_r, h * 1.5, col, np.asarray(col) * 0.75)


def houses(cv, P, rng, pts, size, h, walls, roofs, kind="hip"):
    for u, v in sorted(pts, key=lambda p: p[1]):
        s = rng.uniform(*size)
        box(cv, P, u, v, s, s * rng.uniform(0.7, 1.0), rng.uniform(*h), walls[rng.integers(len(walls))],
            roofs[rng.integers(len(roofs))], kind=kind)


def city_era(cv, P, rng, era):
    i = CITY_ERAS.index(era)
    warm = [C(.82, .74, .60), C(.86, .80, .70), C(.78, .70, .58)]
    tile = [C(.66, .30, .18), C(.58, .26, .16), C(.70, .38, .22)]
    if i == 0:      # Ancient: a palisaded village
        P = over(P, city_ground(cv, 0.8, 201, C(.52, .46, .34)))
        front = wall_ring(cv, P, 0.62, 0.07, C(.45, .32, .20), towers=8, tower_r=0.035)
        pts = rt.place_interior(rng, 34, 0.18, 0.105, lambda u, v: 0.012 < u * u + v * v < 0.27)
        houses(cv, P, rng, pts, (0.06, 0.09), (0.03, 0.04), warm, [C(.62, .52, .30), C(.55, .45, .26), C(.66, .34, .2)])
        box(cv, P, 0.0, 0.0, 0.16, 0.12, 0.07, C(.88, .84, .76), C(.70, .30, .18))
        wall_front(cv, P, front, 0.07, C(.45, .32, .20), towers=8, r=0.62, tower_r=0.035)
    elif i == 1:    # Classical: stone walls, a forum and a temple
        P = over(P, city_ground(cv, 0.84, 202, C(.62, .58, .50), (grid_streets(cv, 7, 0.1) * 0.6, C(.72, .68, .60))))
        front = wall_ring(cv, P, 0.66, 0.09, C(.74, .70, .62), towers=10)
        poly(cv, P, [(-0.12, -0.08), (0.12, -0.08), (0.12, 0.1), (-0.12, 0.1)], C(.86, .84, .78))
        pts = rt.place_interior(rng, 36, 0.18, 0.1, lambda u, v: u * u + v * v < 0.3 and not (abs(u) < 0.16 and -0.12 < v < 0.14))
        houses(cv, P, rng, pts, (0.065, 0.095), (0.04, 0.05), warm, tile)
        box(cv, P, 0.0, -0.2, 0.2, 0.12, 0.09, C(.95, .94, .90), C(.80, .62, .48), kind="gable")
        for k in range(6):
            cyl(cv, P, -0.085 + k * 0.034, -0.14, 0.008, 0.09, C(.96, .95, .92), C(.96, .95, .92), shadow_a=0.2)
        wall_front(cv, P, front, 0.09, C(.74, .70, .62), towers=10, r=0.66)
    elif i == 2:    # Medieval: thick walls, towers, a keep and a cathedral
        P = over(P, city_ground(cv, 0.86, 203, C(.52, .48, .42)))
        front = wall_ring(cv, P, 0.68, 0.13, C(.62, .60, .56), towers=10, tower_r=0.055)
        pts = rt.place_interior(rng, 42, 0.16, 0.095, lambda u, v: u * u + v * v < 0.34 and not (-0.2 < u < 0.25 and -0.3 < v < 0.05))
        houses(cv, P, rng, pts, (0.065, 0.09), (0.05, 0.07), warm, tile + [C(.36, .34, .36)])
        box(cv, P, -0.02, -0.12, 0.3, 0.1, 0.12, C(.70, .68, .64), C(.36, .38, .44), kind="gable")
        box(cv, P, -0.02, -0.12, 0.1, 0.26, 0.12, C(.70, .68, .64), C(.36, .38, .44), kind="gable")
        cyl(cv, P, -0.17, -0.12, 0.04, 0.32, C(.70, .68, .64), C(.36, .38, .44))
        box(cv, P, 0.2, -0.2, 0.12, 0.12, 0.2, C(.60, .58, .54), C(.50, .48, .44), kind="flat")
        flag(cv, P, 0.2, -0.2, 0.36, C(.8, .1, .1))
        wall_front(cv, P, front, 0.13, C(.62, .60, .56), towers=10, r=0.68, tower_r=0.055)
    elif i == 3:    # Renaissance: star-shaped bastions around a dense town with a dome
        P = over(P, city_ground(cv, 0.88, 204, C(.56, .52, .44), (grid_streets(cv, 8, 0.08, 0.3) * 0.5, C(.66, .62, .55))))
        fill_star(cv, P, star_points(6, 0.86, 0.64, 0.0), C(.38, .48, .26))
        fill_star(cv, P, star_points(6, 0.78, 0.6, 0.0), C(.56, .52, .44), lift=0.05)
        P = over(P, city_ground(cv, 0.62, 205, C(.58, .54, .46)))
        pts = rt.place_interior(rng, 44, 0.12, 0.09, lambda u, v: u * u + v * v < 0.3 and (u * u + v * v > 0.03))
        houses(cv, P, rng, pts, (0.065, 0.09), (0.06, 0.08), warm + [C(.92, .88, .80)], tile)
        box(cv, P, 0.0, 0.0, 0.18, 0.18, 0.1, C(.92, .88, .80), C(.80, .76, .70), kind="flat")
        blob(cv, P, 0.0, 0.0, 0.08, 0.075, C(.80, .45, .25), lift=0.14, gloss=0.3)
        blob(cv, P, 0.0, 0.0, 0.02, 0.02, C(.95, .92, .85), lift=0.23)
    elif i == 4:    # Industrial: brick blocks, rail, factories and smoke
        P = over(P, city_ground(cv, 0.9, 206, C(.46, .44, .42), (grid_streets(cv, 8, 0.12), C(.30, .30, .31))))
        seg(cv, P, [(-0.7, 0.42), (0.7, 0.42)], 0.03, C(.35, .32, .30))
        seg(cv, P, [(-0.7, 0.41), (0.7, 0.41)], 0.006, C(.75, .75, .78))
        seg(cv, P, [(-0.7, 0.43), (0.7, 0.43)], 0.006, C(.75, .75, .78))
        pts = []
        for a in range(-6, 7):
            for b in range(-6, 7):
                u, v = (a + 0.5) / 8, (b + 0.5) / 8
                if u * u + v * v < 0.44 and rng.random() > 0.1 and abs(v - 0.42) > 0.06:
                    pts.append((u, v))
        for u, v in sorted(pts, key=lambda p: p[1]):
            big = rng.random() < 0.2
            w = 0.1 if big else rng.uniform(0.07, 0.095)
            box(cv, P, u, v, w, w * rng.uniform(0.7, 1.0), rng.uniform(0.09, 0.14) if big else rng.uniform(0.05, 0.09),
                C(.62, .34, .24) if rng.random() < 0.6 else C(.68, .66, .62),
                C(.30, .30, .32) if big else [C(.42, .40, .40), C(.52, .28, .2), C(.36, .36, .38)][rng.integers(3)],
                kind="flat" if big else "hip")
        box(cv, P, 0.0, -0.05, 0.16, 0.12, 0.2, C(.84, .80, .72), C(.36, .46, .44))
        for u, v in [(-0.32, -0.3), (0.36, -0.24), (-0.4, 0.1)]:
            cyl(cv, P, u, v, 0.025, 0.36, C(.50, .28, .22), C(.15, .15, .15))
            smoke(cv, P, u, v, 0.4, n=6, col=C(.72, .72, .72), size=0.04, drift=0.035, alpha=0.5)
    else:           # Modern / Information: towers on a street grid, parks
        tall = i == 6
        P = over(P, city_ground(cv, 0.93, 207 + i, C(.50, .50, .50), (grid_streets(cv, 7, 0.13), C(.26, .26, .28))))
        d = np.hypot(cv.u - 0.32, cv.v - 0.28)
        P = over(P, premult(ground(cv, C(.32, .52, .24), 208), smooth(0.13, 0.11, d) * footprint(cv)))
        pts = []
        for a in range(-5, 6):
            for b in range(-5, 6):
                u, v = (a + 0.5) / 7, (b + 0.5) / 7
                if u * u + v * v < 0.48 and np.hypot(u - 0.32, v - 0.28) > 0.15:
                    pts.append((u, v))
        for u, v in sorted(pts, key=lambda p: p[1]):
            core = math.exp(-(u * u + v * v) / 0.12)
            hgt = rng.uniform(0.06, 0.12) + core * (0.45 if tall else 0.22) * rng.uniform(0.6, 1.0)
            glassy = tall and core > 0.4
            roof = C(.34, .52, .62) if glassy else [C(.66, .66, .66), C(.58, .58, .60), C(.72, .70, .66), C(.50, .52, .55)][rng.integers(4)]
            facade = C(.20, .32, .40) if glassy else [C(.46, .46, .48), C(.55, .52, .48), C(.40, .42, .46)][rng.integers(3)]
            w = rng.uniform(0.07, 0.1)
            box(cv, P, u, v, w, w * rng.uniform(0.75, 1.0), hgt, facade, roof, kind="flat")
            if not glassy and rng.random() < 0.5:
                blob(cv, P, u + rng.uniform(-0.02, 0.02), v - K * hgt, 0.012, 0.012, C(.85, .85, .85), flat=True)
        if tall:
            box(cv, P, 0.0, 0.0, 0.1, 0.1, 0.62, C(.25, .42, .52), C(.40, .60, .70), kind="flat")
            seg(cv, P, [(0.0, -K * 0.62), (0.0, -K * 0.78)], 0.01, C(.8, .8, .82))
    return P


def render_cities():
    for era in CITY_ERAS:
        rel = "Tiles/City center-" + era
        if not rt.wanted(rel):
            continue
        cv = canvas(BIG_W, overhang=170)
        P = city_era(cv, cv.empty(), np.random.default_rng(300 + CITY_ERAS.index(era)), era)
        finish(cv, P, rel)
    if rt.wanted("Tiles/City center"):   # used before any era is known, e.g. in the map editor
        cv = canvas(BIG_W, overhang=170)
        finish(cv, city_era(cv, cv.empty(), np.random.default_rng(300), CITY_ERAS[0]), "Tiles/City center")


# ============================================================================ natural wonders
def paint_relief(g, rgb, h, overhang):
    """Draw a height field with the tileset's oblique projection (painter's algorithm)."""
    out = Canvas(g.width, overhang=overhang)
    valid = g.hex_alpha() > 0.5
    P = out.empty()
    off = overhang * SS
    for r in range(g.shape[0]):
        top = r + off - K * h[r] * g.spx
        ylo = np.clip(np.floor(top).astype(int), 0, r + off)
        y0 = int(ylo[valid[r]].min()) if valid[r].any() else r + off
        ys = np.arange(y0, r + off + 1)[:, None]
        fillm = (ys >= ylo[None, :]) & valid[r][None, :]
        if not fillm.any():
            continue
        block = P[y0:r + off + 1]
        colr = np.concatenate([np.clip(rgb[r], 0, 1), np.ones((rgb.shape[1], 1))], axis=1)
        block[fillm] = np.broadcast_to(colr[None], fillm.shape + (4,))[fillm]
    return out, P


def relit(g, rgb, h, strength=0.8, tint=C(.86, .9, 1.0)):
    s = 0.5 + 0.55 * np.clip(rt.shade(h, g.spx, strength), 0, 1.6)
    return rt.lit(rgb, s, lo=0.45, hi=1.4, shadow_tint=tint)


def base_ground(g, kind):
    if kind == "rock":
        rgb, h, _ = rt.tex_rock(g)
    else:
        rgb, h, _ = rt.TEXTURES[kind](g, 1)
    return rgb, h


def edge_blend(g, inner_rgb, outer_rgb, width=0.3):
    t = smooth(0.0, width, g.edge)[..., None]
    return outer_rgb * (1 - t) + inner_rgb * t


WONDERS = {}


def wonder(name):
    def deco(fn):
        WONDERS[name] = fn
        return fn
    return deco


def snowcap(rgb, hm, level, n, amount=1.0):
    return mix(rgb, C(.95, .96, .99), smooth(0.0, 0.08, hm - (level + 0.12 * (n - 0.5))) * amount)


@wonder("Mount Fuji")
def w_fuji(g):
    base, bh = base_ground(g, "rock")
    grass, _ = base_ground(g, "Grassland")
    r = np.hypot(g.u, (g.v - 0.1) * 1.05)
    gul = ridged(g.a, g.b, 12, 3, 501)
    hm = (1.45 * np.clip(1 - r / 0.82, 0, 1) ** 1.2 - 0.12 * smooth(0.09, 0.0, r)) * (0.94 + 0.06 * gul)
    hm = hm * smooth(0.02, 0.3, g.edge)
    rgb = edge_blend(g, mix(grass, C(.22, .34, .15), 0.6), base, 0.2)
    rock = ramp(fbm(g.a, g.b, 9, 3, 502), [(0.3, C(.34, .29, .27)), (0.7, C(.45, .39, .35))])
    rgb = mix(rgb, rock, smooth(0.3, 0.6, hm))
    rgb = snowcap(rgb, hm, 0.85, gul)
    return relit(g, rgb, hm + bh), hm + bh, 140


@wonder("Mount Kilimanjaro")
def w_kilimanjaro(g):
    base, bh = base_ground(g, "rock")
    plains, _ = base_ground(g, "Plains")
    r = np.hypot(g.u, (g.v - 0.1) * 1.05)
    n = fbm(g.a, g.b, 6, 3, 511)
    hm = 1.0 * np.clip(1 - r / 0.85, 0, 1) ** 0.9
    hm = np.minimum(hm, 0.72 + 0.04 * n) * smooth(0.02, 0.3, g.edge)
    rgb = edge_blend(g, mix(plains, C(.52, .50, .28), 0.3), base, 0.2)
    rgb = mix(rgb, C(.24, .34, .16), smooth(0.1, 0.25, hm) * smooth(0.5, 0.35, hm))
    rgb = mix(rgb, C(.42, .36, .32), smooth(0.45, 0.6, hm))
    rgb = mix(rgb, C(.96, .97, .99), smooth(0.66, 0.7, hm))
    return relit(g, rgb, hm + bh), hm + bh, 100


@wonder("Mount Kailash")
def w_kailash(g):
    base, bh = base_ground(g, "rock")
    d = np.maximum(np.abs(g.u) * 1.1, np.abs(g.v - 0.08) * 1.25)
    hm = 1.55 * np.clip(1 - d / 0.7, 0, 1) ** 1.05 * smooth(0.02, 0.3, g.edge)
    rock = ramp(fbm(g.a, g.b, 9, 3, 521), [(0.3, C(.30, .27, .28)), (0.7, C(.42, .38, .38))])
    rgb = mix(base, rock, smooth(0.05, 0.3, hm))
    n = fbm(g.a, g.b, 7, 3, 522)
    bands = smooth(0.62, 0.9, np.sin(hm * 16 + n * 5) * 0.5 + 0.5) * smooth(0.5, 0.8, hm) * smooth(0.35, 0.5, n)
    rgb = mix(rgb, C(.93, .95, .98), np.maximum(bands, smooth(1.2, 1.3, hm)))
    return relit(g, rgb, hm + bh, 1.0), hm + bh, 160


@wonder("Mount Sinai")
def w_sinai(g):
    base, bh = base_ground(g, "rock")
    desert, _ = base_ground(g, "Desert")
    aw, bw = warp(g.a, g.b, 0.3, 3, 531)
    env = np.clip(1 - np.hypot(g.u * 0.9, (g.v - 0.1)) / 0.8, 0, 1)
    hm = env * (0.5 + 0.75 * ridged(aw, bw, 4, 5, 532)) * smooth(0.02, 0.3, g.edge)
    rgb = edge_blend(g, desert, base, 0.2)
    rock = ramp(fbm(g.a, g.b, 8, 3, 533), [(0.3, C(.55, .32, .22)), (0.7, C(.68, .42, .28))])
    rgb = mix(rgb, rock, smooth(0.05, 0.25, hm))
    g2 = (g.u + 0.0) ** 2 + (g.v + 0.0) ** 2
    rgb = mix(rgb, C(.9, .88, .82), smooth(0.004, 0.0, g2))
    return relit(g, rgb, hm + bh, 0.9), hm + bh, 120


@wonder("Sri Pada")
def w_sri_pada(g):
    base, bh = base_ground(g, "rock")
    r = np.hypot(g.u, (g.v - 0.1) * 1.05)
    hm = 1.5 * np.clip(1 - r / 0.62, 0, 1) ** 1.0 * (0.9 + 0.1 * ridged(g.a, g.b, 10, 3, 541))
    hm = hm * smooth(0.02, 0.3, g.edge)
    forest = ramp(fbm(g.a, g.b, 14, 3, 542), [(0.3, C(.14, .30, .12)), (0.7, C(.22, .40, .16))])
    rgb = mix(base, forest, smooth(0.02, 0.15, hm))
    path = smooth(0.025, 0.01, np.abs(np.sin(np.arctan2(g.v - 0.1, g.u) * 1.0 + r * 18) * r)) * smooth(0.6, 0.1, r)
    rgb = mix(rgb, C(.85, .78, .62), path * 0.7)
    rgb = mix(rgb, C(.97, .96, .94), smooth(0.06, 0.03, r))
    return relit(g, rgb, hm + bh), hm + bh, 150


@wonder("Cerro de Potosi")
def w_potosi(g):
    base, bh = base_ground(g, "rock")
    r = np.hypot(g.u, (g.v - 0.1) * 1.05)
    hm = 1.3 * np.clip(1 - r / 0.78, 0, 1) ** 1.1 * (0.92 + 0.08 * ridged(g.a, g.b, 8, 3, 551)) * smooth(0.02, 0.3, g.edge)
    rock = ramp(fbm(g.a, g.b, 9, 3, 552), [(0.3, C(.52, .28, .20)), (0.7, C(.66, .38, .26))])
    rgb = mix(base, rock, smooth(0.05, 0.3, hm))
    roads = smooth(0.07, 0.02, np.abs(np.sin(hm * 26 + g.u * 6))) * smooth(0.1, 0.2, hm) * smooth(1.2, 1.0, hm)
    rgb = mix(rgb, C(.78, .70, .62), roads * 0.6)
    mines = smooth(0.82, 0.86, fbm(g.a, g.b, 18, 2, 553)) * smooth(0.15, 0.3, hm)
    rgb = mix(rgb, C(.08, .07, .07), mines)
    rgb = mix(rgb, C(.85, .87, .9), smooth(0.86, 0.9, fbm(g.a, g.b, 26, 2, 554)) * smooth(0.2, 0.4, hm) * 0.8)
    return relit(g, rgb, hm + bh), hm + bh, 130


@wonder("Uluru")
def w_uluru(g):
    base, bh = base_ground(g, "rock")
    desert, _ = base_ground(g, "Desert")
    d = np.hypot(g.u / 0.62, (g.v - 0.08) / 0.4)
    hm = 0.55 * smooth(1.0, 0.82, d) * (1 - 0.15 * d * d) + 0.02 * fbm(g.a, g.b, 14, 3, 561)
    sand = mix(desert, C(.82, .45, .22), 0.55)
    rgb = edge_blend(g, sand, base, 0.2)
    scrub = smooth(0.6, 0.68, fbm(g.a, g.b, 22, 2, 562))
    rgb = mix(rgb, C(.40, .44, .24), scrub * 0.7 * smooth(0.95, 1.1, d))
    rock = ramp(fbm(g.a * 3, g.b * 3, 8, 3, 563), [(0.3, C(.70, .30, .14)), (0.7, C(.82, .40, .18))])
    rock = rock * (0.9 + 0.1 * np.sin(g.u * 60))[..., None]
    rgb = mix(rgb, rock, smooth(0.95, 0.85, d))
    hm = hm * smooth(0.02, 0.2, g.edge)
    return relit(g, rgb, hm + bh, 1.2), hm + bh, 70


@wonder("Grand Mesa")
def w_grand_mesa(g):
    base, bh = base_ground(g, "rock")
    plains, _ = base_ground(g, "Plains")
    n = fbm(g.a, g.b, 5, 3, 571)
    d = np.hypot(g.u / 0.72, (g.v - 0.05) / 0.55) + (n - 0.5) * 0.25
    hm = 0.6 * smooth(1.0, 0.9, d)
    strata = ramp(0.5 + 0.5 * np.sin(hm * 60), [(0, C(.62, .34, .22)), (0.5, C(.78, .52, .34)), (1, C(.55, .30, .20))])
    top = mix(plains, C(.48, .50, .28), 0.4)
    rgb = edge_blend(g, plains, base, 0.2)
    rgb = mix(rgb, strata, smooth(0.02, 0.1, hm))
    rgb = mix(rgb, top, smooth(0.55, 0.59, hm))
    hm = hm * smooth(0.02, 0.15, g.edge)
    return relit(g, rgb, hm + bh, 1.0), hm + bh, 70


@wonder("Barringer Crater")
def w_crater(g):
    base, bh = base_ground(g, "rock")
    desert, _ = base_ground(g, "Desert")
    n = fbm(g.a, g.b, 6, 3, 581)
    r = np.hypot(g.u, (g.v - 0.02) * 1.05) + (n - 0.5) * 0.05
    hm = 0.14 * np.exp(-((r - 0.5) / 0.1) ** 2) - 0.2 * smooth(0.48, 0.1, r) + 0.03 * smooth(0.12, 0.0, r)
    rgb = edge_blend(g, desert, base, 0.25)
    rgb = mix(rgb, C(.72, .62, .48), smooth(0.7, 0.55, r))
    rgb = mix(rgb, C(.58, .50, .40), smooth(0.45, 0.25, r))
    th = np.arctan2(g.v, g.u)
    rays = smooth(0.85, 0.95, np.abs(np.sin(th * 9 + n * 4))) * smooth(0.95, 0.6, r) * smooth(0.5, 0.6, r)
    rgb = mix(rgb, C(.86, .80, .68), rays * 0.5)
    hm = hm * smooth(0.02, 0.2, g.edge)
    return relit(g, rgb, hm + bh, 1.6), hm + bh, 40


@wonder("Krakatoa")
def w_krakatoa(g):
    base, bh = base_ground(g, "rock")
    r = np.hypot(g.u, (g.v - 0.08) * 1.05)
    hm = 1.15 * np.clip(1 - r / 0.75, 0, 1) ** 1.1 - 0.2 * smooth(0.14, 0.0, r)
    hm = hm * (0.92 + 0.08 * ridged(g.a, g.b, 10, 3, 591)) * smooth(0.02, 0.3, g.edge)
    rgb = mix(base * 0.55, C(.16, .15, .15), smooth(0.02, 0.2, hm))
    rgb = mix(rgb, C(.30, .40, .22), smooth(0.3, 0.1, hm) * smooth(0.02, 0.06, hm) * 0.6)
    lava = smooth(0.12, 0.0, r) + smooth(0.86, 0.9, ridged(g.a * 2, g.b * 2, 6, 3, 592)) * smooth(0.5, 0.2, r)
    rgb = mix(rgb, C(1.0, .45, .08), np.clip(lava, 0, 1))
    return relit(g, rgb, hm + bh), hm + bh, 130


@wonder("Lake Victoria")
def w_lake_victoria(g):
    base, bh = base_ground(g, "rock")
    grass, _ = base_ground(g, "Grassland")
    n = fbm(g.a, g.b, 4, 4, 601)
    d = np.hypot(g.u * 1.05, g.v * 1.15) + (n - 0.5) * 0.3
    water = smooth(0.62, 0.58, d)
    rgb = edge_blend(g, grass, base, 0.2)
    lake = ramp(fbm(g.a, g.b, 3, 3, 602), [(0.3, C(.06, .22, .32)), (0.7, C(.09, .28, .38))])
    rgb = mix(rgb, C(.30, .55, .52), smooth(0.68, 0.6, d))
    rgb = mix(rgb, lake, water)
    isl = smooth(0.74, 0.78, fbm(g.a, g.b, 9, 3, 603)) * water
    rgb = mix(rgb, C(.24, .40, .16), isl)
    return rgb, (bh * smooth(0.6, 0.7, d)), 10


@wonder("Great Barrier Reef")
def w_reef(g):
    coast, _ = base_ground(g, "Coast")
    a, b = warp(g.a, g.b, 0.3, 3, 611)
    reef = smooth(0.55, 0.6, fbm(a, b, 6, 4, 612)) * smooth(0.12, 0.3, g.edge)
    lagoon = mix(coast, C(.30, .72, .72), smooth(0.15, 0.35, g.edge) * 0.8)
    coral = ramp(fbm(g.a, g.b, 16, 2, 613), [(0.2, C(.90, .45, .55)), (0.45, C(.95, .62, .25)), (0.65, C(.62, .40, .78)),
                                           (0.85, C(.95, .85, .40))])
    rgb = mix(lagoon, coral, reef)
    surf = smooth(0.02, 0.0, np.abs(fbm(a, b, 6, 4, 612) - 0.55)) * smooth(0.12, 0.3, g.edge)
    rgb = mix(rgb, C(.95, .98, .98), surf * 0.7)
    h = 0.02 * reef
    return relit(g, rgb, h, 1.0), h, 10


@wonder("Old Faithful")
def w_old_faithful(g):
    base, bh = base_ground(g, "rock")
    r = np.hypot(g.u + 0.18, (g.v - 0.14) * 1.1)
    rgb = mix(base, C(.86, .84, .78), smooth(0.7, 0.4, np.hypot(g.u, g.v)))
    spring = ramp(np.clip(r / 0.3, 0, 1), [(0, C(.08, .30, .62)), (0.45, C(.15, .62, .72)), (0.65, C(.95, .85, .30)),
                                           (0.85, C(.92, .48, .12)), (1.0, C(.70, .30, .12))])
    rgb = mix(rgb, spring, smooth(0.32, 0.28, r))
    vent = np.hypot(g.u - 0.25, g.v + 0.05)
    rgb = mix(rgb, C(.55, .50, .45), smooth(0.08, 0.04, vent))
    h = 0.04 * smooth(0.1, 0.0, vent)
    return relit(g, rgb, h + bh), h + bh, 10


@wonder("El Dorado")
def w_el_dorado(g):
    plains, _ = base_ground(g, "Plains")
    jungle = ramp(fbm(g.a, g.b, 16, 3, 621), [(0.3, C(.14, .32, .12)), (0.7, C(.24, .44, .16))])
    n = fbm(g.a, g.b, 5, 3, 622)
    d = np.hypot(g.u, g.v * 1.1) + (n - 0.5) * 0.15
    rgb = edge_blend(g, jungle, plains, 0.25)
    ripple = fbm(g.a, g.b, 12, 3, 623)
    gold = ramp(ripple, [(0.3, C(.80, .58, .12)), (0.55, C(1.0, .80, .26)), (0.75, C(1.0, .92, .55))])
    glints = smooth(0.8, 0.86, fbm(g.a, g.b, 40, 1, 624))
    gold = mix(gold, C(1.0, 1.0, .9), glints)
    rgb = mix(rgb, C(.55, .48, .30), smooth(0.5, 0.45, d))
    rgb = mix(rgb, gold, smooth(0.44, 0.4, d))
    return rgb, np.zeros(g.shape), 10


@wonder("Fountain of Youth")
def w_fountain(g):
    plains, _ = base_ground(g, "Plains")
    n = fbm(g.a, g.b, 5, 3, 631)
    d = np.hypot(g.u, g.v * 1.1) + (n - 0.5) * 0.08
    rgb = mix(plains, C(.30, .55, .22), smooth(0.7, 0.5, d))
    flowers = smooth(0.78, 0.82, fbm(g.a, g.b, 26, 2, 632)) * smooth(0.6, 0.5, d)
    rgb = mix(rgb, C(.95, .75, .85), flowers)
    rgb = mix(rgb, C(.92, .90, .84), smooth(0.36, 0.33, d))
    terr = smooth(0.4, 0.6, np.sin(d * 90) * 0.5 + 0.5) * smooth(0.33, 0.28, d) * smooth(0.2, 0.25, d)
    pool = ramp(d / 0.26, [(0, C(.15, .70, .78)), (1, C(.40, .85, .85))])
    rgb = mix(rgb, pool, smooth(0.27, 0.24, d))
    rgb = mix(rgb, C(.98, .97, .94), terr * 0.6)
    sparkle = smooth(0.86, 0.9, fbm(g.a, g.b, 30, 1, 633)) * smooth(0.24, 0.1, d)
    rgb = mix(rgb, C(1, 1, 1), sparkle)
    return rgb, 0.02 * smooth(0.3, 0.2, d), 10


@wonder("King Solomon's Mines")
def w_solomon(g):
    plains, _ = base_ground(g, "Plains")
    desert, _ = base_ground(g, "Desert")
    aw, bw = warp(g.a, g.b, 0.25, 3, 641)
    hm = 0.35 * ridged(aw, bw, 3, 4, 642) * smooth(0.02, 0.35, g.edge)
    rgb = edge_blend(g, mix(desert, C(.70, .55, .32), 0.4), plains, 0.25)
    rgb = mix(rgb, C(.60, .45, .28), smooth(0.1, 0.25, hm))
    return relit(g, rgb, hm, 1.2), hm, 50


@wonder("Rock of Gibraltar")
def w_gibraltar(g):
    base, bh = base_ground(g, "rock")
    sea, _ = base_ground(g, "Coast")
    ridge = np.exp(-((g.u + 0.12 + 0.1 * g.v) / 0.22) ** 2) * smooth(0.75, 0.2, np.abs(g.v))
    hm = 0.9 * ridge * (0.85 + 0.15 * ridged(g.a, g.b, 8, 3, 651)) * smooth(0.02, 0.3, g.edge)
    rgb = mix(base, C(.84, .82, .76), smooth(0.05, 0.2, hm))
    rgb = mix(rgb, C(.30, .42, .22), smooth(0.3, 0.1, hm) * smooth(0.02, 0.08, hm) * 0.5)
    water = smooth(0.15, 0.3, g.u) * smooth(0.06, 0.2, g.edge)
    rgb = mix(rgb, sea, water)
    hm = hm * (1 - water)
    return relit(g, rgb, hm + bh, 1.0), hm + bh, 110


def render_wonders():
    for name, fn in WONDERS.items():
        rel = "Tiles/" + name
        if not rt.wanted(rel):
            continue
        g = Canvas(BIG_W)
        rgb, h, overhang = fn(g)
        out, P = paint_relief(g, rgb, h, overhang)
        if name == "Old Faithful":    # the geyser's steam column
            for k in range(14):
                blob(out, P, 0.25 + 0.012 * k + 0.02 * math.sin(k), -0.05, 0.055 + 0.014 * k, 0.05 + 0.014 * k,
                     C(.96, .97, .99), lift=0.05 + 0.075 * k, alpha=0.55 - 0.03 * k, shade_lo=0.8)
        if name == "Krakatoa":
            smoke(out, P, 0.0, 0.08, 1.05, n=8, col=C(.45, .44, .44), size=0.07, drift=0.04, alpha=0.7)
        if name == "King Solomon's Mines":
            for u, v in [(-0.35, -0.1), (0.1, -0.3), (0.35, 0.12), (-0.1, 0.25)]:
                blob(out, P, u, v, 0.07, 0.05, C(.06, .05, .05), flat=True)
                seg(out, P, [(u - 0.07, v + 0.01), (u - 0.07, v - K * 0.08), (u + 0.07, v - K * 0.08), (u + 0.07, v + 0.01)],
                    0.016, C(.45, .32, .18))
            rng = np.random.default_rng(7)
            for u, v in scatter(rng, 14, 0.6, 0.1):
                blob(out, P, u, v, 0.025, 0.02, C(1.0, .80, .22), lift=0.02, gloss=0.9)
        if name == "Mount Sinai":
            box(out, P, 0.0, 0.0, 0.05, 0.04, 0.03, C(.9, .88, .82), C(.6, .3, .2))
        finish(out, P, rel)


# ============================================================================ main
def main():
    rt.FILTER = sys.argv[1:] or None
    for name in RESOURCES:
        render_resource(name)
    render_improvements()
    render_cities()
    render_wonders()
    groups = {"resources": 0, "improvements": 0, "cities": 0, "wonders": 0}
    for rel in rt.WRITTEN:
        n = rel.split("/")[-1]
        key = "resources" if n in RESOURCES else "cities" if n.startswith("City center") else "wonders" if n in WONDERS else "improvements"
        groups[key] += 1
    print("Wrote %d images: %s" % (len(rt.WRITTEN), ", ".join("%s %d" % kv for kv in groups.items())))


if __name__ == "__main__":
    main()
