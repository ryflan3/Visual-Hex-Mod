#!/usr/bin/env python3
"""Draw city centres in five regional styles for the base game's civilizations outside Europe.

    python3 tools/render_regional_cities.py           # every regional city centre
    python3 tools/render_regional_cities.py China     # only images whose name contains one of the words

For a city's tile Unciv looks for "City center-<style>-<era>", from the owner's era back to the
first one, before the plain "City center-<era>" images that render_items.py draws. The base game's
nations have no style, so Unciv uses the civilization's name instead: "City center-Japan-Medieval era".
Each style has its own look from the Ancient to the Renaissance era; from the Industrial era on,
every civilization's city is the shared modern one. Every civilization in a style still gets an
image for every era, or a later era would fall back to its last regional one. The copies are
identical, so the atlas packer stores each picture once. Civilizations not listed in REGIONS, and
city-states, keep the plain images.
"""

import math
import sys

import numpy as np

import render_tiles as rt
import render_items as RI
from render_tiles import C, fbm, over, premult, smooth
from render_items import K, poly, seg, shadow, blob, cyl, cone, footprint, col_arr, trees, scatter

SHX, SHY = RI.SHX, RI.SHY
ERAS = ["Ancient era", "Classical era", "Medieval era", "Renaissance era"]
REGIONS = {   # style: the civilizations that build in it
    "East Asian": ["China", "Japan", "Korea", "Mongolia"],
    "Middle Eastern": ["Babylon", "Egypt", "Arabia", "Persia", "The Ottomans", "Carthage"],
    "South Asian": ["India", "Siam"],
    "Mesoamerican": ["Aztecs", "The Maya", "Inca"],
    "African": ["Songhai", "Ethiopia"],
}


# ============================================================================ building blocks
def bld(cv, P, u, v, w, d, h, wall, roof, kind="hip", ov=0.0, flare=0.0, ridge=None, rh=None, lift=0.0,
        shadow_a=0.42, parapet=True):
    """A building in the tileset's oblique view. kind: hip, gable (ridge east-west) or flat. ov: eaves
    overhang; flare: how far the eave corners turn up (East Asian roofs); lift: stand it on something."""
    wall, roof = np.asarray(wall, float), np.asarray(roof, float)
    rh = 0.0 if kind == "flat" else (0.45 * min(w, d) if rh is None else rh)
    hw, hd = w / 2, d / 2
    y = v - K * lift
    if shadow_a and lift == 0:
        top = h + rh
        sx, sy = SHX * top, SHY * top
        poly(cv, P, [(u - hw, y - hd), (u + hw, y - hd), (u + hw + sx, y - hd + sy), (u + hw + sx, y + hd + sy),
                     (u - hw + sx, y + hd + sy), (u - hw, y + hd)], C(.02, .03, .02), shadow_a)
    poly(cv, P, [(u - hw, y + hd), (u + hw, y + hd), (u + hw, y + hd - K * h), (u - hw, y + hd - K * h)], wall * 0.8)
    es, en = y + hd - K * h, y - hd - K * h
    if kind == "flat":
        poly(cv, P, [(u - hw, en), (u + hw, en), (u + hw, es), (u - hw, es)], roof)
        if parapet:
            seg(cv, P, [(u - hw, es), (u - hw, en), (u + hw, en)], 0.006, roof * 1.16)
            seg(cv, P, [(u + hw, en), (u + hw, es), (u - hw, es)], 0.006, roof * 0.8)
        return
    ry = y - K * (h + rh)
    W_, E_ = u - hw - ov, u + hw + ov
    s_, n_ = es + ov, en - ov
    f = K * flare
    if kind == "gable":
        poly(cv, P, [(W_, ry), (E_, ry), (E_, n_ - f), (W_, n_ - f)], roof * 0.68)
        poly(cv, P, [(W_, s_ - f), (u, s_), (E_, s_ - f), (E_, ry), (W_, ry)], roof * 0.95)
        seg(cv, P, [(W_, ry), (E_, ry)], 0.008, ridge if ridge is not None else roof * 0.6)
        return
    ins = min(hw, hd) * 0.9
    rl, rr = u - hw + ins, u + hw - ins
    poly(cv, P, [(W_, n_ - f), (rl, ry), (rr, ry), (E_, n_ - f)], roof * 0.72)
    poly(cv, P, [(W_ - flare * 0.3, n_ - f), (W_ - flare * 0.3, s_ - f), (rl, ry)], roof * 1.12)
    poly(cv, P, [(E_ + flare * 0.3, n_ - f), (rr, ry), (E_ + flare * 0.3, s_ - f)], roof * 0.6)
    poly(cv, P, [(W_ - flare * 0.3, s_ - f), (u, s_), (E_ + flare * 0.3, s_ - f), (rr, ry), (rl, ry)], roof * 0.92)
    if ridge is not None:
        seg(cv, P, [(rl, ry), (rr, ry)], 0.009, ridge)
        if flare:
            for x, sgn in ((rl, -1), (rr, 1)):
                seg(cv, P, [(x, ry), (x + sgn * 0.012, ry - 0.016)], 0.008, ridge)


def tiers(cv, P, u, v, w, d, h, n, wall, roof, step=0.8, shrink=0.16, ridge=None, flare=0.01, ov=0.012, base=0.0):
    """A pavilion or pagoda: n storeys, each a little smaller, each with its own eaves."""
    lift = base
    for k in range(n):
        s = 1 - shrink * k
        last = k == n - 1
        bld(cv, P, u, v, w * s, d * s, h * (1 if k == 0 else step), wall, roof, kind="hip", ov=ov, flare=flare,
            ridge=ridge if last else None, rh=(0.45 if last else 0.22) * min(w, d) * s, lift=lift,
            shadow_a=0.42 if k == 0 and base == 0 else 0.0)
        lift += h * (1 if k == 0 else step) + 0.22 * min(w, d) * s * 0.6


def flat_house(cv, P, u, v, w, d, h, col, rng, court=False, tree=False, extra=None):
    bld(cv, P, u, v, w, d, h, col * 0.95, col, kind="flat")
    top = v - K * h
    if court:
        iw, idd = w * 0.42, d * 0.42
        poly(cv, P, [(u - iw / 2, top - idd / 2), (u + iw / 2, top - idd / 2), (u + iw / 2, top + idd / 2), (u - iw / 2, top + idd / 2)],
             col * 0.5)
        if tree:
            blob(cv, P, u, top, iw * 0.4, idd * 0.4, C(.26, .44, .18), lift=0.0)
    elif extra is not None and rng.random() < extra:
        bld(cv, P, u + w * 0.2, v - d * 0.2, w * 0.3, d * 0.3, 0.02, col * 0.95, col * 1.05, kind="flat", lift=h, shadow_a=0, parapet=False)


def packed_town(cv, P, rng, keep, pitch, heights, cols, court_p=0.3, tree_p=0.3, alley=0.014, merge=0.4, extra=0.15):
    """Flat-roofed houses packed along narrow alleys, merged into bigger blocks here and there."""
    taken, blocks = set(), []
    n = int(1.0 / pitch) + 1
    dz = pitch * 0.92
    for j in range(-n, n + 1):
        for i in range(-n, n + 1):
            if (i, j) in taken:
                continue
            u, v = i * pitch + (j % 2) * pitch * 0.3, j * dz
            if not keep(u, v):
                continue
            wi = 2 if rng.random() < merge and (i + 1, j) not in taken and keep(u + pitch, v) else 1
            di = 2 if rng.random() < merge * 0.5 and keep(u, v + dz) and keep(u + (wi - 1) * pitch, v + dz) else 1
            for a in range(wi):
                for b in range(di):
                    taken.add((i + a, j + b))
            blocks.append((u + (wi - 1) * pitch / 2, v + (di - 1) * dz / 2, wi * pitch - alley, di * dz - alley))
    for u, v, w, d in sorted(blocks, key=lambda b: b[1]):
        col = np.asarray(cols[rng.integers(len(cols))]) * rng.uniform(0.94, 1.05)
        big = w * d > pitch * pitch * 1.5
        flat_house(cv, P, u, v, w, d, rng.uniform(*heights), col, rng, court=big and rng.random() < court_p,
                   tree=rng.random() < tree_p, extra=extra)


def hip_town(cv, P, rng, pts, size, h, walls, roofs, flare=0.0, ridge=None, kind="hip", ov=0.008):
    for u, v in sorted(pts, key=lambda p: p[1]):
        s = rng.uniform(*size)
        rf = np.asarray(roofs[rng.integers(len(roofs))])
        bld(cv, P, u, v, s, s * rng.uniform(0.65, 0.9), rng.uniform(*h), walls[rng.integers(len(walls))], rf,
            kind=kind, ov=ov, flare=flare, ridge=ridge if ridge is None else rf * 0.62)


def wall_line(cv, P, a, b, h, col, thick=0.045, crenel=True):
    """A straight stretch of wall from a to b (u, v), with its face, its walk and crenellations."""
    col = np.asarray(col, float)
    (u0, v0), (u1, v1) = a, b
    if abs(v1 - v0) < 1e-6:     # east-west: the south face shows
        x0, x1 = min(u0, u1), max(u0, u1)
        bld(cv, P, (x0 + x1) / 2, v0, x1 - x0, thick, h, col, col * 1.08, kind="flat", parapet=False, shadow_a=0.25)
        if crenel:
            for x in np.arange(x0 + 0.012, x1 - 0.006, 0.024):
                seg(cv, P, [(x, v0 + thick / 2 - K * h - 0.004), (x + 0.01, v0 + thick / 2 - K * h - 0.004)], 0.008, col * 1.12)
    else:                       # north-south: only the walk shows
        y0, y1 = min(v0, v1), max(v0, v1)
        bld(cv, P, u0, (y0 + y1) / 2, thick, y1 - y0, h, col, col * 1.08, kind="flat", parapet=False, shadow_a=0.25)
        if crenel:
            for y in np.arange(y0 + 0.012, y1 - 0.006, 0.024):
                seg(cv, P, [(u0 - thick / 2, y - K * h), (u0 - thick / 2, y + 0.01 - K * h)], 0.008, col * 1.12)


def square_walls(cv, P, hw, hh, h, col, part, tower=None, n_side=1, thick=0.045, crenel=True, gate=None):
    """A rectangular circuit. part "back": west, north and east walls; "front": the south wall.
    tower(cv, P, u, v) draws a tower; gate(cv, P, u, v) the south gate."""
    xs = np.linspace(-hw, hw, n_side + 2)
    if part == "back":
        wall_line(cv, P, (-hw, -hh), (hw, -hh), h, col, thick, crenel)
        wall_line(cv, P, (-hw, -hh), (-hw, hh), h, col, thick, crenel)
        wall_line(cv, P, (hw, -hh), (hw, hh), h, col, thick, crenel)
        if tower:
            for x in xs:
                tower(cv, P, x, -hh)
            for y in np.linspace(-hh, hh, n_side + 2)[1:-1]:
                tower(cv, P, -hw, y)
                tower(cv, P, hw, y)
    else:
        if gate:
            g = 0.09
            wall_line(cv, P, (-hw, hh), (-g, hh), h, col, thick, crenel)
            wall_line(cv, P, (g, hh), (hw, hh), h, col, thick, crenel)
        else:
            wall_line(cv, P, (-hw, hh), (hw, hh), h, col, thick, crenel)
        if tower:
            for x in xs:
                if not (gate and abs(x) < 0.1):
                    tower(cv, P, x, hh)
        if gate:
            gate(cv, P, 0.0, hh)


def round_tower(col, r=0.045, h=0.16, cap=None):
    def draw(cv, P, u, v):
        cyl(cv, P, u, v, r, h, col, np.asarray(col) * 0.82)
        if cap is not None:
            cone(cv, P, u, v - K * h, r * 1.1, r * 1.6, cap, shadow_a=0.0)
    return draw


def square_tower(col, s=0.075, h=0.16, roof=None, flare=0.0):
    def draw(cv, P, u, v):
        if roof is None:
            bld(cv, P, u, v, s, s * 0.85, h, col, np.asarray(col) * 0.9, kind="flat")
        else:
            bld(cv, P, u, v, s, s * 0.85, h, col, roof, kind="hip", ov=0.01, flare=flare, ridge=np.asarray(roof) * 0.6)
    return draw


def dome(cv, P, u, v, r, base_h, col, drum=None, finial=C(.85, .70, .30), tall=1.0):
    if drum is not None:
        cyl(cv, P, u, v, r * 0.92, base_h, drum, drum, shadow_a=0.0)
    blob(cv, P, u, v, r, r * 0.95 * tall, col, lift=base_h + r * 0.35 * tall, gloss=0.3, shade_lo=0.5)
    seg(cv, P, [(u, v - K * (base_h + r * 0.9 * tall)), (u, v - K * (base_h + r * 0.9 * tall + 0.03))], 0.008, finial)


def minaret(cv, P, u, v, r, h, col, cap=None):
    cyl(cv, P, u, v, r, h, col, np.asarray(col) * 0.92)
    for f in (0.55, 0.8):
        blob(cv, P, u, v, r * 1.7, r * 1.5, np.asarray(col) * 0.88, lift=h * f, flat=True)
    cone(cv, P, u, v - K * h, r * 1.05, r * 3.4, cap if cap is not None else col, shadow_a=0.0)


def palms(cv, P, rng, pts, r=(0.05, 0.065)):
    t = [rt.make_tree(rng, u, v, r, (0.12, 0.16), [C(.28, .44, .16), C(.32, .48, .18)], "palm") for u, v in pts]
    return rt.splat_trees(cv, P, t)


def cypress(cv, P, u, v, h=0.14):
    shadow(cv, P, u, v, 0.02, 0.018, h, alpha=0.35)
    for k in range(4):
        blob(cv, P, u, v, 0.022 * (1 - k * 0.18), 0.02 * (1 - k * 0.18), C(.16, .30, .14), lift=h * (0.25 + 0.22 * k))


def water(cv, P, mask, seed, col=C(.20, .42, .48)):
    rgb = col_arr(col, cv.shape) * (0.9 + 0.2 * fbm(cv.a, cv.b, 8, 3, seed))[..., None]
    return over(P, premult(rgb, mask * footprint(cv)))


def rect_mask(cv, u0, v0, u1, v1, soft=0.006):
    return (smooth(u0 - soft, u0 + soft, cv.u) * smooth(u1 + soft, u1 - soft, cv.u) *
            smooth(v0 - soft, v0 + soft, cv.v) * smooth(v1 + soft, v1 - soft, cv.v))


def pad(cv, P, u0, v0, u1, v1, col, alpha=1.0):
    return over(P, premult(RI.ground(cv, col, int(abs(u0 * 1000 + v1 * 100)) + 7, 0.12), rect_mask(cv, u0, v0, u1, v1) * footprint(cv) * alpha))


def terrace(cv, P, u, v, w, d, h, col):
    """A raised platform (stone terrace, earthen mound)."""
    bld(cv, P, u, v, w, d, h, np.asarray(col) * 0.92, col, kind="flat")


def hut(cv, P, u, v, r=0.05, roof=C(.74, .62, .36), wall=C(.62, .46, .32)):
    shadow(cv, P, u, v, r, r * 0.9, 0.06)
    cyl(cv, P, u, v, r * 0.85, 0.025, wall, np.asarray(wall) * 0.9, shadow_a=0.0)
    cone(cv, P, u, v - K * 0.025, r, r * 1.35, roof, shadow_a=0.0)


def step_pyramid(cv, P, u, v, w, n, th, col, temple=None, stair=None, shrink=0.17):
    col = np.asarray(col, float)
    for k in range(n):
        s = 1 - shrink * k
        bld(cv, P, u, v, w * s, w * s * 0.9, th, col * (0.94 + 0.03 * k), col * 1.08, kind="flat", lift=th * k,
            shadow_a=0.45 if k == 0 else 0.0, parapet=False)
        y = v + w * s * 0.45 - K * th * k
        seg(cv, P, [(u - w * s / 2, y - K * th), (u + w * s / 2, y - K * th)], 0.005, col * 0.8)
    sw = w * 0.14
    top = th * n
    y0, y1 = v + w * 0.45, v + w * (1 - shrink * (n - 1)) * 0.45 - K * top
    poly(cv, P, [(u - sw / 2, y0), (u + sw / 2, y0), (u + sw / 2 * 0.8, y1), (u - sw / 2 * 0.8, y1)],
         stair if stair is not None else col * 0.78)
    for k in range(int((y0 - y1) / 0.012)):
        yy = y0 - k * 0.012
        seg(cv, P, [(u - sw / 2, yy), (u + sw / 2, yy)], 0.003, col * 0.66)
    if temple is not None:
        tw = w * (1 - shrink * (n - 1))
        temples = temple if isinstance(temple, list) else [temple]
        for i, tc in enumerate(temples):
            du = 0.0 if len(temples) == 1 else (i - 0.5) * tw * 0.42
            bld(cv, P, u + du, v - 0.01, tw * (0.32 if len(temples) > 1 else 0.5), tw * 0.32, 0.05, tc, np.asarray(tc) * 0.75,
                kind="gable", lift=top, shadow_a=0.0, rh=0.03)


# ============================================================================ East Asian
EA_ROOF = [C(.38, .40, .44), C(.44, .46, .50), C(.34, .37, .40)]
EA_WALL = [C(.90, .87, .80), C(.84, .78, .68), C(.66, .30, .22)]
EA_RIDGE = C(.22, .23, .26)
GOLD_ROOF = C(.88, .68, .20)
RED = C(.70, .18, .12)


def ea_courtyards(cv, P, rng, keep, pitch=0.2, roofs=EA_ROOF, walls=EA_WALL, flare=0.008):
    """Courtyard houses: a main hall to the north, two side halls, a gate wall, a tree in the yard."""
    cells = [(cu, cvv) for cvv in np.arange(-0.6, 0.62, pitch * 0.9) for cu in np.arange(-0.7, 0.71, pitch)]
    for cu, cvv in sorted(cells, key=lambda c: c[1]):
        if not keep(cu, cvv) or rng.random() < 0.08:
            continue
        s = pitch * rng.uniform(0.86, 0.96)
        P = trees(cv, P, [(cu + rng.uniform(-0.02, 0.02), cvv + 0.005)], rng, r=(0.024, 0.03), ht=(0.04, 0.05))
        rf = roofs[rng.integers(len(roofs))]
        wl = walls[rng.integers(len(walls))]
        bld(cv, P, cu, cvv - s * 0.3, s * 0.72, s * 0.22, 0.04, wl, rf, ov=0.008, flare=flare, ridge=EA_RIDGE)
        for du in (-0.3, 0.3):
            bld(cv, P, cu + du * s, cvv + s * 0.04, s * 0.2, s * 0.36, 0.034, wl, rf, kind="gable", ov=0.006, ridge=EA_RIDGE)
        bld(cv, P, cu, cvv + s * 0.32, s * 0.6, s * 0.08, 0.025, C(.82, .78, .70), rf, kind="gable", ov=0.004)
    return P


def ea_gate(roof, wall, wallc, tiers_n=2, s=1.0):
    def draw(cv, P, u, v):
        bld(cv, P, u, v, 0.2 * s, 0.08 * s, 0.11 * s, wallc, np.asarray(wallc) * 1.06, kind="flat")
        poly(cv, P, [(u - 0.03 * s, v + 0.04 * s), (u + 0.03 * s, v + 0.04 * s), (u + 0.03 * s, v + 0.04 * s - K * 0.07 * s),
                     (u - 0.03 * s, v + 0.04 * s - K * 0.07 * s)], C(.16, .12, .10))
        tiers(cv, P, u, v, 0.17 * s, 0.07 * s, 0.05 * s, tiers_n, wall, roof, ridge=EA_RIDGE, base=0.11 * s, flare=0.012)
    return draw


def east_asian(cv, P, rng, era):
    if era == 0:      # a Shang-style town: rammed-earth walls, thatch, a great hall on an earthen platform
        P = over(P, RI.city_ground(cv, 0.86, 3001, C(.62, .56, .44), (RI.grid_streets(cv, 6, 0.07) * 0.6, C(.70, .64, .52))))
        earth = C(.72, .62, .46)
        hw, hh = 0.62, 0.5
        square_walls(cv, P, hw, hh, 0.06, earth, "back", crenel=False, thick=0.06)
        pts = [(u + rng.uniform(-0.015, 0.015), v + rng.uniform(-0.01, 0.01)) for u in np.arange(-0.54, 0.55, 0.11)
               for v in np.arange(-0.42, 0.45, 0.1) if not (abs(u) < 0.25 and -0.36 < v < 0.08) and rng.random() > 0.22]
        P = trees(cv, P, [p for p in rt.place_interior(rng, 14, 0.2, 0.1, lambda u, v: abs(u) < 0.56 and abs(v) < 0.44
                                                       and not (abs(u) < 0.25 and -0.36 < v < 0.08))], rng, r=(0.03, 0.04), ht=(0.05, 0.06))
        hip_town(cv, P, rng, pts, (0.06, 0.085), (0.025, 0.035), [C(.70, .58, .42), C(.64, .52, .38), C(.76, .66, .50)],
                 [C(.74, .64, .40), C(.64, .54, .32), C(.56, .48, .30)], kind="gable")
        terrace(cv, P, 0.0, -0.16, 0.42, 0.3, 0.04, C(.70, .62, .48))
        bld(cv, P, 0.0, -0.2, 0.34, 0.14, 0.06, C(.56, .34, .22), C(.66, .56, .32), ov=0.02, lift=0.04, shadow_a=0.0)
        bld(cv, P, 0.0, 0.0, 0.16, 0.06, 0.04, C(.56, .34, .22), C(.66, .56, .32), kind="gable", ov=0.01, lift=0.04, shadow_a=0.0)
        P = trees(cv, P, [(-0.5, -0.46), (0.52, 0.36), (-0.12, 0.44)], rng, r=(0.04, 0.05), ht=(0.06, 0.08))
        square_walls(cv, P, hw, hh, 0.06, earth, "front", crenel=False, thick=0.06)
        return P
    if era == 1:      # Han: courtyard houses, gate towers, a palace with twin towers
        P = over(P, RI.city_ground(cv, 0.86, 3002, C(.60, .56, .48), (RI.grid_streets(cv, 6, 0.08) * 0.7, C(.72, .68, .60))))
        wallc = C(.72, .64, .50)
        hw, hh = 0.64, 0.52
        gate = ea_gate(C(.36, .38, .42), C(.62, .30, .22), wallc, 1)
        square_walls(cv, P, hw, hh, 0.09, wallc, "back", tower=square_tower(wallc, 0.08, 0.13, C(.36, .38, .42), 0.01), thick=0.055)
        P = ea_courtyards(cv, P, rng, lambda u, v: abs(u) < hw - 0.09 and -hh + 0.09 < v < hh - 0.09 and not (abs(u) < 0.3 and -0.46 < v < 0.12))
        P = pad(cv, P, -0.28, -0.46, 0.28, 0.1, C(.82, .80, .74))
        terrace(cv, P, 0.0, -0.26, 0.4, 0.2, 0.035, C(.86, .84, .78))
        bld(cv, P, 0.0, -0.28, 0.34, 0.12, 0.07, RED, C(.30, .32, .36), ov=0.02, flare=0.012, ridge=EA_RIDGE, lift=0.035, shadow_a=0.0)
        for du in (-0.16, 0.16):
            tiers(cv, P, du, 0.04, 0.07, 0.06, 0.06, 3, C(.86, .82, .74), C(.30, .32, .36), ridge=EA_RIDGE)
        square_walls(cv, P, hw, hh, 0.09, wallc, "front", tower=square_tower(wallc, 0.08, 0.13, C(.36, .38, .42), 0.01), thick=0.055, gate=gate)
        return P
    if era == 2:      # Tang/Song: brick walls, a pagoda, a palace with green-glazed roofs, a canal
        P = over(P, RI.city_ground(cv, 0.88, 3003, C(.58, .56, .52), (RI.grid_streets(cv, 6, 0.08) * 0.7, C(.70, .68, .64))))
        P = water(cv, P, rect_mask(cv, -0.62, 0.2, 0.62, 0.25), 3004)
        wallc = C(.60, .60, .62)
        hw, hh = 0.66, 0.54
        gate = ea_gate(C(.30, .34, .36), RED, wallc, 2)
        tw = square_tower(wallc, 0.08, 0.15, C(.30, .34, .36), 0.012)
        square_walls(cv, P, hw, hh, 0.11, wallc, "back", tower=tw, thick=0.055)
        P = ea_courtyards(cv, P, rng, lambda u, v: abs(u) < hw - 0.09 and -hh + 0.09 < v < hh - 0.09 and not (-0.32 < u < 0.52 and -0.48 < v < 0.1)
                          and not (0.14 < v < 0.3))
        for x in (-0.4, 0.0, 0.4):
            seg(cv, P, [(x, 0.18), (x, 0.27)], 0.03, C(.80, .78, .74))
        P = pad(cv, P, -0.3, -0.48, 0.22, 0.08, C(.82, .80, .76))
        terrace(cv, P, -0.04, -0.28, 0.42, 0.2, 0.04, C(.88, .86, .82))
        bld(cv, P, -0.04, -0.3, 0.36, 0.13, 0.08, RED, C(.22, .46, .40), ov=0.022, flare=0.014, ridge=C(.14, .30, .26), lift=0.04, shadow_a=0.0)
        bld(cv, P, -0.04, -0.04, 0.2, 0.08, 0.06, RED, C(.22, .46, .40), ov=0.015, flare=0.01, ridge=C(.14, .30, .26))
        tiers(cv, P, 0.38, -0.18, 0.13, 0.12, 0.07, 7, C(.74, .40, .26), C(.26, .28, .30), shrink=0.1, ridge=EA_RIDGE, step=0.75)
        seg(cv, P, [(0.38, -0.18 - K * 0.56), (0.38, -0.18 - K * 0.64)], 0.012, C(.85, .70, .30))
        square_walls(cv, P, hw, hh, 0.11, wallc, "front", tower=tw, thick=0.055, gate=gate)
        return P
    # Ming: massive walls and gate towers, a red-walled palace city with golden roofs, a lake
    P = over(P, RI.city_ground(cv, 0.9, 3005, C(.58, .56, .52), (RI.grid_streets(cv, 7, 0.08) * 0.7, C(.70, .68, .64))))
    wallc = C(.56, .56, .58)
    hw, hh = 0.68, 0.56
    P = water(cv, P, smooth(0.14, 0.12, np.hypot((cv.u + 0.44) / 1.2, cv.v + 0.18)), 3006)
    gate = ea_gate(C(.30, .34, .36), RED, wallc, 2, 1.15)
    tw = square_tower(wallc, 0.09, 0.17, C(.30, .34, .36), 0.014)
    square_walls(cv, P, hw, hh, 0.13, wallc, "back", tower=tw, thick=0.06)
    P = ea_courtyards(cv, P, rng, lambda u, v: abs(u) < hw - 0.09 and -hh + 0.09 < v < hh - 0.09 and not (-0.3 < u < 0.3 and -0.5 < v < 0.36)
                      and np.hypot((u + 0.44) / 1.2, v + 0.18) > 0.2, pitch=0.22)
    P = pad(cv, P, -0.27, -0.5, 0.27, 0.34, C(.80, .76, .68))
    for a, b in [((-0.27, -0.5), (0.27, -0.5)), ((-0.27, -0.5), (-0.27, 0.34)), ((0.27, -0.5), (0.27, 0.34))]:
        wall_line(cv, P, a, b, 0.05, RED, 0.03, crenel=False)
    for vv, w in ((-0.36, 0.3), (-0.14, 0.36), (0.08, 0.26)):
        terrace(cv, P, 0.0, vv, w + 0.06, 0.14, 0.03, C(.94, .93, .90))
        bld(cv, P, 0.0, vv - 0.01, w, 0.1, 0.07, RED, GOLD_ROOF, ov=0.02, flare=0.014, ridge=C(.62, .44, .10), lift=0.03, shadow_a=0.0)
    wall_line(cv, P, (-0.27, 0.34), (0.27, 0.34), 0.05, RED, 0.03, crenel=False)
    tiers(cv, P, 0.0, 0.34, 0.16, 0.06, 0.04, 2, RED, GOLD_ROOF, ridge=C(.62, .44, .10), base=0.05)
    blob(cv, P, -0.44, -0.2, 0.05, 0.05, C(.95, .95, .93), lift=0.05, gloss=0.3)          # a white dagoba on the lake island
    cone(cv, P, -0.44, -0.2 - K * 0.09, 0.015, 0.05, C(.85, .70, .30), shadow_a=0.0)
    square_walls(cv, P, hw, hh, 0.13, wallc, "front", tower=tw, thick=0.06, gate=gate)
    return P



# ============================================================================ Middle Eastern
MUD = [C(.80, .68, .52), C(.84, .74, .58), C(.76, .64, .48), C(.88, .80, .66)]
WHITE = [C(.92, .89, .82), C(.88, .82, .70), C(.84, .74, .58), C(.94, .92, .88)]
TURQ = C(.20, .56, .62)


def ziggurat(cv, P, u, v, w, n, th, col, shrine=C(.24, .42, .72)):
    col = np.asarray(col, float)
    for k in range(n):
        s = 1 - 0.24 * k
        bld(cv, P, u, v, w * s, w * s * 0.8, th, col * (0.95 + 0.04 * k), col * 1.06, kind="flat", lift=th * k,
            shadow_a=0.45 if k == 0 else 0.0, parapet=False)
        for x in np.linspace(u - w * s / 2 + 0.02, u + w * s / 2 - 0.02, max(3, int(w * s / 0.04))):     # buttresses
            seg(cv, P, [(x, v + w * s * 0.4 - K * th * k), (x, v + w * s * 0.4 - K * th * (k + 1))], 0.006, col * 0.82)
    y0 = v + w * 0.4
    poly(cv, P, [(u - 0.025, y0 + 0.06), (u + 0.025, y0 + 0.06), (u + 0.02, v - K * th * n + w * 0.12), (u - 0.02, v - K * th * n + w * 0.12)], col * 0.78)
    for sgn in (-1, 1):
        poly(cv, P, [(u + sgn * w * 0.45, y0 + 0.02), (u + sgn * w * 0.4, y0 + 0.02), (u + sgn * 0.03, v + w * 0.4 - K * th),
                     (u + sgn * 0.05, v + w * 0.4 - K * th)], col * 0.8)
    bld(cv, P, u, v - 0.01, w * 0.22, w * 0.16, 0.05, shrine, np.asarray(shrine) * 1.1, kind="flat", lift=th * n, shadow_a=0)


def mosque(cv, P, u, v, w, d, col, dome_col, minarets=1, court=True, h=0.07):
    """A courtyard mosque: a prayer hall with a big dome on the south side of a court, and minarets."""
    bld(cv, P, u, v, w, d, h * 0.8, col, col, kind="flat")
    top = v - K * h * 0.8
    if court:
        poly(cv, P, [(u - w * 0.36, top + d * 0.02), (u + w * 0.36, top + d * 0.02), (u + w * 0.36, top + d * 0.4), (u - w * 0.36, top + d * 0.4)],
             np.asarray(col) * 0.86)
        blob(cv, P, u, top + d * 0.22, 0.02, 0.016, C(.30, .52, .62), flat=True)
    hall_v = v - d * 0.24
    bld(cv, P, u, hall_v, w * 0.9, d * 0.42, h * 1.15, col, np.asarray(col) * 1.04, kind="flat", shadow_a=0.0)
    dome(cv, P, u, hall_v + 0.01, w * 0.2, h * 1.15 + 0.03, dome_col, drum=np.asarray(col) * 0.96)
    for du in (-w * 0.32, w * 0.32):
        dome(cv, P, u + du, hall_v + 0.01, w * 0.08, h * 1.15, dome_col)
    spots = [(u + w * 0.46, v + d * 0.42), (u - w * 0.46, v + d * 0.42), (u + w * 0.46, v - d * 0.42), (u - w * 0.46, v - d * 0.42)]
    for mu, mv in spots[:minarets]:
        minaret(cv, P, mu, mv, 0.022, 0.34, np.asarray(col) * 1.02, cap=dome_col)


def middle_east(cv, P, rng, era):
    if era == 0:      # Sumer: a dense mud-brick town round a great ziggurat, a canal and palm groves
        P = over(P, RI.city_ground(cv, 0.86, 3101, C(.78, .68, .52)))
        P = water(cv, P, smooth(0.035, 0.025, np.abs(cv.v - 0.5 - 0.08 * np.sin(cv.u * 4))) * smooth(0.95, 0.7, np.abs(cv.u)), 3102)
        P = palms(cv, P, rng, [(x, 0.62 + rng.uniform(-0.03, 0.03)) for x in np.arange(-0.6, 0.62, 0.13)] +
                  [(x, 0.4 + rng.uniform(-0.02, 0.02)) for x in (-0.66, -0.52, 0.5, 0.64)])
        front = RI.wall_ring(cv, P, 0.62, 0.06, C(.74, .62, .46), towers=0)
        packed_town(cv, P, rng, lambda u, v: u * u / 0.33 + v * v / 0.24 < 1 and not (abs(u) < 0.26 and -0.36 < v < 0.14), 0.085,
                    (0.03, 0.05), MUD, court_p=0.4, tree_p=0.2, extra=0.1)
        ziggurat(cv, P, 0.0, -0.1, 0.42, 3, 0.07, C(.74, .60, .44))
        RI.wall_front(cv, P, front, 0.06, C(.74, .62, .46), towers=0, r=0.62)
        return P
    if era == 1:      # Babylon: walls and towers, the blue Ishtar gate, a palace, hanging gardens and the ziggurat
        P = over(P, RI.city_ground(cv, 0.88, 3103, C(.80, .70, .54), (RI.grid_streets(cv, 5, 0.06) * 0.6, C(.86, .78, .62))))
        wallc = C(.78, .64, .46)
        hw, hh = 0.64, 0.52
        tw = square_tower(wallc, 0.07, 0.15)

        def ishtar(cv, P, u, v):
            for du in (-0.07, 0.07):
                bld(cv, P, u + du, v, 0.07, 0.08, 0.2, C(.18, .34, .66), C(.22, .40, .72), kind="flat")
            bld(cv, P, u, v, 0.08, 0.07, 0.15, C(.18, .34, .66), C(.22, .40, .72), kind="flat")
            for k in range(3):
                seg(cv, P, [(u - 0.1, v + 0.04 - K * (0.05 + 0.045 * k)), (u + 0.1, v + 0.04 - K * (0.05 + 0.045 * k))], 0.006, C(.90, .74, .30))
            poly(cv, P, [(u - 0.025, v + 0.04), (u + 0.025, v + 0.04), (u + 0.025, v + 0.04 - K * 0.08), (u - 0.025, v + 0.04 - K * 0.08)], C(.12, .10, .10))
        square_walls(cv, P, hw, hh, 0.12, wallc, "back", tower=tw, n_side=2, thick=0.06)
        packed_town(cv, P, rng, lambda u, v: abs(u) < hw - 0.07 and -hh + 0.07 < v < hh - 0.08 and not (-0.5 < u < 0.42 and -0.44 < v < 0.12),
                    0.085, (0.03, 0.05), MUD, court_p=0.4, tree_p=0.3)
        ziggurat(cv, P, -0.28, -0.24, 0.3, 3, 0.07, C(.74, .60, .44))
        bld(cv, P, 0.2, -0.26, 0.3, 0.2, 0.06, C(.84, .76, .62), C(.86, .78, .64), kind="flat")                 # the palace
        for k in range(3):                                                                                      # hanging gardens
            bld(cv, P, 0.22, -0.02, 0.24 - 0.05 * k, 0.12 - 0.02 * k, 0.04, C(.74, .64, .50), C(.34, .52, .22), kind="flat",
                lift=0.04 * k, shadow_a=0.4 if k == 0 else 0.0, parapet=False)
        P = trees(cv, P, [(0.14, -0.04 - K * 0.12), (0.26, -0.03 - K * 0.12), (0.3, 0.02 - K * 0.08)], rng, r=(0.025, 0.03), ht=(0.04, 0.05))
        square_walls(cv, P, hw, hh, 0.12, wallc, "front", tower=tw, n_side=2, thick=0.06, gate=ishtar)
        return P
    if era == 2:      # Baghdad: the round city, four gates, the green-domed mosque in the middle
        P = over(P, RI.city_ground(cv, 0.9, 3104, C(.80, .70, .54)))
        wallc = C(.80, .68, .52)
        r_out = 0.74
        front = RI.wall_ring(cv, P, r_out, 0.12, wallc, towers=16, tower_r=0.035)
        d = np.hypot(cv.u, cv.v / 0.92)
        roads = (smooth(0.03, 0.02, np.abs(cv.u)) + smooth(0.03, 0.02, np.abs(cv.v))) * smooth(r_out, r_out - 0.02, d)
        P = over(P, premult(col_arr(C(.88, .80, .66), cv.shape), np.clip(roads, 0, 1) * footprint(cv)))
        P = over(P, premult(RI.ground(cv, C(.84, .78, .66), 3105, 0.1), smooth(0.33, 0.32, d) * footprint(cv)))
        ring = lambda u, v: 0.38 < math.hypot(u, v / 0.92) < r_out - 0.08 and abs(u) > 0.05 and abs(v) > 0.05
        packed_town(cv, P, rng, ring, 0.08, (0.03, 0.05), WHITE + MUD[:2], court_p=0.45, tree_p=0.35, extra=0.2)
        RI.wall_ring(cv, P, 0.34, 0.05, wallc, towers=0)
        mosque(cv, P, -0.02, 0.04, 0.3, 0.26, C(.92, .90, .84), TURQ, minarets=1)
        bld(cv, P, 0.0, -0.22, 0.24, 0.12, 0.08, C(.90, .86, .76), C(.92, .88, .80), kind="flat")
        dome(cv, P, 0.0, -0.22, 0.08, 0.1, C(.24, .56, .32), drum=C(.90, .86, .76))
        for a in range(4):
            t = a * math.pi / 2
            bld(cv, P, r_out * math.cos(t), r_out * 0.92 * math.sin(t), 0.1, 0.08, 0.17, wallc, TURQ, kind="flat")
        RI.wall_front(cv, P, front, 0.12, wallc, towers=16, r=r_out, tower_r=0.035)
        return P
    # Ottoman: a great mosque with cascading grey domes and four needle minarets, cypresses, red roofs
    P = over(P, RI.city_ground(cv, 0.9, 3106, C(.74, .68, .58), (RI.grid_streets(cv, 7, 0.07, 0.3) * 0.5, C(.82, .78, .70))))
    wallc = C(.72, .66, .58)
    front = RI.wall_ring(cv, P, 0.72, 0.12, wallc, towers=12, tower_r=0.045)
    keep = lambda u, v: u * u + v * v < 0.42 and not (-0.34 < u < 0.34 and -0.42 < v < 0.2)
    pts = rt.place_interior(rng, 60, 0.12, 0.085, keep)
    hip_town(cv, P, rng, pts, (0.06, 0.085), (0.04, 0.06), [C(.90, .86, .78), C(.84, .76, .62), C(.92, .90, .86)],
             [C(.70, .34, .22), C(.64, .30, .20), C(.74, .40, .26)], ridge=True)
    for u, v in [(-0.4, -0.3), (0.42, -0.28), (-0.44, 0.24), (0.46, 0.26), (-0.2, 0.36), (0.24, 0.4)]:
        cypress(cv, P, u, v)
    P = pad(cv, P, -0.32, -0.42, 0.32, 0.22, C(.86, .82, .74))
    lead = C(.62, .64, .68)
    stone = C(.90, .87, .80)
    bld(cv, P, 0.0, -0.06, 0.4, 0.34, 0.1, stone, stone, kind="flat")
    for du, dv, r in ((0.0, -0.06, 0.15), (0.0, -0.2, 0.09), (0.0, 0.08, 0.09)):
        dome(cv, P, du, dv, r, 0.1 + (0.05 if r > 0.1 else 0.02), lead, drum=stone, finial=C(.85, .70, .30))
    for du in (-0.15, 0.15):
        for dv in (-0.16, 0.04):
            dome(cv, P, du, dv, 0.045, 0.1, lead)
    for mu, mv in ((-0.26, -0.26), (0.26, -0.26), (-0.26, 0.14), (0.26, 0.14)):
        minaret(cv, P, mu, mv, 0.018, 0.48, C(.94, .92, .88), cap=lead)
    RI.wall_front(cv, P, front, 0.12, wallc, towers=12, r=0.72, tower_r=0.045)
    return P


# ============================================================================ South Asian
BRICK = [C(.68, .44, .32), C(.74, .50, .36), C(.62, .40, .30), C(.70, .48, .36)]
SAND = C(.84, .66, .48)
REDSTONE = C(.68, .34, .24)
MARBLE = C(.96, .95, .92)


def gopuram(cv, P, u, v, w, h, col, n=6):
    """A temple gate tower: stacked, shrinking storeys with a barrel-vaulted top."""
    col = np.asarray(col, float)
    lift = 0.0
    for k in range(n):
        s = 1 - 0.1 * k
        hk = h / n
        bld(cv, P, u, v, w * s, w * 0.45 * s, hk, col * (0.92 + 0.02 * k), col * 1.05, kind="flat", lift=lift,
            shadow_a=0.4 if k == 0 else 0.0, parapet=False)
        lift += hk
    blob(cv, P, u, v, w * 0.32, w * 0.12, col * 1.05, lift=lift + 0.01)
    for x in (u - w * 0.22, u, u + w * 0.22):
        seg(cv, P, [(x, v - K * (lift + 0.02)), (x, v - K * (lift + 0.04))], 0.006, C(.85, .70, .30))


def vimana(cv, P, u, v, w, h, col, n=10):
    """A pyramidal temple tower (Chola style) with a domed cap."""
    col = np.asarray(col, float)
    bld(cv, P, u, v, w, w * 0.9, h * 0.25, col * 0.94, col, kind="flat")
    lift = h * 0.25
    for k in range(n):
        s = 1 - 0.075 * (k + 1)
        bld(cv, P, u, v, w * s, w * 0.9 * s, h * 0.06, col * (0.9 + 0.02 * (k % 2)), col * 1.04, kind="flat", lift=lift,
            shadow_a=0.0, parapet=False)
        lift += h * 0.06
    blob(cv, P, u, v, w * 0.14, w * 0.13, col * 1.08, lift=lift + w * 0.06)
    seg(cv, P, [(u, v - K * (lift + w * 0.18)), (u, v - K * (lift + w * 0.26))], 0.01, C(.88, .72, .30))


def chhatri(cv, P, u, v, r, lift, col):
    for du in (-r * 0.7, r * 0.7):
        seg(cv, P, [(u + du, v - K * lift), (u + du, v - K * (lift + r * 1.4))], 0.005, col)
    dome(cv, P, u, v, r, lift + r * 1.4, col, finial=C(.85, .70, .30))


def stupa(cv, P, u, v, r):
    cyl(cv, P, u, v, r * 1.35, 0.02, C(.74, .66, .56), C(.80, .74, .64))
    pts = [(u + r * 1.25 * math.cos(a), v + r * 1.15 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 41)]
    seg(cv, P, pts, 0.006, C(.80, .72, .60), lift=0.035)
    blob(cv, P, u, v, r, r * 0.92, C(.97, .96, .92), lift=0.02 + r * 0.4, gloss=0.3, shade_lo=0.55)
    bld(cv, P, u, v - 0.004, r * 0.3, r * 0.26, 0.03, C(.80, .66, .40), C(.84, .72, .46), kind="flat", lift=0.02 + r * 0.95, shadow_a=0)
    for k in range(3):
        blob(cv, P, u, v, r * (0.3 - 0.07 * k), r * (0.12 - 0.03 * k), C(.85, .70, .30), lift=0.06 + r * 0.95 + 0.025 * k, flat=True)
    for a in range(4):                                                                     # toranas at the four gates
        t = a * math.pi / 2
        gu, gv = u + r * 1.42 * math.cos(t), v + r * 1.3 * math.sin(t)
        for k in (-1, 1):
            seg(cv, P, [(gu + k * 0.016, gv), (gu + k * 0.016, gv - K * 0.06)], 0.007, C(.72, .58, .42))
        for y in (0.05, 0.065):
            seg(cv, P, [(gu - 0.026, gv - K * y), (gu + 0.026, gv - K * y)], 0.006, C(.72, .58, .42))


def south_asian(cv, P, rng, era):
    if era == 0:      # Indus: a grid of baked-brick houses, a citadel mound with the Great Bath and a granary
        P = over(P, RI.city_ground(cv, 0.86, 3201, C(.70, .56, .44)))
        grid = lambda u, v: abs(u) < 0.62 and abs(v) < 0.5 and not (-0.62 < u < -0.12 and -0.5 < v < -0.02) and abs(v - 0.08) > 0.03 and abs(u - 0.2) > 0.03
        packed_town(cv, P, rng, grid, 0.09, (0.03, 0.045), BRICK, court_p=0.6, tree_p=0.15, merge=0.5, extra=0.05)
        terrace(cv, P, -0.37, -0.26, 0.46, 0.4, 0.07, C(.66, .46, .34))
        u, v = -0.42, -0.3
        bld(cv, P, u, v, 0.2, 0.14, 0.03, C(.62, .40, .30), C(.66, .44, .32), kind="flat", lift=0.07, shadow_a=0)
        poly(cv, P, [(u - 0.06, v - K * 0.1 - 0.04), (u + 0.06, v - K * 0.1 - 0.04), (u + 0.06, v - K * 0.1 + 0.03), (u - 0.06, v - K * 0.1 + 0.03)],
             C(.22, .44, .52))
        for k in range(4):                                                                 # the granary's brick plinths
            bld(cv, P, -0.24 + (k % 2) * 0.07, -0.18 + (k // 2) * 0.06, 0.055, 0.04, 0.035, C(.60, .38, .28), C(.64, .42, .30),
                kind="flat", lift=0.07, shadow_a=0)
        return P
    if era == 1:      # Mauryan: brick walls, the great stupa with its gateways, a monastery, tiled houses
        P = over(P, RI.city_ground(cv, 0.86, 3202, C(.68, .58, .44), (RI.grid_streets(cv, 6, 0.06, 0.2) * 0.4, C(.76, .66, .52))))
        wallc = C(.66, .44, .32)
        front = RI.wall_ring(cv, P, 0.7, 0.1, wallc, towers=12, tower_r=0.04)
        keep = lambda u, v: u * u + v * v < 0.36 and math.hypot(u, v + 0.06) > 0.32 and not (0.18 < u < 0.52 and 0.12 < v < 0.44)
        pts = rt.place_interior(rng, 50, 0.14, 0.085, keep)
        hip_town(cv, P, rng, pts, (0.06, 0.08), (0.03, 0.045), [C(.86, .80, .68), C(.80, .70, .56)],
                 [C(.66, .36, .24), C(.72, .62, .38), C(.60, .32, .22)])
        P = pad(cv, P, -0.3, -0.34, 0.3, 0.22, C(.80, .74, .62))
        stupa(cv, P, 0.0, -0.06, 0.17)
        u, v = 0.35, 0.28                                                                  # the monastery round its court
        bld(cv, P, u, v, 0.26, 0.24, 0.05, C(.72, .50, .36), C(.76, .54, .40), kind="flat")
        poly(cv, P, [(u - 0.08, v - K * 0.05 - 0.07), (u + 0.08, v - K * 0.05 - 0.07), (u + 0.08, v - K * 0.05 + 0.07), (u - 0.08, v - K * 0.05 + 0.07)],
             C(.58, .44, .32))
        P = trees(cv, P, [(-0.44, -0.4), (0.44, -0.38), (-0.5, 0.3), (0.0, 0.46)], rng, r=(0.04, 0.05), ht=(0.06, 0.08))
        RI.wall_front(cv, P, front, 0.1, wallc, towers=12, r=0.7, tower_r=0.04)
        return P
    if era == 2:      # Chola: a temple city with gopuram gates, a towering vimana and a stepped tank
        P = over(P, RI.city_ground(cv, 0.88, 3203, C(.72, .60, .46), (RI.grid_streets(cv, 7, 0.06) * 0.4, C(.80, .70, .56))))
        keep = lambda u, v: u * u + v * v < 0.5 and not (abs(u) < 0.36 and -0.46 < v < 0.24) and not (0.3 < u < 0.62 and 0.18 < v < 0.46)
        pts = rt.place_interior(rng, 56, 0.1, 0.08, keep)
        hip_town(cv, P, rng, pts, (0.055, 0.075), (0.03, 0.045), [C(.90, .86, .76), C(.84, .70, .56), C(.88, .76, .58)],
                 [C(.68, .38, .26), C(.62, .34, .22), C(.74, .44, .30)])
        u0, v0 = 0.46, 0.32                                                                # temple tank
        bld(cv, P, u0, v0, 0.26, 0.22, 0.01, C(.80, .72, .60), C(.82, .74, .62), kind="flat")
        P = over(P, premult(col_arr(C(.18, .44, .50), cv.shape), rect_mask(cv, u0 - 0.09, v0 - 0.08, u0 + 0.09, v0 + 0.07) * footprint(cv)))
        bld(cv, P, u0, v0 - 0.01, 0.04, 0.04, 0.05, SAND, SAND, kind="flat", shadow_a=0.2)
        P = pad(cv, P, -0.34, -0.44, 0.34, 0.22, C(.84, .76, .62))
        for a, b in [((-0.34, -0.44), (0.34, -0.44)), ((-0.34, -0.44), (-0.34, 0.22)), ((0.34, -0.44), (0.34, 0.22))]:
            wall_line(cv, P, a, b, 0.05, SAND, 0.03, crenel=False)
        gopuram(cv, P, 0.0, -0.44, 0.14, 0.18, SAND, 5)
        for du in (-0.18, 0.18):
            bld(cv, P, du, -0.06, 0.1, 0.12, 0.05, SAND, SAND * 1.04, kind="flat")
        vimana(cv, P, 0.0, -0.14, 0.2, 0.46, SAND)
        wall_line(cv, P, (-0.34, 0.22), (0.34, 0.22), 0.05, SAND, 0.03, crenel=False)
        gopuram(cv, P, 0.0, 0.22, 0.18, 0.28, SAND, 7)
        return P
    # Mughal: a red sandstone fort, a white marble tomb with four minarets over a walled garden
    P = over(P, RI.city_ground(cv, 0.9, 3204, C(.74, .62, .48), (RI.grid_streets(cv, 7, 0.06, 0.3) * 0.4, C(.82, .72, .58))))
    hw, hh = 0.66, 0.54
    tw = round_tower(REDSTONE, 0.055, 0.15)

    def red_tower(cv, P, u, v):
        tw(cv, P, u, v)
        chhatri(cv, P, u, v, 0.03, 0.15, REDSTONE * 1.15)

    def fort_gate(cv, P, u, v):
        bld(cv, P, u, v, 0.22, 0.09, 0.18, REDSTONE, REDSTONE * 1.08, kind="flat")
        poly(cv, P, [(u - 0.035, v + 0.045), (u + 0.035, v + 0.045), (u + 0.035, v + 0.045 - K * 0.11), (u - 0.035, v + 0.045 - K * 0.11)], C(.30, .14, .10))
        for du in (-0.09, 0.09):
            chhatri(cv, P, u + du, v, 0.03, 0.18, MARBLE)
    square_walls(cv, P, hw, hh, 0.12, REDSTONE, "back", tower=red_tower, n_side=1, thick=0.06)
    keep = lambda u, v: abs(u) < hw - 0.08 and -hh + 0.08 < v < hh - 0.09 and not (abs(u) < 0.4 and -0.5 < v < 0.34)
    packed_town(cv, P, rng, keep, 0.085, (0.035, 0.055), [C(.90, .86, .78), C(.86, .74, .60), C(.72, .42, .30)], court_p=0.4, extra=0.2)
    P = pad(cv, P, -0.4, -0.5, 0.4, 0.34, C(.40, .56, .26))                                   # the charbagh garden
    for (a, b) in [((-0.36, -0.05), (0.36, -0.05)), ((0.0, -0.22), (0.0, 0.32))]:
        seg(cv, P, [a, b], 0.03, C(.86, .80, .70))
        seg(cv, P, [a, b], 0.012, C(.24, .50, .62))
    P = trees(cv, P, [(x, y) for x in (-0.28, -0.12, 0.12, 0.28) for y in (0.06, 0.24)], rng, r=(0.035, 0.04), ht=(0.05, 0.06))
    u, v = 0.0, -0.34
    terrace(cv, P, u, v, 0.48, 0.24, 0.04, MARBLE * 0.97)
    bld(cv, P, u, v, 0.24, 0.18, 0.12, MARBLE, MARBLE, kind="flat", lift=0.04, shadow_a=0.0)
    poly(cv, P, [(u - 0.035, v + 0.09 - K * 0.04), (u + 0.035, v + 0.09 - K * 0.04), (u + 0.035, v + 0.09 - K * 0.14), (u - 0.035, v + 0.09 - K * 0.14)],
         C(.80, .78, .74))
    dome(cv, P, u, v, 0.1, 0.2, MARBLE, drum=MARBLE, tall=1.25)
    for du in (-0.08, 0.08):
        chhatri(cv, P, u + du, v + 0.05, 0.025, 0.16, MARBLE)
    for mu in (-0.22, 0.22):
        for mv in (v - 0.1, v + 0.1):
            minaret(cv, P, mu, mv, 0.016, 0.34, MARBLE, cap=MARBLE)
    square_walls(cv, P, hw, hh, 0.12, REDSTONE, "front", tower=red_tower, n_side=1, thick=0.06, gate=fort_gate)
    return P


# ============================================================================ Mesoamerican and Andean
PLASTER = [C(.90, .88, .82), C(.86, .82, .74), C(.92, .86, .76)]
THATCH = [C(.76, .64, .40), C(.70, .58, .36), C(.80, .70, .46)]


def mesoamerican(cv, P, rng, era):
    if era == 0:      # Olmec: thatched houses round an earthen mound and a plaza with colossal heads, maize fields
        P = over(P, RI.city_ground(cv, 0.86, 3301, C(.58, .52, .38)))
        d = np.hypot(cv.u, cv.v)
        rows = (np.sin(cv.u * 90) * 0.5 + 0.5) * smooth(0.5, 0.56, d) * smooth(0.84, 0.78, d) * (fbm(cv.a, cv.b, 3, 2, 3302) > 0.48)
        P = over(P, premult(col_arr(C(.50, .58, .26), cv.shape), rows * 0.7 * footprint(cv)))
        P = pad(cv, P, -0.22, -0.1, 0.22, 0.2, C(.74, .66, .52))
        cone(cv, P, 0.0, -0.22, 0.2, 0.18, C(.48, .52, .30))
        for u in (-0.12, 0.0, 0.12):
            blob(cv, P, u, 0.12, 0.03, 0.03, C(.40, .40, .40), lift=0.03, shadow_h=0.03)
        for u, v in rt.place_interior(rng, 20, 0.18, 0.11, lambda u, v: 0.28 < math.hypot(u, v) < 0.54):
            blob(cv, P, u, v, 0.05, 0.035, THATCH[rng.integers(3)], lift=0.04, shadow_h=0.04, rough=0.08)
        P = trees(cv, P, [p for p in scatter(rng, 14, 0.86, 0.12, rmin=0.62) if rt.hex_edge_dist(*p) > 0.08], rng, r=(0.05, 0.07), ht=(0.08, 0.1))
        return P
    if era == 1:      # Teotihuacan: the Avenue of the Dead, the Pyramids of the Sun and Moon, apartment compounds
        P = over(P, RI.city_ground(cv, 0.88, 3303, C(.66, .60, .50), (RI.grid_streets(cv, 7, 0.06) * 0.4, C(.76, .70, .60))))
        P = pad(cv, P, -0.07, -0.8, 0.07, 0.8, C(.82, .78, .70))
        keep = lambda u, v: u * u + v * v < 0.52 and abs(u) > 0.12 and not (0.08 < u < 0.56 and -0.32 < v < 0.16) and not (abs(u) < 0.22 and v < -0.4)
        packed_town(cv, P, rng, keep, 0.1, (0.03, 0.04), PLASTER + [C(.80, .64, .52)], court_p=0.7, tree_p=0.1, merge=0.6, extra=0.0)
        step_pyramid(cv, P, 0.0, -0.56, 0.24, 4, 0.05, C(.70, .66, .58), temple=C(.70, .26, .20))
        step_pyramid(cv, P, 0.32, -0.08, 0.42, 5, 0.06, C(.72, .68, .60), temple=C(.70, .26, .20))
        for du in (-0.2, -0.32):
            step_pyramid(cv, P, du, -0.18, 0.1, 2, 0.035, C(.72, .68, .60))
        return P
    if era == 2:      # Tenochtitlan: an island city of canals and causeways, the twin temple and the sacred precinct
        P = over(P, RI.city_ground(cv, 0.9, 3304, C(.64, .58, .46)))
        d = np.hypot(cv.u, cv.v * 1.05)
        lake = smooth(0.6, 0.64, d) * smooth(0.95, 0.9, d)
        rgb = col_arr(C(.20, .44, .46), cv.shape) * (0.9 + 0.2 * fbm(cv.a, cv.b, 8, 3, 3305))[..., None]
        garden = (np.sin(cv.u * 90) > 0.3) & (fbm(cv.a, cv.b, 5, 2, 3306) > 0.52)
        rgb[garden] = C(.36, .50, .26)
        P = over(P, premult(rgb, lake * footprint(cv)))
        canals = (smooth(0.012, 0.006, np.abs(cv.u - 0.36)) + smooth(0.012, 0.006, np.abs(cv.v - 0.32))) * smooth(0.62, 0.58, d)
        P = water(cv, P, np.clip(canals, 0, 1), 3307)
        for a, b in [((0.0, 0.58), (0.0, 0.95)), ((-0.58, -0.05), (-0.95, -0.05)), ((0.58, -0.05), (0.95, -0.05))]:
            seg(cv, P, [a, b], 0.045, C(.74, .68, .56))
        keep = lambda u, v: math.hypot(u, v * 1.05) < 0.54 and not (abs(u) < 0.32 and -0.44 < v < 0.14) and abs(u - 0.36) > 0.03 and abs(v - 0.32) > 0.03
        packed_town(cv, P, rng, keep, 0.08, (0.03, 0.04), PLASTER, court_p=0.5, tree_p=0.3, extra=0.05)
        P = pad(cv, P, -0.32, -0.44, 0.32, 0.14, C(.86, .82, .74))
        for a, b in [((-0.32, -0.44), (0.32, -0.44)), ((-0.32, -0.44), (-0.32, 0.14)), ((0.32, -0.44), (0.32, 0.14))]:
            wall_line(cv, P, a, b, 0.03, C(.90, .88, .82), 0.025, crenel=False)
        step_pyramid(cv, P, 0.0, -0.2, 0.36, 4, 0.065, C(.84, .80, .72), temple=[C(.72, .18, .14), C(.20, .40, .72)])
        cyl(cv, P, 0.0, 0.06, 0.05, 0.05, C(.84, .80, .72), C(.80, .76, .68))                 # round temple of the wind god
        cone(cv, P, 0.0, 0.06 - K * 0.05, 0.05, 0.06, C(.70, .58, .36), shadow_a=0.0)
        bld(cv, P, -0.2, 0.04, 0.12, 0.05, 0.03, C(.80, .76, .68), C(.84, .80, .72), kind="flat")
        bld(cv, P, 0.2, 0.04, 0.1, 0.08, 0.04, C(.84, .80, .72), C(.72, .18, .14), kind="flat")
        wall_line(cv, P, (-0.32, 0.14), (0.32, 0.14), 0.03, C(.90, .88, .82), 0.025, crenel=False)
        return P
    # Inca: terraced hillsides, the fortress's zigzag walls, walled compounds of stone halls, the golden temple
    P = over(P, RI.city_ground(cv, 0.9, 3308, C(.60, .58, .50)))
    d = np.hypot(cv.u, cv.v * 1.05)
    hill = smooth(0.1, -0.2, cv.v) * smooth(0.56, 0.62, d) * smooth(0.95, 0.88, d)
    rgb = col_arr(C(.40, .56, .26), cv.shape) * (0.9 + 0.12 * (np.floor(d * 30) % 2))[..., None] * (0.9 + 0.2 * fbm(cv.a, cv.b, 6, 3, 3309))[..., None]
    P = over(P, premult(rgb, hill * footprint(cv)))
    for k in range(6):
        r = 0.62 + 0.05 * k
        seg(cv, P, [(r * math.cos(t), r * 0.95 * math.sin(t)) for t in np.linspace(math.pi + 0.35, 2 * math.pi - 0.35, 30)], 0.005, C(.58, .54, .46))
    granite = C(.68, .66, .62)
    zig = [(-0.46 + 0.092 * k, -0.5 + (0.05 if k % 2 else 0.0)) for k in range(11)]
    for k in range(2):
        pts = [(x, y + 0.06 * k) for x, y in zig]
        for p0, p1 in zip(pts[:-1], pts[1:]):
            seg(cv, P, [p0, p1], 0.035, granite * 0.75)
            seg(cv, P, [p0, p1], 0.035, granite, lift=0.05 - 0.015 * k)
    P = pad(cv, P, -0.2, -0.04, 0.2, 0.22, C(.60, .66, .48))                                  # the great plaza

    def kancha(cu, cvv, w=0.2, dd=0.16):
        poly(cv, P, [(cu - w / 2, cvv - dd / 2), (cu + w / 2, cvv - dd / 2), (cu + w / 2, cvv + dd / 2), (cu - w / 2, cvv + dd / 2)], C(.56, .54, .48))
        for a2, b2 in [((cu - w / 2, cvv - dd / 2), (cu + w / 2, cvv - dd / 2)), ((cu - w / 2, cvv - dd / 2), (cu - w / 2, cvv + dd / 2)),
                       ((cu + w / 2, cvv - dd / 2), (cu + w / 2, cvv + dd / 2))]:
            wall_line(cv, P, a2, b2, 0.025, granite, 0.018, crenel=False)
        bld(cv, P, cu, cvv - dd * 0.22, w * 0.8, dd * 0.3, 0.035, granite, THATCH[rng.integers(3)], kind="gable", rh=0.05, ov=0.006)
        for du in (-0.28, 0.28):
            bld(cv, P, cu + du * w, cvv + dd * 0.14, w * 0.26, dd * 0.4, 0.03, granite, THATCH[rng.integers(3)], kind="gable", rh=0.04, ov=0.005)
        wall_line(cv, P, (cu - w / 2, cvv + dd / 2), (cu + w / 2, cvv + dd / 2), 0.025, granite, 0.018, crenel=False)

    cells = [(x, y) for y in (-0.3, -0.1, 0.12, 0.34) for x in (-0.48, -0.26, 0.26, 0.48)] + [(-0.1, 0.36), (0.12, 0.36)]
    for cu, cvv in sorted(cells, key=lambda c: c[1]):
        if math.hypot(cu, cvv * 1.05) < 0.62 and not (abs(cu) < 0.24 and -0.42 < cvv < 0.24):
            kancha(cu, cvv)
    bld(cv, P, 0.0, -0.24, 0.36, 0.14, 0.08, granite, C(.92, .74, .26), kind="gable", rh=0.08, ov=0.012)   # the golden temple
    poly(cv, P, [(-0.035, -0.17), (0.035, -0.17), (0.026, -0.17 - K * 0.055), (-0.026, -0.17 - K * 0.055)], C(.18, .16, .14))
    blob(cv, P, 0.0, 0.1, 0.03, 0.03, C(.66, .64, .60), lift=0.02, shadow_h=0.02)              # the ushnu platform
    return P


# ============================================================================ African
THATCH_A = C(.80, .68, .42)


def baobab(cv, P, u, v):
    cyl(cv, P, u, v, 0.03, 0.09, C(.56, .48, .42), C(.50, .44, .38))
    blob(cv, P, u, v, 0.08, 0.06, C(.42, .50, .24), lift=0.12, rough=0.15)


def compound(cv, P, rng, cu, cvv, n=4, r=0.11, hr=0.042, fence=C(.40, .34, .20), granary=True):
    seg(cv, P, [(cu + r * math.cos(a), cvv + r * 0.85 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 22)], 0.014, fence, lift=0.02)
    for a in np.linspace(0, 2 * math.pi, n + 1)[:-1] + rng.uniform(0, 1):
        hut(cv, P, cu + r * 0.58 * math.cos(a), cvv + r * 0.5 * math.sin(a), hr, roof=THATCH_A, wall=C(.62, .42, .28))
    if granary:
        hut(cv, P, cu, cvv, hr * 0.6, roof=C(.70, .58, .34), wall=C(.56, .40, .28))


def stela(cv, P, u, v, h, col=C(.62, .62, .64), w=0.026):
    shadow(cv, P, u, v, w, 0.02, h, alpha=0.4)
    poly(cv, P, [(u - w, v), (u + w, v), (u + w * 0.85, v - K * h), (u, v - K * (h + 0.035)), (u - w * 0.85, v - K * h)], col)
    poly(cv, P, [(u + 0.004, v), (u + w, v), (u + w * 0.85, v - K * h), (u + 0.004, v - K * (h + 0.03))], np.asarray(col) * 0.72)
    for k in range(1, 5):
        seg(cv, P, [(u - w * 0.7, v - K * h * k / 5), (u + w * 0.7, v - K * h * k / 5)], 0.004, np.asarray(col) * 0.6)


def djenne_mosque(cv, P, u, v, w, d, col):
    col = np.asarray(col, float)
    bld(cv, P, u, v, w, d, 0.11, col, col * 1.14, kind="flat")
    top = v - K * 0.11
    poly(cv, P, [(u - w * 0.38, top - d * 0.36), (u + w * 0.05, top - d * 0.36), (u + w * 0.05, top), (u - w * 0.38, top)], col * 0.8)
    for k in range(int(w / 0.04) + 1):                                                    # parapet pinnacles
        x = u - w / 2 + k * 0.04
        cone(cv, P, x, v + d / 2 - K * 0.11, 0.008, 0.025, col * 1.1, shadow_a=0.0)
    for du in (-w * 0.32, 0.0, w * 0.32):
        th = 0.3 if du == 0 else 0.23
        bld(cv, P, u + du, v + d / 2 - 0.02, 0.085, 0.07, th, col, col * 1.12, kind="flat", shadow_a=0.0)
        cone(cv, P, u + du, v + d / 2 - 0.02 - K * th, 0.035, 0.07, col * 1.12, shadow_a=0.0)
        blob(cv, P, u + du, v + d / 2 - 0.02 - K * (th + 0.075), 0.012, 0.012, C(.92, .88, .80), flat=True)
        for k in range(5):
            hh = 0.04 + 0.05 * k
            for sgn in (-1, 1):
                seg(cv, P, [(u + du + sgn * 0.044, v + d / 2 + 0.015 - K * hh), (u + du + sgn * 0.06, v + d / 2 + 0.02 - K * hh)], 0.008, C(.30, .20, .12))
    for k in range(int(w / 0.05)):                                                        # toron along the facade
        x = u - w / 2 + 0.03 + k * 0.05
        for hh in (0.04, 0.08):
            seg(cv, P, [(x, v + d / 2 + 0.002 - K * hh), (x, v + d / 2 + 0.012 - K * hh)], 0.007, C(.30, .20, .12))


def african(cv, P, rng, era):
    if era == 0:      # a kraal village: hut compounds inside a thorn fence, a cattle pen, baobabs
        P = over(P, RI.city_ground(cv, 0.86, 3401, C(.72, .56, .40)))
        ring = [(0.7 * math.cos(a), 0.62 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 60)]
        seg(cv, P, ring, 0.04, C(.30, .36, .18))
        for u, v in ring[::2]:
            blob(cv, P, u, v, 0.03, 0.025, C(.34, .40, .20), lift=0.02, rough=0.3)
        seg(cv, P, [(0.18 * math.cos(a), 0.15 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 30)], 0.016, C(.46, .34, .22), lift=0.025)
        for u, v in scatter(rng, 7, 0.12, 0.05):
            RI.animal(cv, P, u, v, rng.uniform(0, 3), 0.03, 0.017, [C(.30, .22, .16), C(.80, .76, .70), C(.50, .32, .20)][rng.integers(3)])
        for cu, cvv in scatter(rng, 7, 0.5, 0.24, rmin=0.32):
            compound(cv, P, rng, cu, cvv)
        baobab(cv, P, -0.52, -0.42)
        baobab(cv, P, 0.56, 0.36)
        return P
    if era == 1:      # Aksum: a field of tall carved stelae, the stone palace, houses and huts
        P = over(P, RI.city_ground(cv, 0.86, 3402, C(.68, .56, .44)))
        stone = C(.64, .58, .52)
        for cu, cvv in [(-0.44, 0.3), (0.44, 0.34), (-0.5, -0.06), (0.52, 0.0)]:
            compound(cv, P, rng, cu, cvv, n=3, r=0.1, granary=False)
        keep = lambda u, v: u * u + v * v < 0.34 and not (abs(u) < 0.4 and -0.5 < v < 0.1) and math.hypot(u + 0.44, v - 0.3) > 0.14 and math.hypot(u - 0.44, v - 0.34) > 0.14
        packed_town(cv, P, rng, keep, 0.09, (0.03, 0.045), [stone, stone * 1.08, C(.70, .56, .42)], court_p=0.3, extra=0.0)
        u, v = 0.18, -0.06                                                                 # the palace, on a stepped platform
        terrace(cv, P, u, v, 0.32, 0.28, 0.03, stone * 0.95)
        bld(cv, P, u, v, 0.22, 0.18, 0.1, stone, stone * 1.05, kind="flat", lift=0.03, shadow_a=0.0)
        for du in (-0.11, 0.11):
            for dv in (-0.09, 0.09):
                bld(cv, P, u + du, v + dv, 0.06, 0.06, 0.14, stone * 0.95, stone, kind="flat", lift=0.03, shadow_a=0.0)
        for k in range(4):
            seg(cv, P, [(u - 0.1, v + 0.09 - K * (0.06 + 0.02 * k)), (u + 0.1, v + 0.09 - K * (0.06 + 0.02 * k))], 0.004, C(.36, .26, .18))
        for du, h in [(-0.36, 0.5), (-0.26, 0.38), (-0.16, 0.3), (-0.06, 0.24)]:
            stela(cv, P, du, -0.3, h)
        poly(cv, P, [(-0.5, -0.4), (-0.24, -0.43), (-0.24, -0.4), (-0.5, -0.37)], C(.58, .58, .60))   # the great fallen stela
        return P
    if era == 2:      # Djenne: the great mud mosque, a market square, flat-roofed mud houses
        P = over(P, RI.city_ground(cv, 0.88, 3403, C(.80, .66, .50)))
        keep = lambda u, v: u * u / 0.5 + v * v / 0.36 < 1 and not (abs(u) < 0.34 and -0.5 < v < 0.14) and not (-0.2 < u < 0.2 and 0.16 < v < 0.4)
        packed_town(cv, P, rng, keep, 0.085, (0.04, 0.06), [C(.62, .44, .30), C(.58, .40, .26), C(.66, .48, .32)], court_p=0.5, tree_p=0.2, extra=0.1)
        P = pad(cv, P, -0.2, 0.18, 0.2, 0.4, C(.84, .72, .56))
        for u, v in scatter(rng, 10, 0.12, 0.045, center=(0.0, 0.29)):
            poly(cv, P, [(u - 0.016, v - 0.012), (u + 0.016, v - 0.012), (u + 0.016, v + 0.012), (u - 0.016, v + 0.012)],
                 [C(.80, .30, .20), C(.90, .70, .20), C(.20, .50, .60), C(.90, .88, .80)][rng.integers(4)], lift=0.02)
        djenne_mosque(cv, P, 0.0, -0.2, 0.52, 0.34, C(.74, .54, .36))
        return P
    # Gondar: a stone castle with domed round towers in a walled royal enclosure, round churches, huts
    P = over(P, RI.city_ground(cv, 0.9, 3404, C(.78, .70, .56)))
    stone = C(.50, .42, .36)
    P = pad(cv, P, -0.3, -0.3, 0.3, 0.12, C(.70, .64, .52))
    front = RI.wall_ring(cv, P, 0.56, 0.08, stone, towers=10, tower_r=0.035)
    for cu, cvv in scatter(rng, 9, 0.8, 0.2, rmin=0.66):
        if rt.hex_edge_dist(cu, cvv) > 0.14:
            compound(cv, P, rng, cu, cvv, n=3, r=0.08, granary=False)
    for u, v in [(-0.32, 0.26), (0.34, 0.28)]:                                              # round churches
        cyl(cv, P, u, v, 0.08, 0.05, C(.72, .64, .54), C(.72, .64, .54))
        cone(cv, P, u, v - K * 0.05, 0.1, 0.11, THATCH_A, shadow_a=0.0)
        seg(cv, P, [(u, v - K * 0.16), (u, v - K * 0.22)], 0.008, C(.85, .70, .30))
        seg(cv, P, [(u - 0.016, v - K * 0.2), (u + 0.016, v - K * 0.2)], 0.008, C(.85, .70, .30))
    u, v = 0.0, -0.1                                                                         # Fasil Ghebbi
    bld(cv, P, u, v, 0.4, 0.26, 0.2, stone, stone * 1.12, kind="flat")
    for row in range(2):
        for k in range(6):
            x = u - 0.16 + k * 0.064
            poly(cv, P, [(x - 0.008, v + 0.13 - K * (0.12 + 0.045 * row)), (x + 0.008, v + 0.13 - K * (0.12 + 0.045 * row)),
                         (x + 0.008, v + 0.13 - K * (0.145 + 0.045 * row)), (x - 0.008, v + 0.13 - K * (0.145 + 0.045 * row))], C(.22, .17, .14))
    for k in range(int(0.4 / 0.03)):
        seg(cv, P, [(u - 0.2 + k * 0.03 + 0.006, v + 0.13 - K * 0.206), (u - 0.2 + k * 0.03 + 0.02, v + 0.13 - K * 0.206)], 0.009, stone * 1.18)
    for k in range(3):
        poly(cv, P, [(u - 0.12 + k * 0.12 - 0.015, v + 0.13 - K * 0.05), (u - 0.12 + k * 0.12 + 0.015, v + 0.13 - K * 0.05),
                     (u - 0.12 + k * 0.12 + 0.015, v + 0.13 - K * 0.1), (u - 0.12 + k * 0.12 - 0.015, v + 0.13 - K * 0.1)], C(.26, .20, .16))
    bld(cv, P, u + 0.07, v - 0.04, 0.16, 0.12, 0.12, stone, stone * 1.08, kind="flat", lift=0.2, shadow_a=0.0)
    for du, dv, h in [(-0.2, -0.13, 0.3), (0.2, -0.13, 0.28), (-0.2, 0.13, 0.28), (0.2, 0.13, 0.36)]:
        cyl(cv, P, u + du, v + dv, 0.06, h, stone * 0.96, stone)
        dome(cv, P, u + du, v + dv, 0.066, h, C(.66, .30, .20), finial=C(.85, .70, .30))
    RI.wall_front(cv, P, front, 0.08, stone, towers=10, r=0.56, tower_r=0.035)
    return P


STYLES = {"East Asian": east_asian, "Middle Eastern": middle_east, "South Asian": south_asian, "Mesoamerican": mesoamerican,
          "African": african}


def render_era(style, era):
    """One style's city centre for one era, or the shared modern one from the Industrial era on."""
    cv = RI.canvas(RI.BIG_W, overhang=170)
    i = RI.CITY_ERAS.index(era)
    if era in ERAS:
        return cv, STYLES[style](cv, cv.empty(), np.random.default_rng(4000 + i * 17 + len(style)), i)
    return cv, RI.city_era(cv, cv.empty(), np.random.default_rng(300 + i), era)   # as render_items.render_cities draws it


def main():
    rt.FILTER = sys.argv[1:] or None
    for style, civs in REGIONS.items():
        for era in RI.CITY_ERAS:
            rels = ["Tiles/City center-%s-%s" % (civ, era) for civ in civs]
            if any(rt.wanted(rel) for rel in rels):
                cv, P = render_era(style, era)
                for rel in rels:
                    rt.save(cv, P, rel)
    print("Wrote %d regional city centres" % len(rt.WRITTEN))


if __name__ == "__main__":
    main()
