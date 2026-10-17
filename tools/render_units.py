#!/usr/bin/env python3
"""Render the unit set: units as painted miniatures, in the same light and projection as the map.

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
"""

import math
import sys
from contextlib import contextmanager

import numpy as np
from PIL import Image

import render_items as ri
import render_tiles as rt
from render_tiles import C, over, premult

K = rt.K_RISE
SS = rt.SS
UNIT_W = 160            # the same resolution as resources
LAYER_SCALE = 2         # the colour layer is stored at half resolution: it is flat, shaded colour
OVERHANG = 80
MIL = (0.56, 0.36)      # where a military unit stands: the middle of its width and its ground line (hex units,
CIV = (-0.6, 0.32)      # centre at 0, 0, v pointing south); civilians stand on the left
VIEW = np.array([0.0, K, 1.0]) / math.hypot(K, 1.0)   # toward the viewer: up and to the south
LIGHT = rt.LIGHT
HALF = (LIGHT + VIEW) / np.linalg.norm(LIGHT + VIEW)

PAINT = C(.86, .86, .86)    # parts painted in the owner's colour are this light grey in the images

# A few shared colours
SKIN = C(.80, .60, .46)
SKIN2 = C(.62, .44, .32)
HAIR = C(.20, .14, .09)
LEATHER = C(.42, .29, .17)
DARK_LEATHER = C(.26, .18, .11)
WOOD = C(.50, .36, .21)
DARK_WOOD = C(.33, .23, .14)
IRON = C(.55, .56, .58)
STEEL = C(.70, .72, .75)
BRONZE = C(.72, .52, .25)
LINEN = C(.82, .78, .66)
CLOTH = C(.48, .42, .34)
KHAKI = C(.55, .50, .36)
OLIVE = C(.36, .38, .22)
FIELD_GREY = C(.42, .44, .40)
NAVY_GREY = C(.52, .55, .58)
BLACK = C(.08, .08, .09)
WHITE = C(.92, .92, .90)
HORSE = C(.42, .26, .15)
HORSE2 = C(.24, .16, .10)
GOLD = C(.95, .75, .25)


# ============================================================================ layered painting
class Layers:
    """A unit's image and its owner-colour layer, painted together. Whatever is drawn in front
    of a coloured part also hides that part in the colour layer."""

    def __init__(self, cv):
        self.base = cv.empty()
        self.civ = cv.empty()
        self.shape = self.base.shape
        self.civ_on = False


def _put(P, win, rgb, a):
    x0, x1, y0, y1 = win[:4]
    src = premult(rgb, a)
    if not isinstance(P, Layers):
        P[y0:y1, x0:x1] = over(P[y0:y1, x0:x1], src)
        return
    P.base[y0:y1, x0:x1] = over(P.base[y0:y1, x0:x1], src)
    if P.civ_on:
        P.civ[y0:y1, x0:x1] = over(P.civ[y0:y1, x0:x1], src)
    else:
        P.civ[y0:y1, x0:x1] *= 1.0 - np.clip(np.broadcast_to(a, src.shape[:2]), 0, 1)[..., None]


ri._put = _put   # the item primitives draw through this, so they work on Layers too


# ============================================================================ the miniature painter
def shade_normal(n, gloss=0.0):
    """Brightness of a surface with normal(s) n (..., 3): sun from the west, fill from the viewer."""
    n = np.asarray(n, float)
    lam = np.clip(n @ LIGHT, 0, 1)
    fill = np.clip(n @ VIEW, 0, 1)
    out = np.clip(0.45 + 0.62 * lam + 0.3 * fill, 0, 1.15)
    if gloss:
        out = out + gloss * np.clip(n @ HALF, 0, 1) ** 24
    return out


def unit3(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


class Mini:
    """Draws a unit in local coordinates: x east, y south, z up, in units of `s` hex radii, with the
    unit standing at (u0, v0). Inside `turned(yaw)` the local frame is turned about its origin."""

    def __init__(self, cv, P, u0, v0, s=1.0):
        self.cv, self.P, self.u0, self.v0, self.s = cv, P, u0, v0, s
        self.yaw = 0.0

    # --- coordinates
    def w(self, p):
        """Local point to world (still in unit scale, before placing it on the hex)."""
        x, y, z = p
        if self.yaw:
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            x, y = x * c - y * s, x * s + y * c
        return np.array([x, y, z], float)

    def wv(self, d):
        """Local direction to world."""
        return self.w(d)

    def scr_w(self, q):
        return self.u0 + self.s * q[0], self.v0 + self.s * q[1] - K * self.s * q[2]

    def scr(self, x, y, z=0.0):
        return self.scr_w(self.w((x, y, z)))

    def gnd(self, x, y):
        return self.scr(x, y, 0.0)

    @contextmanager
    def turned(self, yaw):
        old, self.yaw = self.yaw, self.yaw + yaw
        try:
            yield
        finally:
            self.yaw = old

    def sub(self, p, k):
        """A painter whose origin is the local point p and whose figures are k times smaller."""
        u0, v0 = self.scr_w(self.w(p))
        m = Mini(self.cv, self.P, u0, v0, self.s * k)
        m.yaw = self.yaw
        return m

    @contextmanager
    def civ(self):
        """Parts drawn inside this block come out in the owner's colour."""
        self.P.civ_on = True
        try:
            yield
        finally:
            self.P.civ_on = False

    # --- primitives
    def shadow(self, x, y, rx, ry, alpha=0.32, ang=0.0):
        q = self.w((x, y, 0.0))
        ri.blob(self.cv, self.P, self.u0 + self.s * q[0], self.v0 + self.s * q[1], rx * self.s, ry * self.s,
                C(.02, .03, .02), ang=ang, alpha=alpha, flat=True)

    def ball(self, p, r, col, gloss=0.0, ry=None, alpha=1.0):
        q = self.w(p)
        ri.blob(self.cv, self.P, self.u0 + self.s * q[0], self.v0 + self.s * q[1], r * self.s, (ry or r) * self.s, col,
                lift=q[2] * self.s, gloss=gloss, shade_lo=0.42, alpha=alpha)

    def _poly_w(self, pts_w, col, alpha=1.0):
        ri.poly(self.cv, self.P, [self.scr_w(q) for q in pts_w], col, alpha=alpha)

    def face(self, pts, col, gloss=0.0, alpha=1.0, light=None):
        """Flat convex polygon through 3D points, shaded by its normal (either side may face us)."""
        p = np.array([self.w(q) for q in pts])
        n = unit3(np.cross(p[1] - p[0], p[2] - p[0]))
        if n @ VIEW < 0:
            n = -n
        k = shade_normal(n, gloss) if light is None else light
        self._poly_w(p, np.asarray(col, float) * k, alpha)

    def capsule(self, a, b, r0, r1=None, col=C(.5, .5, .5), gloss=0.0, alpha=1.0):
        """Tapered rod with rounded ends from a to b (local 3D points), shaded as a cylinder."""
        r1 = r0 if r1 is None else r1
        cv, P = self.cv, self.P
        aw, bw = self.w(a), self.w(b)
        xa, ya = cv.px(*self.scr_w(aw))
        xb, yb = cv.px(*self.scr_w(bw))
        R0, R1 = r0 * self.s * cv.spx, r1 * self.s * cv.spx
        rm = max(R0, R1)
        win = ri._box(cv, P, (xa + xb) / 2, (ya + yb) / 2, abs(xb - xa) / 2 + rm, abs(yb - ya) / 2 + rm)
        if win is None:
            return
        xx, yy = win[4], win[5]
        dx, dy = xb - xa, yb - ya
        ln = math.hypot(dx, dy)
        if ln < 1e-6:
            dx, dy, ln = 0.0, -1e-6, 1e-6
        ax, ay = dx / ln, dy / ln
        nx, ny = -ay, ax
        if nx < 0 or (nx == 0 and ny < 0):
            nx, ny = -nx, -ny
        along = (xx - xa) * ax + (yy - ya) * ay
        t = np.clip(along / ln, 0, 1)
        rr = np.maximum(R0 + (R1 - R0) * t, 0.3)
        ex, ey = xx - (xa + t * dx), yy - (ya + t * dy)
        d = np.hypot(ex, ey)
        cover = np.clip((rr - d) / SS + 0.5, 0, 1) * alpha
        if not cover.any():
            return
        # a 3D frame: the rod's axis, "front" toward the viewer, "right" across the rod on screen
        A = unit3(bw - aw)
        if np.linalg.norm(A) < 1e-9:
            A = np.array([0.0, 0.0, 1.0])
        F = VIEW - (VIEW @ A) * A
        F = unit3(F) if np.linalg.norm(F) > 1e-6 else VIEW
        R = np.cross(A, F)
        if R[0] * nx + (R[1] - K * R[2]) * ny < 0:
            R = -R
        sx = np.clip((ex * nx + ey * ny) / rr, -1, 1)
        over_end = np.where(along < 0, along, np.where(along > ln, along - ln, 0.0))
        cx = np.clip(over_end / rr, -1, 1)
        fz = np.sqrt(np.clip(1 - sx * sx - cx * cx, 0, 1))
        n = sx[..., None] * R + cx[..., None] * A + fz[..., None] * F
        k = shade_normal(n, gloss) * (0.82 + 0.18 * fz)
        ri._put(P, win, np.asarray(col, float) * k[..., None], cover)

    def rod(self, a, b, w, col, gloss=0.0):
        self.capsule(a, b, w / 2, w / 2, col, gloss)

    def disc(self, c, normal, r, col, gloss=0.0, n=20):
        """A flat round shape (shield, wheel) of radius r centred at c, facing `normal`."""
        nrm = unit3(normal)
        e1 = unit3(np.cross(nrm, [0, 0, 1]) if abs(nrm[2]) < 0.9 else np.cross(nrm, [1, 0, 0]))
        e2 = np.cross(nrm, e1)
        self.face([np.asarray(c, float) + r * (math.cos(2 * math.pi * i / n) * e1 + math.sin(2 * math.pi * i / n) * e2)
                   for i in range(n)], col, gloss)

    def loop(self, pts, w, col):
        for i in range(len(pts)):
            self.rod(pts[i], pts[(i + 1) % len(pts)], w, col)

    def _solid(self, quads, centre, cols, gloss=0.0):
        """Draw the faces of a convex solid that face the viewer, farthest first. quads are lists of
        local points; cols gives each face's colour."""
        cw = self.w(centre)
        drawn = []
        for q, col in zip(quads, cols):
            p = np.array([self.w(x) for x in q])
            n = unit3(np.cross(p[1] - p[0], p[2] - p[0]))
            if n @ (p.mean(axis=0) - cw) < 0:
                n = -n
            if n @ VIEW <= 1e-6:
                continue
            drawn.append((p.mean(axis=0) @ VIEW, p, n, col))
        for _, p, n, col in sorted(drawn, key=lambda d: d[0]):
            self._poly_w(p, np.asarray(col, float) * shade_normal(n, gloss))

    def block(self, c, size, col, yaw=0.0, top=None, gloss=0.0):
        """A box standing on c (x, y, bottom z), size (length along its heading, width, height),
        turned `yaw` radians from east toward south."""
        L, W, H = size
        cu, cs = math.cos(yaw), math.sin(yaw)
        fx, fy = np.array([cu, cs, 0.0]), np.array([-cs, cu, 0.0])
        o = np.asarray(c, float)
        P = {(i, j, k): o + sx * L / 2 * fx + sy * W / 2 * fy + np.array([0, 0, sz * H])
             for i, sx in enumerate((-1, 1)) for j, sy in enumerate((-1, 1)) for k, sz in enumerate((0, 1))}
        faces = [((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)), ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)),
                 ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)),
                 ((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1))]
        cols = [top if top is not None else col] + [col] * 4
        self._solid([[P[q] for q in f] for f in faces], o + np.array([0, 0, H / 2]), cols, gloss)

    def side_prism(self, profile, y0, y1, col, top=None, gloss=0.0, near=None):
        """An extrusion of a convex side profile (list of (x, z)) across y from y0 to y1."""
        n = len(profile)
        cx = sum(p[0] for p in profile) / n
        cz = sum(p[1] for p in profile) / n
        quads, cols = [], []
        for i in range(n):
            (x0, z0), (x1, z1) = profile[i], profile[(i + 1) % n]
            quads.append([(x0, y0, z0), (x1, y0, z1), (x1, y1, z1), (x0, y1, z0)])
            up = (x1 - x0) != 0 and abs((z1 - z0) / (x1 - x0)) < 0.6 and (cz < (z0 + z1) / 2)
            cols.append(top if (top is not None and up) else col)
        quads.append([(x, y1, z) for x, z in profile])
        cols.append(col if near is None else near)
        quads.append([(x, y0, z) for x, z in profile])
        cols.append(col if near is None else near)
        self._solid(quads, (cx, (y0 + y1) / 2, cz), cols, gloss)

    def prism(self, outline, z0, z1, col, top=None, gloss=0.0):
        """A vertical extrusion of a convex outline (list of (x, y)) from z0 to z1."""
        n = len(outline)
        cx = sum(p[0] for p in outline) / n
        cy = sum(p[1] for p in outline) / n
        quads = [[(outline[i][0], outline[i][1], z0), (outline[(i + 1) % n][0], outline[(i + 1) % n][1], z0),
                  (outline[(i + 1) % n][0], outline[(i + 1) % n][1], z1), (outline[i][0], outline[i][1], z1)] for i in range(n)]
        cols = [col] * n
        quads.append([(x, y, z1) for x, y in outline])
        cols.append(top if top is not None else col)
        self._solid(quads, (cx, cy, (z0 + z1) / 2), cols, gloss)


# ============================================================================ figures
OWNER = "owner"   # as a colour: paint this part in the owner's colour
FIG = 1.12        # figures are drawn a little larger than the vehicles, so they read as clearly


@contextmanager
def colour(m, col):
    """Yield the colour to draw with, switching to the owner-colour layer for OWNER."""
    if isinstance(col, str):
        with m.civ():
            yield PAINT
    else:
        yield col


def V(*xyz):
    return np.array(xyz, float)


class Body:
    """Joint positions of a figure facing the viewer, 1.0 tall, feet at (x, y, z)."""

    def __init__(self, x=0.0, y=0.0, z=0.0, lean=0.0):
        self.x, self.y, self.z = x, y, z
        f = lambda dx, h: np.array([x + dx + lean * h, y, z + h])
        self.foot_l, self.foot_r = f(-0.06, 0.0), f(0.06, 0.0)
        self.hip_l, self.hip_r = f(-0.055, 0.46), f(0.055, 0.46)
        self.waist = f(0.0, 0.5)
        self.chest = f(0.0, 0.7)
        self.sh_l, self.sh_r = f(-0.125, 0.72), f(0.125, 0.72)
        self.neck = f(0.0, 0.79)
        self.head = f(0.0, 0.885)


def ground(m, x=0.05, rx=0.22, ry=0.065, alpha=0.32):
    m.shadow(x, 0.02, rx, ry, alpha)


def legs(m, b, col, boots=LEATHER, stride=0.0):
    for hip, foot, sx in ((b.hip_l, b.foot_l, -1), (b.hip_r, b.foot_r, 1)):
        foot = foot + V(sx * stride, 0, 0)
        knee = (hip + foot) / 2 + V(sx * stride * 0.3, 0.01, 0.02)
        with colour(m, col) as c:
            m.capsule(hip, knee, 0.056, 0.048, c)
            m.capsule(knee, foot + V(0, 0, 0.1), 0.048, 0.042, c)
        with colour(m, boots) as c:
            m.capsule(foot + V(0, 0, 0.02), foot + V(0, 0, 0.14), 0.045, 0.045, c)
            m.ball(foot + V(0, 0.03, 0.025), 0.045, c, ry=0.032)


def head(m, b, skin=SKIN, hair=HAIR, beard=None):
    m.capsule(b.neck - V(0, 0, 0.02), b.neck + V(0, 0, 0.03), 0.036, 0.036, skin)
    m.ball(b.head, 0.078, skin)
    if hair is not None:
        m.ball(b.head + V(0.0, -0.015, 0.032), 0.07, hair, ry=0.056)
    if beard is not None:
        m.ball(b.head + V(0.0, 0.025, -0.045), 0.054, beard, ry=0.044)


def figure(m, b=None, legs_col=CLOTH, boots=LEATHER, coat=OWNER, skirt=None, skirt_len=0.15, robe=None,
           belt=None, skin=SKIN, hair=HAIR, beard=None, hat=None, hat_col=IRON, plume=None, wide=1.0, armour=None):
    """A standing figure facing the viewer. Returns its Body, for adding arms and what they hold."""
    b = b or Body()
    if robe is not None:
        for foot in (b.foot_l, b.foot_r):
            m.ball(foot + V(0, 0.035, 0.025), 0.045, boots, ry=0.032)
        with colour(m, robe) as c:
            m.capsule(b.waist + V(0, 0, 0.1), V(b.waist[0], b.y, b.z + 0.07), 0.11 * wide, 0.17 * wide, c)
    else:
        legs(m, b, legs_col, boots)
    if skirt is not None:
        with colour(m, skirt) as c:
            m.capsule(b.waist + V(0, 0, 0.02), b.waist - V(0, 0, skirt_len), 0.105 * wide, 0.14 * wide, c)
    with colour(m, coat) as c:
        m.capsule(b.waist, b.chest, 0.105 * wide, 0.13 * wide, c)
        m.capsule(b.sh_l + V(0.035, 0, 0), b.sh_r - V(0.035, 0, 0), 0.062, 0.062, c)
    if armour is not None:   # breastplate or mail over the coat
        m.capsule(b.waist + V(0, 0.015, 0.07), b.chest + V(0, 0.015, 0.0), 0.095 * wide, 0.118 * wide, armour, gloss=0.6)
    if belt is not None:
        m.capsule(b.waist + V(-0.1 * wide, 0.018, 0.03), b.waist + V(0.1 * wide, 0.018, 0.03), 0.026, 0.026, belt)
    head(m, b, skin, hair, beard)
    if hat is not None:
        helmet(m, b, hat_col, hat, plume=plume)
    return b


def arm(m, sh, hand, col, skin=SKIN):
    sh, hand = np.asarray(sh, float), np.asarray(hand, float)
    elbow = (sh + hand) / 2 + V(0, 0.0, -0.035)
    with colour(m, col) as c:
        m.capsule(sh, elbow, 0.042, 0.037, c)
        m.capsule(elbow, hand, 0.037, 0.032, c)
    m.ball(hand, 0.034, skin)


def helmet(m, b, col, kind="round", gloss=0.5, plume=None):
    hp = b.head
    with colour(m, col) as col:
        if kind == "round":
            m.ball(hp + V(0, -0.005, 0.028), 0.086, col, gloss=gloss, ry=0.07)
        elif kind == "cone":
            m.ball(hp + V(0, 0, 0.022), 0.084, col, gloss=gloss, ry=0.064)
            m.capsule(hp + V(0, 0, 0.05), hp + V(0, 0, 0.16), 0.056, 0.008, col, gloss)
        elif kind == "crest":          # Greek helmet; the crest is in the owner's colour
            m.ball(hp + V(0, -0.005, 0.028), 0.088, col, gloss=gloss, ry=0.076)
        elif kind == "brodie":         # First World War helmet with a wide brim
            m.ball(hp + V(0, 0, 0.035), 0.118, col, gloss=0.15, ry=0.045)
            m.ball(hp + V(0, -0.005, 0.06), 0.078, col, gloss=0.15, ry=0.058)
        elif kind == "m1":             # Second World War and later helmet
            m.ball(hp + V(0, -0.005, 0.032), 0.095, col, gloss=0.12, ry=0.08)
            m.rod(hp + V(-0.092, 0.005, 0.0), hp + V(0.092, 0.005, 0.0), 0.022, col * 0.8)
        elif kind == "kepi":
            m.capsule(hp + V(0, 0, 0.04), hp + V(0, -0.012, 0.135), 0.076, 0.066, col)
            m.ball(hp + V(0, 0.055, 0.035), 0.05, BLACK, ry=0.02)
        elif kind == "tricorne":
            m.ball(hp + V(0, 0, 0.045), 0.118, col, ry=0.05)
            m.ball(hp + V(0, 0, 0.075), 0.066, col, ry=0.055)
        elif kind == "bicorne":
            m.capsule(hp + V(-0.16, -0.01, 0.1), hp + V(0, -0.01, 0.18), 0.028, 0.06, col)
            m.capsule(hp + V(0, -0.01, 0.18), hp + V(0.16, -0.01, 0.1), 0.06, 0.028, col)
            m.ball(hp + V(0.02, 0.03, 0.15), 0.022, GOLD, gloss=0.5)
        elif kind == "beret":
            m.ball(hp + V(0.025, 0, 0.065), 0.092, col, ry=0.045)
        elif kind == "hood":
            m.ball(hp + V(0, -0.03, 0.03), 0.1, col, ry=0.09)
            m.ball(hp + V(0, 0.01, -0.005), 0.062, SKIN * 0.8)
        elif kind == "turban":
            m.ball(hp + V(0, -0.005, 0.055), 0.094, col, ry=0.062)
            m.ball(hp + V(0, -0.01, 0.09), 0.07, col * 1.05, ry=0.05)
        elif kind == "czapka":         # lancer's square-topped cap
            m.capsule(hp + V(0, 0, 0.05), hp + V(0, 0, 0.17), 0.062, 0.07, col)
            m.block(hp + V(0, 0, 0.17), (0.16, 0.16, 0.035), col)
        elif kind == "morion":         # crested helmet with an upturned brim
            m.ball(hp + V(0, 0, 0.035), 0.125, col, gloss=gloss, ry=0.045)
            m.ball(hp + V(0, -0.005, 0.07), 0.075, col, gloss=gloss, ry=0.065)
            m.capsule(hp + V(-0.05, 0, 0.13), hp + V(0.05, 0, 0.13), 0.022, 0.022, col, gloss)
        elif kind == "kettle":         # brimmed iron hat
            m.ball(hp + V(0, 0, 0.035), 0.12, col, gloss=gloss, ry=0.045)
            m.ball(hp + V(0, -0.005, 0.06), 0.077, col, gloss=gloss, ry=0.062)
        elif kind == "great":          # knight's great helm
            m.capsule(hp + V(0, 0, -0.045), hp + V(0, 0, 0.075), 0.086, 0.083, col, gloss)
            m.rod(hp + V(-0.065, 0.075, 0.0), hp + V(0.065, 0.075, 0.0), 0.014, BLACK)
        elif kind == "slouch":         # cavalry hat
            m.ball(hp + V(0, 0, 0.04), 0.125, col, ry=0.045)
            m.ball(hp + V(0, -0.005, 0.075), 0.07, col, ry=0.06)
        elif kind == "hardhat":
            m.ball(hp + V(0, 0, 0.035), 0.105, col, gloss=0.3, ry=0.04)
            m.ball(hp + V(0, -0.005, 0.06), 0.08, col, gloss=0.3, ry=0.065)
        elif kind == "cap":            # peaked cap
            m.ball(hp + V(0, -0.005, 0.07), 0.088, col, ry=0.04)
            m.ball(hp + V(0, 0.06, 0.035), 0.05, BLACK, ry=0.02)
        elif kind == "mitre":
            m.capsule(hp + V(0, 0, 0.04), hp + V(0, 0, 0.2), 0.07, 0.035, col)
    if kind == "crest":
        with m.civ():
            m.capsule(hp + V(0, -0.04, 0.1), hp + V(0, 0.06, 0.135), 0.036, 0.028, PAINT)
            m.capsule(hp + V(0, -0.04, 0.1), hp + V(0, -0.08, 0.03), 0.03, 0.02, PAINT)
    if plume is not None:
        with colour(m, plume) as c:
            m.capsule(hp + V(0, 0, 0.1), hp + V(0.045, 0.0, 0.22), 0.032, 0.02, c)


# --- what they carry -------------------------------------------------------------------------------
def spear(m, hand, length=1.15, lean=0.08, below=0.45, shaft=WOOD, tip=BRONZE, w=0.026):
    hand = np.asarray(hand, float)
    d = unit3([lean, 0, 1])
    lo, hi = hand - d * below, hand + d * (length - below)
    m.rod(lo, hi, w, shaft)
    m.capsule(hi - d * 0.01, hi + d * 0.14, w * 1.15, 0.004, tip, gloss=0.6)
    return hi


def sword(m, hand, d=(0.15, 0.0, 1.0), length=0.38, blade=STEEL, guard=BRONZE, w=0.024):
    hand = np.asarray(hand, float)
    d = unit3(d)
    side = unit3(np.cross(d, [0, 1, 0]))
    m.rod(hand - side * 0.05, hand + side * 0.05, 0.022, guard)
    m.capsule(hand + d * 0.02, hand + d * length, w, w * 0.4, blade, gloss=0.9)


def round_shield(m, c, r, col=OWNER, rim=BRONZE, normal=(0.15, 1.0, 0.25), boss=IRON):
    with colour(m, col) as cc:
        m.disc(c, normal, r, cc)
    nrm = unit3(normal)
    e1 = unit3(np.cross(nrm, [0, 0, 1]))
    e2 = np.cross(nrm, e1)
    m.loop([np.asarray(c) + r * (math.cos(t) * e1 + math.sin(t) * e2) for t in np.linspace(0, 2 * math.pi, 20, endpoint=False)],
           0.024, rim)
    if boss is not None:
        m.ball(np.asarray(c) + nrm * 0.02, r * 0.24, boss, gloss=0.6)


def tall_shield(m, c, w, h, col=OWNER, rim=BRONZE, boss=BRONZE, curve=0.03):
    """A tall rectangular or kite shield facing the viewer, centred at c."""
    c = np.asarray(c, float)
    pts = [c + V(-w / 2, -curve, h / 2), c + V(w / 2, -curve, h / 2), c + V(w / 2, -curve, -h / 2), c + V(-w / 2, -curve, -h / 2)]
    with colour(m, col) as cc:
        m.face([c + V(-w / 2, -curve, h / 2), c + V(0, 0, h / 2), c + V(0, 0, -h / 2), c + V(-w / 2, -curve, -h / 2)], cc)
        m.face([c + V(0, 0, h / 2), c + V(w / 2, -curve, h / 2), c + V(w / 2, -curve, -h / 2), c + V(0, 0, -h / 2)], cc)
    m.loop(pts, 0.02, rim)
    if boss is not None:
        m.ball(c + V(0, 0.02, 0), 0.04, boss, gloss=0.6)


def kite_shield(m, c, w, h, col=OWNER, rim=IRON):
    c = np.asarray(c, float)
    top = [c + V(-w / 2, 0, h * 0.35), c + V(w / 2, 0, h * 0.35), c + V(w * 0.42, 0, -h * 0.05), c + V(0, 0, -h * 0.65),
           c + V(-w * 0.42, 0, -h * 0.05)]
    with colour(m, col) as cc:
        m.face(top, cc)
    m.loop(top, 0.02, rim)


def bow_drawn(m, b, sleeve, recurve=False, col=DARK_WOOD):
    """Arms drawing a bow toward the west, with the bow and arrow."""
    hand_bow = b.sh_l + V(-0.32, 0.06, 0.02)
    arm(m, b.sh_l, hand_bow, sleeve)
    pts = []
    for t in np.linspace(-1.0, 1.0, 11):
        bend = -0.07 * (1 - t * t)
        if recurve:
            bend += 0.05 * max(0.0, abs(t) - 0.7) / 0.3
        pts.append(hand_bow + V(bend + 0.02, 0.0, 0.32 * t))
    for p, q in zip(pts[:-1], pts[1:]):
        m.rod(p, q, 0.024, col)
    hand_str = b.head + V(0.03, 0.05, -0.07)
    m.rod(pts[0], hand_str, 0.007, LINEN)
    m.rod(pts[-1], hand_str, 0.007, LINEN)
    m.rod(hand_str, hand_bow + V(-0.07, 0, 0.0), 0.013, LINEN)
    m.capsule(hand_bow + V(-0.06, 0, 0.0), hand_bow + V(-0.11, 0, 0.0), 0.016, 0.003, IRON)
    arm(m, b.sh_r, hand_str, sleeve)


def quiver(m, b):
    m.capsule(b.sh_r + V(-0.02, -0.06, 0.1), b.sh_r + V(0.02, -0.07, -0.22), 0.045, 0.04, LEATHER)
    for i in range(3):
        m.rod(b.sh_r + V(-0.03 + 0.016 * i, -0.06, 0.1), b.sh_r + V(-0.05 + 0.016 * i, -0.06, 0.19), 0.011, LINEN)


def long_gun(m, butt, muzzle, stock=DARK_WOOD, barrel=IRON, bayonet=False, w=0.03):
    butt, muzzle = np.asarray(butt, float), np.asarray(muzzle, float)
    mid = butt + (muzzle - butt) * 0.6
    m.rod(butt, mid, w, stock)
    m.rod(mid - (muzzle - butt) * 0.05, muzzle, w * 0.55, barrel)
    if bayonet:
        d = unit3(muzzle - butt)
        m.capsule(muzzle, muzzle + d * 0.14, 0.009, 0.002, STEEL, gloss=0.8)


def banner(m, base, height, w=0.24, hgt=0.16, pole=DARK_WOOD, finial=GOLD):
    """A pole with a cloth in the owner's colour, flying east."""
    base = np.asarray(base, float)
    top = base + V(0, 0, height)
    m.rod(base, top + V(0, 0, 0.03), 0.022, pole)
    m.ball(top + V(0, 0, 0.04), 0.022, finial, gloss=0.5)
    with m.civ():
        m.face([top, top + V(w, 0.0, -0.025), top + V(w * 0.96, 0.0, -hgt - 0.025), top + V(0, 0, -hgt)], PAINT)


def pennant(m, top, length=0.16, drop=0.03, h=0.06):
    top = np.asarray(top, float)
    with m.civ():
        m.face([top, top + V(length, 0, -drop), top + V(0, 0, -h)], PAINT)


# ============================================================================ horses
def horse(m, x, y, col=HORSE, mane=HORSE2, z=0.0, barding=None, gait=0.0, s=1.0):
    """A horse in profile facing west, about 1.0 long. Returns a function giving points on it
    (dx forward, dz up) for the rider and harness. barding: a colour (or OWNER) for a caparison."""
    P = lambda dx, dz, dy=0.0: np.array([x - dx * s, y + dy, z + dz * s])
    far = col * 0.78
    for dx, sw in ((0.27, 0.06 + gait), (-0.25, -0.05 - gait)):           # far legs
        m.capsule(P(dx, 0.44, -0.035), P(dx + sw * 0.5, 0.22, -0.035), 0.044 * s, 0.032 * s, far)
        m.capsule(P(dx + sw * 0.5, 0.22, -0.035), P(dx + sw, 0.04, -0.035), 0.03 * s, 0.026 * s, far)
        m.ball(P(dx + sw, 0.03, -0.035), 0.03 * s, BLACK, ry=0.022 * s)
    m.capsule(P(-0.38, 0.54), P(-0.48, 0.24, -0.01), 0.04 * s, 0.022 * s, mane)          # tail
    m.capsule(P(-0.3, 0.5), P(0.27, 0.53), 0.14 * s, 0.13 * s, col)                      # body
    m.capsule(P(0.25, 0.56), P(0.43, 0.83), 0.09 * s, 0.062 * s, col)                    # neck
    m.capsule(P(0.41, 0.85), P(0.6, 0.72), 0.065 * s, 0.04 * s, col)                     # head
    m.capsule(P(0.24, 0.66, -0.01), P(0.41, 0.9, -0.01), 0.034 * s, 0.026 * s, mane)     # mane
    m.ball(P(0.44, 0.92), 0.024 * s, col * 0.8)                                         # ear
    m.ball(P(0.5, 0.82, 0.04), 0.012 * s, BLACK)                                        # eye
    if barding is not None:
        with colour(m, barding) as c:
            m.capsule(P(-0.27, 0.5, 0.025), P(0.25, 0.53, 0.025), 0.15 * s, 0.14 * s, c)
            m.face([P(-0.36, 0.48, 0.12), P(0.31, 0.5, 0.12), P(0.29, 0.22, 0.12), P(-0.34, 0.22, 0.12)], c)
            m.capsule(P(0.26, 0.6, 0.02), P(0.41, 0.84, 0.02), 0.095 * s, 0.07 * s, c)
    for dx, sw in ((0.23, -0.03 - gait), (-0.21, 0.06 + gait)):          # near legs
        m.capsule(P(dx, 0.44, 0.04), P(dx + sw * 0.5, 0.22, 0.04), 0.05 * s, 0.036 * s, col)
        m.capsule(P(dx + sw * 0.5, 0.22, 0.04), P(dx + sw, 0.04, 0.04), 0.034 * s, 0.03 * s, col)
        m.ball(P(dx + sw, 0.03, 0.04), 0.034 * s, BLACK, ry=0.025 * s)
    return P


def rider(m, P, coat=OWNER, legs_col=None, hat=None, hat_col=IRON, plume=None, armour=None, skin=SKIN, beard=None,
          cape=None):
    """A rider on the horse made by horse(). Returns the rider's Body for adding arms and weapons."""
    seat = P(-0.02, 0.6)
    b = Body(seat[0], seat[1] + 0.02, seat[2] - 0.46)
    lc = legs_col if legs_col is not None else LEATHER
    if cape is not None:
        with colour(m, cape) as c:
            m.face([b.sh_l + V(0, -0.04, 0), b.sh_r + V(0.06, -0.04, 0), P(-0.32, 0.5, -0.04), P(-0.1, 0.42, -0.04)], c)
    with colour(m, lc) as c:
        m.capsule(b.hip_r + V(0, 0.06, 0), P(0.06, 0.38, 0.13), 0.05, 0.04, c)          # near leg down the flank
    m.ball(P(0.08, 0.34, 0.14), 0.038, BLACK, ry=0.028)
    with colour(m, coat) as c:
        m.capsule(b.waist, b.chest, 0.1, 0.125, c)
        m.capsule(b.sh_l + V(0.035, 0, 0), b.sh_r - V(0.035, 0, 0), 0.06, 0.06, c)
    if armour is not None:
        m.capsule(b.waist + V(0, 0.015, 0.07), b.chest, 0.09, 0.113, armour, gloss=0.6)
    head(m, b, skin, HAIR, beard)
    if hat is not None:
        helmet(m, b, hat_col, hat, plume=plume)
    return b


# ============================================================================ wheels, guns and vehicles
def wheel(m, c, r, col=DARK_WOOD, spokes=6, tyre=None, hub=IRON):
    """A spoked wheel turning east-west (its face toward the viewer)."""
    c = np.asarray(c, float)
    e1, e2 = V(1, 0, 0), V(0, 0, 1)
    m.disc(c, (0, 1, 0), r, col * 0.55)
    m.loop([c + r * (math.cos(t) * e1 + math.sin(t) * e2) for t in np.linspace(0, 2 * math.pi, 20, endpoint=False)],
           r * 0.24, tyre if tyre is not None else col)
    for i in range(spokes):
        t = math.pi * i / spokes
        d = r * 0.86 * (math.cos(t) * e1 + math.sin(t) * e2)
        m.rod(c - d, c + d, max(r * 0.12, 0.012), col)
    m.ball(c + V(0, 0.01, 0), r * 0.22, hub, gloss=0.5)


def solid_wheel(m, c, r, col=C(.12, .12, .12), hub=C(.4, .42, .38)):
    c = np.asarray(c, float)
    m.disc(c, (0, 1, 0), r, col)
    m.disc(c + V(0, 0.005, 0), (0, 1, 0), r * 0.55, hub)
    m.ball(c + V(0, 0.01, 0), r * 0.2, hub * 0.8)


def tracks(m, x0, x1, y, h, w, col=C(.17, .17, .16), wheels=6):
    """A caterpillar track running east-west with its road wheels."""
    m.side_prism([(x0, h * 0.55), (x0 + h * 0.5, 0.0), (x1 - h * 0.5, 0.0), (x1, h * 0.55), (x1 - h * 0.25, h), (x0 + h * 0.25, h)],
                 y - w / 2, y + w / 2, col, top=col * 1.2)
    for i in range(wheels):
        t = x0 + h * 0.5 + (x1 - x0 - h) * i / max(1, wheels - 1)
        m.disc((t, y + w / 2 + 0.004, h * 0.42), (0, 1, 0), h * 0.3, col * 1.9, gloss=0.2)


def truck(m, x0, x1, y, w, col, cab=0.22, h=0.16, bed_h=0.1, wheels=3, tyre=C(.1, .1, .1)):
    """A truck heading west: cab at x0, flat bed behind. Returns the bed height."""
    for i in range(wheels):
        solid_wheel(m, (x0 + 0.1 + (x1 - x0 - 0.2) * i / max(1, wheels - 1), y - w / 2, 0.075), 0.075, tyre)
    m.block(((x0 + x1) / 2, y, 0.08), (x1 - x0, w * 0.9, 0.05), C(.15, .15, .15))
    m.side_prism([(x0, 0.13), (x0 + 0.02, 0.13 + h * 0.6), (x0 + cab * 0.45, 0.13 + h), (x0 + cab, 0.13 + h), (x0 + cab, 0.13)],
                 y - w / 2, y + w / 2, col)
    m.face([(x0 + 0.03, y + w / 2 + 0.002, 0.13 + h * 0.62), (x0 + cab * 0.42, y + w / 2 + 0.002, 0.13 + h * 0.95),
            (x0 + cab * 0.85, y + w / 2 + 0.002, 0.13 + h * 0.95), (x0 + cab * 0.85, y + w / 2 + 0.002, 0.13 + h * 0.62)],
           C(.25, .32, .38), gloss=0.6)
    m.block((x0 + cab + (x1 - x0 - cab) / 2, y, 0.13), (x1 - x0 - cab, w, bed_h), col * 0.9)
    for i in range(wheels):
        solid_wheel(m, (x0 + 0.1 + (x1 - x0 - 0.2) * i / max(1, wheels - 1), y + w / 2, 0.075), 0.075, tyre)
    return 0.13 + bed_h


def barrel(m, a, b, r, col, gloss=0.3, muzzle=True):
    m.capsule(a, b, r, r * 0.85, col, gloss)
    if muzzle:
        d = unit3(np.subtract(b, a))
        m.capsule(np.asarray(b, float) - d * 0.03, np.asarray(b, float) + d * 0.01, r * 1.15, r * 1.15, col * 0.8, gloss)


def side_mark(m, x0, x1, y, z0, z1):
    """A painted panel in the owner's colour on a vehicle's near side."""
    with m.civ():
        m.face([(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)], PAINT)


# ============================================================================ ships and water
def wake(m, x0, x1, y, w=0.08, spray=True):
    """Shadow and foam around a hull, so ships sit in the water."""
    u0, v0 = m.gnd(x0, y)
    u1, v1 = m.gnd(x1, y)
    ri.blob(m.cv, m.P, (u0 + u1) / 2 + 0.03 * m.s, (v0 + v1) / 2 + 0.01, abs(u1 - u0) / 2 + 0.05 * m.s, w * m.s * 1.5,
            C(.02, .06, .08), alpha=0.28, flat=True)
    ri.blob(m.cv, m.P, (u0 + u1) / 2, (v0 + v1) / 2 + 0.01, abs(u1 - u0) / 2 + 0.03 * m.s, w * m.s * 1.05,
            C(.85, .92, .95), alpha=0.45, flat=True)
    if spray:
        ri.blob(m.cv, m.P, u0 + 0.01 * m.s, v0 + 0.005, 0.035 * m.s, w * m.s * 0.6, C(.95, .98, 1.0), alpha=0.45, flat=True)


def hull(m, x0, x1, y, w, h, col, deck=None, bow=0.18, stern=0.04, rail=None, stripe=None, sheer=0.0, boot=None):
    """A ship's hull pointing west (bow at x0), from the waterline (z=0) up to the deck (z=h)."""
    deck = col * 1.15 if deck is None else deck
    hw = w / 2
    outline = [(x0, y), (x0 + bow, y - hw), (x1 - stern, y - hw), (x1, y - hw * 0.7), (x1, y + hw * 0.7),
               (x1 - stern, y + hw), (x0 + bow, y + hw)]
    m.face([(x0, y, 0.0), (x0 + bow, y + hw, 0.0), (x0 + bow, y + hw, h), (x0 - 0.02, y, h + sheer)], col)
    m.face([(x0 + bow, y + hw, 0.0), (x1 - stern, y + hw, 0.0), (x1 - stern, y + hw, h), (x0 + bow, y + hw, h)], col)
    m.face([(x1 - stern, y + hw, 0.0), (x1, y + hw * 0.7, 0.0), (x1, y + hw * 0.7, h + sheer * 0.5), (x1 - stern, y + hw, h)], col)
    if boot is not None:   # waterline band
        m.face([(x0 + bow * 0.3, y + hw * 0.3, 0.0), (x1 - stern, y + hw + 0.002, 0.0), (x1 - stern, y + hw + 0.002, h * 0.22),
                (x0 + bow * 0.3, y + hw * 0.3, h * 0.22)], boot)
    if stripe is not None:
        with m.civ():
            m.face([(x0 + bow * 0.5, y + hw * 0.5 + 0.002, h * 0.6), (x1 - stern, y + hw + 0.002, h * 0.6),
                    (x1 - stern, y + hw + 0.002, h * 0.6 + stripe), (x0 + bow * 0.5, y + hw * 0.5 + 0.002, h * 0.6 + stripe)], PAINT)
    top = [(px, py, h) for px, py in outline]
    for i in range(1, len(top) - 1):
        m.face([top[0], top[i], top[i + 1]], deck, light=shade_normal([0, 0, 1]))
    for dy in (-hw / 3, hw / 3):                                                                    # deck seams
        m.rod((x0 + bow * 0.8, y + dy, h + 0.001), (x1 - stern * 1.5, y + dy, h + 0.001), 0.006, deck * 0.8)
    m.rod((x0 + bow * 0.4, y + hw * 0.6, h), (x0 + bow, y + hw, h), 0.016, col * 0.75)              # gunwale
    m.rod((x0 + bow, y + hw, h), (x1 - stern, y + hw, h), 0.016, rail if rail is not None else col * 0.75)


def mast(m, x, y, z0, h, col=DARK_WOOD, w=0.022):
    m.rod((x, y, z0), (x, y, z0 + h), w, col)


def square_sail(m, x, y, z0, z1, w, col=OWNER, belly=0.035, yard=DARK_WOOD):
    """A square sail across the ship, filled toward the viewer."""
    m.rod((x - w / 2 - 0.02, y, z1 + 0.01), (x + w / 2 + 0.02, y, z1 + 0.01), 0.015, yard)
    with colour(m, col) as c:
        m.face([(x - w / 2, y, z1), (x, y + belly, z1), (x, y + belly * 1.6, z0), (x - w / 2 * 0.94, y + belly, z0)], c)
        m.face([(x, y + belly, z1), (x + w / 2, y, z1), (x + w / 2 * 0.94, y + belly, z0), (x, y + belly * 1.6, z0)], c)


def lateen_sail(m, x, y, z0, z1, w, col=OWNER):
    """A triangular sail on a slanted yard, the long edge toward the bow (west)."""
    a, b, c = (x - w * 0.6, y + 0.01, z0 + 0.02), (x + w * 0.25, y, z1), (x + w * 0.45, y + 0.02, z0)
    m.rod((x - w * 0.68, y, z0 - 0.01), (x + w * 0.3, y, z1 + 0.04), 0.014, DARK_WOOD)
    with colour(m, col) as cc:
        m.face([a, b, c], cc)


def ensign(m, x, y, z, w=0.13, h=0.08, pole=0.18):
    """A flagstaff with a flag in the owner's colour."""
    m.rod((x, y, z), (x, y, z + pole), 0.012, DARK_WOOD)
    with m.civ():
        m.face([(x, y, z + pole), (x + w, y, z + pole - 0.01), (x + w, y, z + pole - h - 0.01), (x, y, z + pole - h)], PAINT)


def turret(m, x, y, z, r, col, guns=2, length=0.22, facing=-1, elev=0.06):
    """A gun turret with its barrels pointing west (facing -1) or east (+1)."""
    m.side_prism([(x - r, z), (x + r, z), (x + r * 0.8, z + r * 0.55), (x - r * 0.7, z + r * 0.6)], y - r * 0.75, y + r * 0.75, col)
    for i in range(guns):
        dy = (i - (guns - 1) / 2) * r * 0.45
        start = V(x + facing * r * 0.8, y + dy, z + r * 0.3)
        barrel(m, start, start + V(facing * length, 0, elev), r * 0.14, col * 0.9)


def funnel(m, x, y, z, h, r, col=C(.25, .25, .26), band=None, smoke_n=4):
    m.capsule((x, y, z), (x, y, z + h), r, r * 0.95, col)
    if band is not None:
        with colour(m, band) as c:
            m.capsule((x, y, z + h * 0.7), (x, y, z + h * 0.85), r * 1.02, r * 1.02, c)
    m.ball((x, y, z + h), r * 0.8, BLACK, ry=r * 0.4)
    for k in range(smoke_n):
        m.ball((x + 0.05 * (k + 1), y - 0.01 * k, z + h + 0.05 + 0.05 * k), r * (1.0 + 0.4 * k), C(.62, .62, .62),
               alpha=0.55 - 0.1 * k)


# ============================================================================ units
UNITS = {}


def unit(name, civilian=False, scale=FIG, at=None, wide=False):
    """Register a unit. wide: a low unit (ship, vehicle) that may reach toward the middle of the hex,
    under where the flags sit."""
    def deco(fn):
        UNITS[name] = dict(fn=fn, civilian=civilian, scale=scale, at=at, wide=wide and not civilian)
        return fn
    return deco


def armband(m, sh, hand):
    """A band in the owner's colour round the upper arm."""
    sh, hand = np.asarray(sh, float), np.asarray(hand, float)
    with m.civ():
        m.capsule(sh + (hand - sh) * 0.18, sh + (hand - sh) * 0.36, 0.046, 0.044, PAINT)


def scarf(m, b):
    with m.civ():
        m.capsule(b.neck + V(-0.065, 0.02, -0.03), b.neck + V(0.065, 0.02, -0.03), 0.04, 0.04, PAINT)


def pack(m, b, col, w=0.2, h=0.26):
    m.block(b.chest + V(0, -0.1, -h + 0.02), (w, 0.09, h), col)


# --- ancient and classical foot ----------------------------------------------------------------------
@unit("Scout")
def u_scout(m):
    ground(m)
    with m.civ():
        m.capsule(V(0.0, -0.05, 0.74), V(0.0, -0.05, 0.22), 0.14, 0.21, PAINT)          # cloak behind
    b = figure(m, legs_col=CLOTH, coat=LEATHER, belt=DARK_LEATHER, hat="hood", hat_col=OWNER)
    hand = b.sh_r + V(0.15, 0.07, -0.08)
    m.rod(hand + V(-0.015, 0, -0.63), hand + V(0.02, 0, 0.38), 0.03, WOOD)            # walking staff
    arm(m, b.sh_r, hand, LEATHER)
    arm(m, b.sh_l, b.head + V(-0.06, 0.08, 0.05), LEATHER)                           # shading the eyes


@unit("Warrior")
def u_warrior(m):
    ground(m)
    b = Body()
    hand = b.sh_l + V(-0.05, -0.03, 0.28)
    m.capsule(hand + V(0.01, 0, -0.05), hand + V(-0.04, 0, 0.32), 0.028, 0.06, DARK_WOOD)   # club raised behind
    arm(m, b.sh_l, hand, SKIN)
    figure(m, b, legs_col=SKIN, coat=SKIN, skirt=OWNER, skirt_len=0.19, hair=C(.12, .09, .06), beard=C(.14, .1, .07))
    m.capsule(b.sh_l + V(0.02, 0.035, 0.0), b.waist + V(0.1, 0.035, 0.0), 0.022, 0.022, C(.72, .66, .55))
    arm(m, b.sh_r, b.sh_r + V(0.08, 0.06, -0.26), SKIN)
    round_shield(m, b.sh_r + V(0.12, 0.13, -0.27), 0.17, col=C(.52, .38, .24), rim=DARK_WOOD, boss=None)


@unit("Archer")
def u_archer(m):
    ground(m)
    b = Body()
    quiver(m, b)
    figure(m, b, legs_col=CLOTH, coat=OWNER, skirt=OWNER, skirt_len=0.15, belt=LEATHER)
    bow_drawn(m, b, OWNER)


@unit("Composite Bowman")
def u_composite_bowman(m):
    ground(m)
    b = Body()
    quiver(m, b)
    figure(m, b, legs_col=C(.38, .3, .22), coat=OWNER, skirt=OWNER, skirt_len=0.17, armour=BRONZE * 0.85,
           belt=LEATHER, hat="cone", hat_col=BRONZE)
    bow_drawn(m, b, OWNER, recurve=True, col=C(.28, .18, .1))


@unit("Spearman")
def u_spearman(m):
    ground(m)
    b = Body()
    hand = b.sh_r + V(0.12, 0.06, -0.18)
    spear(m, hand, length=1.25, lean=0.06, below=0.5)
    figure(m, b, legs_col=SKIN, coat=LINEN, skirt=LINEN, skirt_len=0.15, armour=BRONZE, hat="crest", hat_col=BRONZE)
    for f in (b.foot_l, b.foot_r):
        m.capsule(f + V(0, 0.012, 0.15), f + V(0, 0.012, 0.3), 0.05, 0.05, BRONZE, gloss=0.5)   # greaves
    arm(m, b.sh_r, hand, SKIN)
    arm(m, b.sh_l, b.sh_l + V(-0.03, 0.09, -0.2), SKIN)
    round_shield(m, b.chest + V(-0.06, 0.14, -0.13), 0.23)


@unit("Swordsman")
def u_swordsman(m):
    ground(m)
    b = figure(m, legs_col=C(.45, .36, .28), coat=OWNER, skirt=OWNER, skirt_len=0.16, armour=IRON, belt=LEATHER,
               hat="round", hat_col=IRON, plume=OWNER)
    hand = b.sh_r + V(0.17, 0.06, 0.0)
    arm(m, b.sh_r, hand, OWNER)
    sword(m, hand, d=(0.2, 0, 1), length=0.38)
    arm(m, b.sh_l, b.sh_l + V(-0.02, 0.1, -0.2), OWNER)
    tall_shield(m, b.chest + V(-0.11, 0.15, -0.12), 0.21, 0.44)


# --- medieval foot -----------------------------------------------------------------------------------
@unit("Pikeman")
def u_pikeman(m):
    ground(m)
    b = Body()
    hand = b.sh_r + V(0.11, 0.07, -0.14)
    spear(m, hand, length=1.8, lean=0.025, below=0.62, tip=STEEL, w=0.025)
    figure(m, b, legs_col=OWNER, boots=BLACK, coat=OWNER, armour=STEEL, belt=LEATHER, hat="morion", hat_col=STEEL)
    arm(m, b.sh_r, hand, OWNER)
    arm(m, b.sh_l, hand + V(-0.02, 0.0, 0.24), OWNER)


@unit("Longswordsman")
def u_longswordsman(m):
    ground(m)
    b = figure(m, legs_col=STEEL, boots=STEEL, coat=STEEL, hat="great", hat_col=STEEL)
    with m.civ():                                                                    # surcoat over the armour
        m.capsule(b.waist + V(0, 0.022, -0.2), b.chest + V(0, 0.022, -0.02), 0.135, 0.1, PAINT)
    hand = b.sh_r + V(0.12, 0.09, 0.04)
    arm(m, b.sh_l, hand + V(-0.06, 0.02, -0.06), STEEL, skin=STEEL)
    sword(m, hand, d=(0.32, 0, 1), length=0.6, w=0.03)
    arm(m, b.sh_r, hand, STEEL, skin=STEEL)


@unit("Crossbowman")
def u_crossbowman(m):
    ground(m)
    b = figure(m, legs_col=CLOTH, coat=OWNER, skirt=OWNER, skirt_len=0.17, belt=LEATHER, hat="kettle", hat_col=IRON)
    top = b.chest + V(0.2, 0.14, 0.28)                       # crossbow held upright, the bow across the top
    m.rod(top + V(0, 0, -0.62), top + V(0, 0, 0.02), 0.042, WOOD)
    prod = [top + V(0.22 * t, 0.0, -0.07 * t * t) for t in np.linspace(-1, 1, 9)]
    for p, q in zip(prod[:-1], prod[1:]):
        m.rod(p, q, 0.026, IRON)
    m.rod(prod[0], top + V(0, 0.005, -0.14), 0.007, LINEN)
    m.rod(prod[-1], top + V(0, 0.005, -0.14), 0.007, LINEN)
    arm(m, b.sh_r, top + V(0.0, 0.0, -0.42), OWNER)
    arm(m, b.sh_l, top + V(-0.02, 0.0, -0.2), OWNER)


# --- gunpowder foot --------------------------------------------------------------------------------------
@unit("Musketman")
def u_musketman(m):
    ground(m)
    b = figure(m, legs_col=WHITE * 0.95, boots=BLACK, coat=OWNER, skirt=OWNER, skirt_len=0.13, hair=C(.85, .83, .8),
               hat="tricorne", hat_col=BLACK)
    m.rod(b.sh_l + V(0.02, 0.07, 0.0), b.waist + V(0.1, 0.07, -0.02), 0.026, WHITE)            # cross belts
    m.rod(b.sh_r + V(-0.02, 0.07, 0.0), b.waist + V(-0.1, 0.07, -0.02), 0.026, WHITE)
    hand_hi, hand_lo = b.sh_r + V(0.0, 0.09, 0.06), b.waist + V(0.08, 0.1, 0.04)
    long_gun(m, hand_lo + V(0.02, 0, -0.1), hand_hi + V(0.06, 0, 0.5), w=0.032)                 # musket on the shoulder
    arm(m, b.sh_r, hand_lo, OWNER)
    arm(m, b.sh_l, b.sh_l + V(-0.03, 0.06, -0.26), OWNER)


@unit("Rifleman")
def u_rifleman(m):
    ground(m)
    navy = C(.22, .24, .32)
    b = figure(m, legs_col=navy, boots=BLACK, coat=OWNER, skirt=OWNER, skirt_len=0.1, belt=DARK_LEATHER,
               hat="kepi", hat_col=OWNER)
    hand = b.chest + V(0.06, 0.13, -0.05)
    long_gun(m, hand + V(-0.17, 0, -0.12), hand + V(0.22, 0, 0.62), bayonet=True)               # rifle at port arms
    arm(m, b.sh_r, hand + V(0.1, 0, 0.12), OWNER)
    arm(m, b.sh_l, hand + V(-0.09, 0, -0.05), OWNER)


@unit("Great War Infantry")
def u_great_war_infantry(m):
    ground(m)
    b = Body()
    pack(m, b, C(.4, .37, .26))
    figure(m, b, legs_col=OWNER, boots=DARK_LEATHER, coat=OWNER, skirt=OWNER, skirt_len=0.22, belt=DARK_LEATHER,
           hat="brodie", hat_col=C(.36, .38, .3))
    for f in (b.foot_l, b.foot_r):
        m.capsule(f + V(0, 0.012, 0.08), f + V(0, 0.012, 0.26), 0.05, 0.05, KHAKI)                # puttees
    hand = b.chest + V(0.06, 0.13, -0.05)
    long_gun(m, hand + V(-0.17, 0, -0.12), hand + V(0.22, 0, 0.62), bayonet=True)
    arm(m, b.sh_r, hand + V(0.1, 0, 0.12), OWNER)
    arm(m, b.sh_l, hand + V(-0.09, 0, -0.05), OWNER)


def modern_soldier(m, uniform, helmet_col, gun=BLACK, kit=None):
    ground(m)
    b = Body()
    pack(m, b, kit if kit is not None else uniform * 0.85)
    figure(m, b, legs_col=uniform, boots=C(.15, .12, .1), coat=uniform * 1.06, belt=C(.45, .42, .3), hat="m1",
           hat_col=helmet_col)
    scarf(m, b)
    with m.civ():
        m.capsule(b.head + V(-0.09, 0.0, 0.035), b.head + V(0.09, 0.0, 0.035), 0.024, 0.024, PAINT)   # helmet band
    hand = b.chest + V(0.06, 0.13, -0.04)
    m.rod(hand + V(-0.14, 0, -0.1), hand + V(0.18, 0, 0.42), 0.036, gun)                        # rifle held ready
    m.rod(hand + V(0.0, 0.0, 0.08), hand + V(0.02, 0, -0.06), 0.03, gun)
    lh, rh = hand + V(-0.08, 0, -0.04), hand + V(0.09, 0, 0.12)
    arm(m, b.sh_l, lh, uniform)
    armband(m, b.sh_l, lh)
    arm(m, b.sh_r, rh, uniform)
    return b


@unit("Infantry")
def u_infantry(m):
    modern_soldier(m, OLIVE, OLIVE * 0.9, gun=DARK_WOOD)


@unit("Marine")
def u_marine(m):
    b = modern_soldier(m, C(.3, .33, .22), C(.42, .38, .26))
    for dx, dz in ((-0.05, 0.05), (0.04, 0.0), (0.0, -0.08)):                                    # camouflage
        m.ball(b.chest + V(dx, 0.11, dz), 0.04, C(.22, .2, .14), ry=0.03)


@unit("Paratrooper")
def u_paratrooper(m):
    m.shadow(0.12, 0.02, 0.22, 0.06, alpha=0.22)
    lift = 0.22
    b = Body(z=lift)
    canopy = [V(1.1 * math.cos(t) * 0.42, 0.0, 1.62 + 0.18 * math.sin(t)) for t in np.linspace(math.pi * 0.05, math.pi * 0.95, 9)]
    for p in (canopy[0], canopy[3], canopy[5], canopy[-1]):                                       # rigging lines
        m.rod(p, b.sh_l if p[0] < 0 else b.sh_r, 0.006, LINEN * 0.8)
    figure(m, b, legs_col=OLIVE, boots=C(.15, .12, .1), coat=OLIVE * 1.05, belt=C(.45, .42, .3), hat="m1", hat_col=OLIVE * 0.9)
    arm(m, b.sh_l, b.sh_l + V(-0.03, 0.05, 0.3), OLIVE)
    arm(m, b.sh_r, b.sh_r + V(0.03, 0.05, 0.3), OLIVE)
    with m.civ():                                                                               # the canopy
        for p, q in zip(canopy[:-1], canopy[1:]):
            m.face([p, q, q + V(0, 0.0, -0.1), p + V(0, 0.0, -0.1)], PAINT)
        m.face([canopy[0] + V(0, 0, -0.1), canopy[4] + V(0, 0, 0.05), canopy[-1] + V(0, 0, -0.1)], PAINT)
    for p in canopy[1:-1:2]:
        m.rod(p + V(0, 0.005, -0.1), p + V(0, 0.005, 0.0), 0.008, PAINT * 0.6)


# --- mounted -------------------------------------------------------------------------------------------------
MOUNT = 0.84     # horses are drawn smaller than figures, so horse and rider fit beside the flag
VEHICLE = 1.08


@unit("Horseman", scale=MOUNT)
def u_horseman(m):
    m.shadow(0.02, 0.02, 0.5, 0.08)
    P = horse(m, 0.0, 0.0)
    b = rider(m, P, coat=OWNER, legs_col=SKIN, hat="round", hat_col=BRONZE)
    hand = b.sh_r + V(0.06, 0.09, -0.12)
    m.rod(hand + V(0.18, 0, -0.07), hand + V(-0.46, 0, 0.14), 0.022, WOOD)                     # javelin held forward
    m.capsule(hand + V(-0.45, 0, 0.135), hand + V(-0.56, 0, 0.17), 0.02, 0.003, BRONZE, gloss=0.5)
    arm(m, b.sh_r, hand, SKIN)
    arm(m, b.sh_l, P(0.28, 0.66, 0.02), SKIN)


@unit("Chariot Archer", scale=0.76)
def u_chariot_archer(m):
    m.shadow(0.0, 0.02, 0.62, 0.1)
    P = horse(m, -0.22, 0.0, col=C(.55, .4, .26))
    wheel(m, (0.42, -0.14, 0.2), 0.2)
    m.rod(P(-0.2, 0.42), (0.3, 0.0, 0.3), 0.024, DARK_WOOD)                                       # draught pole
    b = Body(0.42, 0.0, 0.2)
    figure(m, b, legs_col=CLOTH, coat=OWNER, belt=LEATHER, hat="cone", hat_col=BRONZE)
    bow_drawn(m, b, OWNER)
    with m.civ():
        m.side_prism([(0.24, 0.2), (0.6, 0.2), (0.6, 0.42), (0.32, 0.56)], -0.13, 0.13, PAINT)       # the car
    wheel(m, (0.42, 0.15, 0.2), 0.2)


@unit("Knight", scale=MOUNT)
def u_knight(m):
    m.shadow(0.02, 0.02, 0.52, 0.09)
    P = horse(m, 0.0, 0.0, col=C(.66, .64, .62), mane=C(.3, .28, .26), barding=OWNER)
    b = rider(m, P, coat=STEEL, legs_col=STEEL, hat="great", hat_col=STEEL, plume=OWNER, armour=STEEL)
    hand = b.sh_r + V(0.1, 0.07, -0.14)
    top = hand + V(0.06, 0, 0.82)
    m.rod(hand + V(-0.02, 0, -0.32), top, 0.034, WOOD)                                           # lance, upright
    m.capsule(top, top + V(0.01, 0, 0.1), 0.02, 0.004, STEEL, gloss=0.6)
    pennant(m, top + V(0, 0, -0.04), 0.2, 0.05, 0.09)
    arm(m, b.sh_r, hand, STEEL, skin=STEEL)
    kite_shield(m, b.chest + V(-0.14, 0.12, -0.08), 0.2, 0.32)


@unit("Lancer", scale=MOUNT)
def u_lancer(m):
    m.shadow(0.02, 0.02, 0.5, 0.08)
    P = horse(m, 0.0, 0.0, col=C(.32, .2, .12), gait=0.05)
    b = rider(m, P, coat=OWNER, legs_col=C(.2, .2, .26), hat="czapka", hat_col=OWNER)
    hand = b.sh_r + V(0.04, 0.09, -0.1)
    tip = hand + V(-0.7, 0, 0.58)
    m.rod(hand + V(0.2, 0, -0.17), tip, 0.026, WOOD)                                               # lance couched forward
    m.capsule(tip, tip + V(-0.08, 0, 0.065), 0.018, 0.003, STEEL, gloss=0.6)
    pennant(m, tip + V(0.09, 0, -0.06), 0.16, -0.02, 0.07)
    arm(m, b.sh_r, hand, OWNER)
    arm(m, b.sh_l, P(0.28, 0.66, 0.02), OWNER)


@unit("Cavalry", scale=MOUNT)
def u_cavalry(m):
    m.shadow(0.02, 0.02, 0.5, 0.08)
    P = horse(m, 0.0, 0.0, col=C(.2, .14, .1), mane=C(.08, .06, .05), gait=0.08)
    b = rider(m, P, coat=OWNER, legs_col=C(.2, .2, .26), hat="slouch", hat_col=C(.3, .26, .2))
    hand = b.sh_r + V(0.14, 0.05, 0.22)
    arm(m, b.sh_r, hand, OWNER)
    sword(m, hand, d=(-0.25, 0, 1), length=0.42, guard=GOLD)                                         # sabre raised
    arm(m, b.sh_l, P(0.28, 0.66, 0.02), OWNER)




# --- siege and guns ------------------------------------------------------------------------------------------
GUN = C(.36, .38, .3)        # olive gun metal
GUN_GREY = C(.44, .46, .44)


@unit("Catapult", scale=VEHICLE, wide=True)
def u_catapult(m):
    m.shadow(0.04, 0.02, 0.44, 0.12)
    for x in (-0.24, 0.24):
        wheel(m, (x, -0.13, 0.11), 0.11)
    m.block((0.0, 0.0, 0.11), (0.64, 0.26, 0.06), WOOD)
    for y in (-0.11, 0.11):
        m.side_prism([(-0.06, 0.17), (0.06, 0.17), (0.02, 0.48), (-0.02, 0.48)], y - 0.025, y + 0.025, DARK_WOOD)
    m.rod((0.0, -0.13, 0.46), (0.0, 0.13, 0.46), 0.045, DARK_WOOD)                                # crossbar
    m.rod((0.22, 0.0, 0.18), (-0.2, 0.0, 0.62), 0.045, WOOD)                                      # throwing arm
    m.ball((-0.22, 0.0, 0.64), 0.075, LEATHER)
    m.ball((-0.22, 0.0, 0.69), 0.055, C(.52, .5, .47))                                              # the stone
    side_mark(m, 0.1, 0.3, 0.131, 0.11, 0.17)
    for x in (-0.24, 0.24):
        wheel(m, (x, 0.14, 0.11), 0.11)
    banner(m, (0.28, 0.06, 0.17), 0.46, w=0.15, hgt=0.11)


@unit("Trebuchet", scale=VEHICLE, wide=True)
def u_trebuchet(m):
    m.shadow(0.06, 0.02, 0.46, 0.13)
    for y in (-0.15,):
        m.block((0.0, y, 0.0), (0.72, 0.05, 0.05), DARK_WOOD)
        m.rod((-0.26, y, 0.05), (0.0, y, 0.72), 0.045, WOOD)
        m.rod((0.26, y, 0.05), (0.0, y, 0.72), 0.045, WOOD)
    m.rod((0.0, -0.17, 0.72), (0.0, 0.17, 0.72), 0.045, DARK_WOOD)                                # axle
    m.rod((-0.2, 0.0, 0.62), (0.42, 0.0, 1.2), 0.04, WOOD)                                        # the long arm, raised east
    m.rod((0.42, 0.0, 1.2), (0.4, 0.0, 0.92), 0.008, LINEN)                                       # sling with its stone
    m.ball((0.4, 0.0, 0.9), 0.05, C(.52, .5, .47))
    m.rod((-0.2, 0.0, 0.62), (-0.2, 0.0, 0.5), 0.02, IRON)
    with m.civ():                                                                                # counterweight box
        m.block((-0.2, 0.0, 0.22), (0.2, 0.2, 0.28), PAINT)
    m.loop([V(-0.3, 0.101, 0.22), V(-0.1, 0.101, 0.22), V(-0.1, 0.101, 0.5), V(-0.3, 0.101, 0.5)], 0.016, IRON)
    for y in (0.15,):
        m.block((0.0, y, 0.0), (0.72, 0.05, 0.05), DARK_WOOD)
        m.rod((-0.26, y, 0.05), (0.0, y, 0.72), 0.045, WOOD)
        m.rod((0.26, y, 0.05), (0.0, y, 0.72), 0.045, WOOD)


@unit("Cannon", scale=VEHICLE, wide=True)
def u_cannon(m):
    m.shadow(0.08, 0.02, 0.44, 0.1)
    wheel(m, (0.02, -0.15, 0.17), 0.17)
    m.side_prism([(-0.12, 0.16), (0.46, 0.02), (0.5, 0.07), (0.02, 0.3), (-0.12, 0.29)], -0.08, 0.08, WOOD)   # carriage
    side_mark(m, -0.06, 0.12, 0.082, 0.19, 0.25)
    barrel(m, (0.14, 0.0, 0.31), (-0.44, 0.0, 0.37), 0.062, BRONZE, gloss=0.6)
    m.ball((0.16, 0.0, 0.31), 0.06, BRONZE, gloss=0.6)
    wheel(m, (0.02, 0.15, 0.17), 0.17)
    for x, y in ((0.32, 0.16), (0.38, 0.17), (0.35, 0.12)):                                       # shot
        m.ball((x, y, 0.03), 0.032, BLACK, gloss=0.4)


@unit("Artillery", scale=VEHICLE, wide=True)
def u_artillery(m):
    m.shadow(0.1, 0.02, 0.46, 0.11)
    with m.turned(-0.8):                                                                          # aimed toward the viewer
        wheel(m, (0.0, -0.17, 0.18), 0.18, col=GUN * 0.9, tyre=C(.2, .2, .2), spokes=7)
        m.rod((0.04, 0.0, 0.2), (0.6, 0.0, 0.03), 0.06, GUN)                                         # trail
        m.capsule((0.08, 0.0, 0.3), (-0.5, 0.0, 0.36), 0.05, 0.042, GUN, gloss=0.2)                  # barrel and recoil gear
        m.capsule((0.04, 0.0, 0.25), (-0.22, 0.0, 0.28), 0.055, 0.055, GUN * 0.85)
        with m.civ():                                                                            # gun shield
            m.face([(-0.06, -0.16, 0.12), (-0.06, 0.16, 0.12), (-0.06, 0.16, 0.48), (-0.06, -0.16, 0.48)], PAINT)
        m.loop([V(-0.065, -0.16, 0.12), V(-0.065, 0.16, 0.12), V(-0.065, 0.16, 0.48), V(-0.065, -0.16, 0.48)], 0.014, GUN * 0.7)
        wheel(m, (0.0, 0.17, 0.18), 0.18, col=GUN * 0.9, tyre=C(.2, .2, .2), spokes=7)


@unit("Gatling Gun", scale=VEHICLE, wide=True)
def u_gatling_gun(m):
    m.shadow(0.06, 0.02, 0.42, 0.1)
    wheel(m, (0.0, -0.14, 0.16), 0.16)
    b = Body(0.34, -0.02, 0.0)
    figure(m, b, legs_col=C(.22, .24, .32), boots=BLACK, coat=OWNER, belt=DARK_LEATHER, hat="kepi", hat_col=OWNER)
    m.rod((0.04, 0.0, 0.18), (0.3, 0.0, 0.02), 0.05, WOOD)                                         # trail
    for dz, dy in ((0.0, -0.025), (0.035, 0.0), (0.0, 0.025), (-0.03, 0.0)):                      # the barrel cluster
        m.rod((0.12, dy, 0.34 + dz), (-0.4, dy, 0.38 + dz), 0.02, BRONZE * 0.9, gloss=0.5)
    m.capsule((0.14, 0.0, 0.35), (0.0, 0.0, 0.36), 0.07, 0.07, BRONZE, gloss=0.6)
    m.disc((-0.3, 0.0, 0.38), (1.0, 0.25, 0.3), 0.055, BRONZE * 0.8)
    m.ball((0.06, 0.0, 0.47), 0.05, BRONZE * 0.8)                                                  # magazine
    arm(m, b.sh_l, (0.16, 0.05, 0.4), OWNER)
    arm(m, b.sh_r, (0.22, 0.08, 0.36), OWNER)
    wheel(m, (0.0, 0.14, 0.16), 0.16)


@unit("Machine Gun", scale=VEHICLE, wide=True)
def u_machine_gun(m):
    m.shadow(0.04, 0.02, 0.4, 0.11)
    b = Body(0.12, -0.04, -0.3)
    figure(m, b, legs_col=OWNER, coat=OWNER, belt=DARK_LEATHER, hat="brodie", hat_col=C(.36, .38, .3))
    m.rod((-0.16, 0.0, 0.0), (-0.12, 0.0, 0.3), 0.022, GUN)                                         # tripod
    m.rod((-0.04, -0.06, 0.0), (-0.12, 0.0, 0.3), 0.022, GUN)
    m.capsule((0.0, 0.0, 0.33), (-0.48, 0.0, 0.35), 0.045, 0.04, GUN, gloss=0.2)                  # water jacket
    m.rod((-0.48, 0.0, 0.35), (-0.55, 0.0, 0.355), 0.02, BLACK)
    m.block((0.04, 0.0, 0.29), (0.1, 0.08, 0.09), GUN * 0.8)
    arm(m, b.sh_l, (0.0, 0.03, 0.36), OWNER)
    arm(m, b.sh_r, (0.06, 0.06, 0.33), OWNER)
    for i, x in enumerate(np.linspace(-0.4, 0.34, 7)):                                            # sandbags in front
        m.ball((x, 0.14 + 0.02 * math.sin(i), 0.07), 0.075, C(.62, .55, .4), ry=0.05)
    for x in np.linspace(-0.34, 0.28, 6):
        m.ball((x, 0.15, 0.17), 0.072, C(.66, .59, .43), ry=0.048)


@unit("Anti-Aircraft Gun", scale=VEHICLE, wide=True)
def u_anti_aircraft_gun(m):
    m.shadow(0.08, 0.02, 0.46, 0.13)
    m.block((0.0, 0.0, 0.0), (0.8, 0.07, 0.06), GUN * 0.8)                                        # outriggers
    m.block((0.0, 0.0, 0.0), (0.07, 0.42, 0.06), GUN * 0.8)
    m.capsule((0.0, 0.0, 0.06), (0.0, 0.0, 0.16), 0.15, 0.15, GUN)                                # turntable
    for y in (-0.05, 0.05):                                                                       # twin barrels
        barrel(m, (0.06, y, 0.3), (-0.42, y, 0.92), 0.03, GUN * 0.85)
    m.block((0.05, 0.0, 0.16), (0.24, 0.2, 0.18), GUN)
    with m.civ():
        m.face([(-0.04, 0.105, 0.18), (0.16, 0.105, 0.18), (0.16, 0.105, 0.33), (-0.04, 0.105, 0.33)], PAINT)
    m.ball((0.12, 0.13, 0.36), 0.035, BLACK)                                                       # sight


@unit("Anti-Tank Gun", scale=VEHICLE, wide=True)
def u_anti_tank_gun(m):
    m.shadow(0.1, 0.02, 0.48, 0.1)
    with m.turned(-0.8):
        solid_wheel(m, (0.0, -0.15, 0.12), 0.12)
        m.rod((0.02, -0.03, 0.12), (0.62, -0.16, 0.02), 0.04, GUN)                                  # split trail
        m.rod((0.02, 0.03, 0.12), (0.62, 0.16, 0.02), 0.04, GUN)
        m.capsule((0.1, 0.0, 0.22), (-0.6, 0.0, 0.25), 0.032, 0.028, GUN, gloss=0.2)
        m.capsule((-0.58, 0.0, 0.25), (-0.66, 0.0, 0.25), 0.04, 0.04, GUN * 0.8)                     # muzzle brake
        with m.civ():
            m.face([(-0.08, -0.2, 0.06), (-0.08, 0.2, 0.06), (-0.03, 0.17, 0.38), (-0.03, -0.17, 0.38)], PAINT)
        solid_wheel(m, (0.0, 0.15, 0.12), 0.12)


@unit("Rocket Artillery", scale=VEHICLE, wide=True)
def u_rocket_artillery(m):
    m.shadow(0.06, 0.02, 0.5, 0.13)
    bed = truck(m, -0.46, 0.4, 0.0, 0.26, OLIVE, cab=0.24)
    side_mark(m, -0.42, -0.26, 0.131, 0.16, 0.24)
    # the launcher: a box of tubes tilted up toward the front
    a, b = math.radians(28), 0.5
    base = (0.34, bed)
    prof = [base, (base[0] - b * math.cos(a), bed + b * math.sin(a)), (base[0] - b * math.cos(a) + 0.08, bed + b * math.sin(a) + 0.14),
            (base[0] + 0.08, bed + 0.14)]
    m.side_prism(prof, -0.12, 0.12, OLIVE * 0.95)
    fx, fz = prof[1]
    for i in range(3):
        for j in range(2):
            m.disc((fx + 0.02 + 0.025 * j, -0.08 + 0.08 * i, fz + 0.035 + 0.05 * j), (-math.sin(a), 0, math.cos(a) * 0.4 + 0.6),
                   0.022, BLACK)


@unit("Mobile SAM", scale=VEHICLE, wide=True)
def u_mobile_sam(m):
    m.shadow(0.06, 0.02, 0.48, 0.13)
    tracks(m, -0.38, 0.38, -0.13, 0.12, 0.09, wheels=5)
    m.block((0.0, 0.0, 0.09), (0.74, 0.26, 0.14), OLIVE)
    tracks(m, -0.38, 0.38, 0.13, 0.12, 0.09, wheels=5)
    side_mark(m, -0.3, -0.12, 0.131, 0.13, 0.21)
    m.block((0.12, 0.0, 0.23), (0.2, 0.18, 0.08), OLIVE * 0.9)
    for y in (-0.06, 0.06):                                                                       # missiles on their rails
        a0 = V(0.2, y, 0.32)
        d = unit3([-1.0, 0.0, 0.7])
        m.rod(a0 - d * 0.02, a0 + d * 0.5, 0.03, OLIVE * 0.7)
        m.capsule(a0 + d * 0.02 + V(0, 0, 0.04), a0 + d * 0.52 + V(0, 0, 0.04), 0.035, 0.035, WHITE)
        m.capsule(a0 + d * 0.5 + V(0, 0, 0.04), a0 + d * 0.62 + V(0, 0, 0.04), 0.035, 0.004, C(.7, .15, .1))
    m.capsule((0.3, 0.0, 0.31), (0.3, 0.0, 0.42), 0.012, 0.012, GUN)                               # radar
    m.disc((0.3, 0.0, 0.46), (0.2, 0.4, 0.6), 0.07, C(.6, .62, .6))


# --- armour ----------------------------------------------------------------------------------------------------
TRACK = C(.17, .17, .16)


@unit("Landship", scale=VEHICLE, wide=True)
def u_landship(m):
    m.shadow(0.08, 0.02, 0.52, 0.14)
    rhomb = [(-0.5, 0.26), (-0.32, 0.0), (0.34, 0.0), (0.46, 0.18), (0.36, 0.44), (-0.28, 0.44)]
    m.side_prism(rhomb, -0.17, -0.09, TRACK, top=TRACK * 1.3)
    m.block((0.0, 0.0, 0.12), (0.62, 0.2, 0.26), C(.42, .4, .3))
    m.side_prism(rhomb, 0.09, 0.17, TRACK, top=TRACK * 1.3)
    with m.civ():                                                                                # the sponson
        m.side_prism([(-0.12, 0.12), (0.12, 0.12), (0.1, 0.32), (-0.1, 0.32)], 0.17, 0.27, PAINT)
    barrel(m, (-0.1, 0.24, 0.22), (-0.32, 0.27, 0.22), 0.028, C(.3, .3, .26))
    for x in (0.52, 0.62):                                                                        # steering tail wheels
        solid_wheel(m, (x, 0.06, 0.08), 0.08, col=C(.32, .28, .2))
    m.rod((0.4, 0.05, 0.14), (0.62, 0.05, 0.08), 0.02, C(.32, .28, .2))


@unit("Tank", scale=VEHICLE, wide=True)
def u_tank(m):
    m.shadow(0.06, 0.03, 0.47, 0.14)
    tracks(m, -0.36, 0.36, -0.15, 0.14, 0.1)
    m.block((0.0, 0.0, 0.09), (0.7, 0.3, 0.12), OLIVE)
    tracks(m, -0.36, 0.36, 0.15, 0.14, 0.1)
    m.block((0.02, 0.0, 0.2), (0.6, 0.28, 0.06), OLIVE * 1.05)
    ring = [(-0.14, -0.11), (0.1, -0.12), (0.17, 0.0), (0.1, 0.12), (-0.14, 0.11), (-0.19, 0.0)]
    m.prism(ring, 0.26, 0.3, OLIVE)
    with m.civ():                                                                                # turret band
        m.prism(ring, 0.3, 0.34, PAINT)
    m.prism(ring, 0.34, 0.39, OLIVE)
    barrel(m, (-0.14, 0.0, 0.33), (-0.6, 0.0, 0.35), 0.022, OLIVE * 0.85)
    m.ball((0.04, -0.03, 0.41), 0.04, OLIVE * 0.9)


@unit("Modern Armor", scale=VEHICLE, wide=True)
def u_modern_armor(m):
    sand = C(.6, .55, .42)
    m.shadow(0.06, 0.03, 0.5, 0.14)
    tracks(m, -0.4, 0.38, -0.16, 0.12, 0.1, wheels=7)
    m.side_prism([(-0.44, 0.1), (-0.36, 0.2), (0.4, 0.2), (0.42, 0.08), (-0.36, 0.08)], -0.17, 0.17, sand)
    tracks(m, -0.4, 0.38, 0.16, 0.12, 0.1, wheels=7)
    m.block((0.0, 0.17, 0.11), (0.76, 0.03, 0.09), sand * 0.92)                                     # side skirt
    m.prism([(-0.22, -0.13), (0.16, -0.14), (0.2, 0.0), (0.16, 0.14), (-0.22, 0.13), (-0.3, 0.05), (-0.3, -0.05)],
            0.2, 0.31, sand)
    with m.civ():
        m.face([(-0.2, 0.131, 0.22), (0.14, 0.141, 0.22), (0.14, 0.141, 0.29), (-0.2, 0.131, 0.29)], PAINT)
    barrel(m, (-0.28, 0.0, 0.26), (-0.82, 0.0, 0.27), 0.022, sand * 0.85)
    m.capsule((-0.5, 0.0, 0.265), (-0.6, 0.0, 0.266), 0.032, 0.032, sand * 0.8)                       # fume extractor
    m.block((0.06, -0.05, 0.31), (0.08, 0.06, 0.05), sand * 0.85)


@unit("Mechanized Infantry", scale=VEHICLE, wide=True)
def u_mechanized_infantry(m):
    m.shadow(0.06, 0.03, 0.48, 0.14)
    tracks(m, -0.36, 0.36, -0.15, 0.12, 0.09, wheels=6)
    m.side_prism([(-0.42, 0.12), (-0.3, 0.3), (0.38, 0.3), (0.38, 0.08), (-0.34, 0.08)], -0.16, 0.16, OLIVE)
    tracks(m, -0.36, 0.36, 0.15, 0.12, 0.09, wheels=6)
    side_mark(m, 0.06, 0.3, 0.161, 0.15, 0.25)
    m.prism([(-0.12, -0.09), (0.08, -0.09), (0.12, 0.0), (0.08, 0.09), (-0.12, 0.09)], 0.3, 0.38, OLIVE * 0.9)
    barrel(m, (-0.1, 0.0, 0.34), (-0.42, 0.0, 0.35), 0.014, OLIVE * 0.7)
    b = Body(0.24, 0.0, 0.0)
    b.head = V(0.24, 0.0, 0.38)
    m.ball(b.head + V(0, 0, -0.03), 0.06, OLIVE * 0.8)                                               # commander in the hatch
    helmet(m, b, OLIVE * 0.85, "m1")


@unit("Helicopter Gunship", scale=VEHICLE, wide=True)
def u_helicopter_gunship(m):
    m.shadow(0.18, 0.04, 0.5, 0.1, alpha=0.24)
    z = 0.5
    body = C(.3, .34, .26)
    m.rod((-0.1, -0.08, z - 0.12), (0.12, -0.08, z - 0.12), 0.016, BLACK)                             # far skid
    m.capsule((0.1, 0.0, z + 0.03), (0.62, 0.0, z + 0.08), 0.04, 0.022, body)                        # tail boom
    m.face([(0.58, 0.0, z + 0.05), (0.66, 0.0, z + 0.08), (0.7, 0.0, z + 0.24), (0.64, 0.0, z + 0.24)], body)
    m.capsule((-0.3, 0.0, z), (0.14, 0.0, z + 0.03), 0.075, 0.1, body, gloss=0.2)                   # fuselage
    m.capsule((-0.28, 0.02, z + 0.06), (-0.12, 0.02, z + 0.1), 0.05, 0.06, C(.2, .28, .32), gloss=0.8)   # canopy
    with m.civ():
        m.capsule((0.24, 0.0, z + 0.05), (0.36, 0.0, z + 0.065), 0.045, 0.04, PAINT)
    m.rod((-0.08, 0.13, z - 0.03), (0.06, 0.13, z - 0.03), 0.03, body * 0.8)                         # stub wing and rockets
    m.capsule((-0.12, 0.15, z - 0.07), (0.08, 0.15, z - 0.07), 0.03, 0.03, C(.25, .25, .25))
    m.rod((-0.1, 0.08, z - 0.12), (0.12, 0.08, z - 0.12), 0.016, BLACK)                               # near skid
    m.rod((-0.05, 0.08, z - 0.12), (-0.05, 0.04, z - 0.04), 0.012, BLACK)
    m.rod((0.08, 0.08, z - 0.12), (0.08, 0.04, z - 0.04), 0.012, BLACK)
    m.capsule((-0.04, 0.0, z + 0.1), (-0.04, 0.0, z + 0.17), 0.02, 0.02, C(.2, .2, .2))              # rotor mast
    rot = (-0.04, 0.0, z + 0.17)
    m.face([V(*rot) + 0.52 * V(math.cos(t), math.sin(t) * 0.9, 0) for t in np.linspace(0, 2 * math.pi, 24, endpoint=False)],
           C(.75, .78, .8), alpha=0.18)                                                            # spinning blur
    for t in (0.35, 0.35 + math.pi / 2, 0.35 + math.pi, 0.35 + 1.5 * math.pi):
        m.rod(rot, V(*rot) + 0.52 * V(math.cos(t), math.sin(t) * 0.9, 0), 0.02, C(.12, .12, .12))


@unit("Giant Death Robot", scale=0.86)
def u_giant_death_robot(m):
    metal = C(.32, .33, .36)
    m.shadow(0.1, 0.02, 0.36, 0.1)
    for x, yo in ((-0.12, -0.02), (0.12, 0.02)):                                                       # legs
        m.block((x, yo + 0.03, 0.0), (0.16, 0.22, 0.06), metal * 0.8)
        m.capsule((x, yo, 0.06), (x + 0.02, yo, 0.32), 0.07, 0.06, metal)
        m.ball((x + 0.02, yo + 0.02, 0.33), 0.075, metal * 1.1, gloss=0.4)
        m.capsule((x + 0.02, yo, 0.33), (x * 0.6, yo, 0.58), 0.065, 0.075, metal)
    m.block((0.0, 0.0, 0.55), (0.3, 0.2, 0.12), metal * 0.85)
    m.prism([(-0.26, -0.12), (0.26, -0.12), (0.3, 0.06), (0.2, 0.15), (-0.2, 0.15), (-0.3, 0.06)], 0.66, 1.1, metal)
    with m.civ():
        m.face([(-0.2, 0.152, 0.72), (0.2, 0.152, 0.72), (0.2, 0.152, 1.0), (-0.2, 0.152, 1.0)], PAINT)
    m.ball((0.0, 0.16, 0.86), 0.06, C(.15, .15, .16))
    m.block((0.0, 0.0, 1.1), (0.16, 0.14, 0.1), metal * 0.9)                                            # head
    m.ball((0.0, 0.08, 1.15), 0.03, C(1.0, .2, .1), gloss=0.9)
    m.ball((0.0, 0.08, 1.15), 0.06, C(1.0, .3, .1), alpha=0.35)
    for sx in (-1, 1):                                                                                  # arms ending in guns
        m.ball((0.33 * sx, 0.0, 1.0), 0.09, metal * 1.1, gloss=0.4)
        m.capsule((0.34 * sx, 0.02, 0.98), (0.38 * sx, 0.06, 0.66), 0.065, 0.055, metal)
        barrel(m, (0.38 * sx, 0.08, 0.66), (0.36 * sx, 0.3, 0.46), 0.055, C(.2, .2, .22))




# --- ships ---------------------------------------------------------------------------------------------------------
SHIP = 1.15
AT_SEA = (0.54, 0.34)
CIV_SEA = (-0.6, 0.3)
TIMBER = C(.38, .26, .15)
NAVY = C(.5, .53, .56)
NAVY_DECK = C(.36, .38, .4)


def oars(m, xs, y, z, dy, length=0.16, col=WOOD):
    for x in xs:
        m.rod((x, y, z), (x - 0.05, y + dy * length, -0.01), 0.013, col)


@unit("Trireme", scale=SHIP, at=AT_SEA, wide=True)
def u_trireme(m):
    wake(m, -0.52, 0.42, 0.02, 0.12)
    oars(m, np.linspace(-0.24, 0.24, 7), -0.07, 0.1, -1)
    hull(m, -0.48, 0.42, 0.0, 0.17, 0.12, TIMBER, deck=C(.42, .32, .2), bow=0.12, sheer=0.06)
    m.capsule((-0.48, 0.0, 0.03), (-0.57, 0.0, 0.02), 0.022, 0.012, BRONZE, gloss=0.6)                 # ram
    m.capsule((0.38, 0.0, 0.12), (0.45, 0.0, 0.27), 0.024, 0.01, TIMBER)                             # curled stern
    mast(m, -0.02, 0.0, 0.12, 0.62)
    square_sail(m, -0.02, 0.0, 0.3, 0.68, 0.44)
    oars(m, np.linspace(-0.24, 0.24, 7), 0.085, 0.08, 1)
    m.ball((-0.42, 0.083, 0.08), 0.022, WHITE)                                                         # painted eye
    m.ball((-0.415, 0.093, 0.08), 0.01, BLACK)


@unit("Galley", scale=SHIP, at=AT_SEA, wide=True)
def u_galley(m):
    wake(m, -0.44, 0.36, 0.02, 0.1)
    oars(m, np.linspace(-0.18, 0.18, 5), -0.06, 0.09, -1)
    hull(m, -0.4, 0.36, 0.0, 0.15, 0.1, C(.3, .22, .14), deck=C(.4, .3, .19), bow=0.1, sheer=0.08)
    m.capsule((-0.4, 0.0, 0.1), (-0.46, 0.0, 0.26), 0.026, 0.014, C(.3, .22, .14))                 # high prow
    mast(m, 0.0, 0.0, 0.1, 0.55)
    square_sail(m, 0.0, 0.0, 0.26, 0.6, 0.38)
    oars(m, np.linspace(-0.18, 0.18, 5), 0.075, 0.07, 1)
    for x in (-0.1, 0.1, 0.24):                                                                       # shields along the side
        m.disc((x, 0.08, 0.09), (0, 1, 0.2), 0.04, C(.55, .42, .26))


@unit("Galleass", scale=SHIP, at=AT_SEA, wide=True)
def u_galleass(m):
    wake(m, -0.54, 0.44, 0.02, 0.13)
    oars(m, np.linspace(-0.26, 0.26, 8), -0.08, 0.1, -1)
    hull(m, -0.5, 0.44, 0.0, 0.2, 0.14, TIMBER, deck=C(.44, .34, .22), bow=0.16, sheer=0.04, stripe=0.03)
    m.block((-0.36, 0.0, 0.14), (0.14, 0.18, 0.08), TIMBER * 1.1)                                       # forecastle with guns
    for y in (-0.05, 0.05):
        barrel(m, (-0.36, y, 0.19), (-0.5, y, 0.2), 0.016, BRONZE)
    m.block((0.36, 0.0, 0.14), (0.14, 0.18, 0.12), TIMBER * 1.1)
    for x, h in ((-0.2, 0.62), (0.04, 0.72), (0.26, 0.5)):
        mast(m, x, 0.0, 0.14, h)
        lateen_sail(m, x, 0.02, 0.24, 0.14 + h, 0.36 if h > 0.6 else 0.26)
    oars(m, np.linspace(-0.26, 0.26, 8), 0.095, 0.08, 1)


@unit("Caravel", scale=SHIP, at=AT_SEA, wide=True)
def u_caravel(m):
    wake(m, -0.46, 0.4, 0.02, 0.12)
    hull(m, -0.42, 0.4, 0.0, 0.2, 0.15, C(.42, .3, .18), deck=C(.46, .36, .23), bow=0.16, sheer=0.07)
    m.block((0.3, 0.0, 0.15), (0.18, 0.2, 0.1), C(.42, .3, .18) * 1.1)                                    # sterncastle
    m.rod((-0.42, 0.0, 0.17), (-0.56, 0.0, 0.26), 0.016, DARK_WOOD)
    for x, h, w in ((-0.18, 0.62, 0.32), (0.06, 0.78, 0.4), (0.3, 0.6, 0.3)):
        mast(m, x, 0.0, 0.15, h)
        lateen_sail(m, x, 0.02, 0.26, 0.15 + h, w, col=OWNER)
    m.face([(-0.42, 0.0, 0.18), (-0.18, 0.0, 0.7), (-0.2, 0.0, 0.22)], LINEN)                             # jib
    pennant(m, V(0.06, 0.0, 0.95), 0.16)


@unit("Privateer", scale=SHIP, at=AT_SEA, wide=True)
def u_privateer(m):
    wake(m, -0.48, 0.42, 0.02, 0.12)
    hull(m, -0.44, 0.42, 0.0, 0.19, 0.14, C(.1, .09, .08), deck=C(.5, .42, .3), bow=0.16, sheer=0.04)
    for i in range(4):
        x = -0.18 + 0.12 * i
        m.face([(x, 0.096, 0.06), (x + 0.04, 0.096, 0.06), (x + 0.04, 0.096, 0.09), (x, 0.096, 0.09)], C(.5, .1, .08))
    m.rod((-0.44, 0.0, 0.15), (-0.62, 0.0, 0.24), 0.016, DARK_WOOD)
    m.face([(-0.6, 0.0, 0.24), (-0.14, 0.0, 0.86), (-0.16, 0.0, 0.2)], C(.3, .3, .3))                    # jib
    for x, h in ((-0.12, 0.82), (0.18, 0.86)):
        mast(m, x, 0.0, 0.14, h)
        square_sail(m, x, 0.0, 0.3, 0.3 + h * 0.36, 0.3, col=C(.3, .3, .3))
        square_sail(m, x, 0.0, 0.32 + h * 0.38, 0.2 + h * 0.76, 0.24, col=C(.3, .3, .3))
    m.face([(0.18, 0.0, 0.32), (0.5, 0.0, 0.3), (0.18, 0.0, 0.85)], C(.32, .32, .32))                     # spanker
    ensign(m, 0.42, 0.0, 0.14, w=0.14, h=0.09, pole=0.22)


@unit("Frigate", scale=SHIP, at=AT_SEA, wide=True)
def u_frigate(m):
    wake(m, -0.5, 0.44, 0.02, 0.12)
    hull(m, -0.46, 0.44, 0.0, 0.2, 0.16, C(.15, .12, .1), deck=C(.47, .38, .27), bow=0.16, sheer=0.04, stripe=0.04)
    for i in range(6):                                                                                   # gun ports
        x = -0.24 + 0.1 * i
        m.face([(x, 0.101, 0.065), (x + 0.04, 0.101, 0.065), (x + 0.04, 0.101, 0.095), (x, 0.101, 0.095)], BLACK)
    m.rod((-0.46, 0.0, 0.16), (-0.66, 0.0, 0.24), 0.016, DARK_WOOD)                                       # bowsprit
    for x, h in ((-0.24, 0.85), (0.04, 0.95), (0.3, 0.75)):
        mast(m, x, 0.0, 0.16, h)
        square_sail(m, x, 0.0, 0.3, 0.3 + h * 0.4, 0.28, col=WHITE)
        square_sail(m, x, 0.0, 0.32 + h * 0.42, 0.22 + h * 0.8, 0.22, col=WHITE)
    pennant(m, V(0.04, 0.0, 1.11), 0.2)
    ensign(m, 0.44, 0.0, 0.16, w=0.14, h=0.09, pole=0.22)


@unit("Ironclad", scale=SHIP, at=AT_SEA, wide=True)
def u_ironclad(m):
    wake(m, -0.5, 0.42, 0.02, 0.12)
    hull(m, -0.46, 0.42, 0.0, 0.2, 0.08, C(.14, .14, .15), deck=C(.32, .3, .28), bow=0.14, boot=C(.4, .16, .12))
    funnel(m, 0.12, 0.0, 0.08, 0.28, 0.05, col=C(.18, .18, .19), smoke_n=4)
    m.capsule((-0.14, 0.0, 0.08), (-0.14, 0.0, 0.2), 0.12, 0.12, C(.22, .22, .23))                        # turret
    with m.civ():
        m.capsule((-0.14, 0.0, 0.16), (-0.14, 0.0, 0.2), 0.122, 0.122, PAINT)
    for y in (-0.04, 0.04):
        barrel(m, (-0.2, y, 0.14), (-0.36, y, 0.145), 0.022, C(.15, .15, .15))
    mast(m, 0.3, 0.0, 0.08, 0.4, col=C(.2, .2, .2))
    ensign(m, 0.4, 0.0, 0.08, w=0.12, h=0.08, pole=0.2)


@unit("Destroyer", scale=SHIP, at=AT_SEA, wide=True)
def u_destroyer(m):
    wake(m, -0.54, 0.44, 0.02, 0.1)
    hull(m, -0.5, 0.44, 0.0, 0.16, 0.1, NAVY, deck=NAVY_DECK, bow=0.2, sheer=0.04, boot=C(.18, .18, .2), stripe=0.025)
    turret(m, -0.3, 0.0, 0.1, 0.06, NAVY, guns=1, length=0.16)
    m.block((-0.14, 0.0, 0.1), (0.14, 0.12, 0.14), NAVY * 1.05)                                           # bridge
    m.block((-0.15, 0.0, 0.24), (0.08, 0.1, 0.05), C(.2, .24, .28))
    mast(m, -0.08, 0.0, 0.24, 0.3, col=NAVY * 0.8, w=0.016)
    for x in (0.04, 0.16):
        funnel(m, x, 0.0, 0.1, 0.16, 0.035, col=NAVY * 0.9, smoke_n=2)
    turret(m, 0.32, 0.0, 0.1, 0.06, NAVY, guns=1, length=0.14, facing=1)
    ensign(m, 0.42, 0.0, 0.1, w=0.12, h=0.07, pole=0.18)


@unit("Battleship", scale=SHIP, at=AT_SEA, wide=True)
def u_battleship(m):
    wake(m, -0.56, 0.46, 0.02, 0.14)
    hull(m, -0.52, 0.46, 0.0, 0.24, 0.12, NAVY, deck=NAVY_DECK, bow=0.22, sheer=0.03, boot=C(.2, .2, .22), stripe=0.03)
    turret(m, -0.34, 0.0, 0.12, 0.08, NAVY, guns=2, length=0.2)
    turret(m, -0.2, 0.0, 0.16, 0.08, NAVY, guns=2, length=0.2, elev=0.04)
    m.block((-0.2, 0.0, 0.12), (0.1, 0.16, 0.04), NAVY)
    m.prism([(-0.1, -0.07), (0.04, -0.07), (0.06, 0.07), (-0.1, 0.07)], 0.12, 0.42, NAVY * 1.05)         # tower
    m.block((-0.03, 0.0, 0.42), (0.1, 0.1, 0.06), NAVY * 0.95)
    mast(m, -0.03, 0.0, 0.48, 0.2, col=NAVY * 0.8, w=0.016)
    funnel(m, 0.12, 0.0, 0.12, 0.2, 0.05, col=NAVY * 0.9, smoke_n=2)
    turret(m, 0.32, 0.0, 0.12, 0.08, NAVY, guns=2, length=0.2, facing=1)
    ensign(m, 0.44, 0.0, 0.12, w=0.13, h=0.08, pole=0.2)


@unit("Submarine", scale=SHIP, at=AT_SEA, wide=True)
def u_submarine(m):
    wake(m, -0.46, 0.42, 0.02, 0.09)
    hull(m, -0.46, 0.42, 0.0, 0.12, 0.05, C(.22, .24, .25), deck=C(.3, .32, .32), bow=0.18, stern=0.12)
    m.prism([(-0.06, -0.035), (0.1, -0.035), (0.13, 0.0), (0.1, 0.035), (-0.06, 0.035)], 0.05, 0.19, C(.24, .26, .27))
    with m.civ():
        m.face([(-0.06, 0.036, 0.12), (0.1, 0.036, 0.12), (0.1, 0.036, 0.17), (-0.06, 0.036, 0.17)], PAINT)
    mast(m, 0.02, 0.0, 0.19, 0.12, col=C(.2, .2, .2), w=0.014)
    barrel(m, (-0.2, 0.0, 0.08), (-0.3, 0.0, 0.1), 0.014, C(.2, .2, .2))                               # deck gun
    m.block((-0.2, 0.0, 0.05), (0.05, 0.05, 0.04), C(.24, .26, .27))


@unit("Nuclear Submarine", scale=SHIP, at=AT_SEA, wide=True)
def u_nuclear_submarine(m):
    wake(m, -0.52, 0.44, 0.02, 0.1)
    black = C(.1, .1, .11)
    m.capsule((-0.5, 0.0, 0.0), (0.38, 0.0, 0.0), 0.085, 0.07, black, gloss=0.4)                      # rounded hull, half under
    m.face([(0.4, 0.0, 0.0), (0.48, 0.0, 0.0), (0.5, 0.0, 0.12), (0.44, 0.0, 0.1)], black)             # rudder
    m.prism([(-0.22, -0.03), (-0.04, -0.03), (0.0, 0.0), (-0.04, 0.03), (-0.22, 0.03)], 0.05, 0.25, black * 1.3)
    m.rod((-0.15, -0.14, 0.2), (-0.15, 0.14, 0.2), 0.022, black * 1.3)                                  # dive planes
    with m.civ():
        m.face([(-0.22, 0.031, 0.17), (-0.04, 0.031, 0.17), (-0.04, 0.031, 0.22), (-0.22, 0.031, 0.22)], PAINT)


@unit("Carrier", scale=SHIP, at=AT_SEA, wide=True)
def u_carrier(m):
    wake(m, -0.58, 0.5, 0.02, 0.16)
    hull(m, -0.56, 0.5, 0.0, 0.3, 0.12, NAVY, deck=C(.3, .31, .32), bow=0.16, stern=0.02, boot=C(.2, .2, .22))
    m.rod((-0.48, 0.0, 0.122), (0.46, 0.0, 0.122), 0.008, WHITE)                                        # centre line
    for x in (-0.36, -0.2, 0.32):                                                                      # parked aircraft
        m.face([(x - 0.06, 0.0, 0.125), (x + 0.06, 0.0, 0.125), (x + 0.04, 0.0, 0.125), (x - 0.04, -0.07, 0.125)], C(.5, .52, .5))
        m.rod((x - 0.07, -0.03, 0.128), (x + 0.07, -0.03, 0.128), 0.025, C(.55, .57, .55))
        m.rod((x, -0.07, 0.128), (x, 0.03, 0.128), 0.02, C(.55, .57, .55))
    m.block((0.06, 0.11, 0.12), (0.22, 0.06, 0.16), NAVY * 1.05)                                       # the island, starboard
    with m.civ():
        m.face([(-0.05, 0.141, 0.2), (0.17, 0.141, 0.2), (0.17, 0.141, 0.26), (-0.05, 0.141, 0.26)], PAINT)
    mast(m, 0.08, 0.11, 0.28, 0.16, col=NAVY * 0.8, w=0.016)
    m.disc((0.08, 0.11, 0.42), (0.3, 0.5, 0.5), 0.04, C(.7, .72, .72))


@unit("Missile Cruiser", scale=SHIP, at=AT_SEA, wide=True)
def u_missile_cruiser(m):
    wake(m, -0.56, 0.46, 0.02, 0.12)
    hull(m, -0.52, 0.46, 0.0, 0.2, 0.11, NAVY, deck=NAVY_DECK, bow=0.24, sheer=0.04, boot=C(.2, .2, .22), stripe=0.025)
    m.block((-0.3, 0.0, 0.11), (0.12, 0.12, 0.03), C(.3, .3, .32))                                       # missile cells
    for i in range(3):
        for j in range(2):
            m.face([(-0.34 + 0.03 * i, -0.03 + 0.04 * j, 0.142), (-0.32 + 0.03 * i, -0.03 + 0.04 * j, 0.142),
                    (-0.32 + 0.03 * i, -0.01 + 0.04 * j, 0.142), (-0.34 + 0.03 * i, -0.01 + 0.04 * j, 0.142)], BLACK)
    turret(m, -0.42, 0.0, 0.11, 0.045, NAVY, guns=1, length=0.1)
    m.prism([(-0.16, -0.08), (0.12, -0.08), (0.16, 0.0), (0.12, 0.08), (-0.16, 0.08)], 0.11, 0.3, NAVY * 1.05)
    m.prism([(-0.1, -0.05), (0.06, -0.05), (0.08, 0.0), (0.06, 0.05), (-0.1, 0.05)], 0.3, 0.4, NAVY)
    mast(m, -0.02, 0.0, 0.4, 0.2, col=NAVY * 0.8, w=0.016)
    m.ball((-0.02, 0.0, 0.62), 0.04, WHITE)                                                              # radar dome
    m.block((0.28, 0.0, 0.11), (0.12, 0.12, 0.08), NAVY * 0.95)
    for y in (-0.03, 0.03):                                                                             # launcher aft
        m.capsule((0.26, y, 0.22), (0.36, y, 0.26), 0.022, 0.022, WHITE)
    ensign(m, 0.44, 0.0, 0.11, w=0.12, h=0.07, pole=0.18)


# --- civilians -------------------------------------------------------------------------------------------------------
@unit("Worker", civilian=True)
def u_worker(m):
    ground(m)
    b = figure(m, legs_col=C(.4, .34, .26), coat=OWNER, belt=LEATHER, hat="slouch", hat_col=C(.78, .66, .38))
    hand = b.sh_l + V(-0.13, 0.08, -0.2)
    m.rod(hand + V(0.03, 0, 0.2), hand + V(-0.05, 0, -0.44), 0.026, WOOD)                                 # shovel
    m.face([hand + V(-0.1, 0, -0.42), hand + V(0.0, 0, -0.42), hand + V(-0.01, 0, -0.6), hand + V(-0.09, 0, -0.6)], STEEL)
    arm(m, b.sh_l, hand, OWNER)
    arm(m, b.sh_r, hand + V(0.04, 0.0, 0.16), OWNER)


@unit("Settler", civilian=True)
def u_settler(m):
    ground(m, x=-0.1, rx=0.36)
    cart = V(-0.36, -0.04, 0.0)                                                                      # a handcart with the family's goods
    wheel(m, cart + V(0.0, -0.12, 0.13), 0.13)
    m.block(cart + V(0.0, 0.0, 0.14), (0.36, 0.24, 0.06), WOOD)
    with m.civ():
        m.side_prism([(cart[0] - 0.17, 0.2), (cart[0] - 0.12, 0.38), (cart[0] + 0.12, 0.4), (cart[0] + 0.17, 0.2)],
                     cart[1] - 0.12, cart[1] + 0.12, PAINT)
    m.rod(cart + V(0.18, 0.0, 0.18), cart + V(0.36, 0.05, 0.3), 0.022, WOOD)                       # handles
    wheel(m, cart + V(0.0, 0.13, 0.13), 0.13)
    b = Body(0.12, 0.0)
    m.ball(b.chest + V(0.0, -0.1, -0.02), 0.13, C(.62, .54, .38), ry=0.12)                          # bedroll on his back
    figure(m, b, legs_col=C(.38, .32, .24), coat=OWNER, belt=LEATHER, hat="slouch", hat_col=C(.45, .34, .2),
           beard=C(.3, .2, .12))
    hand = b.sh_r + V(0.04, 0.08, 0.06)
    arm(m, b.sh_r, hand, OWNER)
    m.rod(hand + V(-0.04, 0, -0.08), hand + V(0.1, 0, 0.36), 0.022, WOOD)                          # axe over the shoulder
    m.face([hand + V(0.07, 0, 0.3), hand + V(0.17, 0, 0.36), hand + V(0.16, 0, 0.26), hand + V(0.09, 0, 0.25)], STEEL)
    arm(m, b.sh_l, cart + V(0.36, 0.06, 0.3), OWNER)


@unit("Work Boats", civilian=True, scale=SHIP, at=CIV_SEA, wide=True)
def u_work_boats(m):
    wake(m, -0.3, 0.3, 0.02, 0.08)
    hull(m, -0.3, 0.3, 0.0, 0.16, 0.08, C(.45, .32, .18), deck=C(.55, .42, .26), bow=0.1, sheer=0.04, stripe=0.02)
    mast(m, 0.04, 0.0, 0.08, 0.42)
    lateen_sail(m, 0.04, 0.02, 0.14, 0.5, 0.3)
    f = m.sub((-0.14, 0.0, 0.08), 0.6)                                                                # the fisherman
    b = figure(f, legs_col=CLOTH, coat=C(.7, .5, .2), hat="slouch", hat_col=C(.78, .66, .38))
    arm(f, b.sh_r, b.sh_r + V(0.12, 0.06, -0.06), C(.7, .5, .2))
    arm(f, b.sh_l, b.sh_l + V(-0.08, 0.06, -0.12), C(.7, .5, .2))
    pts = [V(0.16 + 0.11 * math.cos(t), 0.14 + 0.035 * math.sin(t), 0.0) for t in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
    m.loop(pts, 0.009, C(.8, .76, .6))                                                                  # net in the water
    m.rod((-0.04, 0.06, 0.4), pts[5], 0.007, C(.8, .76, .6))


def robed(m, robe, sash=None, hair=HAIR, beard=None, hat=None, hat_col=None, skin=SKIN):
    ground(m)
    b = figure(m, robe=robe, coat=robe, boots=DARK_LEATHER, hair=hair, beard=beard, skin=skin,
               hat=hat, hat_col=hat_col if hat_col is not None else robe)
    if sash is not None:                                                                       # a stole
        with colour(m, sash) as c:
            for dx in (-0.055, 0.055):
                m.capsule(b.neck + V(dx * 0.7, 0.05, -0.06), b.waist + V(dx, 0.1, -0.32), 0.03, 0.034, c)
    return b


@unit("Great Artist", civilian=True)
def u_great_artist(m):
    ground(m, x=-0.1, rx=0.3)
    e = V(-0.3, -0.02, 0.0)                                                                             # easel and canvas
    for dx in (-0.08, 0.08):
        m.rod(e + V(dx, 0, 0), e + V(dx * 0.4, -0.02, 0.62), 0.018, WOOD)
    m.face([e + V(-0.12, 0.02, 0.3), e + V(0.12, 0.02, 0.3), e + V(0.12, 0.0, 0.6), e + V(-0.12, 0.0, 0.6)], C(.95, .93, .86))
    for dx, dz, col in ((-0.05, 0.5, C(.2, .45, .75)), (0.04, 0.42, C(.85, .55, .15)), (-0.03, 0.37, C(.3, .6, .25)),
                        (0.06, 0.53, C(.75, .2, .2))):
        m.ball(e + V(dx, 0.03, dz), 0.04, col, ry=0.03)
    b = figure(m, Body(0.06, 0.0), legs_col=C(.25, .22, .2), coat=OWNER, belt=None, hat="beret", hat_col=C(.15, .12, .14),
               beard=C(.3, .2, .12))
    hand = e + V(0.15, 0.06, 0.48)
    arm(m, b.sh_l, hand, OWNER)
    m.rod(hand, hand + V(-0.06, 0, 0.04), 0.01, DARK_WOOD)                                               # brush
    hp = b.sh_r + V(0.1, 0.1, -0.2)
    arm(m, b.sh_r, hp, OWNER)
    m.disc(hp + V(0.04, 0.03, 0.0), (0.0, 1.0, 0.6), 0.075, C(.72, .55, .35))                             # palette
    for i, col in enumerate((C(.8, .2, .2), C(.2, .4, .8), C(.9, .8, .2))):
        m.ball(hp + V(0.01 + 0.03 * i, 0.06, 0.02), 0.014, col)


@unit("Great Scientist", civilian=True)
def u_great_scientist(m):
    ground(m, x=-0.08, rx=0.3)
    t = V(-0.3, -0.04, 0.0)                                                                             # telescope on a tripod
    for dx, dy in ((-0.08, 0.04), (0.08, 0.04), (0.0, -0.06)):
        m.rod(t + V(dx, dy, 0), t + V(0, 0, 0.5), 0.016, DARK_WOOD)
    m.capsule(t + V(0.12, 0.0, 0.44), t + V(-0.22, 0.0, 0.78), 0.035, 0.05, BRONZE, gloss=0.7)
    b = robed(m, OWNER, hair=C(.85, .85, .82), beard=C(.86, .86, .84))
    arm(m, b.sh_l, t + V(0.04, 0.0, 0.52), OWNER)
    hand = b.sh_r + V(0.06, 0.1, -0.2)
    arm(m, b.sh_r, hand, OWNER)
    m.block(hand + V(0.02, 0.03, -0.05), (0.1, 0.03, 0.13), C(.45, .18, .12), yaw=0.2)                    # book


@unit("Great Merchant", civilian=True)
def u_great_merchant(m):
    ground(m, x=-0.08, rx=0.3)
    m.ball((-0.32, -0.04, 0.12), 0.13, C(.72, .62, .44), ry=0.11)                                          # sacks of goods
    m.ball((-0.22, 0.04, 0.09), 0.1, C(.66, .56, .4), ry=0.09)
    b = robed(m, OWNER, sash=GOLD, beard=C(.15, .1, .07), hat="turban", hat_col=WHITE)
    chest = V(0.2, 0.12, 0.0)                                                                           # a chest of gold
    m.block(chest, (0.2, 0.12, 0.1), C(.42, .26, .14))
    for dx in (-0.06, -0.02, 0.02, 0.06):
        m.ball(chest + V(dx, 0.0, 0.12), 0.028, GOLD, gloss=0.9)
    m.ball(chest + V(0.0, -0.01, 0.15), 0.028, GOLD, gloss=0.9)
    arm(m, b.sh_l, b.sh_l + V(-0.12, 0.08, -0.12), OWNER)
    hand = b.sh_r + V(0.08, 0.09, -0.1)
    arm(m, b.sh_r, hand, OWNER)
    m.ball(hand + V(0.0, 0.02, 0.03), 0.03, GOLD, gloss=0.9)


@unit("Great Engineer", civilian=True)
def u_great_engineer(m):
    ground(m, x=-0.08, rx=0.3)
    g = V(-0.32, -0.03, 0.27)                                                                             # a big cog
    teeth = []
    for i in range(24):
        t = 2 * math.pi * i / 24
        r = 0.26 if i % 2 else 0.21
        teeth.append(g + V(r * math.cos(t), 0.0, r * math.sin(t)))
    for i in range(0, 24, 2):
        m.face([g, teeth[i], teeth[(i + 1) % 24], teeth[(i + 2) % 24]], C(.5, .5, .52), gloss=0.4)
    m.disc(g + V(0, 0.01, 0), (0, 1, 0), 0.07, C(.3, .3, .32))
    b = figure(m, legs_col=C(.3, .3, .34), boots=BLACK, coat=OWNER, belt=LEATHER, hat="hardhat", hat_col=C(.92, .72, .15))
    hand = b.sh_r + V(0.06, 0.1, -0.18)
    arm(m, b.sh_r, hand, OWNER)
    m.capsule(hand + V(-0.02, 0.02, -0.02), hand + V(0.14, 0.02, -0.06), 0.03, 0.03, C(.35, .55, .85))     # rolled plans
    arm(m, b.sh_l, b.sh_l + V(-0.1, 0.08, -0.2), OWNER)
    m.rod(b.sh_l + V(-0.1, 0.08, -0.2), b.sh_l + V(-0.12, 0.08, -0.4), 0.022, STEEL, gloss=0.5)            # wrench
    m.ball(b.sh_l + V(-0.12, 0.08, -0.42), 0.03, STEEL, gloss=0.5)


@unit("Great Prophet", civilian=True)
def u_great_prophet(m):
    ground(m)
    b = Body()
    halo = b.head + V(0, -0.04, 0.0)
    m.face([halo + 0.2 * V(math.cos(t), 0, math.sin(t)) for t in np.linspace(0, 2 * math.pi, 20, endpoint=False)],
           C(1.0, .9, .55), alpha=0.4)
    m.face([halo + 0.13 * V(math.cos(t), 0, math.sin(t)) for t in np.linspace(0, 2 * math.pi, 20, endpoint=False)],
           C(1.0, .95, .7), alpha=0.5)
    hand = b.sh_r + V(0.14, 0.07, -0.08)
    m.rod(hand + V(-0.01, 0, -0.64), hand + V(0.01, 0, 0.4), 0.03, WOOD)                                   # staff
    m.capsule(hand + V(0.01, 0, 0.4), hand + V(0.07, 0, 0.46), 0.02, 0.02, WOOD)
    robed(m, WHITE, sash=OWNER, hair=C(.85, .85, .82), beard=C(.88, .88, .86))
    arm(m, b.sh_r, hand, WHITE)
    arm(m, b.sh_l, b.sh_l + V(-0.16, 0.08, 0.12), WHITE)


@unit("Great General", civilian=True)
def u_great_general(m):
    ground(m)
    b = Body()
    with m.civ():                                                                                     # cape behind
        m.face([b.sh_l + V(-0.03, -0.05, 0.02), b.sh_r + V(0.03, -0.05, 0.02), V(0.17, -0.06, 0.12), V(-0.19, -0.06, 0.12)], PAINT)
    navy = C(.14, .16, .28)
    figure(m, b, legs_col=WHITE * 0.92, boots=BLACK, coat=navy, skirt=navy, skirt_len=0.15, belt=GOLD,
           hair=C(.6, .58, .55), hat="bicorne", hat_col=BLACK, plume=OWNER)
    for s in (b.sh_l + V(0.02, 0.02, 0.0), b.sh_r + V(-0.02, 0.02, 0.0)):
        m.ball(s, 0.045, GOLD, gloss=0.6)                                                              # epaulettes
    arm(m, b.sh_l, b.sh_l + V(-0.06, 0.07, -0.25), navy)
    hand = b.sh_r + V(0.2, 0.04, 0.22)
    arm(m, b.sh_r, hand, navy)
    sword(m, hand, d=(0.2, 0, 1), length=0.36, guard=GOLD)


@unit("Great Admiral", civilian=True, scale=SHIP, at=CIV_SEA, wide=True)
def u_great_admiral(m):
    wake(m, -0.34, 0.34, 0.02, 0.09)
    hull(m, -0.34, 0.34, 0.0, 0.18, 0.09, C(.92, .9, .86), deck=C(.6, .48, .32), bow=0.12, sheer=0.04, stripe=0.025)
    ensign(m, 0.3, 0.0, 0.09, w=0.13, h=0.08, pole=0.24)
    a = m.sub((-0.04, 0.0, 0.09), 0.78)                                                                  # the admiral
    navy = C(.12, .14, .26)
    b = figure(a, legs_col=WHITE * 0.92, boots=BLACK, coat=navy, skirt=navy, skirt_len=0.14, belt=GOLD,
               hair=C(.6, .58, .55), hat="bicorne", hat_col=BLACK)
    for p in (b.sh_l + V(0.02, 0.02, 0.0), b.sh_r + V(-0.02, 0.02, 0.0)):
        a.ball(p, 0.04, GOLD, gloss=0.6)
    hand = b.head + V(-0.1, 0.06, -0.02)
    arm(a, b.sh_l, hand, navy)
    a.capsule(hand + V(0.04, 0.0, 0.0), hand + V(-0.24, 0.0, 0.04), 0.024, 0.032, BRONZE, gloss=0.7)         # telescope
    arm(a, b.sh_r, b.sh_r + V(0.06, 0.07, -0.25), navy)


@unit("Missionary", civilian=True)
def u_missionary(m):
    b = robed(m, C(.42, .3, .2), sash=OWNER, hair=HAIR)
    m.capsule(b.waist + V(-0.11, 0.06, 0.02), b.waist + V(0.11, 0.06, 0.02), 0.016, 0.016, LINEN)      # rope belt
    hand = b.chest + V(0.02, 0.14, -0.08)
    m.block(hand + V(0.0, 0.0, -0.04), (0.13, 0.04, 0.16), C(.15, .1, .08))                               # book
    m.rod(hand + V(0.0, 0.025, 0.02), hand + V(0.0, 0.025, 0.08), 0.012, GOLD)
    m.rod(hand + V(-0.025, 0.025, 0.06), hand + V(0.025, 0.025, 0.06), 0.012, GOLD)
    arm(m, b.sh_l, hand + V(-0.05, 0.0, 0.0), C(.42, .3, .2))
    arm(m, b.sh_r, hand + V(0.05, 0.0, 0.0), C(.42, .3, .2))


@unit("Inquisitor", civilian=True)
def u_inquisitor(m):
    b = robed(m, C(.12, .1, .1), sash=OWNER, hair=None, hat="mitre", hat_col=OWNER)
    hand = b.sh_r + V(0.14, 0.07, 0.02)
    arm(m, b.sh_r, hand, C(.12, .1, .1))
    m.rod(hand + V(-0.01, 0, -0.18), hand + V(0.01, 0, 0.16), 0.03, DARK_WOOD)                             # torch
    m.ball(hand + V(0.01, 0, 0.21), 0.06, C(1.0, .55, .12), alpha=0.9)
    m.ball(hand + V(0.01, 0, 0.25), 0.04, C(1.0, .85, .35))
    m.ball(hand + V(0.01, 0, 0.22), 0.12, C(1.0, .6, .2), alpha=0.25)
    arm(m, b.sh_l, b.sh_l + V(-0.06, 0.08, -0.24), C(.12, .1, .1))


# --- spaceship parts, on a transporter -------------------------------------------------------------------------
def transporter(m):
    m.shadow(0.04, 0.02, 0.5, 0.13)
    for x in (-0.28, 0.0, 0.28):
        solid_wheel(m, (x, -0.13, 0.07), 0.07)
    m.block((0.0, 0.0, 0.08), (0.78, 0.26, 0.06), C(.45, .46, .48))
    with m.civ():
        m.face([(-0.39, 0.131, 0.08), (0.39, 0.131, 0.08), (0.39, 0.131, 0.14), (-0.39, 0.131, 0.14)], PAINT)
    for x in (-0.28, 0.0, 0.28):
        solid_wheel(m, (x, 0.14, 0.07), 0.07)
    return 0.14


@unit("SS Booster", civilian=True, scale=VEHICLE, wide=True)
def u_ss_booster(m):
    z = transporter(m)
    m.capsule((-0.34, 0.0, z + 0.13), (0.3, 0.0, z + 0.13), 0.13, 0.13, WHITE, gloss=0.3)
    for x in (-0.2, 0.12):
        m.capsule((x, 0.0, z + 0.13), (x + 0.03, 0.0, z + 0.13), 0.134, 0.134, BLACK)
    m.capsule((-0.34, 0.0, z + 0.13), (-0.48, 0.0, z + 0.13), 0.12, 0.02, WHITE, gloss=0.3)
    for dz in (0.2, -0.0):
        m.face([(0.22, 0.0, z + 0.13 + dz), (0.36, 0.0, z + 0.13 + dz), (0.36, 0.0, z + 0.13 + dz + 0.08), (0.28, 0.0, z + 0.13 + dz + 0.08)],
               C(.7, .15, .12))


@unit("SS Cockpit", civilian=True, scale=VEHICLE, wide=True)
def u_ss_cockpit(m):
    z = transporter(m)
    m.capsule((0.24, 0.0, z + 0.16), (-0.2, 0.0, z + 0.16), 0.17, 0.05, C(.85, .86, .88), gloss=0.4)
    m.capsule((0.24, 0.0, z + 0.16), (0.3, 0.0, z + 0.16), 0.17, 0.17, C(.3, .3, .32))
    for dx in (-0.04, 0.06):
        m.ball((dx, 0.1, z + 0.22), 0.03, C(.15, .3, .45), gloss=0.9)
    with m.civ():
        m.capsule((0.12, 0.0, z + 0.16), (0.16, 0.0, z + 0.16), 0.15, 0.15, PAINT)


@unit("SS Engine", civilian=True, scale=VEHICLE, wide=True)
def u_ss_engine(m):
    z = transporter(m)
    m.block((0.18, 0.0, z), (0.24, 0.24, 0.24), C(.5, .5, .54))
    for y, dz in ((-0.06, 0.16), (0.06, 0.08), (0.06, 0.2)):
        m.capsule((0.06, y, z + dz), (-0.3, y, z + dz), 0.05, 0.12, C(.3, .3, .32), gloss=0.5)            # nozzles
        m.disc((-0.3, y + 0.0, z + dz), (-1, 0.3, 0.2), 0.11, C(.1, .1, .1))
    with m.civ():
        m.face([(0.06, 0.121, z + 0.04), (0.3, 0.121, z + 0.04), (0.3, 0.121, z + 0.1), (0.06, 0.121, z + 0.1)], PAINT)


@unit("SS Stasis Chamber", civilian=True, scale=VEHICLE, wide=True)
def u_ss_stasis_chamber(m):
    z = transporter(m)
    m.capsule((-0.3, 0.0, z + 0.13), (0.3, 0.0, z + 0.13), 0.13, 0.13, C(.6, .62, .66), gloss=0.5)
    m.capsule((-0.22, 0.04, z + 0.15), (0.22, 0.04, z + 0.15), 0.1, 0.1, C(.35, .85, 1.0), alpha=0.85)   # glowing window
    m.capsule((-0.16, 0.06, z + 0.16), (0.14, 0.06, z + 0.16), 0.04, 0.04, C(.8, .95, 1.0), alpha=0.8)
    with m.civ():
        for x in (-0.28, 0.24):
            m.capsule((x, 0.0, z + 0.13), (x + 0.04, 0.0, z + 0.13), 0.135, 0.135, PAINT)


# --- embarked units ---------------------------------------------------------------------------------------------
def longboat(m, cargo):
    wake(m, -0.42, 0.36, 0.02, 0.1)
    oars(m, np.linspace(-0.16, 0.16, 5), -0.06, 0.08, -1)
    hull(m, -0.38, 0.36, 0.0, 0.16, 0.09, TIMBER, deck=C(.42, .32, .2), bow=0.12, sheer=0.07)
    m.capsule((-0.38, 0.0, 0.09), (-0.44, 0.0, 0.22), 0.022, 0.012, TIMBER)
    mast(m, 0.04, 0.0, 0.09, 0.52)
    square_sail(m, 0.04, 0.0, 0.24, 0.56, 0.34)
    cargo(m)
    oars(m, np.linspace(-0.16, 0.16, 5), 0.075, 0.07, 1)


def troops(m, xs, z, kind="ancient"):
    for i, x in enumerate(xs):
        t = m.sub((x, 0.02 * (i % 2), z), 0.62)
        troop(t, kind)


def troop(m, kind):
    """A soldier standing in a boat, seen from the waist up."""
    b = Body(0.0, 0.0, -0.5)
    with colour(m, OWNER) as c:
        m.capsule(b.waist + V(0, 0, 0.12), b.chest, 0.1, 0.12, c)
        m.capsule(b.sh_l + V(0.035, 0, 0), b.sh_r - V(0.035, 0, 0), 0.06, 0.06, c)
    head(m, b, hair=None)
    if kind == "ancient":
        helmet(m, b, BRONZE, "round")
        spear(m, b.sh_r + V(0.06, 0.04, -0.1), length=0.75, lean=0.03, below=0.2)
    elif kind == "musket":
        helmet(m, b, BLACK, "tricorne")
        long_gun(m, b.sh_r + V(0.06, 0.04, -0.2), b.sh_r + V(0.1, 0.04, 0.42))
    else:
        helmet(m, b, OLIVE * 0.9, "m1")


def cargo_bundles(m, xs, z):
    for i, x in enumerate(xs):
        if i % 2:
            m.block((x, 0.0, z), (0.1, 0.1, 0.09), C(.55, .42, .26))
        else:
            m.ball((x, 0.0, z + 0.05), 0.07, C(.72, .64, .46), ry=0.06)


@unit("EmbarkedUnit-Military", scale=SHIP, at=AT_SEA, wide=True)
def u_embarked_military(m):
    longboat(m, lambda m: troops(m, (-0.22, -0.08, 0.2), 0.09))


@unit("EmbarkedUnit-Civilian", civilian=True, scale=SHIP, at=CIV_SEA, wide=True)
def u_embarked_civilian(m):
    longboat(m, lambda m: cargo_bundles(m, (-0.22, -0.1, 0.18, 0.28), 0.09))


def transport_ship(m, cargo):
    wake(m, -0.46, 0.4, 0.02, 0.12)
    hull(m, -0.42, 0.4, 0.0, 0.2, 0.15, C(.4, .28, .17), deck=C(.46, .36, .23), bow=0.16, sheer=0.07)
    m.block((0.3, 0.0, 0.15), (0.18, 0.2, 0.1), C(.4, .28, .17) * 1.1)
    for x, h in ((-0.16, 0.66), (0.1, 0.76)):
        mast(m, x, 0.0, 0.15, h)
        square_sail(m, x, 0.0, 0.32, 0.15 + h * 0.85, 0.3)
    cargo(m)
    ensign(m, 0.38, 0.0, 0.25, w=0.12, h=0.07, pole=0.18)


@unit("EmbarkedUnit-Military-Renaissance era", scale=SHIP, at=AT_SEA, wide=True)
def u_embarked_military_renaissance(m):
    transport_ship(m, lambda m: troops(m, (-0.3, -0.03, 0.2), 0.15, "musket"))


@unit("EmbarkedUnit-Civilian-Renaissance era", civilian=True, scale=SHIP, at=CIV_SEA, wide=True)
def u_embarked_civilian_renaissance(m):
    transport_ship(m, lambda m: cargo_bundles(m, (-0.3, -0.04, 0.2), 0.15))


def landing_craft(m, cargo):
    wake(m, -0.44, 0.4, 0.02, 0.11)
    grey = C(.46, .48, .46)
    m.side_prism([(-0.42, 0.04), (-0.38, 0.16), (0.36, 0.16), (0.4, 0.04), (0.36, 0.0), (-0.38, 0.0)], -0.12, 0.12, grey,
                 top=C(.3, .32, .3))
    m.face([(-0.4, -0.12, 0.16), (-0.4, 0.12, 0.16), (-0.44, 0.12, 0.26), (-0.44, -0.12, 0.26)], grey * 0.95)   # bow ramp
    cargo(m)
    m.block((0.3, 0.0, 0.16), (0.1, 0.16, 0.12), grey)
    with m.civ():
        m.face([(-0.3, 0.121, 0.08), (0.2, 0.121, 0.08), (0.2, 0.121, 0.13), (-0.3, 0.121, 0.13)], PAINT)
    ensign(m, 0.38, 0.0, 0.16, w=0.12, h=0.07, pole=0.18)


@unit("EmbarkedUnit-Military-Modern era", scale=SHIP, at=AT_SEA, wide=True)
def u_embarked_military_modern(m):
    landing_craft(m, lambda m: troops(m, (-0.26, -0.1, 0.06, 0.2), 0.12, "modern"))


@unit("EmbarkedUnit-Civilian-Modern era", civilian=True, scale=SHIP, at=CIV_SEA, wide=True)
def u_embarked_civilian_modern(m):
    landing_craft(m, lambda m: cargo_bundles(m, (-0.26, -0.1, 0.06, 0.2), 0.16))


# ============================================================================ rendering
def save_layers(cv, L, name):
    """Write the unit image, its empty "-1" placeholder and its "-2" owner-colour layer. All three
    are cropped the same way, so they line up when Unciv scales each one to the hex width."""
    def down(P):
        p = cv.pad
        P = P[p:P.shape[0] - p, p:P.shape[1] - p]
        h, w = P.shape[:2]
        return P.reshape(h // SS, SS, w // SS, SS, 4).mean(axis=(1, 3))

    def to_img(P):
        a = P[..., 3:4]
        rgb = np.where(a > 1e-6, P[..., :3] / np.maximum(a, 1e-6), 0.0)
        return (np.concatenate([np.clip(rgb, 0, 1), np.clip(a, 0, 1)], axis=-1) * 255 + 0.5).astype(np.uint8)

    base, civ = down(L.base), down(L.civ)
    rows = np.nonzero(base[..., 3].max(axis=1) > 0.5 / 255)[0]
    top = rows[0] if len(rows) else 0
    top -= (base.shape[0] - top) % LAYER_SCALE      # keep the height divisible, so the layer's aspect matches
    base, civ = base[max(top, 0):], civ[max(top, 0):]
    rel = "Units/" + name
    write(Image.fromarray(to_img(base), "RGBA"), rel)
    h, w = civ.shape[:2]
    small = civ.reshape(h // LAYER_SCALE, LAYER_SCALE, w // LAYER_SCALE, LAYER_SCALE, 4).mean(axis=(1, 3))
    write(Image.fromarray(to_img(small), "RGBA"), rel + "-2")
    write(Image.new("RGBA", (8, 7), (0, 0, 0, 0)), rel + "-1")


def write(img, rel):
    import os
    path = os.path.join(rt.OUT, rel + ".png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, optimize=True)
    rt.WRITTEN.append(rel)


def placement(cv, spec):
    """Where to stand a unit and how big to draw it, so it sits on its side of the hex and nothing
    is cut off at the image edge: draw it small once to measure it, then centre it on its side and
    shrink it if it is too wide to fit."""
    civilian = spec["civilian"]
    u_mid, v0 = spec["at"] or (CIV if civilian else MIL)
    s = spec["scale"]
    k = 0.4
    probe = Layers(cv)
    spec["fn"](Mini(cv, probe, 0.0, v0, s * k))
    cols = np.nonzero(probe.base[..., 3].max(axis=0) > 0.02)[0]
    lo, hi = cv.u[0, cols[0]] / k, cv.u[0, cols[-1]] / k
    # The flags are discs in the middle of the hex, the military one in the upper half and the
    # civilian one in the lower half. Low military units (ships, vehicles) can pass under the
    # military flag; civilians stand at the same height as theirs, so they keep to the left.
    room = (-0.98, -0.18) if civilian else (-0.16, 0.98) if spec["wide"] else (0.04, 0.97)
    if hi - lo > room[1] - room[0]:
        f = (room[1] - room[0]) / (hi - lo)
        s, lo, hi = s * f, lo * f, hi * f
    u0 = min(max(u_mid - (lo + hi) / 2, room[0] - lo), room[1] - hi)
    return u0, v0, s


def render_unit(name, spec):
    rel = "Units/" + name
    if not rt.wanted(rel):
        return
    cv = rt.Canvas(UNIT_W, overhang=OVERHANG)
    u0, v0, s = placement(cv, spec)
    L = Layers(cv)
    spec["fn"](Mini(cv, L, u0, v0, s))
    save_layers(cv, L, name)


def main():
    rt.FILTER = sys.argv[1:] or None
    for name, spec in UNITS.items():
        render_unit(name, spec)
    print("Wrote %d unit images (%d units)" % (len(rt.WRITTEN), len(rt.WRITTEN) // 3))


if __name__ == "__main__":
    main()
