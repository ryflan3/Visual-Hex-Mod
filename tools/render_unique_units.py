#!/usr/bin/env python3
"""Draw the base game's unique units in the unit set's illustrated style.

    python3 tools/render_unique_units.py              # every unique unit
    python3 tools/render_unique_units.py Legion Khan  # only units whose name contains one of the words

Without its own image, Unciv shows a unique unit with the image of the unit it replaces, so each one
here is drawn to read as its own unit next to that one: its own gear, dress and silhouette. They use
render_units.py's people, animals, ships and guns, and register in its UNITS. The Zero and the B17
are aircraft, which Unciv never draws on tiles.
"""

import math
import sys

import render_tiles as rt
import render_units as RU
from render_units import (AT_SEA, OWNER, Shift, arm, beard, bow, figure, ground, hair, hat, head_at, hexcol, horse, hull,
                          lateen, mast, oars, pennant, quiver, rider, rifle, round_shield, spear_up, square_sail, sword_up,
                          tracks, water, wheel, ensign)

UNIQUE = []


def unit(name, **kw):
    UNIQUE.append(name)
    return RU.unit(name, **kw)


JAGUAR = hexcol("#d9a23a")
SPOT = hexcol("#4a2e14")
ELEPHANT = hexcol("#8f8a85")
AFRICAN = hexcol("#9a8f86")
CAMEL = hexcol("#c49a64")
DUN = hexcol("#b08a5a")
INDIGO = hexcol("#2c3a6e")
FUR = hexcol("#6b4b2e")
SABLE = hexcol("#5a4632")
WOLF = hexcol("#8a8378")
BEAR = hexcol("#5a3d26")
WOAD = hexcol("#3a5d9c")
WICKER = hexcol("#c9a25a")
GINGER = hexcol("#b5532a")
QUETZAL = hexcol("#3f8f4a")
JADE = hexcol("#5fb08a")
DUNKELGELB = hexcol("#b9a066")
CAMO_GREEN = hexcol("#5b6a3a")
CAMO_BROWN = hexcol("#7a4a2c")


# ============================================================================ helpers
def shade(col, k):
    return tuple(min(1.0, c * k) for c in col)


def long_beard(p, col="hair"):
    """A long, squared beard with rows of curls, as on Mesopotamian and Persian reliefs."""
    p.path("M-8,-85 L8,-85 L7,-66 Q0,-63 -7,-66 Z", col)
    for y in (-79, -73):
        p.thin(-6.5, y, 6.5, y, 0.9, "black")


def wicker(p, d, lines=6):
    """A wicker shield through path d, with its weave."""
    p.path(d, WICKER)
    pts = RU.path_points(d)
    x0, x1 = min(x for x, _ in pts), max(x for x, _ in pts)
    y0, y1 = min(y for _, y in pts), max(y for _, y in pts)
    for k in range(1, lines):
        y = y0 + (y1 - y0) * k / lines
        p.thin(x0 + 2, y, x1 - 2, y, 0.9, "wood")


def moustache(p, col="hair"):
    p.path("M-7,-83 Q0,-87 7,-83 Q4,-80 0,-82 Q-4,-80 -7,-83 Z", col, ink=False)


def plumed_hat(p, col="black", plume="white", brim=19):
    """A wide cavalier hat with an ostrich plume curling over the brim."""
    p.ellipse(0, -96, brim, 4.6, col)
    p.path("M-9,-97 Q-8,-109 0,-109 Q8,-109 9,-97 Z", col)
    p.path("M-4,-106 Q10,-120 24,-106 Q20,-100 14,-98 Q16,-106 6,-104 Z", plume)


# ============================================================================ animals
def elephant(p, x=0.0, s=1.0, col=ELEPHANT, ears="asian", trunk="down"):
    """An elephant in profile facing west, about 120 long and 95 tall at the back, on y = 0."""
    q = Shift(p, x, 0, s)
    for lx in (-18, 26):                                                                # far legs
        q.rect(lx - 7, -40, 14, 40, shade(col, 0.75), rx=4)
    q.curve("M44,-60 Q52,-48 48,-34", 2.4, col)                                        # tail
    q.path("M-30,-44 Q-34,-86 2,-92 Q38,-94 46,-64 Q50,-38 40,-28 Q10,-22 -24,-28 Z", col)
    if trunk == "raised":
        q.path("M-22,-86 Q-48,-100 -60,-78 Q-62,-62 -50,-50 Q-42,-46 -30,-50 Z", col)    # head
        q.curve("M-56,-66 Q-70,-76 -72,-96 Q-73,-110 -64,-114", 8, col)                 # trunk, raised to trumpet
        tusk = "M-50,-58 Q-64,-58 -70,-68"
    else:
        q.path("M-22,-86 Q-48,-100 -60,-74 Q-64,-52 -62,-32 Q-62,-16 -54,-8 Q-49,-11 -53,-17 "
               "Q-55,-34 -50,-50 Q-42,-56 -30,-50 Z", col)                               # head and trunk
        tusk = "M-48,-54 Q-62,-52 -70,-62"
    q.curve(tusk, 3.4, "white")
    if ears == "african":                                                              # big, fan-shaped ears
        q.path("M-38,-88 Q-22,-104 -6,-90 Q2,-70 -6,-50 Q-16,-38 -28,-44 Q-36,-58 -38,-88 Z", shade(col, 0.88))
    else:
        q.path("M-36,-84 Q-12,-90 -10,-64 Q-12,-44 -28,-46 Q-38,-60 -36,-84 Z", shade(col, 0.88))
    q.circle(-46, -74, 1.6, "black", ink=False)
    for lx in (-24, 18):                                                                # near legs
        q.rect(lx - 7.5, -42, 15, 42, col, rx=4)
        q.rect(lx - 7.5, -4, 15, 4, shade(col, 1.15), rx=1.5, ink=False)


def camel(p, x=0.0, col=CAMEL):
    """A dromedary in profile facing west, about 105 long and 100 tall at the head, on y = 0."""
    q = Shift(p, x, 0)
    far = shade(col, 0.75)
    for top, knee, foot in (((-18, -48), (-20, -26), (-18, -4)), ((26, -48), (28, -26), (26, -4))):
        q.stroke([top, knee, foot], 4.6, far)                                          # far legs
    q.curve("M36,-64 Q42,-56 40,-44", 2.0, col)                                        # tail
    q.path("M-20,-50 Q-26,-66 -14,-74 Q-4,-80 4,-86 Q14,-100 26,-88 Q36,-80 38,-66 Q40,-52 32,-46 "
           "Q6,-42 -20,-50 Z", col)                                                    # body and hump
    q.path("M-14,-72 Q-28,-62 -38,-68 Q-46,-74 -48,-90 Q-48,-98 -56,-100 L-64,-98 Q-68,-94 -66,-90 "
           "L-58,-88 Q-56,-80 -50,-70 Q-40,-56 -22,-54 Z", col)                       # neck and head
    q.poly([(-52, -99), (-50, -105), (-47, -98)], col)                                 # ear
    q.circle(-57, -95, 1.3, "black", ink=False)
    for top, knee, foot in (((-14, -48), (-16, -26), (-14, -4)), ((22, -48), (24, -26), (22, -4))):
        q.stroke([top, knee, foot], 5.2, col)                                          # near legs, long and knobbly
        q.ellipse(foot[0] - 1, -2.5, 5.5, 2.6, shade(col, 0.8))


# ============================================================================ warriors
@unit("Maori Warrior")
def u_maori_warrior(p):
    ground(p)
    figure(p, tunic="skin", pants="skin", boots="skin", bare=True, skirt=False, belt=None)
    p.path("M-15,-53 L15,-53 L19,-24 L-19,-24 Z", OWNER)                                   # piupiu flax skirt
    for k in range(9):
        x = -12 + k * 3
        p.thin(x, -50, x * 1.2, -25, 1.0, "black")
    p.rect(-15, -55, 30, 4.5, "straw", rx=1.5)                                             # woven belt
    hair(p)
    p.circle(0, -102, 5, "hair")                                                           # topknot and a huia feather
    p.limb(2, -105, 6, -121, 2.4, "white")
    p.limb(5, -118, 6, -122, 2.4, "black", ink=False)
    for s in (-1, 1):                                                                      # ta moko
        p.curve("M%g,-92 Q%g,-90 %g,-86" % (2 * s, 7 * s, 4 * s), 0.9, "navy", ink=False, shade=False)
        p.curve("M%g,-84 Q%g,-80 %g,-79" % (3 * s, 7 * s, 2 * s), 0.9, "navy", ink=False, shade=False)
    p.limb(-32, -46, 31, -70, 3.4, "darkwood")                                             # taiaha, held across
    p.path("M-32,-46 L-40,-40 L-38,-37 L-29,-42 Z", "darkwood")                            # its blade
    p.ellipse(25, -67, 4.5, 3.6, "white")                                                  # feather collar
    p.ellipse(26, -66, 3, 2, "red", ink=False)
    arm(p, -13, -72, -16, -51, "skin")
    arm(p, 13, -72, 14, -61, "skin")


@unit("Jaguar")
def u_jaguar(p):
    ground(p)
    p.limb(-13, -72, -24, -94, 7, JAGUAR)
    p.rect(-29, -134, 8, 42, "wood", rx=2)                                                 # macuahuitl
    for y in range(-130, -96, 7):
        p.poly([(-29, y), (-34, y + 3), (-29, y + 5)], "black", ink=False)
        p.poly([(-21, y), (-16, y + 3), (-21, y + 5)], "black", ink=False)
    figure(p, tunic=JAGUAR, pants=JAGUAR, boots="leather", skirt=False, belt=OWNER)
    for x, y in ((-8, -68), (6, -62), (-4, -56), (8, -72), (-6, -40), (6, -30), (-6, -20), (5, -14), (0, -48)):
        p.ellipse(x, y, 2.6, 2, SPOT, ink=False)
    p.path("M-10,-51 L10,-51 L6,-36 L-6,-36 Z", OWNER)                                     # maxtlatl
    p.path("M-14,-82 Q-16,-112 0,-114 Q16,-112 14,-82 Q8,-96 0,-97 Q-8,-96 -14,-82 Z", JAGUAR)   # jaguar helmet
    for x in (-10, 10):
        p.circle(x, -110, 4, JAGUAR)
    for x in range(-9, 10, 3):
        p.poly([(x - 1.4, -96.5), (x + 1.4, -96.5), (x, -93)], "white", ink=False)
    for x, y in ((-6, -106), (5, -108), (0, -102)):
        p.ellipse(x, y, 2.2, 1.6, SPOT, ink=False)
    p.circle(-24, -94, 4, "skin")
    arm(p, 13, -72, 19, -52, JAGUAR)
    for k in range(5):                                                                      # feather fringe
        p.ellipse(10 + k * 6, -32, 3, 7, OWNER if k % 2 else "white")
    round_shield(p, 22, -52, 16, face=OWNER, rim="gold", boss=None)
    p.circle(22, -52, 7, "white", ink=False)


@unit("Brute")
def u_brute(p):
    ground(p, 28)
    p.path("M10,-70 L17,-73 L42,-124 Q40,-134 30,-132 Q26,-128 26,-120 Z", "darkwood")     # spiked club on the shoulder
    for x, y in ((40, -126), (36, -118), (31, -128), (32, -110)):
        p.poly([(x - 2, y), (x + 4, y - 2), (x, y + 3)], "steel", ink=False)
    figure(p, tunic=OWNER, bare=True, wide=1.3)
    p.path("M-19,-77 L2,-77 L-4,-48 L-10,-52 L-14,-46 L-18,-53 L-22,-49 Z", FUR)           # pelt over the shoulder
    for x in (-15, -9, -3):
        p.thin(x, -74, x - 2, -56, 0.9, "darkwood")
    p.ellipse(0, -81, 10, 7, "beard")
    p.thin(-7, -94, -1, -91, 1.8, "black")                                                 # scowling brows
    p.thin(1, -91, 7, -94, 1.8, "black")
    arm(p, -17, -72, -23, -46, "skin", w=9)
    arm(p, 17, -72, 18, -76, "skin", w=9)


# ============================================================================ archers
@unit("Bowman")
def u_bowman(p):
    ground(p)
    wicker(p, "M8,-116 Q17,-124 26,-116 L26,-4 L8,-4 Z", lines=9)                         # tall standing shield
    quiver(p)
    p.rect(-11, -97, 22, 22, "hair", rx=6)                                                 # long hair at the back
    figure(p, robe=OWNER, belt="gold")
    for x in range(-16, 17, 4):                                                            # fringed hem
        p.thin(x, -6, x, -2, 1.0, "gold")
    long_beard(p)
    hat(p, "cone", "bronze")
    p.limb(-13, -72, -40, -76, 6.5, OWNER)
    bow(p, recurve=True)
    arm(p, 13, -72, -5, -80, OWNER)
    p.circle(-40, -76, 3.8, "skin")


@unit("Slinger")
def u_slinger(p):
    ground(p)
    p.ellipse(-15, -44, 6, 7.5, "cloth")                                                   # bag of stones
    figure(p, tunic=OWNER, pants="skin", boots="leather", skirt=False, long_coat=True, belt=None)
    for k in range(6):                                                                     # tocapu yoke
        for j in range(2):
            p.rect(-12 + k * 4, -71 + j * 4, 4, 4, "black" if (k + j) % 2 else "white", ink=False)
    hair(p, "black")
    p.rect(-10.5, -97, 21, 3.6, "red", rx=1.2)                                             # llautu headband
    p.limb(18, -104, 3, -128, 0.9, "rope", ink=False)                                      # the sling, whirling
    p.limb(18, -104, 8, -132, 0.9, "rope", ink=False)
    p.ellipse(5, -131, 5.5, 4, "leather")
    p.circle(5, -133, 2.2, "stone", ink=False)
    p.stroke(RU.ellipse_points(16, -122, 20, 14, n=30, a0=math.radians(150), a1=math.radians(330)), 1.0, "white",
             ink=False, shade=False, alpha=0.5)
    arm(p, 13, -72, 18, -104, "skin")
    arm(p, -13, -72, -30, -80, "skin")


@unit("Atlatlist")
def u_atlatlist(p):
    ground(p)
    for tx, ty in ((-16, -126), (-6, -134), (6, -134), (16, -126)):                       # quetzal plumes
        p.path("M-2,-100 Q%g,%g %g,%g Q%g,%g 2,-100 Z" % (tx - 6, (ty - 100) / 2 - 50, tx, ty, tx + 4, (ty - 100) / 2 - 50), QUETZAL)
    figure(p, tunic=OWNER, pants="skin", boots="leather", skirt=False, belt="white")
    for y in (-68, -62, -56):                                                              # quilted cotton armour
        p.thin(-12, y, 12, y, 0.9, "linen")
    p.path("M-6,-50 L6,-50 L5,-30 L-5,-30 Z", "white")                                     # loincloth
    hair(p, "black")
    p.rect(-11, -99, 22, 5, "gold", rx=2)
    p.circle(0, -97, 3.4, JADE)
    for s in (-1, 1):
        p.circle(10 * s, -88, 2.4, JADE)                                                   # ear spools
    p.limb(34, -104, -30, -114, 2.2, "wood")                                               # dart on the atlatl
    p.path("M-30,-116.5 L-38,-114.5 L-30,-111.5 Z", "black", ink=False)
    p.poly([(30, -106), (38, -110), (36, -102)], "red", ink=False)
    p.limb(20, -100, 32, -104, 3.2, "darkwood")                                            # the atlatl
    arm(p, 13, -72, 20, -100, "skin")
    arm(p, -13, -72, -30, -76, "skin")


@unit("Skirmisher")
def u_skirmisher(p):
    ground(p)
    p.path("M-17,-78 L17,-78 L22,-30 Q0,-24 -22,-30 Z", WOLF)                              # wolf pelt down the back
    for dx in (0, 5, 10):                                                                  # javelins on the back
        p.limb(4 + dx, -40, 14 + dx, -114, 2.0, "wood")
        p.path("M%g,-114 L%g,-121 L%g,-113 Z" % (12 + dx, 15 + dx, 17 + dx), "iron", ink=False)
    figure(p, tunic=OWNER, pants="cloth", boots="leather", belt="leather")
    p.path("M-15,-77 L15,-77 L13,-64 L-13,-64 Z", WOLF)                                    # fur collar
    hat(p, "hood", WOLF)
    p.poly([(-10, -100), (-6, -112), (-3, -102)], WOLF)                                    # wolf ears and muzzle
    p.poly([(3, -102), (6, -112), (10, -100)], WOLF)
    p.ellipse(0, -100, 5, 3.4, WOLF)
    p.circle(0, -101.5, 1.3, "black", ink=False)
    p.limb(28, -100, -30, -118, 2.4, "wood")                                               # javelin, raised to throw
    p.path("M-30,-120.5 L-38,-119 L-30,-115.5 Z", "iron")
    arm(p, 13, -72, 20, -102, OWNER)
    arm(p, -13, -72, -26, -80, OWNER)


# ============================================================================ spearmen
@unit("Hoplite")
def u_hoplite(p):
    ground(p)
    p.limb(34, -118, -36, -84, 3.0, "wood")                                                # spear, held overhand
    p.path("M-36,-84 L-48,-79 L-38,-90 Z", "bronze")
    figure(p, tunic="linen", pants="skin", boots="leather")
    for x in range(-12, 13, 4):                                                            # pteruges
        p.rect(x - 1.6, -52, 3.2, 17, "linen", rx=1)
    p.path("M-13,-74 L13,-74 L12,-52 L-12,-52 Z", "linen")
    p.path("M-13,-74 L13,-74 L12,-68 L-12,-68 Z", OWNER)
    for x in (-6, 6):
        p.rect(x - 4.8, -32, 9.6, 18, "bronze", rx=3)                                      # greaves
    p.path("M-11,-80 Q-12,-106 0,-107 Q12,-106 11,-80 L4,-82 L4,-90 L-4,-90 L-4,-82 Z", "bronze")   # Corinthian helmet
    p.rect(-8, -94, 16, 2.6, "black", ink=False)
    p.path("M-4,-106 Q-2,-130 18,-128 Q22,-116 6,-104 Z", OWNER)                            # tall crest
    arm(p, 13, -72, 30, -104, "skin")
    arm(p, -13, -72, -18, -60, "skin")
    p.circle(-14, -58, 26, "bronze")                                                       # the aspis
    p.circle(-14, -58, 22, OWNER)
    p.limb(-24, -44, -14, -72, 3.2, "gold")                                                # a lambda on the shield
    p.limb(-14, -72, -4, -44, 3.2, "gold")


@unit("Persian Immortal")
def u_persian_immortal(p):
    ground(p)
    p.rect(-21, -88, 8, 30, "leather", rx=3, rot=-15)                                      # bow case
    spear_up(p, 20, -134, tip="steel")
    p.circle(20, -3, 3.8, "gold")                                                          # golden pomegranate
    figure(p, robe=OWNER, belt="gold")
    for x in (-8, -3, 3, 8):                                                               # pleats
        p.thin(x, -48, x * 1.6, -4, 0.9, "black")
    p.rect(-18, -11, 36, 3.4, "gold", ink=False)
    long_beard(p, "black")
    p.path("M-11.5,-82 Q-13,-104 -2,-108 Q4,-118 12,-112 Q8,-108 11,-100 L11.5,-82 L8,-86 L8,-94 Q0,-98 -8,-94 L-8,-86 Z",
           "sand")                                                                         # Median felt cap
    p.rect(-10.5, -98, 21, 3.2, "gold", rx=1)
    p.path("M14,-74 L22,-74 L24,-58 L16,-56 Z", OWNER)                                      # wide sleeve
    arm(p, 13, -72, 20, -58, OWNER)
    arm(p, -13, -72, -18, -60, OWNER)
    wicker(p, "M-36,-84 Q-21,-89 -6,-84 L-8,-68 Q-3,-61 -8,-54 L-6,-38 Q-21,-33 -36,-38 L-34,-54 "
           "Q-39,-61 -34,-68 Z", lines=7)                                                 # violin-shaped wicker shield


@unit("Pictish Warrior")
def u_pictish_warrior(p):
    ground(p)
    p.limb(14, -2, 32, -132, 3.2, "wood")                                                  # spear, leaning
    p.path("M33.6,-144 L36.6,-131 L28.6,-132 Z", "iron")
    figure(p, tunic=OWNER, bare=True, pants="skin", boots="leather")
    for y in (-50, -43, -37):                                                              # plaid
        p.thin(-14 - (y + 52) * 0.2, y, 14 + (y + 52) * 0.2, y, 1.0, "black")
    for x in (-9, -3, 3, 9):
        p.thin(x, -52, x * 1.25, -34, 1.0, "black")
    p.curve("M-9,-72 Q-4,-74 -4,-68 Q-5,-64 -9,-66", 1.4, WOAD, ink=False, shade=False)    # woad spirals
    p.curve("M9,-72 Q4,-74 4,-68 Q5,-64 9,-66", 1.4, WOAD, ink=False, shade=False)
    p.curve("M-6,-58 Q0,-62 6,-58", 1.4, WOAD, ink=False, shade=False)
    p.rect(-9.5, -93, 19, 4, WOAD, ink=False, alpha=0.85)                                  # painted across the eyes
    p.path("M-11,-90 L-13,-104 L-7,-98 L-4,-108 L0,-99 L4,-109 L7,-98 L13,-104 L11,-90 Q0,-97 -11,-90 Z", GINGER)
    p.path("M-7,-83 Q0,-80 7,-83 L8,-76 Q4,-79 0,-81 Q-4,-79 -8,-76 Z", GINGER)           # long moustache
    arm(p, 13, -72, 24, -70, "skin")
    p.limb(14, -64, 17, -60, 7.4, WOAD, ink=False)                                         # painted arm band
    arm(p, -13, -72, -18, -58, "skin")
    p.rect(-34, -76, 24, 30, "bronze", rx=2)                                               # small square shield
    p.rect(-32, -74, 20, 26, OWNER, rx=1.5)
    p.circle(-22, -61, 4.2, "bronze")


@unit("Marauder")
def u_marauder(p):
    ground(p)
    p.path("M-18,-80 L18,-80 L26,-22 Q0,-16 -26,-22 Z", FUR)                               # fur cloak
    figure(p, tunic=OWNER, pants="cloth", boots="leather", belt="leather")
    p.path("M-13,-74 L13,-74 L12,-52 L-12,-52 Z", "iron")                                  # mail shirt
    for y in (-70, -65, -60, -55):
        p.curve("M-11,%g Q-5,%g 0,%g Q5,%g 11,%g" % (y, y + 2, y, y + 2, y), 0.8, "darkgrey", ink=False, shade=False)
    beard(p)
    p.limb(0, -80, 0, -70, 2.6, "beard")                                                   # braid
    hat(p, "round", "iron")
    p.path("M-10,-97 Q-22,-98 -25,-114 Q-18,-107 -9,-103 Z", "linen")                      # horns
    p.path("M10,-97 Q22,-98 25,-114 Q18,-107 9,-103 Z", "linen")
    p.limb(-20, -2, -32, -128, 3.2, "wood")                                                # spear
    p.path("M-31,-126 L-34,-141 L-27,-127 Z", "iron")
    p.limb(-30, -112, -27, -104, 3.6, "red", ink=False)                                    # a rag tied on
    arm(p, -13, -72, -25, -60, OWNER)
    arm(p, 13, -72, 14, -54, OWNER)
    round_shield(p, 22, -56, 16, face=OWNER, rim="iron", boss="iron")
    p.limb(22, -70, 22, -42, 2.0, "red", ink=False)


# ============================================================================ swordsmen
@unit("Legion")
def u_legion(p):
    ground(p)
    p.rect(-29, -150, 3.2, 150, "darkwood", rx=1.2)                                       # the eagle standard
    p.rect(-37, -140, 19, 15, OWNER)
    p.rect(-37, -126, 19, 2.4, "gold", ink=False)
    p.path("M-27.5,-154 Q-40,-166 -46,-158 Q-38,-158 -35,-152 Q-31,-150 -27.5,-150 Q-24,-150 -20,-152 "
           "Q-17,-158 -9,-158 Q-15,-166 -27.5,-154 Z", "gold")
    p.circle(-27.5, -157, 3.4, "gold")
    figure(p, tunic=OWNER, pants="skin", boots="leather")
    p.path("M-13,-74 L13,-74 L12,-52 L-12,-52 Z", "steel")                                # segmented armour
    for y in (-69, -64, -59, -55):
        p.thin(-12.5, y, 12.5, y, 1.1, "darkgrey")
    for x in (-13, 13):
        p.ellipse(x, -73, 6, 4, "steel")
    hat(p, "round", "steel")
    p.path("M-15,-101 Q0,-119 15,-101 Q0,-107 -15,-101 Z", OWNER)                          # crest, worn across
    p.rect(-11, -90, 3, 9, "steel", rx=1)                                                  # cheek guards
    p.rect(8, -90, 3, 9, "steel", rx=1)
    arm(p, -13, -72, -27.5, -78, OWNER)
    arm(p, 13, -72, 18, -56, OWNER)
    p.path("M6,-86 Q19,-89 32,-86 L32,-30 Q19,-27 6,-30 Z", "gold")                        # the curved scutum
    p.path("M8,-84 Q19,-86.5 30,-84 L30,-32 Q19,-29.5 8,-32 Z", OWNER)
    p.circle(19, -58, 4.6, "gold")
    for s in (-1, 1):                                                                      # wings and thunderbolts
        p.thin(19 + 5 * s, -60, 19 + 10 * s, -70, 1.6, "gold")
        p.thin(19 + 5 * s, -56, 19 + 10 * s, -46, 1.6, "gold")


@unit("Mohawk Warrior")
def u_mohawk_warrior(p):
    ground(p)
    figure(p, tunic=OWNER, pants="leather", boots="leather", skirt=False, belt="red")
    p.path("M-8,-51 L8,-51 L7,-30 L-7,-30 Z", OWNER)                                       # breechcloth
    p.limb(-12, -75, 10, -52, 3.6, "white")                                                # wampum sash
    for k in range(5):
        p.circle(-9 + k * 4.6, -71 + k * 4.4, 1.0, "navy", ink=False)
    p.path("M-4,-97 Q-6,-112 0,-117 Q6,-112 4,-97 Z", "red")                               # roach crest
    p.limb(1, -112, 5, -126, 2.2, "white")
    p.limb(4, -123, 5, -127, 2.2, "black", ink=False)
    p.rect(-9, -87, 18, 3, "black", ink=False, alpha=0.8)                                  # war paint
    p.limb(-13, -72, -26, -96, 6.5, OWNER)                                                 # tomahawk, raised
    p.limb(-26, -94, -30, -128, 3.0, "wood")
    p.path("M-31,-122 L-42,-126 L-41,-116 L-31,-118 Z", "steel")
    p.circle(-26, -96, 3.8, "skin")
    arm(p, 13, -72, 19, -52, OWNER)
    p.curve("M19,-52 Q22,-36 18,-22", 3.0, "darkwood")                                     # ball-headed club
    p.circle(17, -19, 5.5, "darkwood")


@unit("Landsknecht")
def u_landsknecht(p):
    ground(p)
    p.poly([(10, -64), (16, -66), (40, -150), (34, -152)], "steel")                        # zweihander on the shoulder
    p.limb(13, -65, 37, -151, 1.0, "metal", ink=False)
    p.limb(5, -54, 12, -68, 3.6, "black")                                                  # grip
    p.limb(4, -70, 20, -62, 2.6, "gold")                                                   # cross-guard
    p.circle(4.5, -52, 2.6, "gold")
    figure(p, tunic=OWNER, pants="linen", boots="black", skirt=False, belt=None)
    p.rect(1, -26, 10, 14, "gold", rx=3)                                                   # odd hose
    for s, col in ((-1, OWNER), (1, "gold")):                                              # puffed, slashed breeches
        p.path("M%g,-52 Q%g,-38 %g,-26 L%g,-26 L%g,-52 Z" % (s * 1, s * 18, s * 13, s * 1, s * 1), col)
    for x in (-12, -7, 6, 11):
        p.thin(x, -48, x * 0.95, -30, 1.3, "black" if x > 0 else "white")
    for x in (-8, -3, 3, 8):                                                               # slashed doublet
        p.thin(x, -74, x, -54, 1.3, "white")
    beard(p)
    p.ellipse(-2, -99, 17, 4.6, "black", rot=-8)                                           # beret and plumes
    p.path("M-10,-101 Q-14,-122 -2,-126 Q-6,-112 -4,-101 Z", "white")
    p.path("M-4,-102 Q0,-128 14,-124 Q4,-114 2,-102 Z", OWNER)
    p.path("M2,-102 Q14,-120 22,-110 Q10,-106 6,-100 Z", "white")
    arm(p, -13, -72, 4, -54, OWNER, w=8)
    arm(p, 13, -72, 8, -60, OWNER, w=8)


# ============================================================================ medieval ranged and heavy foot
@unit("Longbowman")
def u_longbowman(p):
    ground(p)
    p.rect(10, -60, 8, 22, "leather", rx=3, rot=10)                                        # quiver at the hip
    for dx in (0, 3, 6):
        p.thin(12 + dx, -60, 13 + dx, -68, 1.6, "white")
    figure(p, tunic=OWNER, pants="cloth", boots="leather", belt="leather")
    hat(p, "hood", "olive")
    p.path("M-12,-80 L12,-80 L15,-70 L-15,-70 Z", "olive")                                 # hood cape
    p.limb(-13, -72, -36, -76, 6.5, OWNER)
    p.curve("M-34,-136 Q-60,-78 -34,-16", 3.4, "darkwood")                                 # a bow taller than he is
    p.thin(-34, -136, -4, -80, 0.9, "linen")
    p.thin(-4, -80, -34, -16, 0.9, "linen")
    p.thin(-4, -80, -52, -77, 1.4, "linen")
    p.path("M-52,-79.5 L-58,-77 L-52,-74.5 Z", "iron", ink=False)
    arm(p, 13, -72, -4, -80, OWNER)
    p.circle(-38, -76, 3.8, "skin")


@unit("Chu-Ko-Nu")
def u_chu_ko_nu(p):
    ground(p)
    figure(p, tunic=OWNER, pants="cloth", boots="black")
    for x in (-13, 13):                                                                    # lamellar shoulder guards
        p.rect(x - 6, -78, 12, 11, "iron", rx=2)
        p.thin(x - 5, -73, x + 5, -73, 0.9, "darkgrey")
    hat(p, "round", "darkgrey")
    p.rect(-11, -92, 4, 9, "darkgrey", rx=1)                                               # neck flaps
    p.rect(7, -92, 4, 9, "darkgrey", rx=1)
    p.circle(0, -107, 3.6, "red")                                                          # tassel
    p.rect(-38, -68.5, 44, 6, "wood", rx=2)                                                # repeating crossbow, aimed west
    p.curve("M-32,-86 Q-25,-66 -32,-48", 3.2, "darkwood")
    p.thin(-32, -86, -16, -66, 0.9, "linen")
    p.thin(-16, -66, -32, -48, 0.9, "linen")
    p.rect(-27, -92, 17, 24, "darkwood", rx=1.5)                                           # the magazine of bolts
    for x in (-24, -21, -18, -15):
        p.thin(x, -92, x, -96, 1.4, "linen")
    p.limb(-28, -90, 8, -60, 2.6, "darkwood")                                              # cocking lever, pulled back
    p.limb(-13, -72, -22, -62, 6.5, OWNER)
    p.circle(-22, -62, 3.8, "skin")
    arm(p, 13, -72, 8, -60, OWNER)


@unit("Samurai")
def u_samurai(p):
    ground(p)
    p.rect(14, -152, 2.4, 80, "darkwood", rx=1)                                            # sashimono banner
    p.rect(16.4, -150, 14, 34, OWNER)
    p.circle(23.4, -133, 4.4, "white", ink=False)
    figure(p, tunic=OWNER, pants="navy", boots="black", skirt=False, belt="gold")
    p.path("M-14,-52 L14,-52 L19,-28 L-19,-28 Z", OWNER)                                   # kusazuri
    for y in (-62, -55, -45, -37):
        p.thin(-14, y, 14, y, 1.0, "gold")
    for x in (-15, 15):
        p.rect(x - 6, -77, 12, 15, OWNER, rx=1.5)                                          # sode
        p.thin(x - 6, -70, x + 6, -70, 1.0, "gold")
    p.path("M-17,-86 Q-12,-106 0,-106 Q12,-106 17,-86 L11,-90 Q0,-96 -11,-90 Z", "black")   # kabuto
    p.path("M-3,-102 L-14,-122 L-8,-121 L0,-106 L8,-121 L14,-122 L3,-102 Z", "gold")        # kuwagata
    p.rect(-6, -86, 12, 3, "black", ink=False)
    arm(p, -13, -72, 2, -104, OWNER)
    arm(p, 13, -72, 6, -104, OWNER)
    p.curve("M4,-108 Q10,-130 30,-148", 3.2, "steel")                                      # katana, raised
    p.limb(4, -104, 3, -112, 3.2, "black")
    p.limb(-1, -108, 9, -110, 2.2, "gold")


@unit("Berserker")
def u_berserker(p):
    ground(p)
    round_shield(p, 16, -60, 13, face=OWNER, rim="iron", boss="iron")                      # shield slung on the back
    p.path("M-19,-80 L19,-80 L25,-24 Q0,-18 -25,-24 Z", BEAR)                              # bear pelt
    figure(p, tunic="skin", bare=True, pants=OWNER, boots="leather", skirt=False, belt="leather")
    p.ellipse(0, -80, 9.5, 7, "hair")                                                      # wild beard
    for s in (-1, 1):                                                                      # the bear's head as a hood
        p.circle(10 * s, -104, 4.4, BEAR)
    hat(p, "hood", BEAR)
    p.ellipse(0, -100, 8, 5.5, hexcol("#7a5638"))
    p.ellipse(0, -103, 2.6, 1.8, "black", ink=False)
    for s in (-1, 1):
        p.circle(5.5 * s, -105.5, 1.1, "black", ink=False)
        p.path("M%g,-78 L%g,-78 L%g,-64 L%g,-64 Z" % (10 * s, 15 * s, 13 * s, 8 * s), BEAR)          # paws on the chest
        for k in range(3):
            p.thin(8.5 * s + k * 1.8 * s, -64, 8.5 * s + k * 1.8 * s, -61, 0.9, "white")
    p.limb(0, -30, -36, -126, 3.4, "wood")                                                 # dane axe, raised
    p.path("M-33,-123 Q-51,-132 -50,-108 Q-44,-106 -37,-111 Z", "steel")
    arm(p, -13, -72, -20, -83, "skin")
    arm(p, 13, -72, -9, -54, "skin")
    p.limb(-15.6, -77, -16.6, -79, 7.2, "gold", ink=False)                                 # arm ring


# ============================================================================ chariots and riders
@unit("War Chariot", wide=True)
def u_war_chariot(p):
    ground(p, 58, -6)
    horse(Shift(p, -42, -3), col="darkhorse", mane="black", gait=3)                    # the pair: far horse
    horse(Shift(p, -34, 0), col="horse", gait=3)
    p.path("M-50,-50 Q-30,-56 -8,-50 L-8,-34 Q-30,-30 -50,-36 Z", OWNER)              # blanket
    p.rect(-50, -38, 42, 3, "gold", ink=False)
    for hx in (-78, -70):                                                              # feather plumes on their heads
        p.path("M%g,-76 Q%g,-90 %g,-96 Q%g,-98 %g,-92 Q%g,-86 %g,-76 Z" % (hx, hx - 6, hx - 4, hx + 2, hx + 6, hx + 4, hx + 2), OWNER)
        p.rect(hx - 1, -78, 4, 4, "gold", ink=False)
    p.limb(-18, -40, 18, -36, 2.6, "darkwood")                                         # draught pole
    q = Shift(p, 34, -22, 0.86)                                                        # the archer
    figure(q, tunic="linen", pants="skin", boots="leather", belt="gold")
    q.path("M-12,-77 Q0,-70 12,-77 L10,-70 Q0,-64 -10,-70 Z", "gold")                  # broad collar
    q.path("M-11,-86 Q-12,-104 0,-106 Q12,-104 11,-86 L14,-74 L8,-76 L8,-86 L-8,-86 L-8,-76 L-14,-74 Z", OWNER)   # nemes
    for y in (-100, -94):
        q.thin(-10, y, 10, y, 1.0, "gold")
    q.limb(-13, -72, -34, -76, 6.5, "skin")
    bow(q, recurve=True)
    arm(q, 13, -72, -5, -80, "skin")
    q.circle(-36, -76, 3.8, "skin")
    p.path("M16,-50 Q14,-26 22,-20 L52,-20 L52,-44 Q34,-44 16,-50 Z", OWNER)            # the car, open at the back
    p.path("M18,-48 Q34,-42 50,-42 L50,-38 Q34,-38 18,-44 Z", "gold")
    p.rect(42, -62, 6, 28, "leather", rx=2, rot=-24)                                   # quiver on its side
    wheel(p, 46, -15, 15, col="wood", spokes=6, hub="gold")


@unit("Horse Archer")
def u_horse_archer(p):
    ground(p, 40, 0)
    horse(p, col=DUN, mane="mane", gait=4)
    ls, rs, hd = rider(p, coat=OWNER, legs="leather", boots="boots")
    p.path("M-8,-50 L14,-50 L14,-44 L-8,-44 Z", "leather")                             # fur-trimmed hem
    p.rect(14, -66, 7, 22, "leather", rx=2.5, rot=30)                                  # quiver
    for dx in (0, 3):
        p.thin(18 + dx, -66, 24 + dx, -72, 1.4, "linen")
    head_at(p, *hd, hat_kind="cone", hat_col="leather")
    p.ellipse(hd[0], hd[1] - 4, 12, 3.4, "darkwood")                                   # fur brim
    q = Shift(p, 56, -2)                                                               # turning in the saddle to shoot back
    q.curve("M-6,-106 Q10,-78 -6,-48", 3.2, "darkwood")                                # recurve bow, aimed east
    q.curve("M-6,-106 Q-10,-104 -8,-100", 3.0, "darkwood")
    q.curve("M-6,-48 Q-10,-50 -8,-54", 3.0, "darkwood")
    q.thin(-6, -106, -36, -78, 0.9, "linen")
    q.thin(-36, -78, -6, -48, 0.9, "linen")
    q.thin(-36, -78, 6, -77, 1.4, "linen")
    q.path("M6,-79.5 L12,-77 L6,-74.5 Z", "iron", ink=False)
    arm(p, *rs, 48, -78, OWNER)
    arm(p, *ls, 18, -80, OWNER)


@unit("Companion Cavalry")
def u_companion_cavalry(p):
    ground(p, 40, 0)
    horse(p, col="greyhorse", mane="iron", gait=3)
    p.path("M-14,-50 L22,-50 L26,-34 L-16,-36 Z", (0.86, 0.66, 0.36))                  # leopard-skin saddle cloth
    for x, y in ((-8, -44), (2, -40), (12, -45), (18, -38)):
        p.ellipse(x, y, 2.2, 1.6, "darkwood", ink=False)
    p.path("M0,-82 Q26,-84 46,-62 Q38,-54 30,-60 Q18,-70 6,-70 Z", OWNER)              # cloak, flying back
    ls, rs, hd = rider(p, coat="linen", legs="skin", boots="leather", armour="bronze")
    q = Shift(p, hd[0], hd[1] + 89)
    q.ellipse(0, -96, 14, 3.4, "bronze")                                               # Boeotian helmet
    q.path("M-9,-96 Q-9,-106 0,-107 Q9,-106 9,-96 Z", "bronze")
    q.path("M-2,-106 Q2,-116 14,-112 Q8,-108 4,-104 Z", OWNER)
    p.limb(30, -74, -62, -58, 2.8, "wood")                                             # xyston, held low and forward
    p.path("M-62,-61 L-72,-57 L-62,-55 Z", "steel")
    arm(p, *rs, 14, -66, "skin")
    arm(p, *ls, -24, -64, "skin")


@unit("Cataphract")
def u_cataphract(p):
    ground(p, 40, 0)
    horse(p, col="darkhorse", mane="black", barding=OWNER)
    p.path("M-28,-18 L33,-18 L32,-14 L-28,-14 Z", OWNER)                               # barding down to the knees
    for k, y in enumerate((-42, -34, -26, -20)):                                       # scales
        for x in range(-24 + 3 * (k % 2), 32, 6):
            p.thin(x - 3, y, x, y + 3, 1.3, "black")
            p.thin(x, y + 3, x + 3, y, 1.3, "black")
    p.path("M-44,-71 L-34,-74 L-30,-62 L-44,-56 Z", "steel")                           # chamfron
    ls, rs, hd = rider(p, coat="steel", legs="steel", boots="steel")
    for y in (-76, -70, -64, -58, -52):                                                # lamellar
        p.thin(-7, y, 13, y, 0.9, "darkgrey")
    q = Shift(p, hd[0], hd[1] + 89)
    q.path("M-10,-80 L-10,-90 Q-10,-104 0,-114 Q10,-104 10,-90 L10,-80 Z", "steel")    # spangenhelm with a mail aventail
    q.rect(-6, -93, 12, 2.2, "black", ink=False)
    q.circle(0, -114, 2, "gold", ink=False)
    p.limb(28, -64, -60, -80, 3.0, "wood")                                             # kontos, couched
    p.path("M-60,-83 L-70,-82 L-61,-77 Z", "steel")
    p.poly([(-48, -78), (-34, -75), (-38, -69), (-49, -72)], OWNER)
    arm(p, *rs, 16, -66, "steel", hand="steel")
    arm(p, *ls, -16, -70, "steel", hand="steel")
    round_shield(p, 22, -62, 9, face=OWNER, rim="gold", boss="gold")


@unit("Camel Archer")
def u_camel_archer(p):
    ground(p, 40, 0)
    camel(p)
    p.path("M-4,-88 Q8,-102 24,-90 L30,-68 Q12,-62 -6,-70 Z", OWNER)                   # saddle blanket over the hump
    for x in range(-4, 30, 5):
        p.thin(x, -69 + 0.1 * (x - 12) ** 2 * 0.05, x + 1, -64, 1.1, "gold")
    q = Shift(p, 8, -42)
    ls, rs, hd = rider(q, coat="white", legs="white", boots="leather")
    q.path("M-9,-74 L15,-74 L13,-48 L-8,-48 Z", OWNER)                                 # robe
    h = Shift(q, hd[0], hd[1] + 89)
    h.path("M-11,-86 Q-12,-104 0,-104 Q12,-104 11,-86 L13,-74 L8,-78 L8,-86 L-8,-86 L-8,-78 L-13,-74 Z", "white")   # keffiyeh
    h.rect(-10, -101, 20, 3.4, "black", rx=1.5)                                        # agal
    h.limb(-13, -72, -34, -76, 6.5, OWNER)
    bow(h, recurve=True)
    arm(h, 13, -72, -5, -80, OWNER)
    h.circle(-36, -76, 3.8, "skin")


@unit("Conquistador")
def u_conquistador(p):
    ground(p, 40, 0)
    horse(p, col="horse", gait=2)
    p.path("M-12,-50 L22,-50 L24,-30 L-14,-30 Z", OWNER)                               # saddle cloth
    p.rect(-14, -32, 38, 2.6, "gold", ink=False)
    ls, rs, hd = rider(p, coat=OWNER, legs=OWNER, boots="black", armour="steel")
    p.thin(-6, -62, 12, -62, 1.0, "gold")
    q = Shift(p, hd[0], hd[1] + 89)
    beard(q)
    hat(q, "morion", "steel", plume="red")
    arm(p, *rs, -14, -84, OWNER)
    sword_up(p, -16, -86, -50, -112, w=2.6, blade="steel", guard="gold")              # rapier, pointed ahead
    arm(p, *ls, -20, -66, OWNER)


@unit("Mandekalu Cavalry")
def u_mandekalu_cavalry(p):
    ground(p, 40, 0)
    horse(p, col="horse", mane="mane", barding=OWNER, gait=2)
    p.path("M-28,-18 L33,-18 L32,-12 L-28,-12 Z", OWNER)                               # quilted caparison, long
    for x in range(-24, 32, 8):                                                        # its quilting
        p.poly([(x, -42), (x + 4, -34), (x - 4, -34)], "white", ink=False)
        p.poly([(x + 4, -26), (x + 8, -18), (x, -18)], INDIGO, ink=False)
    p.limb(32, -54, 4, -142, 2.6, "wood")                                              # spear, held upright
    p.path("M4,-142 L0,-156 L9,-143 Z", "steel")
    ls, rs, hd = rider(p, coat="white", legs="white", boots="leather")
    p.path("M-8,-72 L12,-72 L18,-40 L-10,-40 Z", "white")                             # boubou robe
    p.thin(-8, -60, 14, -60, 1.2, INDIGO)
    q = Shift(p, hd[0], hd[1] + 89)
    q.path("M-11,-92 Q-13,-109 0,-111 Q13,-109 11,-92 Q0,-95 -11,-92 Z", INDIGO)        # turban
    q.path("M-9,-84 L9,-84 L8,-77 Q0,-73 -8,-77 Z", INDIGO)                            # veil, below the eyes
    arm(p, *rs, 19, -100, "white")
    arm(p, *ls, -16, -68, "white")
    p.ellipse(-16, -64, 9, 12, "leather")                                              # hide shield
    p.ellipse(-16, -64, 5, 8, "wood", ink=False)
    p.circle(-16, -64, 2.4, "gold")


@unit("Keshik")
def u_keshik(p):
    ground(p, 40, 0)
    horse(p, col=DUN, mane="black", gait=4)
    p.rect(16, -64, 8, 22, "leather", rx=2.5, rot=26)                                  # quiver
    for dx in (0, 3):
        p.thin(20 + dx, -64, 25 + dx, -71, 1.4, "linen")
    ls, rs, hd = rider(p, coat=OWNER, legs="leather", boots="boots")
    for y in (-72, -66, -60, -54):                                                     # lamellar
        p.thin(-7, y, 13, y, 0.9, "darkgrey")
    q = Shift(p, hd[0], hd[1] + 89)
    q.path("M-10,-90 Q-10,-104 0,-112 Q10,-104 10,-90 Z", "iron")                      # helmet
    q.path("M-12,-88 L-12,-93 L12,-93 L12,-88 Z", "darkwood")                          # fur brim
    q.path("M-1,-112 Q4,-122 10,-118 Q4,-116 1,-110 Z", "red")                         # tassel
    q.limb(-13, -72, -34, -76, 6.5, OWNER)
    bow(q, recurve=True)
    arm(q, 13, -72, -5, -80, OWNER)
    q.circle(-36, -76, 3.8, "skin")


# ============================================================================ war elephants
@unit("War Elephant", wide=True)
def u_war_elephant(p):
    ground(p, 56, -4)
    elephant(p)
    p.path("M-8,-90 Q16,-96 40,-86 L42,-52 Q16,-46 -8,-54 Z", OWNER)                       # caparison
    p.path("M-8,-58 Q16,-50 42,-56 L42,-52 Q16,-46 -8,-54 Z", "gold")
    p.rect(-4, -114, 40, 22, "wood", rx=2)                                                 # howdah
    for x in (-2, 34):
        p.rect(x - 1, -136, 2.2, 24, "darkwood")
    p.path("M-6,-134 Q16,-150 38,-134 Z", OWNER)
    q = Shift(p, 18, -96, 0.5)                                                             # the archer in the howdah
    q.circle(0, -89, 9.5, "skin")
    hat(q, "turban", "white")
    q.limb(-13, -72, -34, -76, 6.5, OWNER)
    bow(q)
    p.rect(-4, -114, 40, 9, "wood", rx=2)
    p.circle(-26, -104, 5, "skin")                                                         # the mahout on its neck
    p.path("M-31,-106 Q-31,-114 -26,-114 Q-21,-114 -21,-106 Z", OWNER)
    p.limb(-26, -100, -22, -90, 5, OWNER)


@unit("African Forest Elephant", wide=True)
def u_african_forest_elephant(p):
    ground(p, 50, -4)
    elephant(p, s=0.92, col=AFRICAN, ears="african", trunk="raised")
    p.path("M-6,-84 Q16,-90 38,-80 L40,-52 Q16,-46 -6,-54 Z", OWNER)                   # caparison
    p.path("M-6,-58 Q16,-50 40,-56 L40,-52 Q16,-46 -6,-54 Z", "gold")
    for x in range(0, 38, 8):
        p.circle(x, -51, 1.8, "gold", ink=False)
    q = Shift(p, -18, -60, 0.8)                                                        # the rider on its neck
    ls, rs, hd = rider(q, coat=OWNER, legs="skin", boots="leather", armour="bronze")
    h = Shift(q, hd[0], hd[1] + 89)
    hat(h, "crest", "bronze")
    h.path("M-10,-104 Q0,-120 12,-104 Q0,-110 -10,-104 Z", OWNER)
    q.limb(26, -64, -40, -110, 2.6, "wood")                                            # spear, levelled
    q.path("M-40,-110 L-52,-116 L-42,-104 Z", "bronze")
    arm(q, *rs, 4, -90, "skin")
    arm(q, *ls, -14, -68, "skin")
    round_shield(q, 20, -62, 11, face=OWNER, rim="bronze", boss="bronze")


@unit("Naresuan's Elephant", wide=True)
def u_naresuans_elephant(p):
    ground(p, 56, -4)
    elephant(p)
    p.path("M-8,-90 Q16,-96 40,-86 L42,-48 Q16,-42 -8,-50 Z", OWNER)                   # caparison
    for y in (-54, -66):
        p.path("M-8,%g Q16,%g 42,%g L42,%g Q16,%g -8,%g Z" % (y - 4, y + 4, y - 2, y + 2, y + 8, y), "gold")
    p.path("M-30,-90 Q-40,-96 -50,-90 L-46,-82 Q-38,-86 -30,-84 Z", "gold")            # gold headdress
    p.path("M-44,-90 L-40,-104 L-36,-90 Z", "gold")
    p.rect(44, -142, 2.2, 52, "darkwood")                                              # parasol of state, behind
    for w, y in ((19, -124), (15, -132), (11, -140)):
        p.rect(45 - w, y, 2 * w, 7, "white", rx=2.5)
        p.rect(45 - w, y + 6, 2 * w, 1.8, "gold", ink=False)
    p.poly([(43, -140), (47, -140), (45, -150)], "gold")
    p.rect(0, -106, 34, 16, "red", rx=2)                                               # the royal howdah
    p.rect(4, -103, 26, 9, OWNER, rx=1.5)
    p.rect(0, -94, 34, 3, "gold", ink=False)
    p.poly([(-2, -106), (36, -106), (28, -114), (6, -114)], "red")                     # its tiered roof
    p.poly([(8, -114), (26, -114), (21, -121), (13, -121)], "red")
    p.poly([(14, -121), (20, -121), (17, -136)], "gold")
    q = Shift(p, -24, -64, 0.72)                                                       # the king, on its neck
    ls, rs, hd = rider(q, coat="gold", legs=OWNER, boots="gold")
    h = Shift(q, hd[0], hd[1] + 89)
    h.path("M-9,-90 Q-9,-100 0,-104 L0,-120 Q9,-100 9,-90 Z", "gold")                 # pointed crown
    q.limb(22, -64, -44, -116, 3.0, "darkwood")                                        # ngao halberd
    q.path("M-44,-116 Q-60,-120 -62,-132 Q-48,-128 -38,-112 Z", "steel")
    arm(q, *rs, -4, -94, "gold")
    arm(q, *ls, -16, -68, "gold")


# ============================================================================ siege
@unit("Battering Ram", wide=True)
def u_battering_ram(p):
    ground(p, 52, 4)
    p.rect(-30, -26, 72, 8, "darkwood", rx=2)                                          # chassis
    for x in (-24, 36):
        p.rect(x - 2, -72, 4, 48, "wood", rx=1)
    p.thin(6, -84, -4, -48, 1.2, "rope")                                               # the ram's ropes
    p.thin(20, -84, 26, -48, 1.2, "rope")
    p.limb(-56, -46, 40, -46, 9, "wood")                                               # the ram itself
    p.path("M-56,-54 L-70,-50 L-74,-46 L-70,-42 L-56,-38 Z", "iron")                   # its iron head
    p.curve("M-62,-52 Q-58,-60 -52,-56", 2.4, "iron")
    p.path("M-36,-70 L8,-98 L52,-70 L46,-62 L8,-88 L-30,-62 Z", OWNER)                 # roof of hides
    p.path("M-30,-62 L8,-88 L46,-62 L46,-58 L-30,-58 Z", "leather")
    for x in (-18, 2, 22, 38):
        p.thin(x, -62, 8 + (x - 8) * 0.4, -86, 1.0, "darkwood")
    wheel(p, -14, -13, 13)
    wheel(p, 30, -13, 13)


@unit("Ballista", wide=True)
def u_ballista(p):
    ground(p, 44, 4)
    p.poly([(-4, -50), (4, -50), (26, -2), (18, -2)], "darkwood")                       # trestle legs
    p.poly([(-4, -50), (4, -50), (-18, -2), (-26, -2)], "darkwood")
    p.poly([(-2, -50), (2, -50), (4, -2), (-4, -2)], "wood")
    p.rect(-52, -60, 92, 7, "wood", rx=2)                                              # stock, aimed west
    p.rect(-36, -84, 16, 36, OWNER, rx=2)                                              # torsion frame and its plate
    p.rect(-36, -84, 16, 4, "bronze", rx=1)
    p.rect(-36, -52, 16, 4, "bronze", rx=1)
    p.circle(-28, -66, 3.4, "bronze")
    p.limb(-30, -82, -6, -98, 3.6, "darkwood")                                         # arms, drawn back
    p.limb(-30, -50, -6, -38, 3.6, "darkwood")
    p.thin(-6, -98, 26, -57, 1.0, "linen")                                             # bowstring to the trigger
    p.thin(-6, -38, 26, -57, 1.0, "linen")
    p.limb(-60, -62, 22, -62, 2.4, "wood")                                             # the bolt
    p.path("M-60,-65 L-68,-62 L-60,-59 Z", "iron")
    p.poly([(16, -66), (24, -62), (16, -58)], "linen", ink=False)
    p.circle(34, -57, 5, "iron")                                                       # winch


@unit("Hwach'a", wide=True)
def u_hwacha(p):
    ground(p, 46, 2)
    for x, y in ((-44, -106), (-50, -92)):                                             # rockets leaving, trailing smoke
        p.circle(x + 16, y + 6, 4.5, "grey", alpha=0.5, ink=False)
        p.circle(x + 24, y + 11, 3.5, "grey", alpha=0.35, ink=False)
        p.glow(x + 9, y + 4, 4, "flame", alpha=0.8)
        p.limb(x, y, x + 10, y + 5, 1.8, "darkwood", ink=False)
        p.path("M%g,%g L%g,%g L%g,%g Z" % (x - 3, y - 1.5, x + 1, y - 2, x + 1, y + 2), "iron", ink=False)
    p.limb(18, -28, 56, -8, 3.6, "darkwood")                                           # cart shafts
    p.rect(-30, -36, 60, 9, "wood", rx=2)                                              # cart bed
    p.limb(-12, -66, -12, -36, 3.4, "darkwood")                                        # props
    p.limb(18, -86, 20, -36, 3.4, "darkwood")
    p.path("M-40,-82 L20,-110 L32,-84 L-28,-56 Z", "darkwood")                         # launcher box, raised toward the west
    p.path("M-36,-80 L19,-106 L28,-85 L-27,-60 Z", "wood")
    for r in range(4):                                                                 # the honeycomb of rocket tubes
        for c in range(8):
            x = -32 + c * 7 + r * 2.6
            y = -79 + r * 5.4 - c * 3.3
            p.circle(x, y, 1.8, "black", ink=False)
    for c in range(8):                                                                 # arrows loaded in the top row
        x, y = -32 + c * 7, -79 - c * 3.3
        p.limb(x, y, x - 9, y - 5, 1.3, "darkwood", ink=False)
        p.path("M%g,%g L%g,%g L%g,%g Z" % (x - 9, y - 7, x - 13, y - 7, x - 10, y - 3.5), "iron", ink=False)
    p.rect(-30, -36, 60, 9, OWNER, rx=2)                                               # painted cart bed
    p.rect(26, -126, 2, 44, "darkwood")                                                # pennant
    p.poly([(28, -126), (46, -121), (28, -114)], OWNER)
    wheel(p, 0, -18, 18)


# ============================================================================ ships
@unit("Quinquereme", at=AT_SEA, wide=True)
def u_quinquereme(p):
    water(p, -60, 60)
    oars(p, -36, 40, 10, -8)                                                                # two banks of oars
    oars(p, -32, 44, 10, -16)
    hull(p, -58, 58, 22, col="hull", rake=12, stern_h=30)
    p.thin(-46, -14, 56, -14, 1.0, "darkwood")
    p.path("M-58,-14 L-72,-4 L-58,-7 Z", "bronze")                                          # ram
    p.circle(-48, -12, 2.6, "white", ink=False)
    p.circle(-48.6, -12, 1.3, "black", ink=False)
    p.curve("M58,-30 Q72,-42 62,-54", 3.6, "darkwood")                                      # curled stern
    p.rect(-46, -38, 18, 16, "wood", rx=1.5)                                                # fighting tower
    mast(p, 6, -96, -22)
    square_sail(p, 6, -92, -42, 40)
    p.rect(-14, -70, 40, 6, "white", ink=False)                                             # sail band
    for x in range(-24, 52, 9):
        round_shield(p, x, -25, 4.4, face=OWNER, rim="bronze", boss=None)                   # shields on the rail


@unit("Dromon", at=AT_SEA, wide=True)
def u_dromon(p):
    water(p, -56, 56)
    oars(p, -28, 36, 8, -12)
    hull(p, -54, 54, 16, col="darkwood", rake=10, stern_h=26)
    p.rect(-50, -30, 20, 14, "wood", rx=1.5)                                                # forecastle
    p.limb(-46, -26, -64, -31, 3.4, "bronze")                                               # the fire siphon
    p.glow(-80, -33, 16, "flame", 0.55)
    p.path("M-64,-31 Q-74,-42 -92,-36 Q-82,-32 -94,-27 Q-78,-22 -64,-28 Z", "flame")        # Greek fire
    p.path("M-66,-31 Q-74,-36 -84,-33 Q-76,-29 -66,-29 Z", "straw", ink=False)
    for x, top, w in ((-14, -98, 48), (22, -82, 36)):
        mast(p, x, top, -16)
        lateen(p, x, top + 6, -26, w)
    ensign(p, 50, -26, 22, 13)


@unit("Turtle Ship", at=AT_SEA, wide=True)
def u_turtle_ship(p):
    water(p, -60, 60)
    oars(p, -34, 40, 9, -12)
    hull(p, -56, 56, 14, col="darkwood", rake=10, stern_h=18)
    for x in range(-36, 44, 12):
        p.rect(x - 2, -11, 4, 4, "black", ink=False)                                       # gun ports
    mast(p, 6, -100, -40)
    square_sail(p, 6, -96, -52, 34)
    p.path("M-48,-14 Q-46,-48 2,-50 Q48,-48 54,-14 Z", (0.30, 0.34, 0.32))                 # the iron-plated turtle roof
    for k in range(1, 4):
        y = -14 - k * 9
        w = 50 - k * 9
        p.thin(2 - w, y, 2 + w, y, 1.0, "darkgrey")
    for k in range(-4, 5):
        x = 2 + k * 10
        y = -48 + 0.11 * (k * 10) ** 2 * 0.06
        p.poly([(x - 2, y + 1), (x, y - 5), (x + 2, y + 1)], "steel", ink=False)           # spikes
    p.path("M-48,-16 Q-56,-30 -66,-30 Q-72,-26 -68,-22 L-60,-22 Q-58,-14 -50,-12 Z", "gold")   # dragon head
    p.circle(-62, -27, 1.4, "black", ink=False)
    for k in range(3):
        p.circle(-74 - 5 * k, -26 - 2 * k, 3 + 1.5 * k, "grey", alpha=0.6 - 0.15 * k, ink=False)


@unit("Sea Beggar", at=AT_SEA, wide=True)
def u_sea_beggar(p):
    water(p, -50, 50)
    hull(p, -48, 48, 16, col="wood", rake=10, stern_h=22)
    p.path("M30,-20 L52,-22 Q58,-40 50,-50 L34,-46 Z", "darkwood")                         # tall rounded stern
    p.rect(38, -42, 10, 5, "gold", rx=1)
    p.thin(-40, -14, 46, -16, 1.6, "gold")
    p.ellipse(-6, -10, 8, 13, "darkwood")                                                   # leeboard
    p.circle(-6, -20, 1.6, "iron", ink=False)
    p.limb(-48, -18, -66, -28, 2, "darkwood")
    p.poly([(-64, -28), (-30, -84), (-28, -22)], "sail")                                    # jib
    mast(p, -6, -104, -16)
    square_sail(p, -6, -98, -66, 30)
    square_sail(p, -6, -62, -26, 36)
    mast(p, 26, -76, -22)
    p.path("M27,-74 L44,-62 L44,-30 L27,-30 Z", OWNER)                                     # mizzen
    for i, col in enumerate((OWNER, "white", "navy")):                                      # tricolour ensign
        p.rect(52, -62 + 5 * i, 16, 5, col, ink=False)
    p.rect(51, -64, 1.6, 44, "darkwood")
    pennant(p, -5, -106, 22)


@unit("Ship of the Line", at=AT_SEA, wide=True)
def u_ship_of_the_line(p):
    water(p, -64, 64)
    hull(p, -60, 60, 40, col="black", rake=12, stern_h=50)
    for y in (-14, -24, -34):                                                               # three gun decks
        p.rect(-50, y, 108, 5, "straw", ink=False)
        for x in range(-46, 56, 8):
            p.rect(x, y + 1, 3.4, 3, "black", ink=False)
    p.rect(-52, -40, 110, 3.6, OWNER, ink=False)
    p.rect(50, -50, 10, 8, "gold", rx=1)                                                    # stern windows
    p.limb(-60, -38, -84, -52, 2.4, "darkwood")                                             # bowsprit
    p.poly([(-60, -42), (-36, -112), (-38, -46)], "sail")
    for x, h in ((-30, 104), (4, 122), (36, 96)):
        mast(p, x, -40 - h, -40)
        square_sail(p, x, -40 - h * 0.94 + 2, -40 - h * 0.7, 20, "sail")
        square_sail(p, x, -40 - h * 0.66, -40 - h * 0.38, 26, "sail")
        square_sail(p, x, -40 - h * 0.34, -48, 32, "sail")
    pennant(p, 5, -164, 26)
    ensign(p, 60, -50, 26, 18)


# ============================================================================ gunpowder foot
@unit("Musketeer")
def u_musketeer(p):
    ground(p)
    rifle(p, (-24, -2), (-27, -112))                                                       # musket, grounded
    figure(p, tunic="navy", pants="navy", boots="black")
    for x in (-6, 6):
        p.rect(x - 7, -36, 14, 9, "leather", rx=2)                                         # cuffed boots
    p.path("M-12,-76 L12,-76 L15,-34 L-15,-34 Z", OWNER)                                   # the tabard
    p.rect(-2.6, -72, 5.2, 34, "white", ink=False)                                          # its cross
    p.rect(-11, -62, 22, 5.2, "white", ink=False)
    p.ellipse(0, -77, 9, 3.2, "white")                                                     # lace collar
    hair(p)
    p.path("M-10,-90 Q-13,-80 -10,-74 L-7,-76 Q-9,-82 -8,-90 Z", "hair", ink=False)        # long hair
    p.path("M10,-90 Q13,-80 10,-74 L7,-76 Q9,-82 8,-90 Z", "hair", ink=False)
    moustache(p)
    plumed_hat(p, "black", "white")
    arm(p, -13, -72, -25, -62, "navy")
    arm(p, 13, -72, 22, -94, "navy")
    p.limb(23, -96, 38, -138, 1.8, "steel")                                                 # rapier, raised
    p.curve("M19,-95 Q22,-91 27,-97", 2.0, "gold")


@unit("Janissary")
def u_janissary(p):
    ground(p)
    figure(p, tunic=OWNER, pants="red", boots="gold", belt=None, long_coat=True)
    p.rect(-14, -54, 28, 7, "red", rx=2)                                                   # sash
    p.curve("M12,-48 Q20,-34 14,-22", 3.0, "steel")                                         # yataghan at the hip
    p.ellipse(0, -82, 8, 2.8, "hair", ink=False)                                            # big moustache
    p.path("M6,-120 Q20,-108 18,-80 L12,-82 Q13,-104 3,-114 Z", "white")                    # the bork's flap
    p.path("M-9,-96 L-7,-126 Q0,-130 7,-126 L9,-96 Z", "white")                            # tall white bork
    p.rect(-3.5, -112, 7, 10, "gold", rx=1.5)
    p.rect(-10, -99, 20, 4, "gold", rx=1.5)
    rifle(p, (-18, -38), (24, -132))                                                       # long matchlock
    arm(p, 13, -72, 10, -94, OWNER)
    arm(p, -13, -72, -10, -52, OWNER)


@unit("Minuteman")
def u_minuteman(p):
    ground(p, 30, 6)
    p.limb(14, -4, 30, -36, 3, "wood")                                                     # plough left in the field
    p.limb(30, -36, 34, -48, 2.6, "wood")
    p.path("M10,-6 L22,-4 L20,2 L8,0 Z", "iron")
    figure(p, tunic="linen", pants="khaki", boots="black", skirt=False, belt=None)
    for x in (-6, 6):
        p.rect(x - 4.5, -30, 9, 18, "white", rx=3)                                          # stockings
    p.path("M-13,-76 L-4,-76 L-6,-28 L-17,-30 Z", OWNER)                                    # open coat
    p.path("M13,-76 L4,-76 L6,-28 L17,-30 Z", OWNER)
    p.limb(-11, -75, 9, -50, 2.0, "leather")                                                # powder horn strap
    p.ellipse(11, -48, 6, 3.4, "white", rot=-30)
    hair(p)
    hat(p, "tricorne", "black")
    rifle(p, (-20, -40), (22, -118))                                                       # musket, ready
    arm(p, 13, -72, 10, -90, OWNER)
    arm(p, -13, -72, -12, -54, OWNER)


@unit("Tercio")
def u_tercio(p):
    ground(p, 32, 4)
    q = Shift(p, 16, -2, 0.9)                                                              # pikeman behind
    p.rect(30, -168, 2.8, 166, "wood", rx=1.2)
    p.path("M31.4,-180 L33.6,-167 L29.2,-167 Z", "steel")
    p.path("M32.8,-160 L52,-158 L48,-150 L52,-142 L32.8,-140 Z", "white")                  # a Burgundy cross banner
    p.limb(36, -158, 48, -143, 2.2, "red")
    p.limb(48, -158, 36, -143, 2.2, "red")
    figure(q, tunic=OWNER, pants=OWNER, boots="black")
    q.path("M-12,-73 L12,-73 L11,-50 L-11,-50 Z", "steel")
    hat(q, "morion", "steel")
    arm(q, 13, -72, 16, -96, OWNER)
    m = Shift(p, -12, 0)                                                                    # arquebusier in front
    figure(m, tunic=OWNER, pants=OWNER, boots="black")
    for x in (-15, 15):
        m.ellipse(x, -66, 6, 9, OWNER)                                                      # puffed sleeves
        m.thin(x - 3, -72, x - 3, -60, 1.0, "gold")
        m.thin(x + 3, -72, x + 3, -60, 1.0, "gold")
    m.path("M-12,-73 L12,-73 L11,-50 L-11,-50 Z", "steel")                                  # breastplate
    hat(m, "morion", "steel")
    moustache(m)
    rifle(m, (16, -52), (26, -118))                                                         # arquebus on the shoulder
    arm(m, 13, -72, 16, -56, OWNER)
    arm(m, -13, -72, -14, -50, OWNER)


@unit("Norwegian Ski Infantry")
def u_norwegian_ski_infantry(p):
    ground(p, 34, 0)
    p.path("M-42,-6 Q-48,-6 -46,-12 L-42,-10 L38,-4 L38,0 L-40,-2 Z", "wood")             # skis, tips up
    p.limb(-26, -52, -36, -2, 2.0, "darkgrey")                                              # poles
    p.limb(26, -52, 32, -2, 2.0, "darkgrey")
    for x in (-36, 32):
        p.ellipse(x, -4, 4, 1.6, "darkgrey", ink=False)
    rifle(p, (-12, -50), (16, -106))                                                        # rifle slung on the back
    figure(p, tunic="white", pants="white", boots="black", skirt=False, belt="leather")
    p.rect(-9, -80, 18, 5, OWNER, rx=2)                                                    # scarf
    p.limb(-13.2, -69, -12, -65, 7.6, OWNER)                                                # armband
    p.path("M-10.5,-88 Q-11,-104 0,-104 Q11,-104 10.5,-88 Z", "white")                      # knitted cap with a bobble
    p.rect(-11, -92, 22, 5, OWNER, rx=2)
    p.circle(0, -105, 4, OWNER)
    arm(p, -13, -72, -26, -54, "white")
    arm(p, 13, -72, 26, -54, "white")


@unit("Carolean")
def u_carolean(p):
    ground(p)
    rifle(p, (-14, -46), (16, -104))                                                        # musket slung on the back
    figure(p, tunic=OWNER, pants="khaki", boots="black", belt=None, long_coat=True)
    p.rect(-6, -74, 12, 26, "khaki", ink=False)                                             # buff waistcoat
    for y in (-70, -64, -58, -52):
        p.circle(0, y, 1.0, "gold", ink=False)
    p.path("M-9,-93 L-8,-112 Q0,-124 8,-112 L9,-93 Z", OWNER)                               # grenadier mitre cap
    p.path("M-7,-95 L-6,-110 Q0,-118 6,-110 L7,-95 Z", "gold")
    p.circle(0, -105, 2.4, OWNER, ink=False)
    hair(p)
    arm(p, -13, -72, -14, -52, OWNER)
    p.rect(-17.4, -58, 7, 5, "gold", rx=1.5)                                                # cuff
    arm(p, 13, -72, 24, -98, OWNER)
    p.rect(17, -88, 7, 5, "gold", rx=1.5, rot=-50)
    p.limb(25, -100, 44, -136, 2.4, "steel")                                                # sword, raised for the charge
    p.limb(21, -98, 29, -103, 2.2, "gold")


@unit("Mehal Sefari")
def u_mehal_sefari(p):
    ground(p)
    rifle(p, (20, -2), (24, -118))                                                          # rifle, grounded
    figure(p, tunic="white", pants="white", boots="skin", skirt=False, belt=None)
    p.path("M-15,-76 L15,-76 L19,-40 L12,-36 L-14,-46 Z", "white")                         # the shamma
    p.limb(-15, -48, 18, -38, 3.2, OWNER)                                                   # its woven border
    p.limb(-13, -74, 13, -56, 3.0, OWNER)
    p.circle(0, -94, 15, SABLE)                                                               # lion's-mane headdress
    p.circle(0, -89, 9.5, "skin")
    p.rect(-3.2, -84, 6.4, 8, "skin", rx=2)
    p.rect(-9, -100, 18, 4, "gold", rx=1.5)
    arm(p, 13, -72, 21, -60, "white")
    arm(p, -13, -72, -18, -58, "white")
    p.circle(-20, -52, 16, "leather")                                                       # hide shield with brass studs
    p.circle(-20, -52, 4, "gold")
    for k in range(6):
        a = k * math.pi / 3
        p.circle(-20 + 10 * math.cos(a), -52 + 10 * math.sin(a), 1.8, "gold", ink=False)


@unit("Foreign Legion")
def u_foreign_legion(p):
    ground(p)
    p.rect(-12, -80, 24, 27, "sand", rx=3)                                                  # pack
    figure(p, tunic="sand", pants="sand", boots="black", skirt=False, belt=None, long_coat=True)
    p.rect(-14, -54, 28, 8, OWNER, rx=2)                                                    # sash
    for x in (-13, 13):
        p.ellipse(x, -74, 6, 3, "red")                                                      # epaulettes
        for k in range(3):
            p.thin(x - 4 + 4 * k, -72, x - 4 + 4 * k, -67, 1.2, "olive")
    p.path("M-10,-93 L-11,-74 Q0,-71 11,-74 L10,-93 Z", "white")                             # havelock
    p.circle(0, -89, 9.5, "skin")
    p.path("M-9.5,-93 L-8,-106 Q0,-108 8,-106 L9.5,-93 Z", "white")                          # white kepi
    p.path("M-10,-94 L10,-94 L13,-91 L-3,-91 Z", "black")
    rifle(p, (14, -48), (24, -124))                                                         # shoulder arms
    arm(p, 13, -72, 15, -54, "sand")
    arm(p, -13, -72, -14, -50, "sand")


# ============================================================================ mounted
@unit("Sipahi")
def u_sipahi(p):
    ground(p, 40, 0)
    horse(p, col="greyhorse", mane="iron", gait=2)
    p.path("M-10,-52 L20,-52 L24,-28 L-8,-30 Z", OWNER)                                     # saddle cloth
    p.thin(-8, -32, 24, -30, 1.6, "gold")
    p.rect(20, -146, 3, 116, "wood", rx=1.4)                                                # lance, upright
    p.path("M21.5,-158 L24,-146 L19,-146 Z", "steel")
    p.poly([(23, -144), (44, -140), (36, -134), (44, -128), (23, -126)], OWNER)
    ls, rs, hd = rider(p, coat=OWNER, legs="red", boots="gold")
    q = Shift(p, hd[0], hd[1] + 89)                                                         # turban helmet with a spike
    hat(q, "turban", "white")
    q.limb(0, -104, 0, -116, 2.2, "steel")
    arm(p, *rs, 21.5, -76, OWNER)
    arm(p, *ls, -16, -66, OWNER)
    round_shield(p, -16, -64, 13, face=OWNER, rim="gold", boss="gold")


@unit("Hakkapeliitta")
def u_hakkapeliitta(p):
    ground(p, 40, 0)
    horse(p, col="darkhorse", mane="black", gait=3)
    ls, rs, hd = rider(p, coat="leather", legs="navy", boots="black")                      # buff coat
    p.limb(-6, -78, 12, -52, 4.4, OWNER)                                                    # sash
    p.curve("M14,-50 Q26,-40 34,-26", 2.4, "black")                                         # sword in its scabbard
    head_at(p, *hd, hat_kind="slouch", hat_col="black", plume=OWNER)
    arm(p, *ls, -24, -84, "leather")
    p.rect(-38, -88, 16, 4.4, "darkwood", rx=1.5)                                           # wheellock pistol, firing
    p.glow(-44, -86, 10, "flame", 0.6)
    p.circle(-44, -86, 3.4, "flame", ink=False)
    for k in range(3):
        p.circle(-52 - 6 * k, -90 - 3 * k, 4 + 1.6 * k, "white", alpha=0.7 - 0.18 * k, ink=False)
    arm(p, *rs, 16, -66, "leather")


@unit("Cossack")
def u_cossack(p):
    ground(p, 40, 0)
    horse(p, col="horse", mane="mane", gait=3)
    p.path("M-4,-80 L18,-80 L40,-44 Q22,-40 10,-48 Z", "black")                            # burka cloak
    ls, rs, hd = rider(p, coat=OWNER, legs="navy", boots="black")
    for k in range(4):                                                                     # gazyri
        p.rect(-4 + k * 3.2, -74, 2.2, 7, "white", ink=False)
    hair(Shift(p, hd[0], hd[1] + 89))
    beard(Shift(p, hd[0], hd[1] + 89))
    p.rect(hd[0] - 10, hd[1] - 24, 20, 17, "black", rx=4)                                  # papakha
    p.ellipse(hd[0], hd[1] - 23, 9, 2.6, OWNER)
    arm(p, *rs, -14, -102, OWNER)
    p.curve("M-14,-104 Q-30,-118 -50,-116", 2.6, "steel")                                  # shashka, swung forward
    p.limb(-12, -100, -16, -106, 2.2, "gold")
    arm(p, *ls, -20, -66, OWNER)


@unit("Hussar")
def u_hussar(p):
    ground(p, 40, 0)
    horse(p, col="greyhorse", mane="iron", gait=3)
    p.path("M10,-82 L26,-80 L32,-52 L20,-48 Z", OWNER)                                      # pelisse over the shoulder
    p.thin(26, -80, 32, -52, 2.6, "white")
    ls, rs, hd = rider(p, coat=OWNER, legs="navy", boots="black")
    for y in (-72, -67, -62, -57):                                                          # frogging
        p.thin(-5, y, 11, y, 1.4, "gold")
    q = Shift(p, hd[0], hd[1] + 89)                                                         # fur busby, bag and plume
    q.path("M-9.5,-93 L-10,-112 Q0,-115 10,-112 L9.5,-93 Z", SABLE)
    q.path("M8,-110 Q18,-106 16,-94 L11,-98 Z", OWNER)
    q.path("M-2,-112 Q-4,-126 4,-130 Q6,-122 2,-112 Z", "white")
    arm(p, *rs, -14, -100, OWNER)
    p.curve("M-14,-102 Q-28,-120 -48,-120", 2.6, "steel")                                    # sabre, forward
    p.limb(-12, -98, -16, -104, 2.2, "gold")
    arm(p, *ls, -20, -66, OWNER)


# ============================================================================ armour
@unit("Panzer", wide=True)
def u_panzer(p):
    ground(p, 52, 2)
    tracks(p, -48, 48, 0, 20, wheel_col="darkgrey", n=8)
    p.path("M-52,-18 L-46,-32 L48,-32 L52,-18 Z", DUNKELGELB)
    p.rect(-48, -28, 98, 12, DUNKELGELB, rx=1)                                              # side skirts
    for x, w in ((-38, 18), (-6, 14), (22, 20)):
        p.ellipse(x, -22, w / 2, 3.4, CAMO_GREEN, ink=False)
    for x in (-22, 10, 38):
        p.ellipse(x, -25, 5, 2.4, CAMO_BROWN, ink=False)
    p.path("M-26,-32 L-24,-54 L26,-54 L30,-32 Z", DUNKELGELB)                               # boxy turret
    p.ellipse(-6, -44, 9, 4, CAMO_GREEN, ink=False)
    p.ellipse(16, -40, 6, 3, CAMO_BROWN, ink=False)
    p.rect(-24, -48, 52, 5, OWNER)
    p.rect(8, -60, 12, 6, DUNKELGELB, rx=2)                                                  # commander's cupola
    p.rect(-72, -48, 48, 4.2, DUNKELGELB, rx=2)                                              # long gun
    p.rect(-78, -50, 8, 8, "darkgrey", rx=1.5)                                               # muzzle brake
    p.rect(-30, -50, 8, 8, DUNKELGELB, rx=1.5)


# ============================================================================ civilian
@unit("Khan", civilian=True)
def u_khan(p):
    ground(p)
    p.rect(20, -150, 3, 150, "darkwood", rx=1.2)                                            # the sulde, a spirit banner
    p.path("M21.5,-170 L24,-158 L19,-158 Z", "steel")
    p.limb(15, -160, 28, -160, 2.2, "steel")
    p.limb(15, -160, 15, -166, 2.0, "steel")
    p.limb(28, -160, 28, -166, 2.0, "steel")
    p.ellipse(21.5, -150, 9, 3, "steel")                                                    # horse-hair tassels
    for k in range(7):
        x = 14 + k * 2.5
        p.limb(x, -150, x + (k - 3) * 0.8, -124 + abs(k - 3) * 2, 2.2, "black", ink=k in (0, 6))
    figure(p, robe=OWNER, belt="gold", wide=1.05)                                           # deel robe
    p.path("M-3,-77 L9,-77 L-4,-46 L-10,-50 Z", "gold")                                     # robe's crossed front
    p.rect(-17, -40, 34, 4, "gold", ink=False)
    p.path("M-3,-81 L3,-81 L0,-73 Z", "black", ink=False)                                   # goatee
    p.path("M-7,-84 Q0,-86 7,-84 Q9,-80 10,-77 Q6,-81 0,-82 Q-6,-81 -10,-77 Q-9,-80 -7,-84 Z", "black", ink=False)
    p.path("M-12,-93 Q-12,-112 0,-118 Q12,-112 12,-93 Z", OWNER)                            # hat with a fur brim
    p.rect(-13, -97, 26, 7, SABLE, rx=3)
    p.circle(0, -119, 2.6, "gold")
    arm(p, -13, -72, -16, -50, OWNER)
    arm(p, 13, -72, 20, -96, OWNER)


def main():
    rt.FILTER = sys.argv[1:] or None
    for name in UNIQUE:
        RU.render_unit(name, RU.UNITS[name])
    print("Wrote %d unit images (%d unique units)" % (len(rt.WRITTEN), len(rt.WRITTEN) // 3))


if __name__ == "__main__":
    main()
