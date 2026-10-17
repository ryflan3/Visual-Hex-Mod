#!/usr/bin/env python3
"""Draw the unit flags as heraldic banners hanging from a gold rod.

    python3 tools/render_flags.py

Unciv draws a unit's flag from images in UnitFlagIcons (UnitIconGroup.kt), and a visual mod's images
replace the game's own with the same name. For each state (UnitFlag for military units, and the
Civilian, Fortify and Embark versions) it draws, centred on the flag's spot:

* the shape, 30 px wide and keeping its proportions, tinted with the civilization's inner colour,
  so it shows as the border;
* the Inner shape over the same area, tinted with the civilization's outer colour;
* the Mask untinted at 88% of the width: here the cloth's folds and the gold rod;
* the unit's icon, 65% (military) or 50% (civilian) of the width, in the inner colour.

The Selection image is the glow drawn 1.6 times larger behind a selected unit's flag. The banners'
shapes tell the states apart: a swallowtail for military units, a point for civilians, crenellations
for a fortified unit and a wavy hem for an embarked one.
"""

import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Images", "UnitFlagIcons")
SS = 4                 # supersampling
W, H = 200, 250        # the banners are drawn at this size...
SIZE = (160, 200)      # ...and saved at this one; Unciv scales them to 30 px wide, so this is plenty
GLOW = (100, 125)      # the selection glow is soft anyway
CX, CY = 100, 118      # the centre the inner shape shrinks toward
GOLD, GOLD_LIGHT = (226, 180, 67, 255), (240, 200, 90, 255)


def mask(pts):
    img = Image.new("L", (W * SS, H * SS), 0)
    ImageDraw.Draw(img).polygon([(x * SS, y * SS) for x, y in pts], fill=255)
    return img


def white(cover):
    """A white image with the given coverage, ready to be tinted."""
    out = Image.new("RGBA", cover.size, (255, 255, 255, 0))
    out.putalpha(cover)
    return out


def down(img):
    return img.resize((W, H), Image.LANCZOS)


def banner_shape(state):
    """The banner's outline for a flag state."""
    top = 46 if state == "Fortify" else 26
    pts = [(18, top), (182, top)]
    if state in ("", "Fortify"):           # swallowtail
        pts += [(182, 220), (100, 186), (18, 220)]
    elif state == "Civilian":              # a point
        pts += [(182, 160), (100, 236), (18, 160)]
    else:                                  # Embark: a wavy hem
        pts += [(182, 214)] + [(182 - 164 * t, 214 + 12 * math.sin(t * math.pi * 3)) for t in np.linspace(0, 1, 30)] + [(18, 214)]
    if state == "Fortify":                 # crenellations along the top
        pts = [(18, 46), (18, 20), (46, 20), (46, 34), (72, 34), (72, 20), (128, 20), (128, 34), (154, 34), (154, 20),
               (182, 20), (182, 46)] + pts[2:]
    return pts


def cloth(pts):
    """The untinted overlay: folds of light and shade across the cloth, darker toward the hem, and the
    gold rod it hangs from. Unciv draws it at 88% of the flag's width, centred, so it is drawn at full
    size and the middle 88% is scaled up to fill the image."""
    a = np.asarray(mask(pts), np.float32) / 255
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]] / (SS * np.array([H, W]))[:, None, None]
    folds = 0.5 + 0.5 * np.sin(xx * math.pi * 5.0)
    light = np.clip(0.28 - yy * 0.4, 0, 1) * 0.5 + folds * 0.10
    dark = np.clip(yy - 0.45, 0, 1) * 0.35 + (1 - folds) * 0.10
    rgba = np.zeros(a.shape + (4,), np.float32)
    rgba[..., :3] = (light > dark)[..., None] * 1.0
    rgba[..., 3] = np.abs(light - dark) * a
    img = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")
    d = ImageDraw.Draw(img)
    top = min(y for _, y in pts) * SS
    d.rounded_rectangle((14 * SS, top - 9 * SS, 186 * SS, top + 3 * SS), radius=6 * SS, fill=GOLD)
    for x in (12, 174):
        d.ellipse((x * SS, top - 12 * SS, (x + 14) * SS, top + 4 * SS), fill=GOLD_LIGHT)
    w, h = img.size
    m = 0.06
    return img.crop((round(w * m), round(h * m), round(w * (1 - m)), round(h * (1 - m)))).resize((W, H), Image.LANCZOS)


def main():
    os.makedirs(OUT, exist_ok=True)
    written = 0
    for state in ("", "Civilian", "Fortify", "Embark"):
        pts = banner_shape(state)
        images = {
            "UnitFlag" + state: white(down(mask(pts))),
            "UnitFlag" + state + "Inner": white(down(mask([(CX + (x - CX) * 0.84, CY + (y - CY) * 0.84) for x, y in pts]))),
            "UnitFlagMask" + state: cloth(pts),
            "UnitFlagSelection" + state: white(down(mask(pts)).filter(ImageFilter.GaussianBlur(10))),
        }
        for name, img in images.items():
            img = img.resize(GLOW if "Selection" in name else SIZE, Image.LANCZOS)
            img.save(os.path.join(OUT, name + ".png"), optimize=True)
            written += 1
    print("Wrote %d flag images" % written)


if __name__ == "__main__":
    main()
