"""Render the Sign2Text vector-style mark as a macOS app icon."""

from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFilter


ICONSET = Path(sys.argv[1])
ICONSET.mkdir(parents=True, exist_ok=True)
SIZE = 1024

base = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
mask = Image.new("L", (SIZE, SIZE), 0)
ImageDraw.Draw(mask).rounded_rectangle((42, 42, 982, 982), radius=220, fill=255)

gradient = Image.new("RGBA", (SIZE, SIZE))
pixels = gradient.load()
for y in range(SIZE):
    for x in range(SIZE):
        t = (x + y) / (2 * (SIZE - 1))
        pixels[x, y] = (
            round(31 - 17 * t),
            round(105 - 58 * t),
            round(255 - 43 * t),
            255,
        )
base.paste(gradient, (0, 0), mask)

glow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
ImageDraw.Draw(glow).ellipse((-200, -310, 780, 670), fill=(99, 162, 255, 65))
glow = glow.filter(ImageFilter.GaussianBlur(115))
base = Image.alpha_composite(base, glow)
base.putalpha(mask)

draw = ImageDraw.Draw(base)
white = (255, 255, 255, 255)
# A single, deliberately simple S: legible even as a small Dock icon.
points = [(678, 312), (366, 312), (366, 512), (658, 512), (658, 712), (346, 712)]
draw.line(points, fill=white, width=112, joint="curve")
for x, y in (points[0], points[-1]):
    draw.ellipse((x - 56, y - 56, x + 56, y + 56), fill=white)

sizes = {
    "icon_16x16.png": 16, "icon_16x16@2x.png": 32,
    "icon_32x32.png": 32, "icon_32x32@2x.png": 64,
    "icon_128x128.png": 128, "icon_128x128@2x.png": 256,
    "icon_256x256.png": 256, "icon_256x256@2x.png": 512,
    "icon_512x512.png": 512, "icon_512x512@2x.png": 1024,
}
for name, size in sizes.items():
    base.resize((size, size), Image.Resampling.LANCZOS).save(ICONSET / name)
