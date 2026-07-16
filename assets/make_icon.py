"""Generate the application icon (PNG + ICO).

Draws a rounded blue square with a white person-in-frame glyph, matching the
'Person Photo Extractor' theme. Run: python assets/make_icon.py
"""
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 512
ACCENT_TOP = (92, 157, 255)    # #5c9dff
ACCENT_BOT = (47, 116, 208)    # #2f74d0
WHITE = (255, 255, 255, 255)
FRAME = (255, 255, 255, 90)

OUT_DIR = Path(__file__).resolve().parent


def rounded_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return mask


def gradient(size: int) -> Image.Image:
    base = Image.new("RGB", (size, size), ACCENT_TOP)
    top, bot = ACCENT_TOP, ACCENT_BOT
    px = base.load()
    for y in range(size):
        t = y / (size - 1)
        r = int(top[0] + (bot[0] - top[0]) * t)
        g = int(top[1] + (bot[1] - top[1]) * t)
        b = int(top[2] + (bot[2] - top[2]) * t)
        for x in range(size):
            px[x, y] = (r, g, b)
    return base


def build() -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    grad = gradient(SIZE).convert("RGBA")
    img.paste(grad, (0, 0), rounded_mask(SIZE, radius=112))

    d = ImageDraw.Draw(img)

    # Photo-frame corner brackets (subtle).
    m, ln, th = 96, 74, 16
    for (cx, cy, dx, dy) in [
        (m, m, 1, 1), (SIZE - m, m, -1, 1),
        (m, SIZE - m, 1, -1), (SIZE - m, SIZE - m, -1, -1),
    ]:
        d.line([(cx, cy), (cx + dx * ln, cy)], fill=FRAME, width=th)
        d.line([(cx, cy), (cx, cy + dy * ln)], fill=FRAME, width=th)

    # Person glyph: head + shoulders, centered.
    cx = SIZE // 2
    head_r = 70
    head_cy = 212
    d.ellipse([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r],
              fill=WHITE)
    # Shoulders (rounded pill shape).
    sw, top_y, bot_y = 210, 300, 400
    d.rounded_rectangle([cx - sw, top_y, cx + sw, bot_y + 120],
                        radius=110, fill=WHITE)
    return img


def main() -> None:
    icon = build()
    png = OUT_DIR / "icon.png"
    icon.save(png)
    ico = OUT_DIR / "icon.ico"
    icon.save(ico, sizes=[(16, 16), (32, 32), (48, 48),
                          (64, 64), (128, 128), (256, 256)])
    print("wrote", png, "and", ico)


if __name__ == "__main__":
    main()
