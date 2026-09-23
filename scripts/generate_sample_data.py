"""Generate the small synthetic demo catalog in ``data/``.

The real Shopee-style datasets are large and licence-restricted, so the repo
ships a synthetic catalog that exercises the full pipeline end to end:

* 10 product types x 5 colours = 50 ``label_group`` s (one per type/colour),
* 2-4 listings per group, each with a differently rendered image (background,
  scale, rotation, lighting, promo badge) and a differently written title
  (casing, word order, promo noise, optional model code),
* deliberate hard negatives: groups of the same type from the same brand that
  differ only by colour, and groups of different types sharing a colour.

The data is synthetic and simple; metrics on it demonstrate that the pipeline
runs correctly and are NOT indicative of performance on real marketplace data.

Usage:
    python scripts/generate_sample_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter  # noqa: E402

from src.utils.config import PROJECT_ROOT  # noqa: E402

SIZE = 224

COLORS: dict[str, tuple[int, int, int]] = {
    "Red": (200, 40, 45),
    "Blue": (40, 90, 200),
    "Green": (40, 150, 70),
    "Black": (35, 35, 38),
    "Yellow": (235, 195, 40),
}

# type -> (category, [brand for first 3 colours, brand for last 2], noun variants, spec)
PRODUCT_TYPES: dict[str, dict] = {
    "tshirt": {"category": "Fashion", "brands": ["Nordwave", "UrbanKraft"],
               "nouns": ["Cotton T-Shirt", "Cotton Tee", "T Shirt"], "spec": "Size M"},
    "mug": {"category": "Home & Kitchen", "brands": ["Casa Bella", "BrewCo"],
            "nouns": ["Ceramic Coffee Mug", "Coffee Cup", "Mug"], "spec": "350ml"},
    "bottle": {"category": "Sports", "brands": ["HydroPeak", "AquaTrail"],
               "nouns": ["Stainless Steel Water Bottle", "Insulated Bottle", "Water Flask"],
               "spec": "750ml"},
    "sneaker": {"category": "Fashion", "brands": ["Stridex", "Kickline"],
                "nouns": ["Running Sneakers", "Sport Shoes", "Trainers"], "spec": "EU 42"},
    "phone": {"category": "Electronics", "brands": ["Novatel", "Pixelon"],
              "nouns": ["Smartphone", "Android Phone", "Mobile Phone"], "spec": "128GB"},
    "headphones": {"category": "Electronics", "brands": ["SonicArc", "Beatwave"],
                   "nouns": ["Wireless Headphones", "Bluetooth Headset", "Over-Ear Headphones"],
                   "spec": "40h Battery"},
    "backpack": {"category": "Bags", "brands": ["TrekPro", "Campus Gear"],
                 "nouns": ["Laptop Backpack", "Travel Rucksack", "School Bag"], "spec": "25L"},
    "watch": {"category": "Accessories", "brands": ["Chronex", "Timely"],
              "nouns": ["Analog Wrist Watch", "Quartz Watch", "Wristwatch"], "spec": "40mm"},
    "lamp": {"category": "Home & Kitchen", "brands": ["Lumina", "BrightNest"],
             "nouns": ["LED Desk Lamp", "Table Lamp", "Reading Lamp"], "spec": "Dimmable"},
    "cap": {"category": "Fashion", "brands": ["Nordwave", "Kickline"],
            "nouns": ["Baseball Cap", "Sports Cap", "Snapback Hat"], "spec": "Adjustable"},
}


# --------------------------------------------------------------------------- drawing
def _shade(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(int(np.clip(c * factor, 0, 255)) for c in color)  # type: ignore[return-value]


def draw_product(kind: str, color: tuple[int, int, int]) -> Image.Image:
    """Draw a simple product illustration on a transparent canvas."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    dark, light = _shade(color, 0.6), _shade(color, 1.35)
    ol = {"outline": dark, "width": 4}
    if kind == "tshirt":
        d.polygon([(62, 50), (95, 38), (129, 38), (162, 50), (198, 88), (170, 112), (158, 98),
                   (158, 190), (66, 190), (66, 98), (54, 112), (26, 88)], fill=color, **ol)
        d.arc((92, 26, 132, 60), 20, 160, fill=dark, width=4)
    elif kind == "mug":
        d.rounded_rectangle((50, 60, 150, 180), radius=12, fill=color, **ol)
        d.ellipse((140, 85, 195, 150), outline=color, width=14)
        d.rectangle((60, 70, 140, 80), fill=light)
    elif kind == "bottle":
        d.rounded_rectangle((80, 70, 144, 200), radius=18, fill=color, **ol)
        d.rectangle((96, 36, 128, 72), fill=(170, 170, 175), outline=(110, 110, 115), width=3)
        d.rectangle((80, 110, 144, 140), fill=light)
    elif kind == "sneaker":
        d.polygon([(30, 150), (40, 100), (90, 95), (120, 120), (190, 135), (198, 165), (30, 165)],
                  fill=color, **ol)
        d.rectangle((28, 165, 200, 180), fill=(240, 240, 240), outline=(150, 150, 150), width=3)
        for x in (70, 85, 100):
            d.line((x, 105, x + 10, 125), fill=light, width=4)
    elif kind == "phone":
        d.rounded_rectangle((70, 25, 154, 200), radius=14, fill=color, **ol)
        d.rounded_rectangle((78, 40, 146, 180), radius=6, fill=(25, 30, 45))
        d.ellipse((106, 185, 118, 197), fill=light)
    elif kind == "headphones":
        d.arc((45, 30, 179, 170), 180, 360, fill=color, width=16)
        d.rounded_rectangle((35, 100, 75, 175), radius=14, fill=color, **ol)
        d.rounded_rectangle((149, 100, 189, 175), radius=14, fill=color, **ol)
    elif kind == "backpack":
        d.rounded_rectangle((55, 45, 169, 200), radius=26, fill=color, **ol)
        d.rounded_rectangle((75, 120, 149, 185), radius=10, fill=light, outline=dark, width=3)
        d.arc((85, 20, 139, 70), 180, 360, fill=dark, width=6)
    elif kind == "watch":
        d.rectangle((92, 15, 132, 209), fill=color, outline=dark, width=3)
        d.ellipse((62, 62, 162, 162), fill=(230, 230, 232), outline=dark, width=8)
        d.line((112, 112, 112, 80), fill=(20, 20, 20), width=4)
        d.line((112, 112, 136, 120), fill=(20, 20, 20), width=4)
    elif kind == "lamp":
        d.polygon([(70, 40), (154, 40), (180, 110), (44, 110)], fill=color, **ol)
        d.rectangle((106, 110, 118, 185), fill=(120, 120, 125))
        d.ellipse((66, 180, 158, 205), fill=color, outline=dark, width=3)
    elif kind == "cap":
        d.chord((45, 55, 179, 175), 180, 360, fill=color, **ol)
        d.polygon([(150, 115), (205, 118), (200, 132), (140, 125)], fill=dark)
        d.ellipse((106, 50, 118, 62), fill=light)
    else:
        raise ValueError(f"unknown product type {kind}")
    return img


def render_listing(kind: str, color: tuple[int, int, int], rng: np.random.Generator) -> Image.Image:
    """Render one marketplace listing photo with random photographic variation."""
    backgrounds = [(255, 255, 255), (242, 242, 242), (245, 238, 225), (225, 235, 245), (250, 245, 250)]
    bg = backgrounds[rng.integers(len(backgrounds))]
    jitter = rng.uniform(0.9, 1.1)
    product = draw_product(kind, _shade(color, jitter))
    scale = rng.uniform(0.7, 1.0)
    size = int(SIZE * scale)
    product = product.resize((size, size), Image.LANCZOS).rotate(
        rng.uniform(-18, 18), resample=Image.BICUBIC, expand=False)
    canvas = Image.new("RGB", (SIZE, SIZE), bg)
    ox = int(rng.integers(0, SIZE - size + 1))
    oy = int(rng.integers(0, SIZE - size + 1))
    canvas.paste(product, (ox, oy), product)
    if rng.random() < 0.3:  # promo badge, common on marketplace photos
        d = ImageDraw.Draw(canvas)
        d.rectangle((6, 6, 70, 30), fill=(230, 30, 60))
        d.text((14, 12), "SALE", fill=(255, 255, 255))
    canvas = ImageEnhance.Brightness(canvas).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.3:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.2)))
    return canvas


# --------------------------------------------------------------------------- titles
def make_title(brand: str, color: str, noun: str, spec: str, code: str,
               rng: np.random.Generator) -> str:
    """Write one noisy seller title for a product."""
    templates = [
        "{brand} {noun} {color} {spec}",
        "{brand_l} {color_l} {noun_l} {spec} - original",
        "[PROMO] {noun} {brand} {color} {code} free shipping",
        "{color} {noun} by {brand} ({code})",
        "{BRAND} {NOUN} {COLOR} {spec} ready stock",
        "New {brand} {code} {noun} - {color}",
    ]
    t = templates[rng.integers(len(templates))]
    return t.format(brand=brand, noun=noun, color=color, spec=spec, code=code,
                    brand_l=brand.lower(), color_l=color.lower(), noun_l=noun.lower(),
                    BRAND=brand.upper(), NOUN=noun.upper(), COLOR=color.upper())


def generate(out_dir: Path, seed: int = 42) -> pd.DataFrame:
    """Generate images and the CSV catalog; return the catalog DataFrame."""
    rng = np.random.default_rng(seed)
    image_dir = out_dir / "sample_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    pid = 0
    for kind, spec in PRODUCT_TYPES.items():
        for ci, (color_name, rgb) in enumerate(COLORS.items()):
            brand = spec["brands"][0 if ci < 3 else 1]
            code = f"{brand[:2].upper()}-{kind[:2].upper()}{100 + 7 * ci + len(kind)}"
            group = f"{kind}_{color_name.lower()}"
            for _ in range(int(rng.integers(2, 5))):
                pid += 1
                product_id = f"P{pid:04d}"
                rel_path = f"sample_images/{product_id}.jpg"
                render_listing(kind, rgb, rng).save(out_dir / rel_path, quality=88)
                noun = spec["nouns"][rng.integers(len(spec["nouns"]))]
                rows.append({
                    "product_id": product_id,
                    "title": make_title(brand, color_name, noun, spec["spec"], code, rng),
                    "image_path": rel_path,
                    "category": spec["category"],
                    "label_group": group,
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sample_products.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    df = generate(args.out_dir, args.seed)
    print(f"Wrote {len(df)} products in {df['label_group'].nunique()} groups to {args.out_dir}")


if __name__ == "__main__":
    main()
