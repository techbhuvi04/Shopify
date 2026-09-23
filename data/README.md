# Data

This directory holds product catalogs. Only a **small synthetic demo catalog** is
committed; real marketplace datasets are large and licence-restricted and must
never be committed (see `.gitignore`).

## Bundled sample: `sample_products.csv` + `sample_images/`

| | |
|---|---|
| Products | 144 listings |
| Label groups | 50 (10 product types × 5 colours) |
| Listings per group | 2–4 |
| Images | 224×224 JPEG, ~1 MB total |
| Generator | `python scripts/generate_sample_data.py` (deterministic, seed 42) |

Every image is a drawn product illustration with random photographic variation:
background, scale, rotation, brightness, blur and a "SALE" badge. Every title is
a noisy seller-style rewrite, with promo phrases, casing changes, reordered words
and an optional model code. Groups of the same type from the same brand that
differ only by colour act as **hard negatives**.

> ⚠️ The sample is synthetic. Metrics computed on it show that the pipeline runs
> correctly. They are **not** an estimate of performance on real marketplace data.

## Expected CSV format

| column | required | description |
|---|---|---|
| `product_id` | yes | unique listing ID |
| `title` | yes | raw product title (may be empty; handled) |
| `image_path` | yes | image path, relative to `data.image_root` (may be missing; handled) |
| `category` | no | free-text category, used for display only |
| `label_group` | for evaluation / training | listings with the same value are the same product |

Example:

```csv
product_id,title,image_path,category,label_group
P0001,Nordwave Cotton Tee Red Size M,sample_images/P0001.jpg,Fashion,tshirt_red
P0002,New Nordwave NO-TS106 T Shirt - Red,sample_images/P0002.jpg,Fashion,tshirt_red
```

## Using your own catalog

1. Put images anywhere under the project, e.g. `data/raw/my_shop/images/`. The
   `data/raw/` folder is gitignored.
2. Write a CSV in the format above.
3. Point `configs/config.yaml` at it:

   ```yaml
   data:
     products_csv: data/raw/my_shop/products.csv
     image_root: data/raw/my_shop
   ```

4. Rebuild: `python scripts/build_embeddings.py && python scripts/build_index.py`.
5. If you have `label_group`, run `python scripts/evaluate.py`. Then re-tune
   `matching.mode_thresholds`: the bundled values were tuned on the sample catalog.

Without `label_group` you can still search and match. Evaluation and training
need it.

## Using the Shopee Price Match Guarantee data (optional)

This project is *inspired by* multimodal marketplace matching competitions such
as Shopee Price Match Guarantee. It does not reproduce any competition solution.
To try the pipeline on that data:

1. Accept the competition rules on Kaggle, then download (~2 GB):

   ```bash
   kaggle competitions download -c shopee-product-matching -p data/raw/shopee
   unzip data/raw/shopee/shopee-product-matching.zip -d data/raw/shopee
   ```

2. Convert `train.csv` to the expected format:

   ```python
   import pandas as pd
   df = pd.read_csv("data/raw/shopee/train.csv")
   out = pd.DataFrame({
       "product_id": df["posting_id"],
       "title": df["title"],
       "image_path": "train_images/" + df["image"],
       "category": "",
       "label_group": df["label_group"].astype(str),
   })
   out.to_csv("data/raw/shopee/products.csv", index=False)
   ```

3. Set `data.products_csv: data/raw/shopee/products.csv` and
   `data.image_root: data/raw/shopee`, then rebuild.

The competition data is subject to Kaggle's competition terms. Do not
redistribute it or commit it to this repository.
