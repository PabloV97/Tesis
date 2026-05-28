"""LeNet5 Cats vs Dogs: entrenamiento, activaciones, complejidad y Spearman.

Uso desde Jupyter:
    %run train_catvsdog_lenet5_complexity.py --epochs 1,2 --local-data-dir "C:/ruta/dataset" --target-layer fc2

El dataset local puede tener una de estas estructuras:
    dataset/PetImages/Cat + dataset/PetImages/Dog
    dataset/cat.1.jpg + dataset/dog.1.jpg
    dataset/train/cats + dataset/train/dogs + dataset/validation/cats + dataset/validation/dogs
"""

from __future__ import annotations

import argparse
import re
import shutil
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model, layers
from tqdm.auto import tqdm

DATASET_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
LAYER_BASES = ["conv1", "pool1", "conv2", "pool2", "fc1", "fc2"]


def parse_epochs(text: str) -> list[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values or any(x <= 0 for x in values):
        raise argparse.ArgumentTypeError("Use epocas positivas, por ejemplo: 1,2,5")
    return values


def maybe_none_int(text: str) -> int | None:
    return None if text.lower() in {"none", "all", "todo"} else int(text)


def has_expected_structure(path: Path) -> bool:
    return all((path / split / cls).exists() for split in ("train", "validation") for cls in ("cats", "dogs"))


def first_dir(base: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = base / name
        if candidate.is_dir():
            return candidate
    return None


def find_petimages(path: Path) -> Path | None:
    candidates = [path, path / "PetImages"] + [p for p in path.glob("**/PetImages") if p.is_dir()]
    for candidate in candidates:
        if first_dir(candidate, ["Cat", "Cats", "cat", "cats"]) and first_dir(candidate, ["Dog", "Dogs", "dog", "dogs"]):
            return candidate
    return None


def is_decodable_image(path: Path) -> bool:
    """Valida imagenes ruidosas de PetImages antes de que TensorFlow arme batches."""
    try:
        raw = tf.io.read_file(str(path))
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        _ = image.shape
        return True
    except Exception:
        return False


def rebuild_petimages_split(petimages_dir: Path, out_dir: Path, seed: int = 1337) -> Path:
    cat_dir = first_dir(petimages_dir, ["Cat", "Cats", "cat", "cats"])
    dog_dir = first_dir(petimages_dir, ["Dog", "Dogs", "dog", "dogs"])
    if cat_dir is None or dog_dir is None:
        raise FileNotFoundError(f"No se encontro Cat/Dog dentro de {petimages_dir}")

    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    for split in ("train", "validation"):
        for cls in ("cats", "dogs"):
            (out_dir / split / cls).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    skipped = 0
    for src_dir, cls in ((cat_dir, "cats"), (dog_dir, "dogs")):
        files = []
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            files.extend(src_dir.glob(pattern))
        valid_files = []
        for src in files:
            if is_decodable_image(src):
                valid_files.append(src)
            else:
                skipped += 1
        valid_files = list(rng.permutation(valid_files))
        split_at = int(0.8 * len(valid_files))
        for i, src in enumerate(valid_files):
            split = "train" if i < split_at else "validation"
            dst_name = f"{cls[:-1]}.{src.stem}{src.suffix.lower()}"
            shutil.copy2(src, out_dir / split / cls / dst_name)
    if skipped:
        print(f"[WARN] Imagenes corruptas/invalidas omitidas: {skipped}")
    print(f"[OK] Dataset PetImages reorganizado en: {out_dir}")
    return out_dir


def rebuild_flat_split(files: list[Path], out_dir: Path, seed: int = 1337) -> Path:
    files = [p for p in files if p.name.lower().startswith(("cat", "dog")) and is_decodable_image(p)]
    if not files:
        raise FileNotFoundError("No se encontraron imagenes cat.* o dog.* validas")
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    for split in ("train", "validation"):
        for cls in ("cats", "dogs"):
            (out_dir / split / cls).mkdir(parents=True, exist_ok=True)
    files = list(np.random.default_rng(seed).permutation(files))
    split_at = int(0.8 * len(files))
    for i, src in enumerate(files):
        split = "train" if i < split_at else "validation"
        cls = "cats" if src.name.lower().startswith("cat") else "dogs"
        shutil.copy2(src, out_dir / split / cls / src.name)
    return out_dir


def resolve_dataset(data_dir: Path, local_data_dir: Path | None, local_zip: Path | None, dataset_url: str) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    if local_data_dir:
        local_data_dir = local_data_dir.resolve()
        if not local_data_dir.exists():
            raise FileNotFoundError(f"No existe la carpeta local: {local_data_dir}")
        if has_expected_structure(local_data_dir):
            return local_data_dir
        petimages = find_petimages(local_data_dir)
        if petimages:
            return rebuild_petimages_split(petimages, data_dir / "catsdogs_petimages")
        images = list(local_data_dir.glob("**/*.jpg")) + list(local_data_dir.glob("**/*.jpeg"))
        if images:
            return rebuild_flat_split(images, data_dir / "catsdogs_kaggle")
        raise FileNotFoundError("La carpeta local no tiene estructura reconocida para Cats vs Dogs.")

    base_dir = data_dir / "cats_and_dogs_filtered"
    if has_expected_structure(base_dir):
        return base_dir
    if local_zip:
        with zipfile.ZipFile(local_zip, "r") as zf:
            zf.extractall(data_dir)
    else:
        got = tf.keras.utils.get_file("cats_and_dogs_filtered.zip", origin=dataset_url, cache_dir=str(data_dir), cache_subdir=".", extract=True)
        print("[INFO] Descargado:", got)
    if has_expected_structure(base_dir):
        return base_dir
    images = list(data_dir.glob("**/*.jpg")) + list(data_dir.glob("**/*.jpeg"))
    if images:
        return rebuild_flat_split(images, data_dir / "catsdogs_kaggle")
    raise FileNotFoundError("No se pudo resolver el dataset. Use --local-data-dir o --local-zip.")


def load_dataset(base_dir: Path, img_size: int, batch: int, seed: int, augment: bool):
    train_raw = tf.keras.utils.image_dataset_from_directory(base_dir / "train", image_size=(img_size, img_size), batch_size=batch, label_mode="int", seed=seed, shuffle=True)
    val_raw = tf.keras.utils.image_dataset_from_directory(base_dir / "validation", image_size=(img_size, img_size), batch_size=batch, label_mode="int", seed=seed, shuffle=False)
    class_names = train_raw.class_names
    train_raw = train_raw.apply(tf.data.experimental.ignore_errors())
    val_raw = val_raw.apply(tf.data.experimental.ignore_errors())
    print("[INFO] Clases:", class_names)
    normalizer = layers.Rescaling(1.0 / 255)
    aug = tf.keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.05), layers.RandomZoom(0.10)])
    autotune = tf.data.AUTOTUNE
    if augment:
        train = train_raw.map(lambda x, y: (aug(normalizer(x), training=True), y), num_parallel_calls=autotune)
    else:
        train = train_raw.map(lambda x, y: (normalizer(x), y), num_parallel_calls=autotune)
    val = val_raw.map(lambda x, y: (normalizer(x), y), num_parallel_calls=autotune)
    train, val = train.prefetch(autotune), val.prefetch(autotune)
    xs, ys = [], []
    for xb, yb in val:
        xs.append(xb.numpy())
        ys.append(yb.numpy())
    return train, val, np.concatenate(xs), np.concatenate(ys)


def build_lenet5(img_size: int):
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="input")
    conv1 = layers.Conv2D(6, 5, activation="relu", padding="same", name="conv1")(inputs)
    pool1 = layers.AveragePooling2D(2, name="pool1")(conv1)
    conv2 = layers.Conv2D(16, 5, activation="relu", name="conv2")(pool1)
    pool2 = layers.AveragePooling2D(2, name="pool2")(conv2)
    flat = layers.Flatten(name="flatten")(pool2)
    fc1 = layers.Dense(120, activation="relu", name="fc1")(flat)
    drop = layers.Dropout(0.5, name="dropout")(fc1)
    fc2 = layers.Dense(84, activation="relu", name="fc2")(drop)
    logits = layers.Dense(2, activation="softmax", name="logits")(fc2)
    model = Model(inputs, logits, name="lenet5_catsdogs")
    tensors = {"inputs": inputs, "conv1": conv1, "pool1": pool1, "conv2": conv2, "pool2": pool2, "fc1": fc1, "fc2": fc2, "logits": logits}
    return model, tensors


def colnames_map(base: str, h: int, w: int, c: int) -> list[str]:
    return [f"{base}[{i},{j},c{k}]" for i in range(h) for j in range(w) for k in range(c)]


def activation_df(model: Model, tensors: dict, x_eval: np.ndarray, y_eval: np.ndarray, mode: str, batch: int) -> pd.DataFrame:
    outputs, names = [], []
    if mode == "gap":
        for name in ("conv1", "pool1", "conv2", "pool2"):
            outputs.append(layers.GlobalAveragePooling2D(name=f"gap_{name}")(tensors[name]))
            names.append(f"{name}_gap")
    else:
        for name in ("conv1", "pool1", "conv2", "pool2"):
            outputs.append(tensors[name])
            names.append(name)
    for name in ("fc1", "fc2"):
        outputs.append(tensors[name])
        names.append(name)

    probe = Model(tensors["inputs"], outputs)
    acts = probe.predict(x_eval, batch_size=batch, verbose=0)
    if isinstance(acts, np.ndarray):
        acts = [acts]
    blocks, cols = [], []
    for arr, name in zip(acts, names):
        if arr.ndim == 2:
            blocks.append(arr.astype(np.float32, copy=False))
            cols.extend([f"{name}[{i}]" for i in range(arr.shape[1])])
        elif arr.ndim == 4:
            _, h, w, c = arr.shape
            blocks.append(arr.reshape(arr.shape[0], h * w * c).astype(np.float32, copy=False))
            cols.extend(colnames_map(name, h, w, c))
        else:
            raise ValueError(f"Activacion no soportada: {name}, ndim={arr.ndim}")
    df = pd.DataFrame(np.concatenate(blocks, axis=1), columns=cols, dtype=np.float32)
    pred_proba = model.predict(x_eval, batch_size=batch, verbose=0)
    df.insert(0, "label", y_eval.astype(np.int64))
    df.insert(1, "pred", np.argmax(pred_proba, axis=1).astype(np.int64))
    for k in range(pred_proba.shape[1]):
        df.insert(2 + k, f"logits[{k}]", pred_proba[:, k].astype(np.float32))
    return df


def save_df(df: pd.DataFrame, path_no_ext: Path) -> Path:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    try:
        out = path_no_ext.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    except Exception as exc:
        print(f"[WARN] Parquet no disponible ({exc}). Usando CSV.")
        out = path_no_ext.with_suffix(".csv")
        df.to_csv(out, index=False)
    print(f"[OK] Guardado: {out} shape={df.shape}")
    return out


def feature_cols(df: pd.DataFrame) -> list[str]:
    prefixes = tuple([f"{b}[" for b in LAYER_BASES] + [f"{b}_gap[" for b in LAYER_BASES])
    return [c for c in df.columns if c.startswith(prefixes)]


def split_features(df_all: pd.DataFrame, out_dir: Path):
    cols = feature_cols(df_all)
    x_by_epoch, y_by_epoch = {}, {}
    for ep in sorted(df_all.epoch.unique()):
        mask = df_all.epoch == ep
        x_by_epoch[int(ep)] = df_all.loc[mask, cols].astype(np.float32).reset_index(drop=True)
        y_by_epoch[int(ep)] = df_all.loc[mask, "label"].astype(np.int64).to_numpy()
        save_df(x_by_epoch[int(ep)], out_dir / f"X_conv1_fc2_epoch{int(ep)}")
    return x_by_epoch, y_by_epoch


def cols_for_layer(xdf: pd.DataFrame, layer: str, gap_mode: str) -> list[str]:
    flat, gap = f"{layer}[", f"{layer}_gap["
    if gap_mode == "gap":
        return [c for c in xdf.columns if c.startswith(gap)]
    if gap_mode == "flat":
        return [c for c in xdf.columns if c.startswith(flat)]
    cols = [c for c in xdf.columns if c.startswith(flat)]
    return cols or [c for c in xdf.columns if c.startswith(gap)]


def clean_scale(x: np.ndarray, min_var: float, standardize: bool) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x[~np.isfinite(x)] = np.nan
    x = x[:, ~np.isnan(x).any(axis=0)]
    if x.size == 0:
        return x
    x = VarianceThreshold(threshold=min_var).fit_transform(x)
    return StandardScaler().fit_transform(x) if standardize and x.size else x


def metric_names(cc) -> list[str]:
    if hasattr(cc, "metrics"):
        obj = cc.metrics
        names = obj() if callable(obj) else obj
        names = [str(n).upper().strip() for n in list(names)]
        if names:
            return names
    vals = getattr(cc, "complexity", [])
    return [f"M{i + 1}" for i in range(len(list(vals)))]


def calculate_complexity(x_by_epoch, y_by_epoch, layer: str, gap_mode: str, n_samples: int | None, min_var: float, standardize: bool, out_path: Path):
    try:
        import problexity as px
    except ImportError as exc:
        raise ImportError("Instale problexity con: pip install problexity") from exc
    rows = []
    for ep in tqdm(sorted(x_by_epoch), desc=f"Complejidad {layer}"):
        xdf, y = x_by_epoch[ep], y_by_epoch[ep]
        cols = cols_for_layer(xdf, layer, gap_mode)
        meta = {"epoch": ep, "layer": layer, "mode": "gap" if cols and "_gap[" in cols[0] else "flat", "n_available": len(y)}
        if not cols:
            rows.append({**meta, "n_samples": None, "n_features": 0, "diagnostic": "no_columns"})
            continue
        x = xdf[cols].to_numpy(np.float32, copy=False)
        if n_samples and n_samples < len(y):
            _, idx = next(StratifiedShuffleSplit(n_splits=1, test_size=n_samples, random_state=42).split(x, y))
            x, y = x[idx], y[idx]
        if len(np.unique(y)) < 2:
            rows.append({**meta, "n_samples": len(y), "n_features": 0, "diagnostic": "one_class"})
            continue
        x = clean_scale(x, min_var, standardize)
        if x.size == 0:
            rows.append({**meta, "n_samples": len(y), "n_features": 0, "diagnostic": "zero_features"})
            continue
        try:
            with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                warnings.filterwarnings("ignore")
                cc = px.ComplexityCalculator(); cc.fit(x, y)
            names, values = metric_names(cc), np.asarray(cc.complexity, dtype=float)
            row = {**meta, "n_samples": len(y), "n_features": int(x.shape[1])}
            for name, value in zip(names, values):
                row[name] = value
            if hasattr(cc, "score"):
                try: row["score"] = float(cc.score())
                except Exception: pass
            row["diagnostic"] = "ok"
        except Exception as exc:
            row = {**meta, "n_samples": len(y), "n_features": int(x.shape[1]), "diagnostic": f"fit_error:{type(exc).__name__}:{exc}"}
        rows.append(row)
    meta_cols = ["epoch", "layer", "mode", "n_available", "n_samples", "n_features"]
    metric_cols = []
    for row in rows:
        for key in row:
            if key not in meta_cols and key not in {"score", "diagnostic"} and key not in metric_cols:
                metric_cols.append(key)
    out = pd.DataFrame(rows).reindex(columns=meta_cols + metric_cols + ["score", "diagnostic"]).sort_values("epoch")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("[OK] Guardado:", out_path)
    return out


def readable_measure(value) -> str:
    text = str(value)
    match = re.search(r"FUNCTION\s+([A-Za-z0-9_]+)\s+AT", text, re.I)
    return match.group(1) if match else text


def calculate_spearman(complexity: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path, prefix: str):
    joined = complexity.merge(metrics[["epoch", "score", "loss"]], on="epoch", how="inner", suffixes=("_px", "_mdl"))
    joined = joined.rename(columns={"score_mdl": "model_score", "loss": "model_loss"})
    meta = {"epoch", "layer", "mode", "n_available", "n_samples", "n_features", "diagnostic", "score"}
    measures = [c for c in complexity.columns if c not in meta]
    rows = []
    for measure in measures:
        x = joined[measure].to_numpy(float)
        for target in ("model_score", "model_loss"):
            y = joined[target].to_numpy(float)
            mask = np.isfinite(x) & np.isfinite(y)
            rho, p = spearmanr(x[mask], y[mask]) if mask.sum() >= 3 else (np.nan, np.nan)
            rows.append({"measure": measure, "target": target, "spearman": rho, "pvalue": p, "n": int(mask.sum())})
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows).sort_values(["target", "spearman"], ascending=[True, False])
    detail.to_csv(out_dir / f"spearman_{prefix}_vs_model.csv", index=False)
    detail["measure_name"] = detail.measure.map(readable_measure)
    wide = detail.pivot_table(index="target", columns="measure_name", values="spearman", aggfunc="first").sort_index(axis=1)
    wide.to_csv(out_dir / f"spearman_wide_{prefix}.csv")
    return detail, wide


def metrics_summary(df_all: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    cols = [c for c in ["train_loss", "train_acc", "val_loss", "val_acc"] if c in df_all.columns]
    grouped = df_all.groupby("epoch", as_index=False)[cols].first()
    out = pd.DataFrame({
        "epoch": grouped.epoch,
        "score": grouped["val_acc" if "val_acc" in grouped else "train_acc"],
        "loss": grouped["val_loss" if "val_loss" in grouped else "train_loss"],
        "train_loss_last": grouped.get("train_loss", np.nan),
        "train_acc_last": grouped.get("train_acc", np.nan),
        "val_loss_last": grouped.get("val_loss", np.nan),
        "val_acc_last": grouped.get("val_acc", np.nan),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("[OK] Guardado:", out_path)
    return out


def train_collect(args) -> pd.DataFrame:
    tf.keras.utils.set_random_seed(args.seed)
    data_base = resolve_dataset(args.data_dir.resolve(), args.local_data_dir, args.local_zip, args.dataset_url)
    train_ds, val_ds, x_eval, y_eval = load_dataset(data_base, args.img_size, args.batch_size, args.seed, not args.no_augment)
    model, tensors = build_lenet5(args.img_size)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    current_epoch, by_epoch = 0, {}
    for ep in args.epochs:
        print(f"[INFO] Entrenando de epoch {current_epoch} a {ep}")
        hist = model.fit(train_ds, validation_data=val_ds, initial_epoch=current_epoch, epochs=ep, verbose=args.verbose)
        current_epoch = ep
        df = activation_df(model, tensors, x_eval, y_eval, args.mode, args.predict_batch_size)
        df.insert(0, "epoch", ep); df.insert(1, "seed", args.seed)
        df["train_loss"] = np.float32(hist.history["loss"][-1]); df["train_acc"] = np.float32(hist.history["accuracy"][-1])
        df["val_loss"] = np.float32(hist.history["val_loss"][-1]); df["val_acc"] = np.float32(hist.history["val_accuracy"][-1])
        save_df(df, args.out_dir / "activations_by_epoch" / f"activations_epoch{ep}_{args.mode}")
        by_epoch[ep] = df
    df_all = pd.concat([by_epoch[ep] for ep in sorted(by_epoch)], ignore_index=True)
    save_df(df_all, args.out_dir / "activations_by_epoch" / f"activations_ALL_{args.mode}")
    return df_all


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LeNet5 Cats vs Dogs con activaciones, complejidad y Spearman")
    p.add_argument("--epochs", type=parse_epochs, default=parse_epochs("10,20,30,40,50"))
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--predict-batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--mode", choices=["flat", "gap"], default="flat")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--local-data-dir", type=Path, default=None)
    p.add_argument("--local-zip", type=Path, default=None)
    p.add_argument("--dataset-url", default=DATASET_URL)
    p.add_argument("--out-dir", type=Path, default=Path("catvsdog_lenet5_outputs"))
    p.add_argument("--target-layer", choices=LAYER_BASES, default="fc2")
    p.add_argument("--gap-mode", choices=["auto", "flat", "gap"], default="auto")
    p.add_argument("--n-samples", type=maybe_none_int, default=1000)
    p.add_argument("--min-var", type=float, default=0.0)
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--skip-complexity", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    args.out_dir = args.out_dir.resolve(); args.data_dir = args.data_dir.resolve()
    if args.local_data_dir: args.local_data_dir = args.local_data_dir.resolve()
    if args.local_zip: args.local_zip = args.local_zip.resolve()
    print("[CONFIG]", vars(args))
    df_all = train_collect(args)
    metrics = metrics_summary(df_all, args.out_dir / "results" / "metrics_summary.csv")
    x_by_epoch, y_by_epoch = split_features(df_all, args.out_dir / "datasets_conv1_fc2")
    if args.skip_complexity:
        return
    prefix = f"complexity_auto_{args.target_layer}_{args.mode if args.gap_mode == 'auto' else args.gap_mode}_n{args.n_samples or 'ALL'}"
    complexity = calculate_complexity(x_by_epoch, y_by_epoch, args.target_layer, args.gap_mode, args.n_samples, args.min_var, not args.no_standardize, args.out_dir / "results" / "complexity" / f"{prefix}.csv")
    calculate_spearman(complexity, metrics, args.out_dir / "results" / "correlations", prefix)
    print("[OK] Pipeline completo. Salida:", args.out_dir)


if __name__ == "__main__":
    main()
