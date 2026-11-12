"""Feature matrices for fitting: sparse when the table is mostly zeros.

A genus table is about 99% zeros (16S marine: 0.76% non-zero, ~430 of
56,549 genera per sample). Held dense as float32 it is 13.7 GB for 60k
samples, and every parallel CV worker cut its own fold copies out of
it, which is how a single CV job needed 140 GB. As a CSR matrix the
same data is about 0.2 GB.

`feature_matrix` reads the feature table in row slices, keeps only the
requested samples, and stores each slice as CSR when it is mostly zero,
so neither the full-width dense table nor a dense copy of the matrix is
ever held. A table that is not mostly zero (CLR, for instance, has no
zeros at all) is kept dense: CSR stores an index beside every value and
would be larger.

Every model the pipeline trains takes CSR: random forests (sklearn's
sparse splitter only sorts the non-zero values), XGBoost, gradient
boosting, and the SVR behind its MaxAbsScaler. One difference to know:
XGBoost treats an entry absent from a sparse matrix as *missing* rather
than as 0.0, and learns which way missing values go at each split. For
abundances, where absent and zero mean the same, that is the usual way
to feed it count data, but its trees are not identical to a dense fit.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Union, cast

import numpy as np
import polars as pl
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# Above this fraction of non-zeros a CSR matrix is no smaller than dense
# float32 (it stores a 4-byte column index beside each 4-byte value).
SPARSE_MAX_DENSITY = 0.3
# Rows read per slice. One slice is held dense while it is converted:
# 8192 x 56,549 genera x 4 bytes is about 1.9 GB.
BATCH_ROWS = 8192

Matrix = Union[np.ndarray, sp.csr_matrix]


def sparse_mode(value: Union[str, bool, None]) -> str:
    """`cv.sparse`: auto (default) | true | false -> auto | sparse | dense."""
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "sparse" if value else "dense"
    text = str(value).strip().lower()
    if text in ("auto", ""):
        return "auto"
    if text in ("true", "yes", "1", "sparse", "csr"):
        return "sparse"
    if text in ("false", "no", "0", "dense"):
        return "dense"
    raise ValueError(f"cv.sparse must be auto, true or false, got {value!r}")


def feature_matrix(
    features: pl.LazyFrame,
    samples: Sequence[str],
    feature_cols: Sequence[str],
    fillna: float = 0.0,
    sparse: Union[str, bool, None] = "auto",
    batch_rows: int = BATCH_ROWS,
) -> Matrix:
    """The rows of *features* for *samples*, in that order, as float32.

    Args:
        features: Lazy feature table with a `sample` column.
        samples: Output row order. A sample missing from the table gets a
            row of `fillna`, as the left join this replaces did.
        feature_cols: Columns to use, in this order.
        fillna: Value for nulls and for missing samples.
        sparse: "auto" picks CSR when the first slice read is at most
            `SPARSE_MAX_DENSITY` non-zero; True/False force it.
        batch_rows: Rows of the feature table read per slice.

    Returns:
        A CSR matrix or a dense array, shape (len(samples), len(cols)).
    """
    mode = sparse_mode(sparse)
    if mode == "sparse" and fillna != 0:
        raise ValueError("A sparse matrix needs fillna = 0")
    cols = list(feature_cols)
    order = [str(s) for s in samples]
    if not order:
        return np.zeros((0, len(cols)), dtype=np.float32)
    wanted = pl.Series("sample", list(dict.fromkeys(order)), dtype=pl.Utf8)
    total = int(features.select(pl.len()).collect().item())

    blocks: List[Matrix] = []
    row_of: Dict[str, int] = {}
    stacked = 0
    for offset in range(0, total, max(1, int(batch_rows))):
        chunk = (
            features.slice(offset, batch_rows)
            .filter(pl.col("sample").cast(pl.Utf8).is_in(wanted))
            .select(
                pl.col("sample").cast(pl.Utf8),
                pl.col(cols).fill_null(fillna).cast(pl.Float32, strict=False),
            )
            .collect()
        )
        if chunk.height == 0:
            continue
        names = chunk["sample"].to_list()
        keep = [i for i, s in enumerate(names) if s not in row_of]
        if len(keep) < len(names):
            logger.warning(
                "%d duplicated sample row(s) in the feature table; "
                "the first of each is used",
                len(names) - len(keep),
            )
        values = chunk.select(cols).to_numpy()
        if len(keep) < len(names):
            values = values[keep]
            names = [names[i] for i in keep]
        if mode == "auto":
            density = np.count_nonzero(values) / max(1, values.size)
            mode = (
                "sparse"
                if density <= SPARSE_MAX_DENSITY and fillna == 0
                else "dense"
            )
            logger.info(
                "Feature matrix: %.2f%% non-zero in the first slice -> %s",
                100 * density,
                mode,
            )
        blocks.append(
            sp.csr_matrix(values, dtype=np.float32)
            if mode == "sparse"
            else np.ascontiguousarray(values, dtype=np.float32)
        )
        for name in names:
            row_of[name] = stacked
            stacked += 1
        del values, chunk

    if mode == "auto":  # nothing read at all
        mode = "dense" if fillna != 0 else "sparse"
    missing = [s for s in dict.fromkeys(order) if s not in row_of]
    if missing:
        logger.warning(
            "%d sample(s) have no feature row; filled with %s",
            len(missing),
            fillna,
        )
        fill = np.full((1, len(cols)), fillna, dtype=np.float32)
        blocks.append(sp.csr_matrix(fill) if mode == "sparse" else fill)
        for name in missing:
            row_of[name] = stacked
        stacked += 1

    index = np.fromiter((row_of[s] for s in order), dtype=np.int64)
    if mode == "sparse":
        stacked_matrix = sp.vstack(blocks, format="csr", dtype=np.float32)
        csr: Any = sp.csr_matrix(stacked_matrix[index])
        csr.sort_indices()
        return cast(Matrix, csr)
    dense = blocks[0] if len(blocks) == 1 else np.vstack(blocks)
    if len(index) == dense.shape[0] and np.array_equal(
        index, np.arange(len(index))
    ):
        return np.asarray(dense, dtype=np.float32)
    return np.asarray(dense[index], dtype=np.float32)
