"""Spatial holdout: grid cells by distance, a haversine buffer, auto sizes.

A grouped holdout (bioproject) keeps studies apart but not places: in the
18S_marine bioproject holdout, 24% of test samples had a training sample
within 10 km and 59% within 50 km. Labels such as temperature and pH are
spatially autocorrelated, so a model can score on the test set by
recognising a location. This holdout separates them in space:

1. Every sample gets a grid cell about `cell_km` across. Cells have equal
   height (latitude bands of `cell_km`) and close to equal width (each
   band has as many columns as `cell_km` fits around that latitude), so a
   cell is the same size near the pole as at the equator -- unlike a
   degree grid or Web Mercator (EPSG:3857) metres, which shrink with
   latitude.
2. Whole cells go to test, picked at random (seeded) until the test set
   holds about `test_size` of the samples. The label is not looked at:
   the point is that test is somewhere else, not that it looks like train.
3. Training samples closer than `buffer_km` to any test sample are taken
   out of training (split = "buffer"), so no training sample sits next to
   a test sample across a cell edge. Test samples are never removed, so
   the test set stays the `test_size` asked for; the training set shrinks.

All distances are great-circle (haversine) distances, found with
sklearn's BallTree on unique sites -- a sample set of 300k with 40k sites
is one tree and one query, not a distance matrix.

"auto" sizes. Either distance may be "auto" (the default). Auto takes the
label's empirical variogram over unique sites: the semivariance of label
differences, binned by distance. Its sill is the label's variance, which
is what a pair of unrelated sites gives. The auto range is the shortest
distance at which the semivariance reaches `auto_fraction` of the sill
(two bins in a row, so one noisy bin does not decide it): beyond it two
sites are no more alike than two picked at random. Clipped to
[auto_min_km, auto_max_km].

  cell_km: auto    -> the range: test cells are at least as wide as the
                      distance over which the label is correlated.
  buffer_km: auto  -> the range too, unless that would remove more than
                      `max_buffer_loss` of the training samples; it is
                      then shortened to the distance that removes exactly
                      that share. A number is used as given.

The range comes from the label, not from model residuals, so it is an
upper bound on what a model can exploit: conservative, by design. It uses
every labelled sample, test included, to set the size of the split -- no
model sees it.

CLUSTERS INSTEAD OF CELLS (token `spatial_cluster`). Grid lines cut
through dense patches of sites, and the buffer then deletes every training
site near the cut. Clusters put the boundaries in the gaps instead, and
delete nothing:

1. Sites closer than `gap_km` are joined, and joined sites form a cluster
   (single linkage, what DBSCAN gives with min_samples=1). Any two
   clusters are then more than `gap_km` apart. Exact: a minimum spanning
   tree of the sites with every edge longer than the gap cut, built once
   in O(n) memory.
2. Whole clusters go to test, picked at random (seeded, the label not
   looked at) until test holds about `test_size` of the samples. A
   cluster with more than `max_cluster_share` of the samples (a dense
   region chained into one, or one heavily sampled station) is never
   picked: it stays in training whole, since cutting it would put close
   neighbours on both sides.
3. No buffer. A training sample near a test sample would be in the same
   cluster, so it is in test too; every training sample is more than
   `gap_km` from every test sample, and no sample is deleted.

  gap_km: auto -> the largest of 25, 50, 100, 200 and 400 km at which no
                  more than `max_oversize_share` of the samples sit in
                  clusters too big to go to test.

Every step is deterministic: same data, config and seed, same split.

Config (all optional), under split.holdout:

    spatial:
      cell_km: auto          # grid: km, or auto
      buffer_km: auto        # grid: km (0 = no buffer), or auto
      max_buffer_loss: 0.25  # auto buffer removes at most this share of train
      auto_min_km: 10
      auto_max_km: 500
      auto_fraction: 0.95
      gap_km: auto           # cluster: km, or auto
      max_cluster_share: 0.05
      max_oversize_share: 0.5
      lat_column: lat
      lon_column: lon

Selected with the holdout tokens `spatial` (grid) and `spatial_cluster`
(sweep.holdout_groupings).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


EARTH_RADIUS_KM = 6371.0088
KM_PER_DEGREE = np.pi * EARTH_RADIUS_KM / 180.0  # about 111.195
HALF_CIRCUMFERENCE_KM = np.pi * EARTH_RADIUS_KM

AUTO = "auto"
SPLIT_BUFFER = "buffer"
CELL_COLUMN = "spatial_cell"
CLUSTER_COLUMN = "spatial_cluster"
NEAREST_COLUMN = "nearest_test_km"
METHODS = ("grid", "cluster")
_GROUP_COLUMN = {"grid": CELL_COLUMN, "cluster": CLUSTER_COLUMN}

# gap_km: auto picks the largest of these.
_GAP_CANDIDATES_KM = (25.0, 50.0, 100.0, 200.0, 400.0)

# Variogram bins: log-spaced from 100 m to half the globe. A bin needs
# this many pairs to be reported.
_BIN_EDGES = np.geomspace(0.1, HALF_CIRCUMFERENCE_KM, 41)
_MIN_PAIRS = 30
# Coordinates rounded to this many decimals (~1 m) define a site.
_SITE_DECIMALS = 5

VARIOGRAM_SCHEMA = {
    "label": pl.Utf8,
    "distance_km": pl.Float64,
    "semivariance": pl.Float64,
    "relative": pl.Float64,
    "n_pairs": pl.Int64,
}


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
def _distance(value: Any, name: str, allow_zero: bool) -> Optional[float]:
    """A distance in km, or None for auto."""
    if value is None or (
        isinstance(value, str) and value.strip().lower() == AUTO
    ):
        return None
    try:
        km = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"split.holdout.spatial.{name} must be a distance in km or "
            f"'auto', got {value!r}"
        ) from None
    if not np.isfinite(km) or km < 0 or (km == 0 and not allow_zero):
        raise ValueError(
            f"split.holdout.spatial.{name} must be "
            f"{'>= 0' if allow_zero else '> 0'} km, got {value!r}"
        )
    return km


@dataclass(frozen=True)
class SpatialSettings:
    """How a spatial holdout is built; None for a distance means auto."""

    cell_km: Optional[float] = None
    buffer_km: Optional[float] = None
    max_buffer_loss: float = 0.25
    auto_min_km: float = 10.0
    auto_max_km: float = 500.0
    auto_fraction: float = 0.95
    gap_km: Optional[float] = None
    max_cluster_share: float = 0.05
    max_oversize_share: float = 0.5
    lat_column: str = "lat"
    lon_column: str = "lon"
    # Variogram sampling: every site is a candidate, these bound the work.
    variogram_sites: int = 5000
    variogram_neighbours: int = 50
    variogram_random_pairs: int = 200_000

    @classmethod
    def from_config(cls, cfg: Optional[Mapping[str, Any]]) -> "SpatialSettings":
        cfg = dict(cfg or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(cfg) - known)
        if unknown:
            raise ValueError(
                f"Unknown split.holdout.spatial setting(s) {unknown}; "
                f"known: {sorted(known)}"
            )
        settings = cls(
            cell_km=_distance(cfg.get("cell_km", AUTO), "cell_km", False),
            buffer_km=_distance(cfg.get("buffer_km", AUTO), "buffer_km", True),
            max_buffer_loss=float(cfg.get("max_buffer_loss", 0.25)),
            auto_min_km=float(cfg.get("auto_min_km", 10.0)),
            auto_max_km=float(cfg.get("auto_max_km", 500.0)),
            auto_fraction=float(cfg.get("auto_fraction", 0.95)),
            gap_km=_distance(cfg.get("gap_km", AUTO), "gap_km", False),
            max_cluster_share=float(cfg.get("max_cluster_share", 0.05)),
            max_oversize_share=float(cfg.get("max_oversize_share", 0.5)),
            lat_column=str(cfg.get("lat_column", "lat")),
            lon_column=str(cfg.get("lon_column", "lon")),
            variogram_sites=int(cfg.get("variogram_sites", 5000)),
            variogram_neighbours=int(cfg.get("variogram_neighbours", 50)),
            variogram_random_pairs=int(
                cfg.get("variogram_random_pairs", 200_000)
            ),
        )
        if not 0 <= settings.max_buffer_loss <= 1:
            raise ValueError(
                "split.holdout.spatial.max_buffer_loss is a fraction from 0 "
                f"to 1, got {settings.max_buffer_loss}"
            )
        if not 0 < settings.auto_min_km <= settings.auto_max_km:
            raise ValueError(
                "split.holdout.spatial needs 0 < auto_min_km <= auto_max_km, "
                f"got {settings.auto_min_km} and {settings.auto_max_km}"
            )
        if not settings.auto_fraction > 0:
            raise ValueError(
                "split.holdout.spatial.auto_fraction must be > 0, got "
                f"{settings.auto_fraction}"
            )
        for name in ("max_cluster_share", "max_oversize_share"):
            value = getattr(settings, name)
            if not 0 < value <= 1:
                raise ValueError(
                    f"split.holdout.spatial.{name} is a fraction above 0 "
                    f"and at most 1, got {value}"
                )
        return settings


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def coordinates(
    metadata: pl.DataFrame, lat_column: str = "lat", lon_column: str = "lon"
) -> pl.DataFrame:
    """{sample, lat, lon} for every sample with valid coordinates."""
    missing = [c for c in (lat_column, lon_column) if c not in metadata.columns]
    if missing:
        raise ValueError(
            f"A spatial holdout needs coordinates; metadata has no {missing} "
            "(set split.holdout.spatial.lat_column / lon_column)"
        )
    return (
        metadata.select(
            pl.col("sample").cast(pl.Utf8),
            pl.col(lat_column).cast(pl.Float64, strict=False).alias("lat"),
            pl.col(lon_column).cast(pl.Float64, strict=False).alias("lon"),
        )
        .filter(
            pl.col("lat").is_between(-90.0, 90.0)
            & pl.col("lon").is_between(-180.0, 180.0)
        )
        .unique(subset="sample", keep="first", maintain_order=True)
    )


def _floats(values: ArrayLike) -> np.ndarray:
    out: np.ndarray = np.asarray(values, dtype=np.float64)
    return out


def haversine_km(
    lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike
) -> np.ndarray:
    """Great-circle distance in km, elementwise (broadcasts)."""
    p1: np.ndarray = np.radians(_floats(lat1))
    p2: np.ndarray = np.radians(_floats(lat2))
    dlam: np.ndarray = np.radians(_floats(lon2)) - np.radians(_floats(lon1))
    a: np.ndarray = (
        np.sin((p2 - p1) / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    )
    out: np.ndarray = (
        2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    )
    return out


def cell_ids(lat: ArrayLike, lon: ArrayLike, cell_km: float) -> List[str]:
    """Grid cell of each point: `<band>_<column>`, about `cell_km` across.

    Bands are `cell_km` of latitude tall. A band holds as many columns as
    `cell_km` fits around the Earth at its central latitude, so cells keep
    their width towards the poles rather than narrowing.
    """
    lat_a, lon_a = _floats(lat), _floats(lon)
    if lat_a.size == 0:
        return []
    dlat = cell_km / KM_PER_DEGREE
    n_bands = max(1, int(np.ceil(180.0 / dlat)))
    band: np.ndarray = np.clip(
        np.floor((lat_a + 90.0) / dlat), 0, n_bands - 1
    ).astype(np.int64)
    centre: np.ndarray = np.clip(-90.0 + (band + 0.5) * dlat, -90.0, 90.0)
    around_km: np.ndarray = (
        2 * np.pi * EARTH_RADIUS_KM * np.cos(np.radians(centre))
    )
    n_cols: np.ndarray = np.maximum(1, np.floor(around_km / cell_km)).astype(
        np.int64
    )
    col: np.ndarray = (
        np.floor((lon_a + 180.0) / 360.0 * n_cols).astype(np.int64) % n_cols
    )
    return [f"{b}_{c}" for b, c in zip(band.tolist(), col.tolist())]


def cell_bounds(
    cell: str, cell_km: float
) -> Tuple[float, float, float, float]:
    """(lon_min, lat_min, lon_max, lat_max) of a cell from `cell_ids`."""
    band_text, col_text = cell.split("_", 1)
    band, col = int(band_text), int(col_text)
    dlat = cell_km / KM_PER_DEGREE
    lat0 = -90.0 + band * dlat
    centre = float(np.clip(lat0 + dlat / 2, -90.0, 90.0))
    around_km = 2 * np.pi * EARTH_RADIUS_KM * np.cos(np.radians(centre))
    n_cols = max(1, int(np.floor(around_km / cell_km)))
    width = 360.0 / n_cols
    lon0 = -180.0 + col * width
    return lon0, lat0, lon0 + width, min(90.0, lat0 + dlat)


def _sites(lat: ArrayLike, lon: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Unique (lat, lon) sites, and each point's index into them."""
    points: np.ndarray = np.column_stack(
        [
            np.round(_floats(lat), _SITE_DECIMALS),
            np.round(_floats(lon), _SITE_DECIMALS),
        ]
    )
    sites, inverse = np.unique(points, axis=0, return_inverse=True)
    index: np.ndarray = np.asarray(inverse, dtype=np.intp).reshape(-1)
    return _floats(sites), index


def _tree(sites: np.ndarray) -> Any:
    from sklearn.neighbors import BallTree

    return BallTree(np.radians(sites), metric="haversine")


def nearest_km(
    query_lat: ArrayLike,
    query_lon: ArrayLike,
    ref_lat: ArrayLike,
    ref_lon: ArrayLike,
) -> np.ndarray:
    """Distance (km) from each query point to its nearest reference point.

    One BallTree over the unique reference sites, one query per unique
    query site. Infinite when there are no reference points.
    """
    q_lat = _floats(query_lat)
    r_lat = _floats(ref_lat)
    out: np.ndarray
    if q_lat.size == 0:
        out = np.empty(0, dtype=np.float64)
        return out
    if r_lat.size == 0:
        out = np.full(q_lat.size, np.inf, dtype=np.float64)
        return out
    ref_sites, _ = _sites(r_lat, ref_lon)
    q_sites, inverse = _sites(q_lat, query_lon)
    dist, _ = _tree(ref_sites).query(np.radians(q_sites), k=1)
    out = _floats(dist[:, 0][inverse] * EARTH_RADIUS_KM)
    return out


# ---------------------------------------------------------------------------
# clusters
# ---------------------------------------------------------------------------
def _spanning_tree(site_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Minimum spanning tree of the sites: (parent of each site, edge km).

    Prim's algorithm on unit vectors: the nearest site is the one with the
    largest dot product, and great-circle distance only grows with that,
    so this is the haversine tree. O(n) memory and n vectorised steps --
    seconds for tens of thousands of sites, no distance matrix. Cutting
    every edge longer than a gap leaves exactly the single-linkage
    clusters at that gap, for any gap, from the one tree. Ties go to the
    lowest site index, so the tree is always the same. Site 0 is the
    root (parent -1, edge 0).
    """
    lat = np.radians(site_xy[:, 0])
    lon = np.radians(site_xy[:, 1])
    xyz: np.ndarray = np.column_stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )
    n = len(xyz)
    parent: np.ndarray = np.full(n, -1, dtype=np.intp)
    edge_cos: np.ndarray = np.ones(n, dtype=np.float64)
    if n == 1:
        return parent, np.zeros(1, dtype=np.float64)
    remaining: np.ndarray = np.arange(1, n, dtype=np.intp)
    best: np.ndarray = xyz[remaining] @ xyz[0]
    via: np.ndarray = np.zeros(n - 1, dtype=np.intp)
    while remaining.size:
        k = int(np.argmax(best))
        site = int(remaining[k])
        parent[site] = via[k]
        edge_cos[site] = best[k]
        remaining = np.delete(remaining, k)
        best = np.delete(best, k)
        via = np.delete(via, k)
        if remaining.size:
            cos_new: np.ndarray = xyz[remaining] @ xyz[site]
            closer: np.ndarray = cos_new > best
            best[closer] = cos_new[closer]
            via[closer] = site
    edge_km: np.ndarray = EARTH_RADIUS_KM * np.arccos(
        np.clip(edge_cos, -1.0, 1.0)
    )
    edge_km[0] = 0.0
    return parent, edge_km


def _components(
    parent: np.ndarray, edge_km: np.ndarray, gap_km: float
) -> np.ndarray:
    """Single-linkage clusters at gap_km: the tree with long edges cut."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = parent.size
    keep: np.ndarray = (parent >= 0) & (edge_km <= gap_km)
    child: np.ndarray = np.flatnonzero(keep)
    graph = coo_matrix(
        (np.ones(child.size), (child, parent[keep])), shape=(n, n)
    )
    _, labels = connected_components(graph, directed=False)
    out: np.ndarray = np.asarray(labels, dtype=np.intp)
    return out


def auto_gap_km(
    parent: np.ndarray,
    edge_km: np.ndarray,
    weights: np.ndarray,
    max_cluster_share: float,
    max_oversize_share: float,
    candidates: Tuple[float, ...] = _GAP_CANDIDATES_KM,
) -> float:
    """Largest candidate gap keeping oversize clusters to a share of samples.

    A cluster is oversize when it holds more than `max_cluster_share` of
    the samples (it has to be cut to be held out). The gap is the largest
    candidate at which oversize clusters hold no more than
    `max_oversize_share` of the samples, else the smallest candidate.
    """
    total = float(weights.sum())
    best = float(min(candidates))
    for gap in sorted(candidates):
        sizes: np.ndarray = np.bincount(
            _components(parent, edge_km, gap), weights=weights
        )
        oversize = float(sizes[sizes > max_cluster_share * total].sum())
        if oversize <= max_oversize_share * total:
            best = float(gap)
    return best


def cluster_ids(
    lat: ArrayLike,
    lon: ArrayLike,
    gap_km: Optional[float] = None,
    *,
    max_cluster_share: float = 0.05,
    max_oversize_share: float = 0.5,
) -> Tuple[List[str], Dict[str, Any]]:
    """Cluster of each point (single linkage at the gap), and how it went.

    The info dict holds gap_km, gap_source, n_clusters, n_too_big (clusters
    with more than `max_cluster_share` of the samples, which never go to
    test), too_big_share (the samples they hold) and largest_cluster_share.
    """
    lat_a, lon_a = _floats(lat), _floats(lon)
    info: Dict[str, Any] = {
        "gap_km": gap_km,
        "gap_source": "config" if gap_km is not None else "auto",
        "n_clusters": 0,
        "n_too_big": 0,
        "too_big_share": None,
        "largest_cluster_share": None,
    }
    if lat_a.size == 0:
        return [], info
    site_xy, inverse = _sites(lat_a, lon_a)
    weights: np.ndarray = np.bincount(
        inverse, minlength=len(site_xy)
    ).astype(np.float64)
    parent, edge_km = _spanning_tree(site_xy)
    gap = (
        float(gap_km)
        if gap_km is not None
        else auto_gap_km(
            parent, edge_km, weights, max_cluster_share, max_oversize_share
        )
    )
    comp: np.ndarray = _components(parent, edge_km, gap)
    total = float(weights.sum())
    sizes: np.ndarray = np.bincount(comp, weights=weights)
    too_big: np.ndarray = sizes > max_cluster_share * total
    info["gap_km"] = gap
    info["n_clusters"] = int(sizes.size)
    info["n_too_big"] = int(np.count_nonzero(too_big))
    info["too_big_share"] = float(sizes[too_big].sum()) / total
    info["largest_cluster_share"] = float(sizes.max()) / total
    names = [str(c) for c in comp.tolist()]
    return [names[i] for i in inverse.tolist()], info


def pick_test_groups(
    sizes: Mapping[str, int],
    test_size: float,
    seed: int,
    max_size: Optional[float] = None,
) -> List[str]:
    """Whole groups for test, at random, until about `test_size` of samples.

    Groups are visited in a seeded random order and taken while the test
    set stays within the target; then the one group that brings it closest
    to the target is added if that helps. Groups above `max_size` samples
    are never taken. The label plays no part. Same sizes and seed, same
    groups.
    """
    names = sorted(sizes)
    total = sum(sizes.values())
    target = int(round(test_size * total))
    order = np.random.default_rng(seed).permutation(len(names)).tolist()
    eligible = [
        names[i]
        for i in order
        if max_size is None or sizes[names[i]] <= max_size
    ]
    chosen: List[str] = []
    n = 0
    for name in eligible:
        if n >= target:
            break
        if n + sizes[name] <= target:
            chosen.append(name)
            n += sizes[name]
    if n < target:
        taken = set(chosen)
        best: Optional[str] = None
        best_gap = target - n
        for name in eligible:
            gap = abs(target - (n + sizes[name]))
            if name not in taken and gap < best_gap:
                best, best_gap = name, gap
        if best is not None:
            chosen.append(best)
    return sorted(chosen)


# ---------------------------------------------------------------------------
# auto sizes
# ---------------------------------------------------------------------------
def variogram(
    lat: ArrayLike,
    lon: ArrayLike,
    values: ArrayLike,
    *,
    sites: int = 5000,
    neighbours: int = 50,
    random_pairs: int = 200_000,
    seed: int = 0,
) -> Tuple[pl.DataFrame, float]:
    """Empirical variogram of *values* over unique sites, and its sill.

    Values are averaged per site first, so many samples at one site do not
    swamp the short distances. Pairs come from two samples: each of up to
    `sites` sites with its `neighbours` nearest sites (short distances are
    rare among random pairs), plus `random_pairs` random pairs (the long
    distances). Returns {distance_km, semivariance, n_pairs} per distance
    bin with enough pairs, and the variance of the site values (the sill).
    """
    lat_a, lon_a, val_a = _floats(lat), _floats(lon), _floats(values)
    keep: np.ndarray = (
        np.isfinite(lat_a) & np.isfinite(lon_a) & np.isfinite(val_a)
    )
    empty = pl.DataFrame(
        schema={
            "distance_km": pl.Float64,
            "semivariance": pl.Float64,
            "n_pairs": pl.Int64,
        }
    )
    if int(np.count_nonzero(keep)) < 3:
        return empty, float("nan")
    site_xy, inverse = _sites(lat_a[keep], lon_a[keep])
    n = len(site_xy)
    z = _floats(
        np.bincount(inverse, weights=val_a[keep], minlength=n)
        / np.bincount(inverse, minlength=n)
    )
    sill = float(np.var(z))
    if n < 3 or not sill > 0:
        return empty, sill

    rng = np.random.default_rng(seed)
    chosen: np.ndarray = np.asarray(
        rng.choice(n, size=min(int(sites), n), replace=False), dtype=np.intp
    )
    k = min(int(neighbours) + 1, n)
    _, idx = _tree(site_xy).query(np.radians(site_xy[chosen]), k=k)
    near_i: np.ndarray = np.repeat(chosen, k)
    near_j: np.ndarray = np.asarray(idx, dtype=np.intp).reshape(-1)
    rand_i: np.ndarray = np.asarray(
        rng.integers(0, n, int(random_pairs)), dtype=np.intp
    )
    rand_j: np.ndarray = np.asarray(
        rng.integers(0, n, int(random_pairs)), dtype=np.intp
    )
    all_i: np.ndarray = np.concatenate([near_i, rand_i])
    all_j: np.ndarray = np.concatenate([near_j, rand_j])
    distinct: np.ndarray = all_i != all_j
    i: np.ndarray = all_i[distinct]
    j: np.ndarray = all_j[distinct]

    d = haversine_km(site_xy[i, 0], site_xy[i, 1], site_xy[j, 0], site_xy[j, 1])
    gamma: np.ndarray = 0.5 * (z[i] - z[j]) ** 2
    n_bins = len(_BIN_EDGES) - 1
    b: np.ndarray = np.asarray(np.digitize(d, _BIN_EDGES), dtype=np.intp) - 1
    inside: np.ndarray = (b >= 0) & (b < n_bins)
    counts: np.ndarray = np.bincount(b[inside], minlength=n_bins)
    sums: np.ndarray = np.bincount(
        b[inside], weights=gamma[inside], minlength=n_bins
    )
    centres: np.ndarray = np.sqrt(_BIN_EDGES[:-1] * _BIN_EDGES[1:])
    enough: np.ndarray = counts >= _MIN_PAIRS
    table = pl.DataFrame(
        {
            "distance_km": centres[enough],
            "semivariance": sums[enough] / counts[enough],
            "n_pairs": counts[enough].astype(np.int64),
        }
    )
    return table, sill


def auto_range_km(
    table: pl.DataFrame,
    sill: float,
    fraction: float = 0.95,
    min_km: float = 10.0,
    max_km: float = 500.0,
) -> float:
    """Shortest distance at which the variogram reaches `fraction` of the sill.

    Two bins in a row must reach it. No variogram, or one that never gets
    there, gives `max_km`; the result is clipped to [min_km, max_km].
    """
    if table.height == 0 or not sill > 0:
        return float(max_km)
    ratio: np.ndarray = table.get_column("semivariance").to_numpy() / sill
    dist: np.ndarray = table.get_column("distance_km").to_numpy()
    above: np.ndarray = ratio >= fraction
    for k in range(len(dist)):
        if above[k] and (k + 1 == len(dist) or above[k + 1]):
            return float(np.clip(dist[k], min_km, max_km))
    return float(max_km)


def choose_buffer(
    nearest_test_km: ArrayLike,
    requested_km: Optional[float],
    target_km: float,
    max_loss: float,
) -> Tuple[float, str]:
    """Buffer distance, and where it came from.

    A requested distance is used as given. Auto is `target_km` unless
    that removes more than `max_loss` of the training samples (those
    closer than the buffer to a test sample); it is then the largest
    distance that removes no more than that.
    """
    if requested_km is not None:
        return float(requested_km), "config"
    every = _floats(nearest_test_km)
    near: np.ndarray = every[np.isfinite(every)]
    if near.size == 0 or float(np.mean(near < target_km)) <= max_loss:
        return float(target_km), "auto"
    capped = float(np.quantile(near, max_loss, method="lower"))
    return capped, "auto, capped by max_buffer_loss"


# ---------------------------------------------------------------------------
# the holdout
# ---------------------------------------------------------------------------
def _share_within(dist: np.ndarray, km: float) -> Optional[float]:
    return float(np.mean(dist <= km)) if dist.size else None


def _quantile(dist: np.ndarray, q: float) -> Optional[float]:
    finite: np.ndarray = dist[np.isfinite(dist)]
    return float(np.quantile(finite, q)) if finite.size else None


def spatial_holdout(
    split_manager: Any,
    data: pl.DataFrame,
    coords: pl.DataFrame,
    settings: SpatialSettings,
    *,
    test_size: float = 0.2,
    n_bins: int = 5,
    random_state: int = 42,
    metadata: Optional[pl.DataFrame] = None,
    method: str = "grid",
) -> Tuple[Dict[str, Any], pl.DataFrame]:
    """Build *split_manager*'s holdout in space; returns (report, variogram).

    *data* is {sample, <label>}; *coords* is {sample, lat, lon} (see
    `coordinates`). Labelled samples without coordinates are left out of
    the holdout, like samples without a label. *method* is "grid" (cells,
    then a buffer) or "cluster" (no buffer needed). Test groups are picked
    without looking at the label, so *n_bins* is not used. The holdout
    keeps its usual columns plus the group (`spatial_cell` or
    `spatial_cluster`) and, for training and buffer rows,
    `nearest_test_km`.
    """
    if method not in METHODS:
        raise ValueError(f"spatial method must be one of {METHODS}, got {method!r}")
    group_col = _GROUP_COLUMN[method]
    label = split_manager.label
    labelled = data.select("sample", label).drop_nulls(subset=[label])
    joined = labelled.join(coords, on="sample", how="inner")
    n_without = labelled.height - joined.height
    if n_without:
        logger.warning(
            "Spatial holdout for '%s': %d labelled sample(s) have no valid "
            "coordinates and are left out",
            label,
            n_without,
        )
    lat = joined.get_column("lat").to_numpy()
    lon = joined.get_column("lon").to_numpy()

    vario = pl.DataFrame(schema=VARIOGRAM_SCHEMA)
    sill = float("nan")
    range_km: Optional[float] = None
    # The variogram sizes the grid's cells and buffer; clusters size both
    # from the gap instead, so they do not need it.
    if method == "grid" and (
        settings.cell_km is None or settings.buffer_km is None
    ):
        if joined.height and joined.schema[label].is_numeric():
            table, sill = variogram(
                lat,
                lon,
                joined.get_column(label).to_numpy(),
                sites=settings.variogram_sites,
                neighbours=settings.variogram_neighbours,
                random_pairs=settings.variogram_random_pairs,
                seed=random_state,
            )
            range_km = auto_range_km(
                table,
                sill,
                settings.auto_fraction,
                settings.auto_min_km,
                settings.auto_max_km,
            )
            vario = table.with_columns(
                pl.lit(label, dtype=pl.Utf8).alias("label"),
                (
                    pl.col("semivariance") / sill
                    if sill > 0
                    else pl.lit(None, dtype=pl.Float64)
                ).alias("relative"),
            ).select(list(VARIOGRAM_SCHEMA))
        else:
            range_km = settings.auto_max_km
            if joined.height:
                logger.warning(
                    "Spatial holdout for '%s': auto distances need a numeric "
                    "label; using auto_max_km = %g",
                    label,
                    range_km,
                )
    cell_km: Optional[float] = None
    cluster_info: Dict[str, Any] = {}
    if method == "grid":
        cell_km = settings.cell_km if settings.cell_km is not None else range_km
        assert cell_km is not None
        groups = cell_ids(lat, lon, cell_km)
        target = range_km if range_km is not None else settings.auto_max_km
    else:
        groups, cluster_info = cluster_ids(
            lat,
            lon,
            settings.gap_km,
            max_cluster_share=settings.max_cluster_share,
            max_oversize_share=settings.max_oversize_share,
        )
        target = float(
            cluster_info["gap_km"]
            if cluster_info["gap_km"] is not None
            else min(_GAP_CANDIDATES_KM)
        )

    frame = joined.select("sample", label).with_columns(
        pl.Series(group_col, groups, dtype=pl.Utf8)
    )
    group_sizes: Dict[str, int] = {}
    for name in groups:
        group_sizes[name] = group_sizes.get(name, 0) + 1
    test_groups = pick_test_groups(
        group_sizes,
        test_size,
        random_state,
        max_size=(
            settings.max_cluster_share * frame.height
            if method == "cluster"
            else None
        ),
    )
    split_manager.holdout = split_manager._enrich_holdout(
        holdout_df=frame.select(
            "sample",
            pl.when(
                pl.col(group_col).is_in(pl.Series(test_groups, dtype=pl.Utf8))
            )
            .then(pl.lit("test"))
            .otherwise(pl.lit("train"))
            .cast(pl.Utf8)
            .alias("split"),
        ),
        data=frame,
        target_col=label,
        metadata=metadata,
        split_col_name="split",
    )
    holdout: pl.DataFrame = split_manager.holdout

    placed = (
        holdout.select("sample", "split")
        .join(frame.select("sample", group_col), on="sample", how="left")
        .join(joined.select("sample", "lat", "lon"), on="sample", how="left")
    )
    test = placed.filter(pl.col("split") == "test")
    train = placed.filter(pl.col("split") != "test")
    t_lat, t_lon = test["lat"].to_numpy(), test["lon"].to_numpy()
    r_lat, r_lon = train["lat"].to_numpy(), train["lon"].to_numpy()

    near = nearest_km(r_lat, r_lon, t_lat, t_lon)
    if method == "cluster":
        # Clusters are more than the gap apart: nothing to buffer.
        buffer_km, buffer_source = 0.0, "none: clusters are gap_km apart"
    else:
        buffer_km, buffer_source = choose_buffer(
            near, settings.buffer_km, target, settings.max_buffer_loss
        )
    in_buffer: np.ndarray = np.asarray(near < buffer_km, dtype=bool)
    before = nearest_km(t_lat, t_lon, r_lat, r_lon)
    after = nearest_km(t_lat, t_lon, r_lat[~in_buffer], r_lon[~in_buffer])

    extra = pl.concat(
        [
            train.select(
                "sample",
                pl.Series(
                    "_split", np.where(in_buffer, SPLIT_BUFFER, "train")
                ).cast(pl.Utf8),
                group_col,
                pl.Series(NEAREST_COLUMN, near, dtype=pl.Float64),
            ),
            test.select(
                "sample",
                pl.lit("test", dtype=pl.Utf8).alias("_split"),
                group_col,
                pl.lit(None, dtype=pl.Float64).alias(NEAREST_COLUMN),
            ),
        ]
    )
    rest = [
        c
        for c in holdout.columns
        if c not in ("sample", "split", label, group_col, NEAREST_COLUMN)
    ]
    split_manager.holdout = (
        holdout.join(extra, on="sample", how="left")
        .with_columns(pl.col("_split").alias("split"))
        .select(["sample", "split", label, group_col, NEAREST_COLUMN, *rest])
    )

    n_buffer = int(np.count_nonzero(in_buffer))
    n_train = int(train.height - n_buffer)
    report: Dict[str, Any] = {
        "label": label,
        "method": method,
        "n_labelled": labelled.height,
        "n_no_coordinates": n_without,
        "n_sites": int(len(_sites(lat, lon)[0])) if joined.height else 0,
        "sill": sill,
        "range_km": range_km,
        "cell_km": None if cell_km is None else float(cell_km),
        "cell_source": (
            None
            if method != "grid"
            else "config" if settings.cell_km is not None else "auto"
        ),
        "gap_km": cluster_info.get("gap_km"),
        "gap_source": cluster_info.get("gap_source"),
        "n_too_big": cluster_info.get("n_too_big"),
        "too_big_share": cluster_info.get("too_big_share"),
        "largest_cluster_share": cluster_info.get("largest_cluster_share"),
        "n_groups": frame.get_column(group_col).n_unique(),
        "n_test_groups": test.get_column(group_col).n_unique(),
        "buffer_km": buffer_km,
        "buffer_source": buffer_source,
        "n_test": test.height,
        "n_train": n_train,
        "n_buffer": n_buffer,
        "buffer_loss": (n_buffer / train.height) if train.height else 0.0,
        # Leakage before the buffer: test samples with a training sample
        # this close (what the bioproject holdout looks like here).
        "test_within_1km_before": _share_within(before, 1.0),
        "test_within_10km_before": _share_within(before, 10.0),
        "test_within_50km_before": _share_within(before, 50.0),
        # After it: by construction no closer than buffer_km.
        "test_nearest_train_km_min": _quantile(after, 0.0),
        "test_nearest_train_km_median": _quantile(after, 0.5),
    }
    sizing = (
        f"cells {cell_km:.0f} km ({report['cell_source']})"
        if method == "grid"
        else f"clusters, gap {target:.0f} km ({report['gap_source']}), "
        f"{report['n_too_big']} too big for test"
    )
    logger.info(
        "Spatial holdout for '%s': %s, %d of %d groups to test; buffer "
        "%.1f km (%s) removed %d of %d training samples (%.1f%%)",
        label,
        sizing,
        report["n_test_groups"],
        report["n_groups"],
        buffer_km,
        buffer_source,
        n_buffer,
        train.height,
        100 * report["buffer_loss"],
    )
    return report, vario


REPORT_SCHEMA = {
    "label": pl.Utf8,
    "method": pl.Utf8,
    "n_labelled": pl.Int64,
    "n_no_coordinates": pl.Int64,
    "n_sites": pl.Int64,
    "sill": pl.Float64,
    "range_km": pl.Float64,
    "cell_km": pl.Float64,
    "cell_source": pl.Utf8,
    "gap_km": pl.Float64,
    "gap_source": pl.Utf8,
    "n_too_big": pl.Int64,
    "too_big_share": pl.Float64,
    "largest_cluster_share": pl.Float64,
    "n_groups": pl.Int64,
    "n_test_groups": pl.Int64,
    "buffer_km": pl.Float64,
    "buffer_source": pl.Utf8,
    "n_test": pl.Int64,
    "n_train": pl.Int64,
    "n_buffer": pl.Int64,
    "buffer_loss": pl.Float64,
    "test_within_1km_before": pl.Float64,
    "test_within_10km_before": pl.Float64,
    "test_within_50km_before": pl.Float64,
    "test_nearest_train_km_min": pl.Float64,
    "test_nearest_train_km_median": pl.Float64,
}


def report_frame(reports: List[Dict[str, Any]]) -> pl.DataFrame:
    """One row per label, with a fixed schema (empty labels included)."""
    return pl.DataFrame(
        [{k: r.get(k) for k in REPORT_SCHEMA} for r in reports],
        schema=REPORT_SCHEMA,
    )
