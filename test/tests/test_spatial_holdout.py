"""The spatial holdout: grid cells, the haversine buffer, auto sizes."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

pytest.importorskip("sklearn")

from microbiome_ml.wrangle import spatial as sp  # noqa: E402
from microbiome_ml.wrangle.dataset import Dataset  # noqa: E402
from microbiome_ml.wrangle.splits import SplitManager  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import sweep_tokens  # noqa: E402


def _brute_nearest(q_lat, q_lon, r_lat, r_lon):
    d = sp.haversine_km(
        np.asarray(q_lat)[:, None],
        np.asarray(q_lon)[:, None],
        np.asarray(r_lat)[None, :],
        np.asarray(r_lon)[None, :],
    )
    return d.min(axis=1)


def _clustered(n_sites=120, per_site=6, seed=0):
    """Sites in a few regions; the label follows latitude plus noise."""
    rng = np.random.default_rng(seed)
    centres = np.array([[-35, 145], [10, 20], [45, 5], [60, -100], [0, -60]])
    which = rng.integers(0, len(centres), n_sites)
    site_lat = centres[which, 0] + rng.normal(0, 4, n_sites)
    site_lon = centres[which, 1] + rng.normal(0, 4, n_sites)
    lat = np.repeat(site_lat, per_site)
    lon = np.repeat(site_lon, per_site)
    n = lat.size
    samples = [f"s{i}" for i in range(n)]
    label = 30 - 0.4 * np.abs(lat) + rng.normal(0, 1, n)
    data = pl.DataFrame({"sample": samples, "temp": label})
    coords = pl.DataFrame({"sample": samples, "lat": lat, "lon": lon})
    return data, coords


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lat", [0.0, 45.0, 70.0])
def test_cells_keep_their_width_towards_the_poles(lat):
    """A degree grid narrows with latitude; these cells should not."""
    cell_km = 100.0
    # points 1 km apart along 1000 km of one parallel
    step_deg = 1.0 / (sp.KM_PER_DEGREE * np.cos(np.radians(lat)))
    lon = -20 + step_deg * np.arange(1000)
    cells = sp.cell_ids(np.full(lon.size, lat), lon, cell_km)
    assert 9 <= len(set(cells)) <= 12


def test_cells_are_cell_km_tall():
    lat = np.linspace(0, 9.0, 1000)  # ~1000 km of latitude
    cells = sp.cell_ids(lat, np.zeros(lat.size), 100.0)
    assert 9 <= len(set(cells)) <= 12


def test_nearest_matches_brute_force():
    rng = np.random.default_rng(1)
    q_lat, q_lon = rng.uniform(-80, 80, 300), rng.uniform(-180, 180, 300)
    r_lat, r_lon = rng.uniform(-80, 80, 400), rng.uniform(-180, 180, 400)
    # repeated sites on both sides, and the dateline
    q_lat[:20], q_lon[:20] = q_lat[0], q_lon[0]
    q_lat[20], q_lon[20], r_lat[0], r_lon[0] = 10.0, 179.9, 10.0, -179.9
    got = sp.nearest_km(q_lat, q_lon, r_lat, r_lon)
    want = _brute_nearest(q_lat, q_lon, r_lat, r_lon)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-3)
    assert got[20] < 25  # 0.2 degrees across the dateline, not 359.8


def test_nearest_without_references_is_infinite():
    assert np.isinf(sp.nearest_km([0.0], [0.0], [], [])).all()
    assert sp.nearest_km([], [], [0.0], [0.0]).size == 0


# ---------------------------------------------------------------------------
# auto sizes
# ---------------------------------------------------------------------------
def test_auto_range_is_short_for_noise_and_long_for_a_smooth_field():
    rng = np.random.default_rng(2)
    lat, lon = rng.uniform(-60, 60, 3000), rng.uniform(-60, 60, 3000)
    noise = rng.normal(0, 1, lat.size)
    smooth = np.sin(np.radians(lat) * 3) + np.cos(np.radians(lon) * 2)

    def rng_km(values):
        table, sill = sp.variogram(lat, lon, values, sites=1000, seed=0)
        return sp.auto_range_km(table, sill, 0.95, 10.0, 5000.0)

    # Noise is unrelated at every distance: the range is the first well
    # populated bins (about 70 km here). A field that varies over
    # thousands of km does not reach its sill before the cap.
    assert rng_km(noise) < 300.0
    assert rng_km(smooth) > 2000.0


def test_auto_range_without_a_variogram_is_the_maximum():
    empty = pl.DataFrame(
        schema={"distance_km": pl.Float64, "semivariance": pl.Float64}
    )
    assert sp.auto_range_km(empty, float("nan"), 0.95, 10, 300) == 300


def test_auto_buffer_is_capped_by_max_loss():
    near = np.linspace(0, 1000, 1001)  # one training sample per km
    km, source = sp.choose_buffer(near, None, 500.0, 0.25)
    assert source.startswith("auto, capped")
    assert np.mean(near < km) <= 0.25
    assert km == pytest.approx(250.0, abs=1.0)
    # short enough already: used as is
    assert sp.choose_buffer(near, None, 100.0, 0.25) == (100.0, "auto")
    # a configured buffer is never capped
    assert sp.choose_buffer(near, 800.0, 100.0, 0.25) == (800.0, "config")


def test_settings_read_auto_and_numbers():
    s = sp.SpatialSettings.from_config(None)
    assert s.cell_km is None and s.buffer_km is None
    s = sp.SpatialSettings.from_config(
        {"cell_km": 150, "buffer_km": "AUTO", "max_buffer_loss": 0.1}
    )
    assert (s.cell_km, s.buffer_km, s.max_buffer_loss) == (150.0, None, 0.1)
    assert sp.SpatialSettings.from_config({"buffer_km": 0}).buffer_km == 0.0
    with pytest.raises(ValueError, match="Unknown"):
        sp.SpatialSettings.from_config({"cell": 100})
    with pytest.raises(ValueError, match="cell_km"):
        sp.SpatialSettings.from_config({"cell_km": 0})
    with pytest.raises(ValueError, match="buffer_km"):
        sp.SpatialSettings.from_config({"buffer_km": "far"})
    with pytest.raises(ValueError, match="max_buffer_loss"):
        sp.SpatialSettings.from_config({"max_buffer_loss": 2})


# ---------------------------------------------------------------------------
# the holdout
# ---------------------------------------------------------------------------
def _split(settings, seed=42):
    data, coords = _clustered()
    sm = SplitManager("temp")
    report, vario = sp.spatial_holdout(
        sm, data, coords, settings, test_size=0.2, random_state=seed
    )
    return sm.holdout, coords, report, vario


def test_no_training_sample_is_within_the_buffer_of_a_test_sample():
    settings = sp.SpatialSettings(cell_km=300.0, buffer_km=200.0)
    holdout, coords, report, _ = _split(settings)
    h = holdout.join(coords, on="sample")
    test = h.filter(pl.col("split") == "test")
    train = h.filter(pl.col("split") == "train")
    buffered = h.filter(pl.col("split") == "buffer")
    assert test.height and train.height and buffered.height

    d = _brute_nearest(train["lat"], train["lon"], test["lat"], test["lon"])
    assert d.min() >= 200.0
    d = _brute_nearest(
        buffered["lat"], buffered["lon"], test["lat"], test["lon"]
    )
    assert d.max() < 200.0
    # whole cells: no cell on both sides
    assert not set(test["spatial_cell"]) & set(
        h.filter(pl.col("split") != "test")["spatial_cell"]
    )
    assert report["n_buffer"] == buffered.height
    assert report["test_nearest_train_km_min"] >= 200.0


def test_the_test_set_is_the_size_asked_for():
    holdout, _, report, _ = _split(sp.SpatialSettings(cell_km=300.0))
    n = holdout.height
    n_test = holdout.filter(pl.col("split") == "test").height
    assert abs(n_test / n - 0.2) < 0.08
    assert report["n_test"] + report["n_train"] + report["n_buffer"] == n


def test_the_split_is_the_same_every_time():
    settings = sp.SpatialSettings()  # all auto
    a, _, ra, va = _split(settings)
    b, _, rb, vb = _split(settings)
    assert a.sort("sample").equals(b.sort("sample"))
    assert ra == rb
    assert va.equals(vb)


def test_auto_settings_respect_the_bounds_and_the_loss():
    settings = sp.SpatialSettings(
        auto_min_km=50.0, auto_max_km=800.0, max_buffer_loss=0.2
    )
    _, _, report, vario = _split(settings)
    assert report["cell_source"] == "auto"
    assert 50.0 <= report["cell_km"] <= 800.0
    assert report["buffer_loss"] <= 0.2 + 1e-9
    assert vario.height and set(vario["label"]) == {"temp"}


def test_samples_without_coordinates_are_left_out():
    data, coords = _clustered()
    coords = coords.with_columns(
        pl.when(pl.col("sample").is_in(["s0", "s1"]))
        .then(None)
        .otherwise(pl.col("lat"))
        .alias("lat")
    ).drop_nulls()
    sm = SplitManager("temp")
    report, _ = sp.spatial_holdout(
        sm, data, coords, sp.SpatialSettings(cell_km=300.0, buffer_km=0)
    )
    assert report["n_no_coordinates"] == 2
    assert {"s0", "s1"}.isdisjoint(set(sm.holdout["sample"]))
    assert report["n_buffer"] == 0


def test_coordinates_drop_impossible_values():
    meta = pl.DataFrame(
        {
            "sample": ["a", "b", "c", "d"],
            "lat": ["10.5", "95", None, "-45"],
            "lon": [20.0, 0.0, 0.0, 200.0],
        }
    )
    assert sp.coordinates(meta)["sample"].to_list() == ["a"]
    with pytest.raises(ValueError, match="coordinates"):
        sp.coordinates(meta.drop("lat"))


# ---------------------------------------------------------------------------
# the dataset and the pipeline
# ---------------------------------------------------------------------------
def _dataset():
    data, coords = _clustered()
    ds = Dataset()
    ds.add_labels(data)
    # Only .metadata.collect() is used, so a stand-in will do.
    ds.metadata = SimpleNamespace(metadata=coords.lazy())
    return ds


def test_dataset_writes_the_holdout_and_both_reports(tmp_path):
    ds = _dataset().create_spatial_holdout_split(
        spatial={"cell_km": 300, "buffer_km": 150}, output_dir=tmp_path
    )
    holdout = pl.read_csv(tmp_path / "temp" / "holdout.csv")
    assert set(holdout["split"]) == {"train", "test", "buffer"}
    assert holdout.columns[:5] == [
        "sample",
        "split",
        "temp",
        "spatial_cell",
        "nearest_test_km",
    ]
    report = pl.read_csv(tmp_path / "spatial_report.tsv", separator="\t")
    assert report["buffer_km"].item() == 150
    assert (tmp_path / "spatial_variogram.tsv").exists()
    assert ds.spatial_report.height == 1


def test_cv_folds_leave_the_buffer_out():
    ds = _dataset().create_spatial_holdout_split(
        spatial={"cell_km": 300, "buffer_km": 150}
    )
    ds.create_cv_folds(
        label="temp", grouping="random", use_holdout=True, force=True
    )
    folds = ds.splits["temp"].cv_schemes["random"]
    holdout = ds.splits["temp"].holdout
    train = set(holdout.filter(pl.col("split") == "train")["sample"])
    assert set(folds["sample"]) == train


def test_the_spatial_token():
    assert sweep_tokens.holdout_spec("spatial") == ("spatial", None)
    assert sweep_tokens.holdout_spec("Spatial") == ("spatial", None)
    assert sweep_tokens.spec_dir_name("spatial") == "spatial"
    specs = sweep_tokens.run_specs(holdout_groupings=["bioproject", "spatial"])
    assert list(specs) == ["holdout_bioproject", "holdout_spatial"]


# ---------------------------------------------------------------------------
# the map
# ---------------------------------------------------------------------------
def test_cell_bounds_contain_their_points():
    rng = np.random.default_rng(4)
    lat, lon = rng.uniform(-89, 89, 500), rng.uniform(-180, 179.99, 500)
    for km in (50.0, 300.0, 1000.0):
        for x, y, cell in zip(lon, lat, sp.cell_ids(lat, lon, km)):
            lon0, lat0, lon1, lat1 = sp.cell_bounds(cell, km)
            assert lon0 <= x <= lon1 and lat0 <= y <= lat1


def test_the_map_counts_every_side(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from microbiome_ml.visualise.visualisations import Visualiser

    ds = _dataset().create_spatial_holdout_split(
        spatial={"cell_km": 300, "buffer_km": 150}, output_dir=tmp_path
    )
    holdout = ds.splits["temp"].holdout
    stats = Visualiser(tmp_path / "maps").plot_holdout_map(
        tmp_path / "temp" / "holdout.csv",
        cell_km=300,
        buffer_km=150,
        output="holdout_map_temp",
    )
    assert (tmp_path / "maps" / "holdout_map_temp.png").exists()
    for side in ("train", "test", "buffer"):
        n = holdout.filter(pl.col("split") == side).height
        assert stats[f"n_{side}"] == n


def test_maps_are_drawn_for_spatial_runs_unless_asked(tmp_path):
    import run_cv_to_final_eval as pipeline

    _dataset().create_spatial_holdout_split(
        spatial={"cell_km": 300, "buffer_km": 150}, output_dir=tmp_path
    )
    vis = {"enabled": True, "formats": ["png"]}
    pipeline.plot_holdout_maps(tmp_path, vis, tmp_path / "a", spatial=False)
    assert not (tmp_path / "a" / "holdout_map_temp.png").exists()
    pipeline.plot_holdout_maps(tmp_path, vis, tmp_path / "b", spatial=True)
    assert (tmp_path / "b" / "holdout_map_temp.png").exists()
    pipeline.plot_holdout_maps(
        tmp_path, {**vis, "holdout_map": True}, tmp_path / "c"
    )
    assert (tmp_path / "c" / "holdout_map_temp.png").exists()
    pipeline.plot_holdout_maps(
        tmp_path, {**vis, "holdout_map": False}, tmp_path / "d", spatial=True
    )
    assert not (tmp_path / "d" / "holdout_map_temp.png").exists()


# ---------------------------------------------------------------------------
# clusters (spatial_cluster)
# ---------------------------------------------------------------------------
ISLANDS = [(0, 0), (0, 20), (20, 0), (-30, 100), (45, -120), (60, 179.95)]


def _islands(per_site=4, sites=50, seed=0):
    """Tight islands of sites (~10 km across) over 1000 km apart."""
    rng = np.random.default_rng(seed)
    lat, lon, island = [], [], []
    for k, (a, b) in enumerate(ISLANDS):
        la = a + rng.normal(0, 0.05, sites)
        lo = b + rng.normal(0, 0.05, sites)
        lat += list(np.repeat(la, per_site))
        lon += list(np.repeat(lo, per_site))
        island += [k] * sites * per_site
    lon_a = (np.asarray(lon) + 180.0) % 360.0 - 180.0
    return np.asarray(lat), lon_a, np.asarray(island)


def test_islands_become_one_cluster_each():
    lat, lon, island = _islands()
    ids, info = sp.cluster_ids(lat, lon, 100.0, max_cluster_share=0.5)
    ids_a = np.asarray(ids)
    assert info["n_clusters"] == len(ISLANDS) and info["n_too_big"] == 0
    for k in range(len(ISLANDS)):
        assert len(set(ids_a[island == k])) == 1
    # the last island straddles the dateline and is still one cluster
    assert lon[island == 5].min() < -179 and lon[island == 5].max() > 179


def test_clusters_are_the_same_every_time():
    lat, lon, _ = _islands()
    rng = np.random.default_rng(1)
    lat = np.concatenate([lat, rng.uniform(40, 50, 800)])
    lon = np.concatenate([lon, rng.uniform(0, 10, 800)])
    first = sp.cluster_ids(lat, lon, None, max_cluster_share=0.05)
    second = sp.cluster_ids(lat, lon, None, max_cluster_share=0.05)
    assert first == second


def test_a_dense_region_stays_whole_and_out_of_test():
    """Nothing is cut: a dense region is one cluster, too big for test."""
    rng = np.random.default_rng(2)
    lat, lon = rng.uniform(40, 50, 2000), rng.uniform(0, 10, 2000)
    ids, info = sp.cluster_ids(lat, lon, 100.0, max_cluster_share=0.05)
    assert info["n_clusters"] == 1 and len(set(ids)) == 1
    assert info["n_too_big"] == 1 and info["too_big_share"] == 1.0


def test_a_heavy_station_keeps_its_neighbours():
    """The Plymouth case: one site with most of the samples and a site
    234 m away. They are one cluster, so neither ends up buffered."""
    lat = [50.25] * 230 + [50.252] * 3 + [10.0] * 40 + [-20.0] * 40
    lon = [-4.2167] * 230 + [-4.2167] * 3 + [100.0] * 40 + [60.0] * 40
    ids, info = sp.cluster_ids(lat, lon, 50.0, max_cluster_share=0.3)
    assert ids[0] == ids[230]  # the station and its neighbour
    assert info["n_too_big"] == 1


def test_test_groups_ignore_the_label_and_hit_the_size():
    sizes = {f"g{i}": s for i, s in enumerate([30, 5, 12, 8, 50, 3, 20, 9])}
    picked = sp.pick_test_groups(sizes, 0.2, seed=42, max_size=40)
    assert picked == sp.pick_test_groups(sizes, 0.2, seed=42, max_size=40)
    assert "g4" not in picked  # 50 > max_size
    n = sum(sizes[g] for g in picked)
    assert abs(n - 0.2 * sum(sizes.values())) <= 3
    seeds = {
        tuple(sp.pick_test_groups(sizes, 0.2, seed=s, max_size=40))
        for s in range(10)
    }
    assert len(seeds) > 1  # another seed, another holdout
    assert sp.pick_test_groups({"a": 100}, 0.2, seed=1, max_size=50) == []


def test_auto_gap_is_the_largest_that_keeps_clusters_holdable():
    """Ten islands, in pairs 80 km apart, pairs far apart. Each island is
    10% of the samples and the share limit 15%: joining a pair (any gap
    of 100 km or more) makes every cluster oversize, so 50 km it is."""
    rng = np.random.default_rng(3)
    lat, lon = [], []
    for pair in range(5):
        for side in (0.0, 80.0 / sp.KM_PER_DEGREE):
            lat += list(rng.normal(0, 0.02, 40))
            lon += list(rng.normal(pair * 30.0 + side, 0.02, 40))
    _, info = sp.cluster_ids(lat, lon, None, max_cluster_share=0.15)
    assert info["gap_source"] == "auto"
    assert info["gap_km"] == 50.0
    assert info["n_clusters"] == 10 and info["n_too_big"] == 0


def test_cluster_holdout_costs_nothing_between_islands():
    lat, lon, _ = _islands()
    n = lat.size
    samples = [f"s{i}" for i in range(n)]
    data = pl.DataFrame({"sample": samples, "temp": 30 - 0.3 * np.abs(lat)})
    coords = pl.DataFrame({"sample": samples, "lat": lat, "lon": lon})
    sm = SplitManager("temp")
    settings = sp.SpatialSettings(gap_km=100.0, max_cluster_share=0.5)
    report, vario = sp.spatial_holdout(
        sm, data, coords, settings, method="cluster"
    )
    h = sm.holdout.join(coords, on="sample")
    test = h.filter(pl.col("split") == "test")
    train = h.filter(pl.col("split") == "train")
    assert test.height and train.height
    assert report["n_buffer"] == 0  # clusters are more than the gap apart
    assert set(sm.holdout["split"]) == {"train", "test"}  # nothing deleted
    assert report["method"] == "cluster" and report["cell_km"] is None
    assert report["gap_km"] == 100.0 and report["buffer_km"] == 0.0
    assert not set(test["spatial_cluster"]) & set(train["spatial_cluster"])
    d = _brute_nearest(train["lat"], train["lon"], test["lat"], test["lon"])
    assert d.min() >= 100.0
    assert vario.height == 0  # clusters need no variogram


def test_dataset_and_token_for_clusters(tmp_path):
    ds = _dataset().create_spatial_holdout_split(
        spatial={"gap_km": 100, "max_cluster_share": 0.2},
        method="cluster",
        output_dir=tmp_path,
    )
    holdout = pl.read_csv(tmp_path / "temp" / "holdout.csv")
    assert holdout.columns[3] == "spatial_cluster"
    assert ds.spatial_report["method"].to_list() == ["cluster"]
    assert sweep_tokens.holdout_spec("spatial_cluster") == ("spatial", "cluster")
    assert sweep_tokens.holdout_spec("spatial:cluster") == ("spatial", "cluster")
    assert sweep_tokens.spec_dir_name("spatial_cluster") == "spatial_cluster"
    specs = sweep_tokens.run_specs(
        holdout_groupings=["spatial", "spatial_cluster"]
    )
    assert list(specs) == ["holdout_spatial", "holdout_spatial_cluster"]
    with pytest.raises(ValueError, match="method"):
        sp.spatial_holdout(
            SplitManager("temp"),
            pl.DataFrame({"sample": ["a"], "temp": [1.0]}),
            pl.DataFrame({"sample": ["a"], "lat": [0.0], "lon": [0.0]}),
            sp.SpatialSettings(),
            method="hexagons",
        )


def test_cluster_settings_are_checked():
    s = sp.SpatialSettings.from_config({"gap_km": "auto"})
    assert s.gap_km is None and s.max_cluster_share == 0.05
    with pytest.raises(ValueError, match="gap_km"):
        sp.SpatialSettings.from_config({"gap_km": 0})
    with pytest.raises(ValueError, match="max_cluster_share"):
        sp.SpatialSettings.from_config({"max_cluster_share": 0})
    with pytest.raises(ValueError, match="Unknown"):
        sp.SpatialSettings.from_config({"cluster_neighbours": 0})
