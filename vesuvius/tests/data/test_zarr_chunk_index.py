import json
import os

from vesuvius.data.zarr_chunk_index import ENV_OVERRIDE_URL, build_chunk_occupancy


def test_sidecar_cache_hit_is_materialized_to_override(tmp_path, monkeypatch):
    array_path = tmp_path / "volume.zarr" / "0"
    array_path.mkdir(parents=True)
    (array_path / ".zarray").write_text(
        json.dumps(
            {
                "zarr_format": 2,
                "shape": [4, 4, 4],
                "chunks": [2, 2, 2],
                "dtype": "|u1",
                "compressor": None,
                "fill_value": 0,
                "order": "C",
                "filters": None,
                "dimension_separator": ".",
            }
        ),
        encoding="ascii",
    )
    (array_path / "0.0.0").write_bytes(b"non-empty")

    monkeypatch.setenv("VESUVIUS_CACHE_DIR", str(tmp_path / "cache"))
    first = build_chunk_occupancy(
        str(array_path),
        chunks=(2, 2, 2),
        shape=(4, 4, 4),
        use_cache=True,
    )
    assert first is not None
    assert (array_path / ".chunk_occupancy.npz").exists()

    override = tmp_path / "shared" / "chunk-occupancy.npz"
    monkeypatch.setenv(ENV_OVERRIDE_URL, str(override))
    second = build_chunk_occupancy(
        str(array_path),
        chunks=(2, 2, 2),
        shape=(4, 4, 4),
        use_cache=True,
    )

    assert second is not None
    assert second.shape == first.shape
    assert override.exists()


def _write_array(array_path, *, separator, chunk_keys):
    array_path.mkdir(parents=True)
    (array_path / ".zarray").write_text(
        json.dumps(
            {
                "zarr_format": 2,
                "shape": [4, 4, 4],
                "chunks": [2, 2, 2],
                "dtype": "|u1",
                "compressor": None,
                "fill_value": 0,
                "order": "C",
                "filters": None,
                "dimension_separator": separator,
            }
        ),
        encoding="ascii",
    )
    for key in chunk_keys:
        _add_chunk(array_path, key)


def _add_chunk(array_path, key, *, seconds_later=0):
    chunk = array_path / key
    chunk.parent.mkdir(parents=True, exist_ok=True)
    chunk.write_bytes(b"non-empty")
    if seconds_later:
        # Filesystem timestamps tick every few milliseconds; a test adds files faster
        # than any real download does, so say explicitly that this one came later.
        stamp = chunk.parent.stat().st_mtime_ns + seconds_later * 1_000_000_000
        os.utime(chunk.parent, ns=(stamp, stamp))


def _occupied(array_path):
    occupancy = build_chunk_occupancy(
        str(array_path), chunks=(2, 2, 2), shape=(4, 4, 4), use_cache=True
    )
    assert occupancy is not None
    return {tuple(int(v) for v in index) for index in zip(*occupancy.nonzero())}


def test_chunks_that_arrive_later_are_not_left_marked_empty(tmp_path, monkeypatch):
    """A download that completes after the first run must not keep the old index."""
    monkeypatch.setenv("VESUVIUS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv(ENV_OVERRIDE_URL, raising=False)
    for separator, first, in_same_dir, in_new_dir in (
        ("/", "0/0/0", "0/0/1", "1/1/1"),
        (".", "0.0.0", "0.0.1", "1.1.1"),
    ):
        array_path = tmp_path / f"volume{ord(separator)}.zarr" / "0"
        _write_array(array_path, separator=separator, chunk_keys=[first])
        assert _occupied(array_path) == {(0, 0, 0)}
        assert (array_path / ".chunk_occupancy.npz").exists()

        _add_chunk(array_path, in_same_dir, seconds_later=60)
        assert _occupied(array_path) == {(0, 0, 0), (0, 0, 1)}

        _add_chunk(array_path, in_new_dir, seconds_later=120)
        assert _occupied(array_path) == {(0, 0, 0), (0, 0, 1), (1, 1, 1)}


def test_unchanged_local_array_still_hits_its_sidecar(tmp_path, monkeypatch, capsys):
    """Writing the sidecar into the array directory must not invalidate it."""
    monkeypatch.setenv("VESUVIUS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv(ENV_OVERRIDE_URL, raising=False)
    for separator, key in (("/", "0/0/0"), (".", "0.0.0")):
        array_path = tmp_path / f"volume{ord(separator)}.zarr" / "0"
        _write_array(array_path, separator=separator, chunk_keys=[key])
        _occupied(array_path)
        capsys.readouterr()
        assert _occupied(array_path) == {(0, 0, 0)}
        assert "cache HIT (sidecar" in capsys.readouterr().out
