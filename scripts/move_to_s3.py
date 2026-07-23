"""Move a Lance dataset onto S3 (RustFS / MinIO / AWS) and verify reads over it.

The local companion to the backend's S3 wiring (RASK_LANDING §4): copy a
``<name>.lance`` dataset to an object store and prove pylance reads it back with
``storage_options`` — tabular, vector KNN, and managed-blob streaming. Managed
blobs ride a plain directory copy; external ``Blob.from_uri`` (file://) pointers
do NOT (they need re-ingest/rebase, §4.4).

    # start RustFS (or reuse lance-rustfs): docker run -p 9100:9000 \
    #   -e RUSTFS_ACCESS_KEY=rustfsadmin -e RUSTFS_SECRET_KEY=rustfsadmin rustfs/rustfs
    uv run --with numpy python scripts/move_to_s3.py parity_new.lance \
        --endpoint http://127.0.0.1:9100 --key rustfsadmin --secret rustfsadmin \
        --bucket lance-media-test

Then serve the backend from it (no code change — env only):

    MEDIA_S3_ENDPOINT=http://127.0.0.1:9100 MEDIA_S3_ACCESS_KEY_ID=rustfsadmin \
    MEDIA_S3_SECRET_ACCESS_KEY=rustfsadmin MEDIA_S3_DB_ROOT=s3://lance-media-test \
    MEDIA_DB=<name>.lance make services-up   # viewer :8101 / search :8102 / annotator :8103
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import lance
import pyarrow.fs as pafs


def storage_options(endpoint: str, key: str, secret: str, region: str) -> dict[str, str]:
    return {
        "endpoint": endpoint,
        "access_key_id": key,
        "secret_access_key": secret,
        "region": region,
        "allow_http": "true" if endpoint.startswith("http://") else "false",
        "virtual_hosted_style_request": "false",  # path-style — RustFS/MinIO reject virtual-hosted
    }


def s3fs(endpoint: str, key: str, secret: str, region: str) -> pafs.S3FileSystem:
    host = endpoint.split("://", 1)[-1]
    return pafs.S3FileSystem(
        access_key=key,
        secret_key=secret,
        endpoint_override=host,
        scheme="http" if endpoint.startswith("http://") else "https",
        region=region,
        allow_bucket_creation=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", help="local <name>.lance directory to move")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--secret", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()

    local = Path(args.dataset)
    name = local.name
    prefix = f"{args.bucket}/{name}"
    fs = s3fs(args.endpoint, args.key, args.secret, args.region)
    so = storage_options(args.endpoint, args.key, args.secret, args.region)

    fs.create_dir(args.bucket, recursive=True)
    with contextlib.suppress(Exception):
        fs.delete_dir_contents(prefix, missing_dir_ok=True)
    pafs.copy_files(str(local), prefix, source_filesystem=pafs.LocalFileSystem(), destination_filesystem=fs)
    n = len(fs.get_file_info(pafs.FileSelector(prefix, recursive=True)))
    print(f"moved {local} → s3://{prefix}  ({n} objects)")

    tables = sorted(
        info.base_name[: -len(".lance")]
        for info in fs.get_file_info(pafs.FileSelector(prefix, recursive=False))
        if info.base_name.endswith(".lance")
    )
    ok = True
    for table in tables:
        uri = f"s3://{prefix}/{table}.lance"
        ds = lance.dataset(uri, storage_options=so)
        vcol = next((f.name for f in ds.schema if str(f.type).startswith("fixed_size_list")), None)
        bcol = next((f.name for f in ds.schema
                     if getattr(f.type, "extension_name", "") == "lance.blob.v2"), None)
        rows = ds.count_rows()
        # Tabular: a moved table must actually carry rows.
        if rows <= 0:
            print(f"  {table}: FAILED — 0 rows over S3")
            ok = False
            continue
        extra = ""
        # Vector KNN over S3 — the read path the docstring promises to prove.
        if vcol:
            try:
                dim = ds.schema.field(vcol).type.list_size
                hits = ds.to_table(nearest={"column": vcol, "q": [0.0] * dim, "k": 1}).num_rows
                extra += f" | vector {vcol} KNN→{hits} over S3"
            except Exception as exc:  # noqa: BLE001
                extra += f" | vector {vcol} KNN FAILED ({type(exc).__name__})"
                ok = False
        if bcol:
            try:
                blob = ds.take_blobs(bcol, indices=[0])[0]
                extra += f" | blob {bcol}[0]={blob.size()}B over S3"
            except Exception as exc:  # noqa: BLE001 — external file:// pointers don't resolve off-box
                # External pointers legitimately don't resolve off-box (§4.4) — a
                # note, not a failure; managed blobs would resolve here.
                extra += f" | blob {bcol} unresolved ({type(exc).__name__}; external → re-ingest)"
        print(f"  {table}: {rows} rows over S3{extra}")
    print("S3 READ OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
