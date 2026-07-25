#!/usr/bin/env python3
"""Download and prepare all 35 ArxivMathGradingBench papers.

For every annotated benchmark row this script stores the exact-version PDF,
the original arXiv source response, a safely extracted source tree, and a
single UTF-8 proof bundle containing every text source file with explicit file
boundaries. The generated manifest is consumed by ``prepare_dataset``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.arxiv_bench import (  # noqa: E402
    ARXIV_MATH_GRADING_BENCH,
    DEFAULT_ARXIV_DATA_DIR,
    MANIFEST_NAME,
)


PDF_URL = "https://arxiv.org/pdf/{arxiv_id}{version}"
SOURCE_URL = "https://export.arxiv.org/e-print/{arxiv_id}{version}"
USER_AGENT = (
    "pverify-ArxivMathGradingBench/1.0 "
    "(research dataset preparation; https://github.com/)"
)
TEXT_SUFFIXES = {
    ".tex", ".ltx", ".sty", ".cls", ".bib", ".bbl", ".bst",
    ".txt", ".md", ".cfg", ".def", ".clo",
}


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, minimum_size: int = 1) -> bool:
    if destination.is_file() and destination.stat().st_size >= minimum_size:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(5):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                with temporary.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            if temporary.stat().st_size < minimum_size:
                raise ValueError(f"response is only {temporary.stat().st_size} bytes")
            temporary.replace(destination)
            return True
        except Exception as exc:
            last_error = exc
            if temporary.exists():
                temporary.unlink()
            if attempt < 4:
                time.sleep(min(2 ** attempt, 8))
    assert last_error is not None
    raise last_error


def _validate_pdf(path: Path) -> None:
    with path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise ValueError(f"{path} is not a PDF")


def _safe_members(archive: tarfile.TarFile):
    for member in archive.getmembers():
        member_path = Path(member.name)
        if (
            member_path.is_absolute()
            or ".." in member_path.parts
            or member.issym()
            or member.islnk()
            or member.isdev()
        ):
            continue
        yield member


def _extract_source(archive_path: Path, destination: Path) -> None:
    marker = destination / ".complete"
    if marker.is_file():
        return
    destination.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(destination, members=_safe_members(archive))
    else:
        raw = archive_path.read_bytes()
        try:
            raw = gzip.decompress(raw)
        except gzip.BadGzipFile:
            pass
        (destination / "main.tex").write_bytes(raw)
    marker.write_text("ok\n", encoding="utf-8")


def _text_files(source_dir: Path) -> list[Path]:
    files = [
        path
        for path in source_dir.rglob("*")
        if path.is_file()
        and path.name != ".complete"
        and path.suffix.lower() in TEXT_SUFFIXES
    ]

    def priority(path: Path) -> tuple[int, str]:
        try:
            head = path.read_text(encoding="utf-8", errors="replace")[:100_000]
        except OSError:
            head = ""
        is_main = "\\documentclass" in head and "\\begin{document}" in head
        return (0 if is_main else 1, path.relative_to(source_dir).as_posix())

    return sorted(files, key=priority)


def _write_bundle(row: dict[str, Any], source_dir: Path, output: Path) -> list[str]:
    files = _text_files(source_dir)
    if not files:
        raise ValueError(f"No textual TeX sources found in {source_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(
            "% ArxivMathGradingBench complete source bundle\n"
            f"% arXiv: {row['arxiv_id']}{row['version']}\n"
            f"% title: {row.get('title_extracted_from_tex', '')}\n"
            "% Each source file is included verbatim between FILE markers.\n\n"
        )
        for path in files:
            name = path.relative_to(source_dir).as_posix()
            content = path.read_text(encoding="utf-8", errors="replace")
            handle.write(f"\n% ===== BEGIN FILE: {name} =====\n")
            handle.write(content)
            if not content.endswith("\n"):
                handle.write("\n")
            handle.write(f"% ===== END FILE: {name} =====\n")
    return [path.relative_to(source_dir).as_posix() for path in files]


def _load_rows(metadata_jsonl: Path | None) -> list[dict[str, Any]]:
    if metadata_jsonl:
        with metadata_jsonl.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    from datasets import load_dataset

    return list(load_dataset(ARXIV_MATH_GRADING_BENCH, split="train"))


def prepare(args: argparse.Namespace) -> None:
    root = args.out_dir.resolve()
    rows = _load_rows(args.metadata_jsonl)
    if len(rows) != 35 and not args.allow_nonstandard_count:
        raise ValueError(f"Expected 35 benchmark papers, received {len(rows)}")

    metadata_path = root / "metadata.jsonl"
    root.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    manifest: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, row in enumerate(rows, 1):
        arxiv_id = str(row["arxiv_id"])
        version = str(row.get("version") or "")
        if not version:
            failures.append(f"{arxiv_id}: missing benchmark version")
            print(f"[{index:02d}/{len(rows)}] {arxiv_id}: FAILED: missing version")
            continue
        key = f"{arxiv_id}{version}"
        pdf_path = root / "pdfs" / f"arXiv-{key}.pdf"
        archive_path = root / "source_archives" / f"arXiv-{key}.src"
        source_dir = root / "sources" / key
        proof_path = root / "proof_bundles" / f"arXiv-{key}.tex"
        try:
            changed = _download(
                PDF_URL.format(arxiv_id=arxiv_id, version=version),
                pdf_path,
                minimum_size=10_000,
            )
            _validate_pdf(pdf_path)
            if changed and args.delay:
                time.sleep(args.delay)
            changed = _download(
                SOURCE_URL.format(arxiv_id=arxiv_id, version=version),
                archive_path,
                minimum_size=100,
            )
            if changed and args.delay:
                time.sleep(args.delay)
            _extract_source(archive_path, source_dir)
            source_files = _write_bundle(row, source_dir, proof_path)
            prepared = dict(row)
            prepared.update({
                "pdf_path": _relative(pdf_path, root),
                "source_archive_path": _relative(archive_path, root),
                "source_dir": _relative(source_dir, root),
                "proof_path": _relative(proof_path, root),
                "source_files": source_files,
                "pdf_sha256": _sha256(pdf_path),
                "source_archive_sha256": _sha256(archive_path),
                "proof_sha256": _sha256(proof_path),
            })
            manifest.append(prepared)
            print(
                f"[{index:02d}/{len(rows)}] {key}: "
                f"{len(source_files)} text sources, ready"
            )
        except Exception as exc:
            failures.append(f"{key}: {type(exc).__name__}: {exc}")
            print(f"[{index:02d}/{len(rows)}] {key}: FAILED: {exc}")

    manifest_path = root / MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(f"Prepared {len(manifest)}/{len(rows)} papers; manifest: {manifest_path}")
    if failures:
        raise SystemExit("Preparation failures:\n" + "\n".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ARXIV_DATA_DIR)
    parser.add_argument(
        "--metadata-jsonl",
        type=Path,
        default=None,
        help="Use local metadata instead of downloading it from Hugging Face.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Delay after each network download (arXiv-friendly default: 3s).",
    )
    parser.add_argument("--allow-nonstandard-count", action="store_true")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
