"""Audit an anonymous source repository without embedding private identifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


TEXT_SUFFIXES = {
    ".bib",
    ".cfg",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".ps1",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def repository_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    )


def load_identity_terms(path: Path | None) -> list[str]:
    if path is None:
        return []
    terms = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    return [term for term in terms if term and not term.startswith("#")]


def audit_content(root: Path, identity_terms: list[str]) -> dict[str, int]:
    files = repository_files(root)
    require(files, "repository is empty")
    require(not any(path.is_symlink() for path in root.rglob("*")), "symlink found")

    cache_dirs = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir() and path.name in {"__pycache__", ".pytest_cache"}
    ]
    require(not cache_dirs, f"generated cache directories found: {cache_dirs}")

    oversized = [
        path.relative_to(root).as_posix()
        for path in files
        if path.stat().st_size > 20_000_000
    ]
    require(not oversized, f"unexpected file larger than 20 MB: {oversized}")

    email_pattern = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
    absolute_path_pattern = re.compile(
        r"(?i)(?<![A-Z0-9_])[A-Z]:(?:\\|/)|/(?:Users|home|root)/[^/\s]+"
    )
    remote_shell_pattern = re.compile(
        r"(?i)(?:\bssh\s+(?:-[A-Za-z]\s+\S+\s+)*\S+@|\bconnect\.[A-Za-z0-9.-]+)"
    )
    credential_pattern = re.compile(
        r"(?im)^\s*(?:api[_-]?key|access[_-]?token|password|passwd|secret)"
        r"\s*[:=]\s*['\"]?[^\s'\"]{8,}"
    )
    private_key_pattern = re.compile(
        "-----BEGIN " + "(?:RSA |EC |OPENSSH )?" + "PRIVATE KEY-----"
    )
    generic_hits: list[str] = []
    identity_hits: list[str] = []
    for path in files:
        if path.suffix.casefold() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8-sig")
        categories = []
        emails = {
            match.group(0).casefold() for match in email_pattern.finditer(text)
        } - {"anonymous@invalid.example"}
        if emails:
            categories.append("email")
        if absolute_path_pattern.search(text):
            categories.append("absolute-user-path")
        if remote_shell_pattern.search(text):
            categories.append("remote-shell-endpoint")
        if credential_pattern.search(text) or private_key_pattern.search(text):
            categories.append("credential-shape")
        generic_hits.extend(f"{relative}: {category}" for category in categories)
        folded = text.casefold()
        for index, term in enumerate(identity_terms, start=1):
            if term.casefold() in folded:
                identity_hits.append(f"{relative}: identity-term-{index}")

    require(not generic_hits, f"generic anonymity checks failed: {generic_hits}")
    require(not identity_hits, f"provided identity terms found: {identity_hits}")
    return {
        "files": len(files),
        "oversized_files": 0,
        "generic_hits": 0,
        "identity_term_hits": 0,
    }


def audit_manifest(root: Path) -> dict[str, int]:
    path = root / "release-manifest.json"
    require(path.is_file(), "release-manifest.json is absent")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("source_git_history_included") is False, "source history flag drifted")
    require(payload.get("title_page_included") is False, "title-page flag drifted")

    declared: set[str] = set()
    for record in payload.get("files", []):
        relative = Path(record["path"])
        require(not relative.is_absolute(), f"absolute manifest path: {relative}")
        candidate = root / relative
        require(candidate.is_file(), f"manifest file is absent: {relative}")
        require(sha256(candidate) == record["sha256"], f"hash drift: {relative}")
        declared.add(relative.as_posix())
    require(declared, "release manifest contains no files")

    actual = {
        path.relative_to(root).as_posix()
        for path in repository_files(root)
        if path.name != "release-manifest.json"
    }
    require(declared == actual, f"unmanifested or absent files: {sorted(declared ^ actual)}")
    return {"manifest_files": len(declared)}


def audit_git(root: Path) -> dict[str, object]:
    if not (root / ".git").exists():
        return {"git_initialized": False, "commits": 0, "remotes": 0}
    authors = subprocess.run(
        ["git", "log", "--format=%an <%ae>"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    require(
        set(authors) <= {"Anonymous Authors <anonymous@invalid.example>"},
        f"non-anonymous commit identity found: {sorted(set(authors))}",
    )
    remotes = subprocess.run(
        ["git", "remote", "-v"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    require(not remotes, f"repository remote may expose an account: {remotes}")
    return {
        "git_initialized": True,
        "commits": len(authors),
        "remotes": 0,
        "authors": sorted(set(authors)),
    }


def audit_media_metadata(root: Path) -> dict[str, int]:
    try:
        import fitz
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("media metadata audit requires the analysis dependencies") from error

    pdf_count = 0
    image_count = 0
    for path in repository_files(root):
        suffix = path.suffix.casefold()
        if suffix == ".pdf":
            pdf_count += 1
            document = fitz.open(path)
            require(
                not (document.metadata.get("author") or "").strip(),
                f"PDF author metadata is not empty: {path.relative_to(root)}",
            )
            require(document.xref_xml_metadata() == 0, f"PDF carries XMP: {path.relative_to(root)}")
            require(not document.embfile_names(), f"PDF carries attachments: {path.relative_to(root)}")
        elif suffix in {".png", ".jpg", ".jpeg"}:
            image_count += 1
            with Image.open(path) as image:
                metadata = {"info": image.info, "exif": dict(image.getexif())}
            require(not metadata["exif"], f"image carries EXIF: {path.relative_to(root)}")
    return {"pdf_files": pdf_count, "image_files": image_count, "metadata_hits": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument(
        "--identity-file",
        type=Path,
        help="optional newline-delimited private terms; keep this file outside the repository",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    result = {
        "content": audit_content(root, load_identity_terms(args.identity_file)),
        "manifest": audit_manifest(root),
        "media_metadata": audit_media_metadata(root),
        "git": audit_git(root),
        "status": "pass",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
