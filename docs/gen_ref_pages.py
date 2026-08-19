"""Генератор страниц документации для MkDocs.

Делает две вещи при каждой сборке:

1. Собирает `README.md` из папок модулей в раздел «Модули». README лежат
   рядом с кодом — так их правят вместе с ним, а не забывают в отдельной папке.
2. Создаёт страницу API-справочника на каждый Python-модуль пакета `mir`.

Ничего из этого не хранится в репозитории: справочник пересобирается
из исходников и потому не может разойтись с кодом.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

PACKAGE_DIR = Path("mir")
REFERENCE_DIR = Path("reference")

MODULE_READMES: dict[str, Path] = {
    "mir": Path("mir/README.md"),
    "common": Path("mir/common/README.md"),
    "ingest": Path("mir/ingest/README.md"),
    "vision": Path("mir/vision/README.md"),
    "audio": Path("mir/audio/README.md"),
    "fusion": Path("mir/fusion/README.md"),
    "notation": Path("mir/notation/README.md"),
    "export": Path("mir/export/README.md"),
    "core": Path("core/README.md"),
    "app": Path("app/README.md"),
    "config": Path("config/README.md"),
    "tests": Path("tests/README.md"),
    "data": Path("data/README.md"),
    "scripts": Path("scripts/README.md"),
}


def _copy_module_readmes() -> None:
    """Перенести README модулей в раздел «Модули»."""
    for name, source in MODULE_READMES.items():
        target = Path("modules") / f"{name}.md"
        if not source.exists():
            with mkdocs_gen_files.open(target, "w") as fd:
                fd.write(f"# {name}\n\nОписание модуля пока не написано.\n")
            continue
        with mkdocs_gen_files.open(target, "w") as fd:
            fd.write(source.read_text(encoding="utf-8"))
        mkdocs_gen_files.set_edit_path(target, Path("..") / source)


def _generate_api_reference() -> None:
    """Создать страницу справочника на каждый модуль пакета."""
    nav = mkdocs_gen_files.Nav()

    for source_path in sorted(PACKAGE_DIR.rglob("*.py")):
        module_path = source_path.relative_to(PACKAGE_DIR).with_suffix("")
        doc_path = source_path.relative_to(PACKAGE_DIR).with_suffix(".md")
        parts = tuple(module_path.parts)

        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
        elif parts[-1].startswith("_"):
            continue

        identifier = ".".join(("mir", *parts))
        full_doc_path = REFERENCE_DIR / doc_path
        nav[("mir", *parts)] = doc_path.as_posix()

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            fd.write(f"# `{identifier}`\n\n::: {identifier}\n")
        mkdocs_gen_files.set_edit_path(full_doc_path, Path("..") / source_path)

    with mkdocs_gen_files.open(REFERENCE_DIR / "SUMMARY.md", "w") as nav_file:
        nav_file.writelines(nav.build_literate_nav())


_copy_module_readmes()
_generate_api_reference()
