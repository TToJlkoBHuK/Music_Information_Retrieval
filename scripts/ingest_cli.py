"""CLI для проверки этапа загрузки.

Пока нет графического интерфейса, конвейер гоняется отсюда.

Examples:
    Метаданные без скачивания::

        python -m scripts.ingest_cli probe "https://youtu.be/VIDEO_ID"

    Подготовка локального файла::

        python -m scripts.ingest_cli fetch ./video.mp4 -o ./out

    Скачивание через прокси::

        python -m scripts.ingest_cli fetch "https://youtu.be/ID" --proxy socks5://127.0.0.1:1080
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from mir.common.enums import Stage
from mir.common.errors import MirError
from mir.common.logging import setup_logging
from mir.config import load_config
from mir.ingest import MediaCache, detect_platform, ingest, normalize_url, probe

_BAR_WIDTH = 30


def _print_progress(stage: Stage, percent: float) -> None:
    filled = int(percent * _BAR_WIDTH)
    bar = "#" * filled + "." * (_BAR_WIDTH - filled)
    line = f"{stage.title_ru:22} [{bar}] {percent:5.1%}"
    end = "\n" if percent >= 1.0 else ""
    print(f"\r{line}", end=end, flush=True)
    if percent >= 1.0:
        sys.stdout.flush()


def _cmd_probe(args: argparse.Namespace) -> int:
    config = load_config(overrides=_overrides(args))
    info = probe(args.source, config)
    if args.json:
        print(json.dumps(info.__dict__, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"Название:    {info.title}")
        print(f"Площадка:    {info.platform.value}")
        print(f"Длительность:{info.duration:8.1f} с")
        if info.width:
            print(f"Разрешение:  {info.width}x{info.height} @ {info.fps} fps")
        if info.filesize_approx:
            print(f"Размер:      ~{info.filesize_approx / 1024**2:.1f} МБ")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    config = load_config(overrides=_overrides(args))
    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = ingest(
        args.source,
        config=config,
        work_dir=out_dir,
        progress=None if args.quiet else _print_progress,
        use_cache=not args.no_cache,
    )

    print()
    print(f"Видео:       {bundle.video_path}")
    print(f"Аудио:       {bundle.audio_path}")
    print(f"Название:    {bundle.title or '—'}")
    print(f"Длительность:{bundle.duration:8.2f} с")
    print(f"Кадры:       {bundle.width}x{bundle.height} @ {bundle.fps:.3f} fps")
    print(f"Всего кадров:{bundle.frame_count:8d}")
    print(f"Источник:    {bundle.platform.value}")
    return 0


def _cmd_url(args: argparse.Namespace) -> int:
    print(f"Площадка:    {detect_platform(args.source).value}")
    print(f"Нормализация:{normalize_url(args.source)}")
    print(f"Ключ кэша:   {MediaCache.key_for(args.source)}")
    return 0


def _cmd_check_proxy(args: argparse.Namespace) -> int:
    from mir.ingest.proxycheck import COMMON_PROXY_PORTS, check_proxy, scan_common_ports

    if args.address:
        host, _, port_text = args.address.rpartition(":")
        host = host.split("://")[-1] or "127.0.0.1"
        probe = check_proxy(host, int(port_text))
        if probe.reachable:
            print(f"Доступен: {probe.url}")
            return 0
        print(f"Недоступен: {host}:{port_text} — {probe.error}")
        return 1

    print("Проверка типичных портов VPN-клиентов...\n")
    found = scan_common_ports()

    if not found:
        print("Работающий прокси не найден.\n")
        print("Что делать:")
        print("  1. Переключите VPN-клиент из режима TUN в режим Proxy.")
        print("     Контейнер Docker не видит TUN-интерфейс хоста: у WSL2")
        print("     собственный сетевой стек, системный туннель на него не действует.")
        print("  2. Посмотрите в настройках клиента, на каком порту он слушает.")
        print("  3. Проверьте конкретный адрес:")
        print("     mir-ingest check-proxy --address 127.0.0.1:10808\n")
        print("Порты, которые проверялись:")
        for port, description in COMMON_PROXY_PORTS:
            print(f"  {port:>6}  {description}")
        return 1

    print("Найдены работающие адреса:\n")
    for probe in found:
        print(f"  {probe.url:<40} {probe.description}")
    print("\nИспользуйте так:")
    print(f'  ... fetch "ССЫЛКА" -o data/output --proxy {found[0].url}')
    return 0


def _cmd_cache(args: argparse.Namespace) -> int:
    config = load_config()
    cache = MediaCache(config.ingest.cache_path, config.ingest.cache_max_gb)
    if args.clear:
        freed = cache.clear(older_than_days=args.older_than)
        print(f"Освобождено {freed / 1024**2:.1f} МБ")
    else:
        print(f"Каталог:  {cache.root}")
        print(f"Записей:  {len(cache)}")
        print(f"Объём:    {cache.total_size / 1024**2:.1f} МБ")
    return 0


def _overrides(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    ingest_over: dict[str, object] = {}
    if getattr(args, "proxy", None):
        ingest_over["proxy"] = args.proxy
    if getattr(args, "max_height", None):
        ingest_over["max_height"] = args.max_height
    return {"ingest": ingest_over} if ingest_over else {}


def build_parser() -> argparse.ArgumentParser:
    """Собрать разбор аргументов командной строки."""
    parser = argparse.ArgumentParser(
        prog="mir-ingest",
        description="Этап загрузки: ссылка или файл → видео + аудио",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог")
    parser.add_argument("-q", "--quiet", action="store_true", help="без прогресса")
    sub = parser.add_subparsers(dest="command", required=True)

    p_probe = sub.add_parser("probe", help="метаданные без скачивания")
    p_probe.add_argument("source", help="ссылка или путь к файлу")
    p_probe.add_argument("--json", action="store_true", help="вывод в JSON")
    p_probe.add_argument("--proxy", help="прокси для yt-dlp")
    p_probe.set_defaults(func=_cmd_probe)

    p_fetch = sub.add_parser("fetch", help="скачать и подготовить материал")
    p_fetch.add_argument("source", help="ссылка или путь к файлу")
    p_fetch.add_argument("-o", "--output", default="./output", help="каталог результата")
    p_fetch.add_argument("--proxy", help="прокси для yt-dlp")
    p_fetch.add_argument("--max-height", type=int, help="ограничение разрешения")
    p_fetch.add_argument("--no-cache", action="store_true", help="игнорировать кэш")
    p_fetch.set_defaults(func=_cmd_fetch)

    p_url = sub.add_parser("url", help="разбор ссылки без обращения к сети")
    p_url.add_argument("source")
    p_url.set_defaults(func=_cmd_url)

    p_proxy = sub.add_parser("check-proxy", help="найти работающий прокси")
    p_proxy.add_argument("--address", help="проверить конкретный адрес, например 127.0.0.1:10808")
    p_proxy.set_defaults(func=_cmd_check_proxy)

    p_cache = sub.add_parser("cache", help="состояние кэша")
    p_cache.add_argument("--clear", action="store_true", help="очистить")
    p_cache.add_argument("--older-than", type=float, help="только записи старше N дней")
    p_cache.set_defaults(func=_cmd_cache)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Точка входа. Возвращает код завершения процесса."""
    args = build_parser().parse_args(argv)
    setup_logging(level=logging.DEBUG if args.verbose else logging.WARNING)

    try:
        result: int = args.func(args)
        return result
    except MirError as exc:
        print(f"\nОшибка: {exc.user_message}", file=sys.stderr)
        if args.verbose:
            print(f"Подробности: {exc.technical}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nПрервано пользователем", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
