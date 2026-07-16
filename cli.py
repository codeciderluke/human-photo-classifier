"""Command-line interface for Human Photo Classifier.

Runs the same classification pipeline as the GUI, without any GUI dependency.

Examples:
    python cli.py ./photos ./sorted
    python cli.py ./photos ./sorted --device cuda --face --gender
    python cli.py ./photos ./sorted --model yolo11s.pt --confidence 0.15 --quiet
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys

from core import ClassifyConfig, PipelineError, run_classification


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="human-photo-classifier",
        description="Detect people in photos and sort them into folders.",
    )
    p.add_argument("source", help="Folder of images to scan.")
    p.add_argument("destination", help="Folder to copy results into.")
    p.add_argument("--model", default="yolo11n.pt", help="YOLO model (default: yolo11n.pt).")
    p.add_argument("--confidence", type=float, default=0.20,
                   help="Detection confidence 0-1 (default: 0.20).")
    p.add_argument("--image-size", type=int, default=960,
                   help="Inference image size (default: 960).")
    p.add_argument("--device", default="cpu",
                   help="cpu, cuda, or cuda:0 (default: cpu).")
    p.add_argument("--no-recursive", action="store_true",
                   help="Do not scan subfolders.")
    p.add_argument("--face", action="store_true",
                   help="Sort person photos into with_face / no_face.")
    p.add_argument("--gender", action="store_true",
                   help="Sort person photos by gender (male / female).")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="Show only a progress bar, not per-image lines.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    config = ClassifyConfig(
        source_folder=args.source,
        destination_folder=args.destination,
        model_name=args.model,
        confidence=args.confidence,
        image_size=args.image_size,
        device=args.device,
        recursive=not args.no_recursive,
        detect_face=args.face,
        detect_gender=args.gender,
    )

    # Ctrl+C requests a graceful stop (finish the current image, print summary).
    state = {"stop": False}
    signal.signal(signal.SIGINT, lambda *_: state.__setitem__("stop", True))

    def on_log(message: str) -> None:
        if not args.quiet:
            print(message)

    def on_progress(current: int, total: int) -> None:
        if args.quiet:
            pct = int(current / total * 100) if total else 0
            bar = "#" * (pct // 4)
            print(f"\r[{bar:<25}] {pct:3d}%  ({current}/{total})", end="", flush=True)

    try:
        summary = run_classification(
            config,
            on_progress=on_progress,
            on_log=on_log,
            should_stop=lambda: state["stop"],
        )
    except PipelineError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1

    if args.quiet:
        print()

    state_label = "Stopped" if summary.stopped else "Done"
    print(
        f"\n{state_label}: total {summary.total} | person {summary.with_person} | "
        f"others {summary.others} | damaged->others {summary.damaged} | "
        f"copied {summary.copied} | corrupt skipped {summary.corrupt} | "
        f"failed {summary.failed}"
    )
    if summary.category_counts:
        print("Category breakdown:")
        for category, count in sorted(summary.category_counts.items()):
            print(f"  {category or '(unclassified)'}: {count}")

    return 0


if __name__ == "__main__":
    # Required for frozen builds so a spawned worker process does not re-run
    # the whole program.
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(main())
