"""Command line: python -m straightrows {submit,evaluate,report,robustness,validate}."""
import argparse
import json
import sys
from pathlib import Path

from . import data, pipeline
from .constants import DEV_END, DEV_START, EVAL_END, EVAL_START


def _windows(name):
    return {"eval": (EVAL_START, EVAL_END), "dev": (DEV_START, DEV_END)}[name]


def cmd_submit(args):
    ds = data.load(args.data)
    decisions, _ = pipeline.analyse(ds, EVAL_START, EVAL_END)
    sub = pipeline.submission(decisions, args.github_handle, args.repo_url, args.display_name)
    Path(args.output).write_text(json.dumps(sub, indent=1) + "\n")
    counts = {}
    for d in decisions:
        counts[d.cls] = counts.get(d.cls, 0) + 1
    print(f"Wrote {len(decisions)} detections to {args.output}: {counts}")
    for d in decisions:
        if d.cls != "MULTIPATH":
            print(f"  {d.start:%H:%M:%S}-{d.end:%H:%M:%S}  {d.cls:16} {','.join(d.tractors):9} conf {d.confidence:.2f}")


def cmd_evaluate(args):
    ds = data.load(args.data)
    result, _ = pipeline.evaluate_dev(ds, use_sda=not args.farm_only)
    for ep in result["episodes"]:
        status = ep.get("predicted_class", "MISSED")
        error = f'{ep["onset_error_s"] / 60:+.1f} min' if "onset_error_s" in ep else "-"
        print(f'{ep["episode"]:8} {ep["truth_class"]:16} {status:16} {error:>10}  action ok={ep["action_agreement"]}')
    for c, m in result["class_metrics"].items():
        print(f'{c:16} P={m["precision"]:.3f} R={m["recall"]:.3f} F1={m["f1"]:.3f}')
    print(f'DEV DIAGNOSTIC: {result["diagnostic_score"]:.2f}/40  macro-F1 {result["macro_f1"]:.3f}  '
          f'onset credit {result["mean_timing_credit"]:.3f}  false positives {len(result["false_positives"])}  '
          f'benign multipath {result["benign_multipath_reports"]}')
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=1, default=str))


def cmd_report(args):
    from . import report
    ds = data.load(args.data)
    start, end = _windows(args.window)
    decisions, sig = pipeline.analyse(ds, start, end)
    output = Path(args.output or f"out/report_{args.window}.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.write(output, ds, sig, decisions, start, end, args.window)
    print(f"Wrote {output}")


def cmd_robustness(args):
    from . import robustness
    ds = data.load(args.data)
    robustness.run(ds, Path(args.output))


def cmd_validate(args):
    from .official_validator import main
    sys.exit(main(args.path))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="straightrows", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("submit", help="analyse the evaluation window and write submission.json")
    p.add_argument("--data", default="data")
    p.add_argument("--output", default="submission.json")
    p.add_argument("--github-handle", required=True)
    p.add_argument("--repo-url", required=True)
    p.add_argument("--display-name")
    p.set_defaults(func=cmd_submit)
    p = sub.add_parser("evaluate", help="score the labelled development window with the official scorer")
    p.add_argument("--data", default="data")
    p.add_argument("--json")
    p.add_argument("--farm-only", action="store_true", help="ablation: ignore SDA feeds in the regional model")
    p.set_defaults(func=cmd_evaluate)
    p = sub.add_parser("report", help="write the self-contained HTML operator report")
    p.add_argument("--data", default="data")
    p.add_argument("--window", choices=["eval", "dev"], default="eval")
    p.add_argument("--output", help="default out/report_<window>.html")
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("robustness", help="threshold sensitivity, ablations and noise injection on dev")
    p.add_argument("--data", default="data")
    p.add_argument("--output", default="out/robustness.json")
    p.set_defaults(func=cmd_robustness)
    p = sub.add_parser("validate", help="run the official submission validator")
    p.add_argument("path")
    p.set_defaults(func=cmd_validate)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
