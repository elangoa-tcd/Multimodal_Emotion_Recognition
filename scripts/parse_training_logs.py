#!/usr/bin/env python3
"""Turn emotion training logs (the stdout of train.py) into CSV files.

For each log it writes
  results/training/<run>_history.csv     one row per epoch: train loss, dev loss/macro-F1,
                                         test loss/macro-F1, new-best flag, learning-rate changes
  results/training/<run>_final_test.csv  the last per-emotion test table in the log, i.e. the
                                         best-on-dev model's test scores reported in the dissertation

Usage (from the repo root):
  python scripts/parse_training_logs.py test_results_final/*.log
Logs that contain no emotion training epochs (sentiment, ablation, probe logs) are skipped.
"""
import argparse
import csv
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NAME = re.compile(r"'name': '([^']+)'")
EPOCH = re.compile(r"^Epoch (-?\d+) - Training loss: ([\d.]+)")
DEV = re.compile(r"^\[dev\] macro-F1: ([\d.]+)\s+macro-Recall: ([\d.]+)")
VAL = re.compile(r"^val loss: ([\d.]+)")
TEST = re.compile(r"^test loss: ([\d.]+), test acc: ([\d.]+)")
LR = re.compile(r"Current learning rate: ([\d.eE-]+)")
ROW = re.compile(r"^(happy|sad|anger|surprise|disgust|fear)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")
MACRO = re.compile(r"^MACRO\s+([\d.]+)\s+([\d.]+)\s*$")


def parse(path):
    name, history, table, last_table = None, [], [], None
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            if name is None and (m := NAME.search(line)):
                name = m.group(1)
            if m := EPOCH.match(line):
                history.append({'epoch': int(m.group(1)), 'train_loss': float(m.group(2)),
                                'dev_loss': '', 'dev_macro_f1': '', 'dev_macro_recall': '',
                                'test_loss': '', 'test_macro_f1': '', 'new_best_on_dev': False, 'lr': ''})
            elif history and (m := DEV.match(line)):
                history[-1]['dev_macro_f1'] = float(m.group(1))
                history[-1]['dev_macro_recall'] = float(m.group(2))
            elif history and (m := VAL.match(line)):
                history[-1]['dev_loss'] = float(m.group(1))
            elif history and (m := TEST.match(line)):
                history[-1]['test_loss'] = float(m.group(1))
                history[-1]['test_macro_f1'] = float(m.group(2))
            elif history and line.startswith('Found new best model on dev set'):
                history[-1]['new_best_on_dev'] = True
            elif history and (m := LR.search(line)):
                history[-1]['lr'] = float(m.group(1))   # set when patience runs out and the LR is halved

            if m := ROW.match(line):
                e, sup, acc, prec, rec, f1 = m.groups()
                table.append({'emotion': e, 'support': int(sup), 'accuracy': float(acc),
                              'precision': float(prec), 'recall': float(rec), 'f1': float(f1)})
            elif (m := MACRO.match(line)) and len(table) == 6:
                table.append({'emotion': 'MACRO', 'support': '', 'accuracy': '', 'precision': '',
                              'recall': float(m.group(1)), 'f1': float(m.group(2))})
                last_table, table = table, []
    return name, history, last_table


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+', help='training log files')
    ap.add_argument('--out_dir', default=os.path.join(REPO_ROOT, 'results', 'training'))
    args = ap.parse_args()

    for path in args.logs:
        name, history, final = parse(path)
        if not history or final is None:
            print(f"skipped {path} (no emotion training epochs found)")
            continue
        run = name or os.path.splitext(os.path.basename(path))[0]
        print(f"{path} -> run '{run}': {len(history)} epochs, final macro-F1 {final[-1]['f1']:.3f}")
        write_csv(os.path.join(args.out_dir, f'{run}_history.csv'), history)
        write_csv(os.path.join(args.out_dir, f'{run}_final_test.csv'), final)


if __name__ == '__main__':
    main()
