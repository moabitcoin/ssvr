from pathlib import Path

import argparse

import ssvr.train

parser = argparse.ArgumentParser(prog="ssvr")
subcmd = parser.add_subparsers(title="commands", metavar="")
subcmd.required = True

Fmt = argparse.ArgumentDefaultsHelpFormatter

train = subcmd.add_parser("train", help="trains model on unlabaled dataset", formatter_class=Fmt)
train.add_argument("--dataset", type=Path, required=True, help="directory to unlabeled dataset")
train.add_argument("--lr", type=float, default=1e-4, help="learning rate for optimizer")
train.add_argument("--epochs", type=int, default=100, help="number of total epochs")
train.set_defaults(main=ssvr.train.main)

args = parser.parse_args()
args.main(args)
