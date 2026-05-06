import argparse

from cs336_scaling.client import get_budget, list_experiments


def main():
  p = argparse.ArgumentParser()
  sub = p.add_subparsers(dest="cmd", required=True)
  sub.add_parser("budget")
  sub.add_parser("experiments")
  args = p.parse_args()

  if args.cmd == "budget":
    print(get_budget().model_dump(mode="json"))
  elif args.cmd == "experiments":
    xs = [x.model_dump(mode="json") for x in list_experiments()]
    print(xs)


if __name__ == "__main__":
  main()
