"""Command-line entry points; every pipeline stage can be run independently."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="MuJoCo pick-and-place imitation learning lab")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect", help="Collect scripted-expert demonstrations")
    p.add_argument("--output", type=Path, default=Path("artifacts/data/expert.npz"))
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--validation-episodes", type=int, default=32)
    p = sub.add_parser("train", help="Train a behavior-cloning policy")
    p.add_argument("--data", type=Path, default=Path("artifacts/data/expert.npz"))
    p.add_argument("--output", type=Path, default=Path("artifacts/checkpoints/bc256.pt"))
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("dagger", help="Aggregate corrective labels on learner-visited states")
    p.add_argument("--data", type=Path, default=Path("artifacts/data/expert.npz"))
    p.add_argument("--initial", type=Path, default=Path("artifacts/checkpoints/bc256.pt"))
    p.add_argument("--output", type=Path, default=Path("artifacts/checkpoints/dagger.pt"))
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--episodes-per-round", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("evaluate", help="Evaluate all baselines on paired held-out episodes")
    p.add_argument("--checkpoints", type=Path, default=Path("artifacts/checkpoints"))
    p.add_argument("--output", type=Path, default=Path("artifacts/results"))
    p.add_argument("--episodes", type=int, default=100)
    p = sub.add_parser("report", help="Build the experiment report and browser data bundle")
    p.add_argument("--root", type=Path, default=Path.cwd())
    p = sub.add_parser("render", help="Render an actual MuJoCo replay to GIF")
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--replay-id")
    p = sub.add_parser("serve", help="Serve the local dashboard and fresh policy rollouts")
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.command == "collect":
        from .data import collect

        collect(args.output, args.episodes, args.validation_episodes)
    elif args.command == "train":
        from .train import train

        train(args.data, args.output, episodes=args.episodes, epochs=args.epochs, seed=args.seed)
    elif args.command == "dagger":
        from .train import dagger

        dagger(
            args.data,
            args.initial,
            args.output,
            rounds=args.rounds,
            episodes_per_round=args.episodes_per_round,
            epochs=args.epochs,
            seed=args.seed,
        )
    elif args.command == "evaluate":
        from .evaluate import evaluate
        from .policy import Policy

        policies = {"random": "random", "expert": "expert"}
        for name in ("bc32", "bc256", "dagger"):
            path = args.checkpoints / f"{name}.pt"
            if path.exists():
                policies[name] = Policy.load(path)
        evaluate(policies, args.output, episodes=args.episodes)
    elif args.command == "report":
        from .report import build_report

        build_report(args.root)
    elif args.command == "render":
        from .report import render_replay

        render_replay(args.root, args.replay_id)
    elif args.command == "serve":
        from .server import serve

        serve(args.root, args.port)


if __name__ == "__main__":
    main()
