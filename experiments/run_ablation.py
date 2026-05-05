import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["acii", "cgc"], default="acii")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    distributions = ["iid", "noniid"]
    commands = []
    if args.kind == "acii":
        metrics = ["entropy", "std", "random"]
        for dist in distributions:
            for metric in metrics:
                commands.append(
                    [
                        "python",
                        "-m",
                        "experiments.train",
                        "profile=kaggle_matrix_shard",
                        "dataset=ham10000",
                        f"distribution={dist}",
                        "method=sl_acc",
                        f"acii.importance_metric={metric}",
                    ]
                )
    else:
        methods = ["sl_acc", "powerquant_sl", "easyquant"]
        for dist in distributions:
            for m in methods:
                commands.append(
                    [
                        "python",
                        "-m",
                        "experiments.train",
                        "profile=kaggle_matrix_shard",
                        "dataset=ham10000",
                        f"distribution={dist}",
                        f"method={m}",
                    ]
                )
    for cmd in commands:
        print(" ".join(cmd))
        if args.execute:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
