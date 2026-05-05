import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    datasets = ["ham10000", "mnist", "cifar100"]
    distributions = ["iid", "noniid"]
    methods = ["sl_acc", "splitfc", "powerquant_sl", "randtopk_sl"]
    commands = []
    for d in datasets:
        for dist in distributions:
            for m in methods:
                commands.append(
                    [
                        "python",
                        "-m",
                        "experiments.train",
                        "profile=kaggle_matrix_shard",
                        f"dataset={d}",
                        f"distribution={dist}",
                        f"method={m}",
                        "seed=0",
                        "client.num_clients=5",
                    ]
                )
    for cmd in commands:
        print(" ".join(cmd))
        if args.execute:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
