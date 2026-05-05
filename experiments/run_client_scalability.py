import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    methods = ["sl_acc", "splitfc", "powerquant_sl", "randtopk_sl"]
    distributions = ["iid", "noniid"]
    clients = [5, 10, 15, 20, 25]
    commands = []
    for dist in distributions:
        for n in clients:
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
                        "seed=0",
                        f"client.num_clients={n}",
                    ]
                )
    for cmd in commands:
        print(" ".join(cmd))
        if args.execute:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
