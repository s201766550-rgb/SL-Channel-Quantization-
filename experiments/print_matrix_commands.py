def main() -> None:
    datasets = ["ham10000", "mnist", "cifar100"]
    distributions = ["iid", "noniid"]
    methods = ["sl_acc", "splitfc", "powerquant_sl", "randtopk_sl"]
    seeds = [0]
    for d in datasets:
        for dist in distributions:
            for m in methods:
                for s in seeds:
                    print(
                        "python -m experiments.train "
                        f"profile=kaggle_matrix_shard dataset={d} distribution={dist} method={m} seed={s}"
                    )


if __name__ == "__main__":
    main()
