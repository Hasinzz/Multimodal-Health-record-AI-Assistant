from src.datasets._shared import run_dataset_loader


def main() -> None:
    run_dataset_loader(
        dataset_name="multicare",
        default_dir="data/external/multicare",
        default_output="data/structured/multicare_index.csv",
    )


if __name__ == "__main__":
    main()
