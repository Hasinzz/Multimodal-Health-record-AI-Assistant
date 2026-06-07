from src.datasets._shared import run_dataset_loader


def main() -> None:
    run_dataset_loader(
        dataset_name="synthea",
        default_dir="data/external/synthea",
        default_output="data/structured/synthea_index.csv",
    )


if __name__ == "__main__":
    main()
