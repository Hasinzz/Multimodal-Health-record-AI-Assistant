from src.datasets._shared import run_dataset_loader


def main() -> None:
    run_dataset_loader(
        dataset_name="openi",
        default_dir="data/external/openi",
        default_output="data/structured/openi_index.csv",
    )


if __name__ == "__main__":
    main()
