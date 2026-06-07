from src.datasets._shared import run_dataset_loader


def main() -> None:
    run_dataset_loader(
        dataset_name="pmc_patients",
        default_dir="data/external/pmc_patients",
        default_output="data/structured/pmc_patients_index.csv",
    )


if __name__ == "__main__":
    main()
