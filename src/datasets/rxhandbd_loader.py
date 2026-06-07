from src.datasets._shared import run_dataset_loader


def main() -> None:
    run_dataset_loader(
        dataset_name="rxhandbd",
        default_dir="data/external/rxhandbd",
        default_output="data/structured/rxhandbd_index.csv",
    )


if __name__ == "__main__":
    main()
