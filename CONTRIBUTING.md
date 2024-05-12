# Contributing
## Structure
### Data folder
It's separated into these sources:
- `external` - Data from third party sources.
- `interim` - Intermediate data that has been transformed.
- `processed` - The final, canonical data sets for modeling.
- `raw` - The original, immutable data dump.

## Coding guidelines
Project structure inspired by popular [cookiecutter project templates](https://drivendata.github.io/cookiecutter-data-science/)
### Files, Data and Paths Naming

- `[<dataname>_]filename` - filename.

    For example, `filename = "test.csv` or `cifar_filename = "Cifar100.tar.gz`.

- `[<dataname>_][<source>_]_path` - path to directory.

    For example, `cifar_external_data_path = Path("data/external/cifar100")`.

- `[<dataname>_][<source>_]file_path` - path to file.

    For example, `cifar_external_data_file_path = Path("data/external/cifar100/cifar100.csv")`.

`source` is the name of the subdirectories in data. Most of the time it's
`external`, `interim` or `processed`. In most cases though it's changed
depending on context:
- `external` is often omitted because it's usually the first step in working with
    data and it's quite obvious that it's external. For example, just
    `cifar_data_path`.
- `interim` is changed to the actual step in the processing to make it more
    clear: `cifar_with_my_classes_data_path`.
- `processed` is ok overall but like with `interim` it's always better to point out
    how or for what purpose it was processed: `cifar_for_autoencoder_data_path`.

It's tempting to make the filenames as rigid as possible for easier
autocomplete but in most cases readability is more important. Tools should help
us, not vice versa. For example, it's better to name `cleaned_cifar_data_path`
instead of `cifar_cleaned_data_path` if you think that first variant is more
accurate.

