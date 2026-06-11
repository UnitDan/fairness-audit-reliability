# Data

This repository does not redistribute third-party datasets.  Use the helper script or download the files from the original providers.

## Adult

Source: UCI Machine Learning Repository, Adult dataset.

Expected local files:

```text
adult/adult.data
adult/adult.test
adult/adult.names
adult/old.adult.names
adult/Index
```

Download:

```bash
python scripts/download_data.py --adult
```

Citation:

```text
Becker, B. & Kohavi, R. (1996). Adult [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20
```

The UCI page lists the Adult dataset under CC BY 4.0.  Cite the dataset and follow the provider terms.

## MovieLens-1M

Source: GroupLens MovieLens 1M dataset.

Expected local files:

```text
ml-1m/users.dat
ml-1m/ratings.dat
ml-1m/movies.dat
```

Download:

```bash
python scripts/download_data.py --ml1m
```

Citation:

```text
F. Maxwell Harper and Joseph A. Konstan. 2015.
The MovieLens Datasets: History and Context.
ACM Transactions on Interactive Intelligent Systems 5, 4, Article 19.
https://doi.org/10.1145/2827872
```

The MovieLens 1M README states that the data may be used for research purposes but should not be redistributed without separate permission.  Keep the raw `ml-1m/` directory out of public Git history.
