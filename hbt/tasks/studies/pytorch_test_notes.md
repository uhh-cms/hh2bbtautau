# read via dask dataframe

```python
dd = maybe_import("dask.dataframe")

signal_dfs = dd.read_parquet(
    [
        t.path
        for collections in signal_targets
        for targets in collections.targets.values()
        for t in targets.values() 
    ]
)
background_dfs = dd.read_parquet(
    [
        t.path
        for collections in backgrounds_targets
        for targets in collections.targets.values()
        for t in targets.values() 
    ]
)
```

# read via dask awkward

```python
signal_daks = dak.from_parquet(
    signal_target_paths,
    split_row_groups=True,
)
```

pro:
- can read multiple parquet files w/o loading to memory
- can read only the columns we need
- option to split between row groups
con:
- when accessing single elements, there seems to be a lot of overhead/
  leaked memory
- each compute step takes time


# read via awkward array
```python
signal_daks = ak.from_parquet(
    signal_target_paths,
)
```

pro:
- can read multiple parquet files
- can read only the columns we need
- fast
con:
- eager loading of all data (problem?)
