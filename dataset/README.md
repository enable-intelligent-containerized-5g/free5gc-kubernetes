# DATASET

## Implementation

Run the next comands to create the **dataset.csv**:

```
python3 get-data.py -n free5gc -t 1h
python3 build-dataset.py -o output.csv -d data/ 
```

## Flags description

- **-n**: namespace name.
- **-t**: duration in hour or minutes (e.g., 1h for 1 hour or 30m for 30 minutes).
- **-o**: file to save the dataset.
- **-d**: directory to get the data to build the dataset.