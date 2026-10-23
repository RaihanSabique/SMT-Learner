# Sample data

`MSL_sample.csv` and `UCPMP_sample.csv` are small extracts of the two study datasets,
included so the preprocessing, model and analysis code can be run without the full
release. They carry the complete column schema of the originals.

| File | Trials | Participants | Coverage |
|---|---|---|---|
| `MSL_sample.csv` | 20 | 2 (1 term, 1 preterm) | both task types, 5 trials each |
| `UCPMP_sample.csv` | 20 | 2 (1 TD, 1 UCP) | both hands, 5 trials each |

The full de-identified MSL and MP datasets are released on OSF. Both studies were
approved by the Marquette University IRB (#4376, #4667); participant identifiers are
pseudonymous and no directly identifying fields are present.

```python
import pandas as pd
from DataProcessing.Normalization import normalize_trajectory_sequence_3d

df = pd.read_csv("Dataset/MSL_sample.csv")
traj = normalize_trajectory_sequence_3d(df.loc[0, "path"], df.loc[0, "time_diff_ms"])  # (512, 3)
```

`MSL_SampleRawData/` holds the raw MATLAB exports consumed by
`MATLAB_Scripts/CreateTrajectoryDataset.m`.

Note: `Dataset/` is listed in `.gitignore`, so these files are tracked by explicit
`git add -f`. Use `-f` when adding anything else here.
