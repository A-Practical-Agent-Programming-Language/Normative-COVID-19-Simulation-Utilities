import sys
import pyarrow as pa

# Call this script with the dumped data frame as the first argument
df = pa.ipc.open_file(sys.argv[1]).get_batch(0).to_pandas()
print(df.loc[(df["coughing"] < 0) | (df["mask"] < 0) | (df["sdist"] < 0), ["n_contacts", "coughing", "mask", "sdist"]])
