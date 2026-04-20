import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.eim_combine_score_global_surface import process_pdb_robust


def main(args):

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.dataset_csv_file)

    pdbid_to_pk = dict(zip(df['PDBID'], df['pK']))

    results = []

    print(f"🔄 Processing {len(df)} complexes (GLOBAL)...")

    for pdbid in tqdm(df['PDBID']):

        result = process_pdb_robust(pdbid)

        if result is not None:
            pdbid, features, pk = result

            row = {'PDBID': pdbid, 'pK': pk}

            for i, val in enumerate(features):
                row[f"f_{i}"] = val

            results.append(row)

    if len(results) == 0:
        print("❌ No features extracted")
        return

    df_out = pd.DataFrame(results)

    # save
    csv_path = os.path.join(args.out_dir, "features_global.csv")
    npz_path = os.path.join(args.out_dir, "features_global.npz")

    df_out.to_csv(csv_path, index=False)

    np.savez_compressed(
        npz_path,
        features=df_out.drop(columns=['PDBID', 'pK']).values,
        labels=df_out['pK'].values,
        pdbids=df_out['PDBID'].values
    )

    print(f"✅ Saved: {csv_path}")
    print(f"✅ Saved: {npz_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_csv_file", required=True)
    parser.add_argument("--out_dir", default="../features")

    args = parser.parse_args()

    main(args)
