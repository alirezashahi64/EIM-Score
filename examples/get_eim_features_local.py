import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.eim_combine_score_local_surface import EIM_Combine_Score_Local_Surface


def main(args):

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.dataset_csv_file)

    results = []

    print(f"🔄 Processing {len(df)} complexes (LOCAL)...")

    for pdbid, pk in tqdm(zip(df['PDBID'], df['pK']), total=len(df)):

        try:
            model = EIM_Combine_Score_Local_Surface(
                path=args.data_folder,
                pdbid=pdbid,
                kernel_type='exponential',
                kernel_tau=1.0,
                kernel_power=2.0,
                cutoff=7.0,
                isovalue=0.25
            )

            feat = model.get_features()

            if feat is None:
                continue

            feat = feat.flatten()
            feat = np.nan_to_num(feat)

            row = {'PDBID': pdbid, 'pK': pk}

            for i, val in enumerate(feat):
                row[f"f_{i}"] = val

            results.append(row)

        except:
            continue

    if len(results) == 0:
        print("❌ No features extracted")
        return

    df_out = pd.DataFrame(results)

    csv_path = os.path.join(args.out_dir, "features_local.csv")
    npz_path = os.path.join(args.out_dir, "features_local.npz")

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
    parser.add_argument("--data_folder", required=True)
    parser.add_argument("--out_dir", default="../features")

    args = parser.parse_args()

    main(args)
