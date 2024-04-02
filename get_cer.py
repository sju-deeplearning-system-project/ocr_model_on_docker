import argparse
import os
import sys
import timeit
from datetime import datetime
import pandas as pd
from jiwer import cer


def parse_args(argv):
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    # set the argument formats

    parser.add_argument(
        '--label_data', '-label_data', required=True,
        help='file that has source, target, prediction corpus data (.csv, .tsv, .xlsx)')

    return parser.parse_args(argv[1:])


if __name__ == '__main__':
    args = parse_args(sys.argv)

    label_data = args.label_data

    print(f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] CER scoring started...")
    print(f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] Loading data...")
    file_type = label_data.split(f'/')[-1].split('.')[-1]

    if file_type == 'csv':
        df = pd.read_csv(f'{label_data}')

    elif file_type == 'tsv':
        df = pd.read_csv(f'{label_data}', sep='\t')

    elif file_type == 'xlsx':
        df = pd.read_excel(f'{label_data}', na_values='NaN')

    ground_truth = df['gt_text'].tolist()
    hypothesis = df['prediction'].tolist()

    print(f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] Calculating CER score...")

    cer_list = []
    for gt, prd in zip(df['gt_text'].tolist(), df['prediction'].tolist()):
        cer_list.append(cer(str(gt), str(prd)))

    df['cer'] = cer_list

    cer_mean = df['cer'].mean(axis=0)
    df.loc['Mean', 'cer'] = cer_mean

    print(f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] CER calculation is done!")

    result_file = f'{label_data.split(os.sep)[-1].strip().split(f".{file_type}")[0]}_cer_result_{datetime.today().strftime("%m%d%H%M%S")}.xlsx'
    result_path = f'data/cer/{result_file}'
    df.to_excel(result_path, index=False)
    print(f"\n[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] CER scoring completed... (Avg. CER: {cer_mean}) \n")