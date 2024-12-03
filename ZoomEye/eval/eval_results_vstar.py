import argparse
import json
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", type=str, required=True)
    args = parser.parse_args()

    infos = []
    with open(args.answers_file, 'r') as f:
        for line in f:
            infos.append(json.loads(line))
    print("len:", len(infos))
    all_acc = {}
    all_pop = {}
    for info in tqdm(infos):
        if info['test_type'] not in all_acc:
            all_acc[info['test_type']] = []
            all_pop[info['test_type']] = []
        all_acc[info['test_type']].append(info['output']==0)
        # all_acc[info['test_type']].append(info['correct']==1)
        if 'num_pop' not in info:
            info['num_pop'] = [0]
        all_pop[info['test_type']].extend(info['num_pop'])

    total_acc = []
    total_pop = []
    for test_type in all_acc:
        print(test_type)
        print("acc:", 100*np.mean(all_acc[test_type]))
        print("pop:", np.mean(all_pop[test_type]))
        total_acc.extend(all_acc[test_type])
        total_pop.extend(all_pop[test_type])
        print('='*50)

    print("total acc:", 100*np.mean(total_acc))
    print("total pop:", np.mean(total_pop))

    