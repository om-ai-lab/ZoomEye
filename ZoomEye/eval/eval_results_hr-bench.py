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
        if info['category'] not in all_acc:
            all_acc[info['category']] = []
            all_pop[info['category']] = []
        for ans, cho in zip(info['answer'], info['output']):
            x = ''
            if len(cho) == 1:
                x = cho[0]
            else:
                for c in cho:
                    if c in ['A', 'B', 'C', 'D']:
                        x = c
                        break
            all_acc[info['category']].append(ans == x)
        if 'num_pop' not in info:
            info['num_pop'] = [0]
        all_pop[info['category']].extend(info['num_pop'])

    total_acc = []
    total_pop = []
    for category in all_acc:
        print(category)
        print("acc:", 100*np.mean(all_acc[category]))
        if len(all_pop[category]) == 0:
            all_pop[category].append(0)
        print("pop:", np.mean(all_pop[category]))
        total_acc.extend(all_acc[category])
        total_pop.extend(all_pop[category])
        print('='*50)

    print("total acc:", 100*np.mean(total_acc))
    print("total pop:", np.mean(total_pop))

    