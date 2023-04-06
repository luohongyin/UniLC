import os
import sys
import json
import argparse

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from collections import Counter

matplotlib.use('Agg')

parser = argparse.ArgumentParser(
    prog='python analysis.py',
    description='Performance analysis for unified language safety checking with LLMs.',
    epilog='Submit issues on Github for addtional help.'
)

parser.add_argument('-t', '--task', type=str, help = 'Language safety checking task.')
parser.add_argument('-m', '--mode', type=str, help = 'Prompting mode [zero | fp | cot].')
parser.add_argument('-n', '--exp-name', type=int, help = 'Name tag for the experiment log file.')

def evaluate(case_str):
    if 'Label: SUPPORTS' in case_str:
        label = 'SUPPORTS'
    else:
        label = 'REFUTES'
    
    if 'Prediction: SUPPORTS' in case_str:
        pred = 'SUPPORTS'
    else:
        pred = 'REFUTES'
    
    return float(label == 'REFUTES'), float(pred == 'REFUTES'), float(label == pred), float(label == pred and label == 'REFUTES')


def fact_class(case_str):
    if not ('Related' in case_str and 'fact:' in case_str):
        return 'None'
    keyword = case_str.split('Related ')[1].split(' fact:')[0]
    if len(keyword) > 20:
        keyword = 'None'
    return keyword


def load_eval_data(task = 'climate', mode = 'cot', exp_name = 'fact'):
    log_path = f'log/{task}_{mode}_check_{exp_name}.log'

    os.system(f'tail {log_path}')

    log_data = open(log_path).read()
    case_list = log_data.split('----------------------------------------------------\n\n')
    pred_list = [evaluate(x) for x in case_list]

    true_r = sum([x[0] for x in pred_list])
    pred_r = sum([x[1] for x in pred_list])
    crr_r = sum([x[3] for x in pred_list])

    rec = crr_r / true_r
    prec = crr_r / pred_r

    f1 = 2 * rec * prec / (rec + prec)
    return f1


def analyze_task_recog(task = 'climate', mode = 'base_few', exp_name = 'fact'):

    def count_crr(crr_list, task_list, target_task):
        count_list = [0, 0]
        for crr, task in zip(crr_list, task_list):
            if task != target_task:
                continue
            count_list[1 - int(crr)] += 1
        return np.array(count_list)

    log_data = open(f'log/{task}_{mode}_check_{exp_name}.log').read()
    
    case_list = log_data.split('----------------------------------------------------\n\n')
    
    crr_list = [evaluate(x)[2] for x in case_list]
    task_list = [fact_class(x) for x in case_list]

    task_top_list = [
        x[0] for x in Counter(task_list).most_common(10)
    ]

    weight_counts = {}

    for task_recog in task_top_list:
        task_crr = count_crr(crr_list, task_list, task_recog)
        
        acc = int(task_crr[0] / task_crr.sum() * 100)
        label = f'{task_recog}. acc. :{acc} %'
        weight_counts[label] = task_crr
    
    correctness = ('Correct', 'Incorrect')
    width = 0.8

    fig, ax = plt.subplots()
    plt.ion()

    bottom = np.zeros(2)

    for boolean, weight_count in weight_counts.items():
        p = ax.bar(correctness, weight_count, width = width, label=boolean, bottom=bottom)
        bottom += weight_count

    # ax.set_title("Number of penguins with above average body mass")
    # ax.legend(loc="upper right")
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.4, box.height])

    font = font_manager.FontProperties(size=12)

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), prop=font, frameon=False)

    plt.ioff()
    plt.savefig(f'images/{task}_{mode}_{split}_task_cls.pdf')
    plt.clf()


def compare(task):
    task = sys.argv[1]
    # task = 'health'

    pred_list_joint, case_list_joint = load_eval_data(task = task, split = 'fbias')
    pred_list_fact, case_list_fact = load_eval_data(task = task, split = 'moral4')

    num_case = len(case_list_joint)

    pred_selected = []
    case_j_selected = []
    case_f_selected = []

    for i in range(num_case):
        pred_j, crr_j = pred_list_joint[i]
        pred_f, crr_f = pred_list_fact[i]

        case_j = case_list_joint[i]
        case_f = case_list_fact[i]

        if crr_f and not crr_j:
            pred_selected.append(pred_j)
            case_j_selected.append(case_j)
            case_f_selected.append(case_f)
    
    print(Counter(pred_selected))

    open(f'log/diff_{task}_j.log', 'w').write(
        '\n\n'.join(case_j_selected)
    )
    
    open(f'log/diff_{task}_f.log', 'w').write(
        '\n\n'.join(case_f_selected)
    )


if __name__ == '__main__':
    args = parser.parse_args()

    f1 = load_eval_data(task = args.task, mode = args.mode, exp_name = args.exp_name)
    print(f'\nF1 = {f1}\n')