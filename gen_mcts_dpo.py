from functools import lru_cache
import os
import json
from glob import glob
import random
import re
from tqdm import tqdm
from itertools import groupby
import numpy as np
from datetime import datetime

pattern = re.compile(r'\-?\d+\.\d+|\-?\d+')


@lru_cache(1024)
def extract_label(text: str) -> str:
    if '\n####' in text:
        text = text.split('\n####')[-1].replace(',', '')
    elif 'The answer is' in text:
        text = text.split('The answer is')[-1].replace(',', '')
    numbers = pattern.findall(text)
    if not numbers:
        return None
    return numbers[0]


def check(gt, ans):
    gt_label = extract_label(gt)
    ans_label = extract_label(ans)
    # print(gt_label,ans_label)
    if gt_label is None or ans_label is None:
        return False
    if ans_label == gt_label or abs(float(ans_label) - float(gt_label)) < 1e-5:
        return True
    else:
        return False


final_json_list = []


def get_json(query, good, bad, history=[]):
    return {'input': '', 'instruction': query, 'output': [good, bad], 'history': history}


def get_node_id(answers, ans):
    return answers.index(ans)


def get_oldest_father(answers, ans, childs):
    possible_fathers = []
    for possible_father in childs:
        if ans in childs[possible_father]:
            possible_fathers.append(possible_father)
    print(len(possible_fathers))
    possible_father_ids = []
    for possible_father in possible_fathers:
        possible_father_ids.append(get_node_id(answers, possible_father))
    return possible_fathers[possible_father_ids.index(min(possible_father_ids))]


def get_fathers(answers, ans, childs):
    possible_fathers = []
    for possible_father in childs:
        if ans in childs[possible_father]:
            possible_fathers.append(possible_father)
    return possible_fathers


def fix_loops(answers, fathers, childs):
    fathers_no_loop = {}
    for node in childs:
        for child in childs[node]:
            if child not in fathers_no_loop:
                fathers_no_loop[child] = node
            elif child in fathers_no_loop:
                # 保持最早的父节点
                if get_node_id(answers, node) < get_node_id(answers, fathers_no_loop[child]):
                    fathers_no_loop[child] = node

    for ans in answers:
        if ans not in fathers_no_loop:
            fathers_no_loop[ans] = None

    return fathers_no_loop


def collect_paths(answers, gt, fathers, childs):
    gold = []
    for answer in answers:
        if check(gt, answer):
            gold.append(answer)
    paths = []
    for g in gold:
        if g is None:
            continue
        path = [g, ]
        while g in fathers and g is not None:
            father = None
            for t in fathers:
                if t in path:
                    continue
                else:
                    father = t
            g = father
            if g is not None:
                path.append(g)
            else:
                break
        paths.append(path)
    return paths


def get_refined_ans(history_bank, hints_list, answer_list):
    hints_map = {}
    for ans in history_bank:
        if len(history_bank[ans]) > 2:
            hint = history_bank[ans][-3]
            hints_map[hint] = ans
    for hint in hints_list:
        if hint not in hints_map:
            for history in history_bank.values():
                if hint in history:
                    hints_map[hint] = history[history.index(hint) + 2]
                    break
    dummys = ["I Don't Know", "I can't understand this question.", "I can't help with this question.",
              "I don't know how to solve this question.", "I don't know the answer to this question.",
              "I don't know the answer to this question, sorry."]
    startpoint = 1
    for dummy in dummys:
        if dummy in answer_list:
            startpoint = answer_list.index(dummy) + 1
    for hint in hints_list:
        if hint not in hints_map:
            hints_map[hint] = answer_list[hints_list.index(hint) + startpoint]
    return hints_map


def collect_refine(paths, hints_reward_imp_bank, hints_map, structure_reward):
    re_hints_reward_imp_bank = {}
    for ans in hints_reward_imp_bank:
        if len(hints_reward_imp_bank[ans]) >= 2:
            re_hints_reward_imp_bank[ans] = []
            for hint_item in hints_reward_imp_bank[ans]:
                if len(hint_item) >= 2:
                    hint = hint_item[0]
                    _ = hint_item[1]

                    if ans not in structure_reward:
                        print(f"Warning: Reward for hint '{hint}' with answer '{ans}' not found in structure_reward")
                        continue

                    reward0 = structure_reward[ans]

                    refined_ans = hints_map[hint]

                    if refined_ans not in structure_reward:
                        print(
                            f"Warning: Reward for refined hint '{hint}' with answer '{refined_ans}' not found in structure_reward")
                        continue

                    reward1 = structure_reward[refined_ans]

                    re_hints_reward_imp_bank[ans].append([hint, reward1 - reward0])
            re_hints_reward_imp_bank[ans] = sorted(re_hints_reward_imp_bank[ans], key=lambda x: x[1], reverse=True)
            re_hints_reward_imp_bank[ans] = [random.choice(list(g))[0] for k, g in
                                             groupby(re_hints_reward_imp_bank[ans], key=lambda x: x[1])]

    return re_hints_reward_imp_bank


def refine_prompt(query, ans):
    q = f'Since we have a weak Answer, could you provide me with a relection or feedback to correct this answer better? Analyze this Answer Strictly and Critic, point out every flaw for ervery possible imperfect to minus every possible score!\nLet\'s think step by step.'
    return q


def rereward(paths, answers, gt, fathers, childs, gemma=0.9, ucb_bank=None):
    structue_reward = {}
    step_scores = {ans: [] for ans in answers}
    step_sentences = {ans: [] for ans in answers}

    def get_ucb_score(sentence):
        if ucb_bank and sentence in ucb_bank:
            return ucb_bank[sentence]
        return 1  # Default score if not found in ucb_bank

    for path in paths:
        for i, ans in enumerate(path):
            ucb_score = get_ucb_score(ans)
            step_scores[ans].append(ucb_score)
            step_sentences[ans].append(answers[max(0, i)])
            if len(path) > 8:
                structue_reward[ans] = gemma ** max(0, i - 8)
            else:
                structue_reward[ans] = ucb_score

    if not structue_reward:
        return {}, {}

    gemma2 = 0.5 * gemma
    root_reward = min(structue_reward.values()) * gemma if structue_reward else 0

    def get_reward(ans):
        if ans is None or ans in structue_reward:
            return structue_reward.get(ans, root_reward)
        if ans in fathers:
            parent = fathers[ans]
            if parent is None:
                structue_reward[ans] = root_reward * gemma2
            else:
                structue_reward[ans] = get_reward(parent) * gemma2
        return structue_reward[ans]

    for ans in answers:
        if ans in structue_reward:
            get_reward(ans)

    structue_reward = {k: v for k, v in structue_reward.items() if v > 0}
    step_scores = {k: v for k, v in step_scores.items() if k in structue_reward}
    step_sentences = {k: v for k, v in step_sentences.items() if k in structue_reward}

    # Printing out the contents of structure_reward for debugging
    print("Structure Rewards:")
    for key, value in structue_reward.items():
        print(f"Node: {key}\tReward: {value}")

    return structue_reward, step_scores, step_sentences


def pair_importance_sampling(rewards, actions, nums):
    if len(actions) < 2:
        print(f"Warning: Not enough actions for sampling. Actions: {actions}")
        return []

    weights = []
    action_pairs = []

    for i in range(len(actions)):
        for j in range(i + 1, len(actions)):
            reward_diff = abs(rewards[i] - rewards[j])
            weights.append(reward_diff)
            if rewards[i] >= rewards[j]:
                action_pairs.append([actions[i], actions[j]])
            else:
                action_pairs.append([actions[j], actions[i]])

    if not weights or sum(weights) == 0:
        print(f"Warning: All weights are zero. Using uniform distribution. Rewards: {rewards}")
        weights = [1] * len(action_pairs)

    weights = [weight / sum(weights) for weight in weights]
    action_pairs_index = list(range(len(action_pairs)))

    # Count non-zero weight entries and ensure sampling size is valid
    non_zero_weights_count = sum(1 for weight in weights if weight > 0)
    nums = min(nums, non_zero_weights_count)  # Ensure we don't sample more pairs than available non-zero weights
    if non_zero_weights_count == 0 or nums == 0:
        print(f"Warning: No valid action pairs available for sampling. Action pairs: {action_pairs}")
        return []

    sampled_actions_index = np.random.choice(action_pairs_index, size=nums, p=weights, replace=False)
    sampled_actions = [action_pairs[index] for index in sampled_actions_index]

    return sampled_actions


def process_file(file_path):
    print(f"Processing file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return []

    print(f"Data loaded. Answers list length: {len(data['answers_list'])}")

    data['fathers'] = fix_loops(data['answers_list'], data['fathers'], data['childs'])
    golden_paths = collect_paths(data['answers_list'], data['ground_truth'], data['fathers'], data['childs'])

    print(f"Number of golden paths: {len(golden_paths)}")
    if not golden_paths:
        print(f"No golden paths found for file: {file_path}")
        return []

    structue_reward, step_scores, step_sentences = rereward(golden_paths, data['answers_list'], data['ground_truth'],
                                                            data['fathers'], data['childs'])

    print(f"Structure reward calculated. Length: {len(structue_reward)}")
    if not structue_reward:
        print(f"No valid rewards found for file: {file_path}")
        return []

    hints_map = get_refined_ans(data['history_bank'], data['hints_list'], data['answers_list'])
    if not hints_map:
        print(f"No hints map found for file: {file_path}")
        return []

    print(f"Hints map created. Length: {len(hints_map)}")

    re_hints_reward_imp_bank = collect_refine(golden_paths, data['hints_reward_imp_bank'], hints_map, structue_reward)

    dpo_pairs = []
    for path in golden_paths:
        valid_path = [node for node in path if node in structue_reward]
        if len(valid_path) < 2:
            continue
        path_rewards = [structue_reward[node] for node in valid_path]
        if len(valid_path) == 2:
            good_step_list_score = []
            bad_step_list_score = []

            for good_list in step_sentences[valid_path[0]]:
                try:
                    good_step_list_score.append(data['ucb_bank'][good_list])
                except Exception as e:
                    good_step_list_score.append(data['to_explore_reward'][good_list][0])
            for bad_list in step_sentences[valid_path[-1]]:
                try:
                    bad_step_list_score.append(data['ucb_bank'][bad_list])
                except Exception as e:
                    bad_step_list_score.append(data['to_explore_reward'][bad_list][0])

            dpo_pairs.append({
                'query': data['query'],
                'good': valid_path[0],
                'bad': valid_path[-1],
                'good_score': structue_reward[valid_path[0]],
                'bad_score': structue_reward[valid_path[-1]],
                'good_step_list': step_sentences[valid_path[0]],
                'good_step_scores': good_step_list_score,
                'bad_step_list': step_sentences[valid_path[-1]],
                'bad_step_scores': bad_step_list_score
            })
            print(f"Appending pair: Good: {valid_path[0]} | Bad: {valid_path[-1]} | Good Steps: {step_sentences[valid_path[0]]} | Bad Steps: {step_sentences[valid_path[-1]]}")
        else:
            pairs_to_sample = min((len(valid_path) ** 2) // 2, len(valid_path) * (len(valid_path) - 1) // 2)
            pairs = pair_importance_sampling(path_rewards, valid_path, pairs_to_sample)
            for pair in pairs:
                good_step_list_score = []
                bad_step_list_score = []
                for good_list in step_sentences[pair[0]]:
                    try:
                        good_step_list_score.append(data['ucb_bank'][good_list])
                    except Exception as e:
                        good_step_list_score.append(data['to_explore_reward'][good_list][0])
                for bad_list in step_sentences[pair[1]]:
                    try:
                        bad_step_list_score.append(data["ucb_bank"][bad_list])
                    except Exception as e:
                        bad_step_list_score.append(data['to_explore_reward'][bad_list][0])
                dpo_pairs.append({
                    'query': data['query'],
                    'good': pair[0],
                    'bad': pair[1],
                    'good_score': structue_reward[pair[0]],
                    'bad_score': structue_reward[pair[1]],
                    'good_step_list': step_sentences[pair[0]],
                    'good_step_scores': good_step_list_score,
                    'bad_step_list': step_sentences[pair[1]],
                    'bad_step_scores': bad_step_list_score
                })
                print(f"Appending sampled pair: Good: {pair[0]} | Bad: {pair[1]} | Good Steps: {step_sentences[pair[0]]} | Bad Steps: {step_sentences[pair[1]]}")

    print(f"Generated {len(dpo_pairs)} DPO pairs for file: {file_path}")
    return dpo_pairs


# 创建一个函数来将 JSON 对象写入文件，每个对象一行
def write_json_objects(file_path, json_objects):
    with open(file_path, 'w') as f:
        f.write('[\n')
        flag_write = 0
        # 将每个对象转换为格式化字符串并写入文件
        for obj in json_objects:
            if flag_write == 0:
                flag_write = 1
            else:
                f.write(',\n')
            json.dump(obj, f)
        f.write('\n]\n')


# # 随机打乱列表
# random.shuffle(final_json_list)
# print(len(final_json_list))

# # 写入前百分之一的 JSON 对象，每个对象一行
# output_file_path = 'data_mistral7b_pathfinder_new_mcts_answers_10_percent.json'
# write_json_objects(output_file_path, final_json_list[:len(final_json_list) // 2])

def create_output_directory(base_path, dataset_name):
    """创建输出目录"""
    output_dir = os.path.join(base_path, 'output', dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_output_filename(output_dir, prefix):
    """生成带有时间戳的输出文件名"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return os.path.join(output_dir, f"{prefix}_dpo_pairs_with_scores_{timestamp}.json")


def main():
    data_folders = [
        './AIME-gpt-4o-mini-mcts-2-20240719054541/jsons'
    ]

    for folder in data_folders:
        print(f"Processing folder: {folder}")
        dataset_name = os.path.basename(os.path.dirname(folder))
        output_dir = create_output_directory('.', dataset_name)

        all_dpo_pairs = []

        for file in tqdm(glob(os.path.join(folder, '*.json'))):
            dpo_pairs = process_file(file)
            all_dpo_pairs.extend(dpo_pairs)

        # 随机打乱并取前10%
        random.shuffle(all_dpo_pairs)
        selected_pairs = all_dpo_pairs[:len(all_dpo_pairs) // 10]

        # 保存结果
        output_file = get_output_filename(output_dir, dataset_name)
        with open('data_mistral7b_pathfinder_new_mcts_answers_10_percent.json', 'w', encoding='utf-8') as f:
            json.dump(selected_pairs, f, indent=2)

        print(f"Processed {len(all_dpo_pairs)} total pairs for {dataset_name}")
        print(f"Selected {len(selected_pairs)} pairs (10%)")
        print(f"Results saved to {output_file}")

        # 打印一些示例结果
        for item in selected_pairs[:5]:
            print(f"Query: {item['query']}")
            print(f"Good answer: {item['good']}")
            print(f"  Final score: {item['good_score']}")
            print(f"  Step scores: {item['good_step_scores']}")
            print(f"Bad answer: {item['bad']}")
            print(f"  Final score: {item['bad_score']}")
            print(f"  Step scores: {item['bad_step_scores']}")
            print(f"Good step list: {item['good_step_list']}")
            print(f"Bad step list: {item['bad_step_list']}")
            if 'intermediate_scores' in item:
                print(f"Intermediate scores: {item['intermediate_scores']}")
            print("---")


if __name__ == "__main__":
    main()


