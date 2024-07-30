import os
import random
import copy
import pickle
import numpy as np
from pathlib import Path

# 데이터 생성 함수 정의
def make_train_data_dict(env_params, num_prob, base_dir):
    current_path = Path(os.getcwd())
    params = copy.deepcopy(env_params)
    np.random.seed(1)
    for path in range(num_prob):
        problems_dict = generate_data_dict(env_params)
        dir_path = os.path.join(base_dir, "data_train", f"{params['n_m']}x{params['n_j']}x{params['num_families']}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_path = os.path.join(dir_path, f"{params['n_m']}x{params['n_j']}x{params['num_families']}_{path}.pickle")
        with open(file_path, 'wb') as f:
            pickle.dump(problems_dict, f, pickle.HIGHEST_PROTOCOL)

def make_eval_data_dict(env_params, num_prob, base_dir):
    current_path = Path(os.getcwd())
    params = copy.deepcopy(env_params)
    np.random.seed(1)
    for path in range(num_prob):
        problems_dict = generate_data_dict(env_params)
        dir_path = os.path.join(base_dir, "data_eval")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_path = os.path.join(dir_path, f"{params['n_m']}x{params['n_j']}x{params['num_families']}_{path}.pickle")
        with open(file_path, 'wb') as f:
            pickle.dump(problems_dict, f, pickle.HIGHEST_PROTOCOL)

def generate_data_dict(env_params):
    # machine-job 당 process time
    machine_processing_times = [[] for _ in range(env_params['n_m'])]
    for job_index in range(env_params['n_j']):
        for machine_index in range(env_params['n_m']):
            machine_processing_times[machine_index].append(random.randint(env_params['low'], env_params['high']))

    # ready time (= 이전 코드의 release time)
    ready_times = [random.randint(0, 100) for _ in range(env_params['n_j'])]

    # due date
    due_dates = [random.randint(env_params['due_low'], env_params['due_high']) for _ in range(env_params['n_j'])]

    # family
    group = [[] for _ in range(env_params['num_families'])]
    for job_index in range(env_params['n_j']):
        family = random.choice(range(env_params['num_families']))
        group[family].append(job_index)
    family_group = {job: i for i, family in enumerate(group) for job in family}

    # job requiring resource
    required_resource = [random.randint(0, env_params['num_resource_type']-1) for _ in range(env_params['n_j'])]

    # eligible machine set
    eligible_machines = [random.sample(range(env_params['n_m']), 3) for _ in range(env_params['n_j'])]

    random_generate_data = {
        'machine_processing_times': machine_processing_times,
        'ready_times': ready_times,
        'due_dates': due_dates,
        'family_group': family_group,
        'eligible_machines': eligible_machines,
        'required_resource': required_resource,
        'n_j': env_params['n_j'],
        'n_m': env_params['n_m'],
        'low': env_params['low'],
        'high': env_params['high'],
        'due_low': env_params['due_low'],
        'due_high': env_params['due_high'],
        'num_families': env_params['num_families'],
        'num_resource_type': env_params['num_resource_type']
    }
    return random_generate_data

# 메인 함수 - 데이터 생성
if __name__ == '__main__':
    # parameters
    env_params = {
        'n_j': 100,
        'n_m': 5,
        'low': 5,
        'high': 50,
        'due_low': 5,
        'due_high': 100,
        'num_families': 6,
        'num_resource_type': 15
    }

    base_dir = './'  # 로컬 디렉토리 설정

    make_train_data_dict(env_params, num_prob=200, base_dir=base_dir)

"""
# 데이터 불러오기 예시
problem_path = './data_train/5x100x6/5x100x6_77.pickle'

with open(problem_path, 'rb') as fr:
    problem = pickle.load(fr)
    print(problem)
"""
