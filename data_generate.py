import os
import random
import copy
import pickle
import numpy as np
from pathlib import Path

# 데이터 생성 함수 정의

def generate_symmetric_matrix(size, min_value, max_value):
    # Initialize a size x size matrix with zeros
    matrix = [[0] * size for _ in range(size)]

    # Fill in the upper triangle and mirror it to the lower triangle
    for i in range(size):
        for j in range(i, size):
            value = random.randint(min_value, max_value)  # Generate a random integer (e.g., between 0 and 10)
            matrix[i][j] = value
            matrix[j][i] = value
    
    for i in range(size):
        matrix[i][i] = 0

    return matrix

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
<<<<<<< HEAD
    #job 종류 생성
    job_type = [[] for _ in range(env_params['num_families'])]
    for i in range(env_params['num_families']):
        length = random.randint(env_params['min_job_length'], env_params['max_job_length'])
        for _ in range(length-1):
            job_type[i].append(random.sample(env_params['operation_type'], 1)[0])
        photo_index = random.randint(0, length)
        job_type[i].insert(photo_index, env_params['important_operation'])
    
    #job type list 생성
    job_type_list = []
    for _ in range(env_params['n_j']):
        job_type_list.append(random.randint(0, env_params['num_families']-1))

    #prts[job type][operation type][machine type]의 순서
    for_prts_op = copy.deepcopy(env_params['operation_type'])
    for_prts_op.append(env_params['important_operation'])
    machine_processing_times = [{i:[] for i in for_prts_op} for _ in range(env_params['num_families'])]
    for_mean = []
    for fa_type in range(env_params['num_families']):
        for op in for_prts_op:
            for _ in range (env_params['n_m']):
                machine_processing_times[fa_type][op].append(random.randint(env_params['low'], env_params['high']))
                for_mean.append(machine_processing_times[fa_type][op][-1])

    #ready time 생성
    average_prts = np.mean(for_mean)
    
    ready_times = [random.randint(0, int(average_prts/2)) for _ in range(env_params['n_j'])]
    #ready_times = [0 for _ in range(env_params['n_j'])]
    # due date 생성
    due_dates = [random.randint(int(ready_times[i]+(average_prts-ready_times[i])*(1-env_params['T']-env_params['R']/2)), int(ready_times[i]+(average_prts-ready_times[i])*(1-env_params['T']+env_params['R']/2))) for i in range(env_params['n_j']) ] 
    
    """
=======
    # machine-job 당 process time
    distribution_choice = random.choice(['uniform', 'gaussian', 'quasi-bimodal'])
    machine_processing_times = [[] for _ in range(env_params['n_m'])]
    for job_index in range(env_params['n_j']):
        for machine_index in range(env_params['n_m']):
            if distribution_choice == 'uniform':
                value = random.randint(0, 100)
            elif distribution_choice == 'gaussian':
                value = max(0, min(100, int(random.gauss(50, 15))))
            elif distribution_choice == 'quasi-bimodal':
                value = random.choice([random.randint(0, 50), random.randint(51, 100)])
            machine_processing_times[machine_index].append(value)

    job_weights = [random.uniform(0, 1) for _ in range(env_params['n_j'])]
    # ready time (= 이전 코드의 release time)
    #ready_times = [random.randint(0, 100) for _ in range(env_params['n_j'])]
    
    # p_hat 계산 후 ready time
    total_processing_time = sum(sum(times) for times in machine_processing_times)
    p_hat = total_processing_time / (env_params['n_m'] ** 2)
    ready_times = [random.randint(0, int(p_hat / 2)) for _ in range(env_params['n_j'])]
    #min_prts = [min(row[i] for row in machine_processing_times) for i in range(env_params['n_j'])]
    #average_prts = np.mean(min_prts)
    # due date
    #due_dates = [random.randint(env_params['due_low'], env_params['due_high']) for _ in range(env_params['n_j'])]
    due_dates = [random.randint(int(ready_times[i] + (p_hat - ready_times[i]) * (1 - env_params['T'] - env_params['R'] / 2)),
                                int(ready_times[i] + (p_hat - ready_times[i]) * (1 - env_params['T'] + env_params['R'] / 2))) for i in range(env_params['n_j'])]

>>>>>>> 7f6a8c47228a8ab98d7ca77e91e01472a4293a61
    # family
    group = [[] for _ in range(env_params['num_families'])]
    for job_index in range(env_params['n_j']):
        family = random.choice(range(env_params['num_families']))
        group[family].append(job_index)
    family_group = {job: i for i, family in enumerate(group) for job in family}
    
    """
    # job requiring resource
    required_resource = [random.randint(0, env_params['num_resource_type']-1) for _ in range(env_params['num_families'])]

    # eligible machine set
    eligible_machines = [{i:[] for i in for_prts_op} for _ in range(env_params['num_families'])]
    for fa_type in range(env_params['num_families']):
        for op in for_prts_op:
            for _ in range (env_params['n_m']):
                eligible_machines[fa_type][op] = random.sample(range(env_params['n_m']), 2)


    # operation이 바뀔 때
    operation_change_time = {op: {opp: random.randint(env_params['etc_min'], env_params['etc_max']) for opp in for_prts_op} for op in for_prts_op}
    operation_change_time['A']['A'] = 0
    operation_change_time['B']['B'] = 0
    operation_change_time['Photo']['Photo'] = 0
    # job이 다른 머신으로 이동할 때
    job_transfer_time = generate_symmetric_matrix(env_params['n_m'], env_params['etc_min'], env_params['etc_max'])

    # job type이 바뀔 때
    job_change_time = generate_symmetric_matrix(env_params['num_families'], 0, env_params['max_setup'])
    # 리소스가 이동할 때
    transfer_time = generate_symmetric_matrix(env_params['n_m'], env_params['min_transfer'], env_params['max_transfer'])
    # 리소스 설치는 고정

    random_generate_data = {
        'job_type': job_type,
        'job_type_list': job_type_list,
        'machine_processing_times': machine_processing_times,
        'job_weights': job_weights,
        'ready_times': ready_times,
        'due_dates': due_dates,
        'eligible_machines': eligible_machines,
        'required_resource': required_resource,
        'transfer_time': transfer_time,
        'job_change_time' : job_change_time,
        'operation_change_time': operation_change_time,
        'job_transfer_time': job_transfer_time,
        'n_j': env_params['n_j'],
        'n_m': env_params['n_m'],
        'num_families': env_params['num_families'],
        'num_resource_type': env_params['num_resource_type'],
        'operation_type': env_params['operation_type'],
        'important_operation': env_params['important_operation'],
        'num_operation': env_params['num_operation']
    }
    return random_generate_data

# 메인 함수 - 데이터 생성
if __name__ == '__main__':
    # parameters
    env_params = {
        'n_j': 12,
        'n_m': 3,
        'low': 10,
        'high': 30,
<<<<<<< HEAD
        'num_families': 5,
        'num_resource_type': 5,
=======
        'num_families': 3,
        'num_resource_type': 15,
>>>>>>> 7f6a8c47228a8ab98d7ca77e91e01472a4293a61
        'min_transfer': 20,
        'max_transfer': 30,
        'T' : 0.2,
        'R': 0.2,
        'operation_type' : ['A', 'B'],
        'important_operation': 'Photo',
        'max_job_length': 5,
        'min_job_length': 2,
        'num_operation': 3,
        'max_setup': 5,
        'etc_min': 3,
        'etc_max': 7
    }

    base_dir = './'  # 로컬 디렉토리 설정

    make_train_data_dict(env_params, num_prob=200, base_dir=base_dir)


# 데이터 불러오기 예시
problem_path = './data_train/3x12x5/3x12x5_77.pickle'

with open(problem_path, 'rb') as fr:
    problem = pickle.load(fr)
    print(problem)
