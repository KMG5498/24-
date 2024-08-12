# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')  # Agg 백엔드 설정
import pickle
import numpy as np
import math
import random
import copy
import matplotlib.pyplot as plt
from datetime import datetime

import simulator as sim
import tree_class as tc
import data_generate as dg

import sys
sys.setrecursionlimit(10**8)

from joblib import Parallel, delayed

base_path = './data_train/9x30x5/9x30x5_%d.pickle'

num_iteration = 50
num_population = 100
num_elite = 4
num_parent = num_population-num_elite
num_child = int(num_parent/2)
num_tournament = 3
percentage_mutation = 0.5
population = []
tardiness_list = []
min_depth = 1
max_depth = 3
mutation_minimum, mutation_maximum = 1, 3
terminal_node_list = ['machine_available_time', 'job_prts_at_machine', 'job_due', 'is_there_setup', 'is_there_job_transfer', 'slack', 'is_there_resource_setup', 'is_there_resource_transfer']
function_node_list = ['+', '-', '*', 'neg', 'is_positive']
train_set = 50
test_set = 50

# 시뮬레이션을 병렬로 실행하는 함수 정의
def parallel_simulation(problem, individual):
    return sim.run_the_simulator(problem, individual)

if __name__ == "__main__":
    start_time = datetime.now()

    # 초기 population 생성
    population = tc.making_random_population(terminal_node_list, function_node_list, min_depth, max_depth, num_population)
    for_graph = []

    # 문제들을 미리 로드 (사전 로딩)
    problems = []
    for j in range(train_set):
        problem_path = base_path % j
        with open(problem_path, 'rb') as fr:
            problems.append(pickle.load(fr))

    # iteration 수 만큼 시뮬레이터 실행 + evolution
    for i in range(num_iteration):
        next_population = []

        # 병렬로 시뮬레이션 실행
        tardiness_list_for_tl = Parallel(n_jobs=-1)(
            delayed(parallel_simulation)(problem, ind) for problem in problems for ind in population
        )
        tardiness_list_for_tl = np.array(tardiness_list_for_tl).reshape(train_set, num_population)
        tardiness_list = np.mean(tardiness_list_for_tl, axis=0)

        print(i)

        # elitism
        elite_index = []
        elite_tardiness = []
        for tt in range(num_elite):
            elite_index.append(np.argmin(tardiness_list))
            elite_tardiness.append(tardiness_list[elite_index[tt]])
            if tt == 0:
                for_graph.append(elite_tardiness[0])
            tardiness_list[elite_index[tt]] = math.inf
        for tt in range(len(elite_index)):
            tardiness_list[elite_index[tt]] = elite_tardiness[tt]
            next_population.append(tc.copy_tree(population[elite_index[tt]], None))

        # parent selection with tournament k=num_tournament
        selected_parent = []
        parent_indices = []
        for _ in range(num_parent):
            tournament_candidates = random.sample(list(zip(population, tardiness_list)), num_tournament)
            selected_parent.append(tc.copy_tree(min(tournament_candidates, key=lambda x: x[1])[0], None))

        # mutation + crossover
        for klkl in range(num_child):
            first_parent = None
            second_parent = None
            child1 = None
            child2 = None
            first_parent = tc.copy_tree(selected_parent[random.randint(0, num_parent-1)], None)
            second_parent = tc.copy_tree(selected_parent[random.randint(0, num_parent-1)], None)
            if random.random() <= percentage_mutation:
                child1 = tc.mutation(first_parent, terminal_node_list, function_node_list, mutation_minimum, mutation_maximum)
                child2 = tc.mutation(second_parent, terminal_node_list, function_node_list, mutation_minimum, mutation_maximum)
            else:
                child1, child2 = tc.crossover(first_parent, second_parent)
            next_population.append(tc.copy_tree(child1, None))
            next_population.append(tc.copy_tree(child2, None))

        population = copy.deepcopy(next_population)

    # 평가 및 시각화
    tardiness_list = []
    tardiness_list_for_tl = Parallel(n_jobs=-1)(
        delayed(parallel_simulation)(problem, ind) for problem in problems for ind in population
    )
    tardiness_list_for_tl = np.array(tardiness_list_for_tl).reshape(train_set, num_population)
    tardiness_list = np.mean(tardiness_list_for_tl, axis=0)
    first_elite = np.argmin(tardiness_list)
    best_individual = population[first_elite]
    tc.print_tree(best_individual)
    end_time = datetime.now()

    elapsed_time = end_time - start_time
    print("코드 실행 시간: {}".format(elapsed_time))

    plt.plot(for_graph, marker='o') 
    plt.xlabel('Index') 
    plt.ylabel('Value')
    plt.title('Line Graph of for_graph')
    plt.grid(True)  
    plt.savefig('GP_train_curve.png')
    plt.show()  

    rules = ['GP', 'SPT', 'M-SPT', 'EDD', 'M-EDD', 'ATCS', 'M-ATCS']
    total_dict = {rule: [] for rule in rules}
    problem_path = base_path % 199
    with open(problem_path, 'rb') as fr:
        ex_problem = pickle.load(fr)
    print(ex_problem['job_type'])
    for rule_type in rules:
        print(rule_type)
        start_time = datetime.now()
        sim.run_the_simulator_last(ex_problem, best_individual if rule_type == 'GP' else rule_type)
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        print("코드 실행 시간: {}".format(elapsed_time))

    for iteration in range(test_set):
        iteration += 100
        problem_path = base_path % iteration
        with open(problem_path, 'rb') as fr:
            problem = pickle.load(fr)
        tardiness_list = []
        for rule_type in rules:
            total_dict[rule_type].append(sim.run_the_simulator(problem, best_individual if rule_type == 'GP' else rule_type))

    for rule_type in rules:
        print(rule_type, np.mean(total_dict[rule_type]))




