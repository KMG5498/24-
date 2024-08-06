import pickle
import numpy as np
import math
import random
import copy

import simulator as sim
import tree_class as tc
import data_generate as dg

import sys
sys.setrecursionlimit(10**7)

base_path = './data_train/3x12x5/3x12x5_%d.pickle'

num_iteration = 5
num_population = 30
num_parent = num_population-2
num_child = int(num_parent/2)
num_tournament = 3
percentage_mutation = 0.9
population = []
tardiness_list = []
min_depth = 1
max_depth = 3
mutation_minimum, mutation_maximum = 1, 2
terminal_node_list = ['machine_available_time', 'job_prts_at_machine', 'job_due', 'is_there_setup', 'is_there_transfer', 'slack', 'is_there_resource_setup']
function_node_list = ['+', '-', '*', 'neg', 'is_positive']


if __name__ == "__main__":
    # 초기 population 생성
    population = tc.making_random_population(terminal_node_list, function_node_list, min_depth, max_depth, num_population)
    # iteration 수 만큼 시뮬레이터 실행 + evolution
    for i in range(num_iteration):
        next_population = []
        # 문제 변경 (train data 변경)
        problem_path = base_path % i
        with open(problem_path, 'rb') as fr:
            problem = pickle.load(fr)
        tardiness_list = []
        # 모든 individual에 대해 시뮬레이션 후 tardiness list 획득
        for individual in population:
            tardiness_list.append(sim.run_the_simulator(problem, individual)) # total tardiness를 리턴하도록 변경
        print(i)
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # elitism
        first_elite = np.argmin(tardiness_list)
        first_elite_tardiness = tardiness_list[first_elite]
        tardiness_list[first_elite] = math.inf
        second_elite = np.argmin(tardiness_list)
        tardiness_list[first_elite] = first_elite_tardiness
        next_population.append(tc.copy_tree(population[first_elite], None))
        next_population.append(tc.copy_tree(population[second_elite], None))

        # parent selction with tournament k=7
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
            #print(first_parent==second_parent)
            #tc.print_tree(first_parent)
            #print("p1")
            #tc.print_tree(second_parent)
            #print("p2")
            if random.random() <= percentage_mutation:
              child1 = tc.mutation(first_parent, terminal_node_list, function_node_list, mutation_minimum, mutation_maximum)
              child2 = tc.mutation(second_parent, terminal_node_list, function_node_list, mutation_minimum, mutation_maximum)
            else:
              child1, child2 = tc.crossover(first_parent, second_parent)
            #tc.print_tree(child1)
            #print("c1")
            #tc.print_tree(child2)
            #print("c2")
            next_population.append(tc.copy_tree(child1, None))
            next_population.append(tc.copy_tree(child2, None))

        population = copy.deepcopy(next_population)




    problem_path = './data_train/3x12x5/3x12x5_77.pickle'
    with open(problem_path, 'rb') as fr:
        problem = pickle.load(fr)
    tardiness_list = []
    for individual in population:
        tc.print_tree(individual)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        tardiness_list.append(sim.run_the_simulator(problem, individual)) # total tardiness를 리턴하도록 변경
    best_individual_index = np.argmin(tardiness_list)
    best_individual = population[best_individual_index]
    tc.print_tree(best_individual)
    """
    problem_path = './data_train/3x12x3/3x12x3.pickle'
    with open(problem_path, 'rb') as fr:
        problem = pickle.load(fr)
    """

<<<<<<< HEAD
    """
=======

>>>>>>> 9d06223ea5c9e90ad019f2c2c948074b01e71c7d
    rules = ['GP', 'SPT', 'EDD', 'LPT', 'FIFO', "CR", 'CO', 'ATCS', 'CUSTOM']
    total_dict = {
        'GP':[],
        'SPT':[],
        'LPT':[],
        'EDD':[],
        'FIFO':[],
        'CR':[],
        'CO':[],
        'ATCS':[],
        'CUSTOM':[]
    }
    """
    rules = ['GP']
    total_dict = {
        'GP':[]
    }
    """




    for rule_type in rules:
        print(rule_type)
        sim.run_the_simulator_last(problem, best_individual if rule_type=='GP' else rule_type)

    for iteration in range(30):
        iteration += 80
        problem_path = base_path % iteration
        with open(problem_path, 'rb') as fr:
            problem = pickle.load(fr)
        tardiness_list = []
        for individual in population:
            tardiness_list.append(sim.run_the_simulator(problem, individual)) # total tardiness를 리턴하도록 변경
        best_individual_index = np.argmin(tardiness_list)
        best_individual = population[best_individual_index]
        for rule_type in rules:
            total_dict[rule_type].append(sim.run_the_simulator(problem, best_individual if rule_type=='GP' else rule_type))


    for rule_type in rules:
        print(rule_type, np.mean(total_dict[rule_type]))



