import pickle
import numpy as np
import math
import random
import copy
import matplotlib.pyplot as plt


import simulator as sim
import tree_class as tc
import data_generate as dg

import sys
sys.setrecursionlimit(10**8)

base_path = './data_train/9x12x5/9x12x5_%d.pickle'

num_iteration = 30
num_population = 100
num_parent = num_population-2
num_child = int(num_parent/2)
num_tournament = 3
percentage_mutation = 0.5
population = []
tardiness_list = []
min_depth = 1
max_depth = 3
mutation_minimum, mutation_maximum = 1, 3
terminal_node_list = ['machine_available_time', 'job_prts_at_machine', 'job_due', 'is_there_setup', 'is_there_job_transfer', 'slack', 'is_there_resource_setup', 'Photo_indicator', 'is_there_resource_transfer']
function_node_list = ['+', '-', '*', 'neg', 'is_positive', 'is_Photo']
train_set = 50





if __name__ == "__main__":
    # 초기 population 생성
    population = tc.making_random_population(terminal_node_list, function_node_list, min_depth, max_depth, num_population)
    for_graph = []
    # iteration 수 만큼 시뮬레이터 실행 + evolution

    for i in range(num_iteration):
        next_population = []
        # 문제 변경 (train data 변경)
        tardiness_list = []
        tardiness_list_for_tl = []
        for j in range(train_set):
            tardiness_list_train = []
            problem_path = base_path % j
            with open(problem_path, 'rb') as fr:
                problem = pickle.load(fr)
            # 모든 individual에 대해 시뮬레이션 후 tardiness list 획득
            for individual in population:
                tardiness_list_train.append(sim.run_the_simulator(problem, individual)) # total tardiness를 리턴하도록 변경
            tardiness_list_for_tl.append(tardiness_list_train)
        tardiness_list_for_tl = np.array(tardiness_list_for_tl)
        tardiness_list = np.mean(tardiness_list_for_tl, axis=0)

        print(i)
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # elitism
        first_elite = np.argmin(tardiness_list)
        first_elite_tardiness = tardiness_list[first_elite]
        for_graph.append(first_elite_tardiness)
        tardiness_list[first_elite] = math.inf
        second_elite = np.argmin(tardiness_list)
        tardiness_list[first_elite] = first_elite_tardiness
        next_population.append(tc.copy_tree(population[first_elite], None))
        next_population.append(tc.copy_tree(population[second_elite], None))

        # parent selction with tournament k=num_tournament
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




    problem_path =  base_path % 199
    with open(problem_path, 'rb') as fr:
        problem = pickle.load(fr)
    tardiness_list = []
    for individual in population:
        tardiness_list.append(sim.run_the_simulator(problem, individual))
    best_individual_index = np.argmin(tardiness_list)
    best_individual = population[best_individual_index]
    tc.print_tree(best_individual)



    plt.plot(for_graph, marker='o') 
    plt.xlabel('Index') 
    plt.ylabel('Value')
    plt.title('Line Graph of for_graph')
    plt.grid(True)  
    plt.show()  


    """
    problem_path = './data_train/3x12x3/3x12x3.pickle'
    with open(problem_path, 'rb') as fr:
        problem = pickle.load(fr)
    """

    rules = ['GP', 'SPT', 'M-SPT', 'EDD', 'M-EDD', 'ATCS', 'M-ATCS']
    total_dict = {
        'GP':[],
        'SPT':[],
        'M-SPT':[],
        'EDD':[],
        'M-EDD':[],
        'ATCS':[],
        'M-ATCS':[]
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

    for iteration in range(50):
        iteration += 100
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
    averages = {key: np.mean(value) for key, value in total_dict.items()}


    labels = list(averages.keys())
    mean_values = list(averages.values())

    bars = plt.bar(labels, mean_values, color='skyblue')
    plt.xlabel('Categories') 
    plt.ylabel('Average Values') 
    plt.title('Average Values of Each Category') 
    plt.xticks(rotation=45)  
    plt.grid(axis='y')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')
    plt.show()

    selected_keys = ['GP', 'M-SPT', 'M-EDD', 'M-ATCS']
    indices = range(len(next(iter(total_dict.values())))) 
    plt.figure(figsize=(10, 6))
    for key in selected_keys:
        plt.plot(indices, total_dict[key], label=key)
    plt.title('Comparison of Selected Values by Index')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend() 
    plt.grid(True)
    plt.show()
