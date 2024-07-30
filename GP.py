import pickle
import numpy as np
import math
import random

import simulator as sim
import tree_class as tc

import sys
sys.setrecursionlimit(10**7)

base_path = './data_train/5x100x6/5x100x6_%d.pickle'

num_iteration = 5
num_population = 50
num_parent = 48
num_child = 24
num_tournament = 1
percentage_mutation = 0.3
population = []
tardiness_list = []
min_depth = 1
max_depth = 4
mutation_minimum, mutation_maximum = 1, 3
terminal_node_list = ['machine_available_time', 'job_prts_at_machine', 'job_due', 'is_there_setup', 'is_there_transfer']
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
        next_population.append(tc.copy_tree(population[first_elite]))
        next_population.append(tc.copy_tree(population[second_elite]))

        # parent selction with tournament k=7
        selected_parent = []
        parent_indices = []
        for _ in range(num_parent):
            tournament_candidates = random.sample(list(zip(population, tardiness_list)), num_tournament)
            selected_parent.append(tc.copy_tree(min(tournament_candidates, key=lambda x: x[1])[0]))


        # mutation + crossover
        first_parent = None
        second_parent = None
        for klkl in range(num_child):
            child1 = None
            child2 = None
            first_parent = selected_parent[random.randint(0, num_parent-1)]
            second_parent = selected_parent[random.randint(0, num_parent-1)]
            if random.random() <= percentage_mutation:
              child1 = tc.mutation(first_parent, terminal_node_list, function_node_list, mutation_minimum, mutation_maximum)
              child2 = tc.mutation(second_parent, terminal_node_list, function_node_list, mutation_minimum, mutation_maximum)
            else:
              child1, child2 = tc.crossover(first_parent, second_parent)

            next_population.append(child1)
            next_population.append(child2)


        population = next_population




    problem_path = './data_train/5x100x6/5x100x6_77.pickle'
    with open(problem_path, 'rb') as fr:
        problem = pickle.load(fr)
    tardiness_list = []
    for individual in population:
        tardiness_list.append(sim.run_the_simulator(problem, individual)) # total tardiness를 리턴하도록 변경
    best_individual_index = np.argmin(tardiness_list)
    best_individual = population[best_individual_index]
    tc.print_tree(best_individual)
    """
    problem_path = './data_train/5x100x6/5x100x6_77.pickle'
    with open(problem_path, 'rb') as fr:
        problem = pickle.load(fr)
    """



    print('GP')
    sim.run_the_simulator_last(problem, best_individual) # last는 스케쥴까지 출력 + tardiness 출력
    print('SPT')
    sim.run_the_simulator_last(problem, 'SPT')
    print('EDD')
    sim.run_the_simulator_last(problem, 'EDD')
    print('LPT')
    sim.run_the_simulator_last(problem, 'LPT')
    print('FIFO')
    sim.run_the_simulator_last(problem, 'FIFO')
    print('CR')
    sim.run_the_simulator_last(problem, 'CR')
    print('CO')
    sim.run_the_simulator_last(problem, 'CO')
    print('ATCS')
    sim.run_the_simulator_last(problem, 'ATCS')

    total_dict = {
        'GP':[],
        'SPT':[],
        'LPT':[],
        'EDD':[],
        'FIFO':[],
        'CR':[],
        'CO':[],
        'ATCS':[]
    }
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
        total_dict['GP'].append(sim.run_the_simulator(problem, best_individual))
        total_dict['SPT'].append(sim.run_the_simulator(problem, 'SPT'))
        total_dict['LPT'].append(sim.run_the_simulator(problem, 'LPT'))
        total_dict['EDD'].append(sim.run_the_simulator(problem, 'EDD'))
        total_dict['FIFO'].append(sim.run_the_simulator(problem, 'FIFO'))
        total_dict['CR'].append(sim.run_the_simulator(problem, 'CR'))
        total_dict['CO'].append(sim.run_the_simulator(problem, 'CO'))
        total_dict['ATCS'].append(sim.run_the_simulator(problem, 'ATCS'))

    print('GP', np.mean(total_dict['GP']))
    print('SPT', np.mean(total_dict['SPT']))
    print('LPT', np.mean(total_dict['LPT']))
    print('EDD', np.mean(total_dict['EDD']))
    print('FIFO', np.mean(total_dict['FIFO']))
    print('CR', np.mean(total_dict['CR']))
    print('CO', np.mean(total_dict['CO']))
    print('ATCS', np.mean(total_dict['ATCS']))

