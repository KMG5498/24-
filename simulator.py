# @title 시뮬레이터 수정중  (GP 적용)
import math
#import dataloader
import random
import matplotlib.pyplot as plt

import tree_class as tc


class Job():
    def __init__(self, job_id, prts=0, due=0, machines=None, ready_time=0, job_family=[], required_resource=0):
        # static info
        self.id = job_id
        self.prts = prts  # processing times
        self.due = due
        self.machines = machines # eligible machine set 이거 인덱스 리스트임 !
        # 내가 추가한거
        self.ready_time = ready_time
        self.job_family = job_family # 이거는 group number임 !
        self.required_resource = required_resource

        # dynamic info
        self.now_remain_t = 0 # remaining time

class Machine():
    def __init__(self, machine_id):
        # static info
        self.id = machine_id

        # dynamic info
        self.available_time = 0
        self.processing_job = None

class Resource():
    def __init__(self, resource_id, resource_location):
        self.id = resource_id
        self.location = resource_location
        self.available_time = 0

class Simulator:
    def __init__(self, dataset = None):
        # input parameters
        self.num_job = dataset['n_j']# job 개수
        self.num_machine = dataset['n_m'] # machine 개수
        self.setup_time = 50
        self.total_tardiness = 0
        self.num_resource = 15
        self.resource_setup_time = 5
        self.transfer_time = []

        # static info
        self.machine_info = dict()
        self.job_info = dict()

        # dynamic info
        self.sim_time = 0
        self.available_job_list = list()
        self.completed_job_list = list()
        self.done = False

        self.job_info, self.machine_info, self.resource_info = self.load_data(dataset)

        # schedule info
        # 기존 self.schedule = dict.fromkeys(list(self.machine_info.keys()), list())
        self.schedule = {key: [] for key in self.machine_info.keys()}

    def load_data(self, dataset):
        """
        pickle 파일 및 environment parameter input에서 데이터 불러와서 job/machine class에 저장
        """
        # dataset이 env_params라고 가정
        # processing time을 job당으로 바꾸기
        processing_time_for_load_data = [[] for _ in range(self.num_job)]
        for i in range(self.num_job):
          for j in range(self.num_machine):
              processing_time_for_load_data[i].append(dataset['machine_processing_times'][j][i])
        # 룰 적용할때는 machine 당이 편할거 같아서 data generate 부분은 유지

        job_info = {i:Job(i, processing_time_for_load_data[i], dataset['due_dates'][i], dataset['eligible_machines'][i], dataset['ready_times'][i], dataset['family_group'][i], dataset['required_resource'][i]) for i in range(self.num_job) }
        machine_info = {j:Machine(j) for j in range(self.num_machine) }
        resource_info = {k:Resource(k, random.randint(0, self.num_machine-1)) for k in range(self.num_resource) }
        self.transfer_time = dataset['transfer_time']
        return job_info, machine_info, resource_info

    def reset(self):
        self.sim_time = 0
        for_reset = []
        for i in range(self.num_job):
          if self.job_info[i].ready_time == 0:
            for_reset.append(i)
        self.available_job_list = for_reset
        # ready time 0인 job들만 저장


    def get_available_job(self, available_machines):
        # ready time이 self.sim_time 이하인 job들만 가져오기
        for_available_job = []
        for i in range(self.num_job):
          if self.job_info[i].ready_time <= self.sim_time and self.job_info[i] not in self.completed_job_list and self.job_info[i].now_remain_t == 0 and self.resource_info[self.job_info[i].required_resource].available_time <= self.sim_time:
              jj = 0
              for j in available_machines:
                if j.id in self.job_info[i].machines:
                    jj += 1
              if jj > 0:
                for_available_job.append(self.job_info[i])
        self.available_job_list = for_available_job
        return self.available_job_list


    def get_next_time_step(self):
        """
        의사결정 가능한 다음 time step 구하기
        """
        # differnece인거 인지!!
        next_job_time = min([job.now_remain_t for job in self.job_info.values() if job.now_remain_t > 0])
        next_machine_time = min([machine.available_time for machine in self.machine_info.values() if machine.available_time >= self.sim_time])
        next_machine_time -= self.sim_time
        return min(next_job_time, next_machine_time)


    def step(self, action):
        """
        시뮬레이터 및 environment 정보를 실질적으로 업데이트해 주는 함수
        action: 내가 테스트하고자 하는 rule에 따라 결정 (SPT, LPT, EDD, FIFO, GP, RL, ...)
        """
        # 얘 액션 완성하고 schedule 부분수정
        job, machine, resource = action
        setup_time = self.if_setup(job, machine)
        if setup_time > 0:
            self.schedule[machine.id].append((Job('s'), self.sim_time, self.sim_time + setup_time))
        transfer_time = self.if_transfer(resource, machine)
        if transfer_time > 0:
            resource.location = machine.id
            self.schedule[machine.id].append((Job('t'), self.sim_time + setup_time, self.sim_time + setup_time + transfer_time))
        resource_setup_time = self.if_resource_setup(job, machine)
        if resource_setup_time > 0:
            self.schedule[machine.id].append((Job('rs'), self.sim_time + setup_time + transfer_time, self.sim_time + setup_time + transfer_time + resource_setup_time))
        job.now_remain_t = job.prts[machine.id] + setup_time + transfer_time + resource_setup_time
        machine.available_time += job.prts[machine.id] + setup_time + transfer_time + resource_setup_time
        resource.available_time += job.prts[machine.id] + setup_time + transfer_time + resource_setup_time
        tardiness = 0 if machine.available_time - job.due <= 0 else machine.available_time - job.due
        self.total_tardiness += tardiness
        #machine.resource -= job.required_resource
        self.schedule[machine.id].append((job, self.sim_time + setup_time + transfer_time + resource_setup_time, machine.available_time))
        #print(self.sim_time)
        #print(job.id)
        #print(machine.id)
        #print("~~~~~~~~~~~~~~~~~~~time indicator~~~~~~~~~~~~")


    def move_to_next_sim_t(self, passid):
        """
        시뮬레이터 time 및 job/machine의 dynamic info 업데이트
        """
        if passid == 0:
          time_diff = self.get_next_time_step()
        else:
          time_diff = 1
          for machine_index in range(self.num_machine):
              if self.machine_info[machine_index].available_time == self.sim_time:
                  self.machine_info[machine_index].available_time += 1
          for resource_index in range(self.num_resource):
              if self.resource_info[resource_index].available_time == self.sim_time:
                  self.resource_info[resource_index].available_time += 1
        self.sim_time += time_diff
        for job in list(self.job_info.values()):
            if job.now_remain_t > 0:
                job.now_remain_t -= time_diff # job remaining time 업데이트
                if job.now_remain_t == 0:
                    self.completed_job_list.append(job) # 완료된 job 업데이트


    def is_done(self):
        """
        return True if the simulation is done
        """
        # 수정할거 없을듯
        if len(self.completed_job_list) == self.num_job:
            #print(self.total_tardiness)
            return True
        return False

    def plot_gantt_chart(self):
        """
        Gantt Chart를 출력하는 함수
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        for machine_id, jobs in self.schedule.items():
            for job in jobs:
                job_id, start_time, end_time = job
                ax.barh(machine_id, end_time - start_time, left=start_time, edgecolor='black', align='center', alpha=0.8)
                ax.text(start_time + (end_time - start_time) / 2, machine_id, f'{job_id.id}', color='black', ha='center', va='center')

        ax.set_xlabel('Time')
        ax.set_ylabel('Machine')
        ax.set_title('Schedule')
        plt.show()

    def if_setup(self, job, machine):
        if len(self.schedule[machine.id])==0 or self.schedule[machine.id][-1][0].job_family == job.job_family:
          return 0
        else:
          return self.setup_time

    def if_transfer(self, resource, machine):
        if resource.location == machine.id:
            return 0
        else:
            transfer_time = self.transfer_time[resource.location][machine.id]
            return transfer_time

    def if_resource_setup(self, job, machine):
        if len(self.schedule[machine.id])==0 or self.schedule[machine.id][-1][0].required_resource == job.required_resource:
          return 0
        else:
          return self.resource_setup_time
        
def get_action(env, jobs, machines, method):
    """
    action selection
    """
    if method == 'SPT':  # shortest processing time
        job_index=0
        machine_index=0
        minimum_prts=math.inf
        for i in machines:
          for j in jobs:
            for jj in j.machines:
              if jj==i.id and j.prts[i.id] < minimum_prts:
                job_index = j.id
                machine_index = i.id
                minimum_prts = j.prts[i.id]
        selected_job = env.job_info[job_index]
        selected_machine = env.machine_info[machine_index]
        selected_machine.processing_job = selected_job
    elif method == "FIFO":
        job_index=0
        machine_index=0
        minimum_ready_time=math.inf
        for j in jobs:
          if j.ready_time < minimum_ready_time:
              job_index = j.id
              machine_index = random.choice(list(set(j.machines)&set([machine.id for machine in machines])))
              minimum_ready_time = j.ready_time
        selected_job = env.job_info[job_index]
        selected_machine = env.machine_info[machine_index]
        selected_machine.processing_job = selected_job
    elif method == 'EDD':
        job_index=0
        machine_index=0
        minimum_due_time=math.inf
        for j in jobs:
          if j.due < minimum_due_time:
              job_index = j.id
              machine_index = random.choice(list(set(j.machines)&set([machine.id for machine in machines])))
              minimum_due_time = j.due
        selected_job = env.job_info[job_index]
        selected_machine = env.machine_info[machine_index]
        selected_machine.processing_job = selected_job
    elif method == 'LPT':
        job_index=0
        machine_index=0
        maximum_prts=-math.inf
        for i in machines:
          for j in jobs:
            for jj in j.machines:
              if jj==i.id and j.prts[i.id] > maximum_prts:
                job_index = j.id
                machine_index = i.id
                maximum_prts = j.prts[i.id]
        selected_job = env.job_info[job_index]
        selected_machine = env.machine_info[machine_index]
        selected_machine.processing_job = selected_job
    elif method == 'CR':
        job_index=0
        machine_index=0
        maximum_cr=-math.inf
        for i in machines:
          for j in jobs:
            for jj in j.machines:
              if jj==i.id and j.prts[i.id]/j.due > maximum_cr:
                job_index = j.id
                machine_index = i.id
                maximum_cr = j.prts[i.id]
        selected_job = env.job_info[job_index]
        selected_machine = env.machine_info[machine_index]
        selected_machine.processing_job = selected_job
    elif method == 'CO':
        job_index=0
        machine_index=0
        minimum_co=math.inf
        for i in machines:
          for j in jobs:
            for jj in j.machines:
              if jj==i.id and j.prts[i.id]*j.due < minimum_co:
                job_index = j.id
                machine_index = i.id
                minimum_co = j.prts[i.id]
        selected_job = env.job_info[job_index]
        selected_machine = env.machine_info[machine_index]
        selected_machine.processing_job = selected_job
    elif method == 'ATCS':
        job_index=0
        machine_index=0
        maximum_prts=-math.inf
        for i in machines:
          for j in jobs:
            for jj in j.machines:
              if jj==i.id and ((-j.due+j.prts[i.id]-env.if_setup(j,i))*2+4*(-env.if_transfer(env.resource_info[j.required_resource], i)))/j.prts[i.id] > maximum_prts:
                job_index = j.id
                machine_index = i.id
                maximum_prts = j.prts[i.id]
        selected_job = env.job_info[job_index]
        selected_machine = env.machine_info[machine_index]
        selected_machine.processing_job = selected_job
    else:
        job_index = 0
        machine_index = 0
        maximum_priority = -math.inf
        for i in machines:
          for j in jobs:
            for jj in j.machines:
              if jj==i.id:
                value_dict = dict()
                value_dict['machine_available_time'] = i.available_time
                value_dict['job_prts_at_machine'] = j.prts[i.id]
                value_dict['job_due'] = -j.due
                value_dict['is_there_setup'] = env.if_setup(j, i)
                value_dict['is_there_transfer'] = env.if_transfer(env.resource_info[j.required_resource], i)
                value_dict['is_there_resource_setup'] = env.if_resource_setup(j, i)
                if tc.translate_to_priority(method, value_dict) > maximum_priority:
                  job_index = j.id
                  machine_index = i.id
                  maximum_priority = tc.translate_to_priority(method, value_dict)
        selected_job = env.job_info[job_index]
        selected_machine = env.machine_info[machine_index]
        selected_machine.processing_job = selected_job
    return selected_job, selected_machine, env.resource_info[selected_job.required_resource]


def run_the_simulator_last(problem, rule):
    data = problem
    sim = Simulator(data)
    sim.reset()
    while not sim.is_done():
        available_machines = [machine for machine in sim.machine_info.values() if machine.available_time <= sim.sim_time] # machine.resource>0 이거 사실상 의미 없음 step에서 쓰자마자 10이하면 리필하니까
        available_jobs = sim.get_available_job(available_machines) # ready time, 종료 여부 등 고려하여 작업 가능한 job 가져오기
        if available_jobs != []:
          action = get_action(sim, available_jobs, available_machines, rule)
          sim.step(action)
          sim.move_to_next_sim_t(0)
        else:
          sim.move_to_next_sim_t(1)

    # 결과 출력 부분
    for i in range(sim.num_machine):
        print(f"machine {i}:", end=" ")
        for j in sim.schedule[i]:
            print(j[0].id, end=" ")
            #print('('+str(j[1])+','+str(j[2])+')')
        print()
    print(sim.total_tardiness)
    sim.plot_gantt_chart()

def run_the_simulator(problem, rule):
    data = problem
    sim = Simulator(data)
    sim.reset()
    while not sim.is_done():
        available_machines = [machine for machine in sim.machine_info.values() if machine.available_time <= sim.sim_time]
        available_jobs = sim.get_available_job(available_machines) # ready time, 종료 여부 등 고려하여 작업 가능한 job 가져오기
        if available_jobs != []:
          action = get_action(sim, available_jobs, available_machines, rule)
          sim.step(action)
          sim.move_to_next_sim_t(0)
        else:
          sim.move_to_next_sim_t(1)
    return sim.total_tardiness

"""
problem_path = './data_train/5x100x6/5x100x6_77.pickle'
with open(problem_path, 'rb') as fr:
    problem = pickle.load(fr)

run_the_simulator_last(problem, 'SPT')
run_the_simulator(problem, 'SPT')
"""