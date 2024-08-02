# @title 시뮬레이터 수정중  (GP 적용)
import math
#import dataloader
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import tree_class as tc


class Job():
    def __init__(self, job_id, dataset):
        if dataset != None:
            self.id = job_id
            self.due = dataset['due_dates'][job_id]
            self.ready_time = dataset['ready_times'][job_id]
            self.job_type = dataset['job_type_list'][job_id] # 이거는 group number임 !
            self.required_resource = dataset['required_resource'][self.job_type]
            self.operations = []
            self.now = 0
            self.next = 0
            self.last = -1
            order = 0
            for op in dataset['job_type'][self.job_type]:
                self.operations.append(Operation(op, dataset['machine_processing_times'][self.job_type][op], dataset['eligible_machines'][self.job_type][op], order,self.id, self.job_type))
                order += 1
        else:
           self.id = job_id
           self.parent = ""


    def is_done(self):
        for op in self.operations:
            if op.state == 'not done' or op.state == 'doing':
                return False
        return True
    
    def is_doing(self):
        for op in self.operations:
            if op.state == 'doing':
                return True
        return False
    
    def next_op(self):
        for op in self.operations:
            if op.state == 'not done':
                return op
        return None
    
    def now_op(self):
        for op in self.operations:
            if op.state == 'doing':
                return op
        return None
    
            

class Operation():
    def __init__(self, id, prts, eligible_machine, order, parent, type):
        self.id = id
        self.prts = prts
        self.eligible_machine = eligible_machine
        self.now_remain_t = 0 # remaining time
        self.state = 'not done'
        self.order = order
        self.parent = parent
        self.type = type
        # not done, doing, done

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
        self.total_tardiness = 0
        self.num_resource = dataset['num_resource_type']
        self.resource_setup_time = 5
        self.transfer_time = dataset['transfer_time']
        self.job_change_time = dataset['job_change_time']
        self.operation_change_time = dataset['operation_change_time']
        self.job_transfer_time = dataset['job_transfer_time']

        # static info
        self.machine_info = dict()
        self.job_info = dict()
        self.resource_info = dict()

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
        # 룰 적용할때는 machine 당이 편할거 같아서 data generate 부분은 유지
        job_info = {i:Job(i, dataset) for i in range(self.num_job)}
        machine_info = {j:Machine(j) for j in range(self.num_machine) }
        resource_info = {k:Resource(k, random.randint(0, self.num_machine-1)) for k in range(self.num_resource) }
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
        for_available_job = []
        op = None
        for job in list(self.job_info.values()):
            if job.ready_time <= self.sim_time and not job.is_done() and not job.is_doing():
                op = job.next_op()
                k=0
                for ma in available_machines:
                    if ma.id in op.eligible_machine:
                        k+=1
                if k > 0:
                    if op.id == 'Photo':
                        if self.resource_info[job.required_resource].available_time <= self.sim_time:
                            for_available_job.append(job)
                    else:
                        for_available_job.append(job)
        self.available_job_list = for_available_job
        return self.available_job_list


    def get_next_time_step(self):
        next_job_time = min([job.operations[job.now].now_remain_t for job in self.job_info.values() if job.now_op() != None and job.operations[job.now].now_remain_t > 0])
        next_machine_time = min([machine.available_time for machine in self.machine_info.values() if machine.available_time >= self.sim_time])
        next_machine_time -= self.sim_time
        return min(next_job_time, next_machine_time)


    def step(self, action):
        job, machine, resource = action
        next_operation = job.next_op()
        setup_time = self.if_setup(job, machine)
        transfer_time = self.if_transfer(resource, machine)
        resource_setup_time = self.if_resource_setup(job, machine)
        job_transfer_time = self.if_job_transfer(job, machine)
        operation_change_time = self.if_operation_change(job, machine)
        now_time = self.sim_time
        if job_transfer_time>0:
            self.schedule[machine.id].append((Job('jt', None), now_time, now_time + job_transfer_time))
        now_time += job_transfer_time
        if setup_time > 0:
            self.schedule[machine.id].append((Job('s', None), now_time, now_time + setup_time))
        now_time += setup_time
        if operation_change_time > 0:
            self.schedule[machine.id].append((Job('oc', None), now_time, now_time + operation_change_time))
        now_time += operation_change_time

        if next_operation.id == 'Photo':
            if transfer_time > 0:
                resource.location = machine.id
                self.schedule[machine.id].append((Job('t', None), now_time, now_time + transfer_time))
            now_time += transfer_time
            
            if resource_setup_time > 0:
                self.schedule[machine.id].append((Job('rs', None), now_time, now_time + resource_setup_time))
            now_time += resource_setup_time

        job.operations[job.next].now_remain_t = next_operation.prts[machine.id] + now_time - self.sim_time
        job.operations[job.next].state = 'doing'
        job.now = job.next
        job.next += 1
        job.last = machine.id
        machine.available_time += next_operation.prts[machine.id] + now_time - self.sim_time
        resource.available_time += next_operation.prts[machine.id] + now_time - self.sim_time
        self.schedule[machine.id].append((next_operation, now_time, machine.available_time))


    def move_to_next_sim_t(self, passid):
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
            if job.now_op() != None:
                if job.operations[job.now].now_remain_t > 0:
                    job.operations[job.now].now_remain_t -= time_diff # job remaining time 업데이트
                    if job.operations[job.now].now_remain_t == 0:
                        job.operations[job.now].state = 'done'
                        if job.is_done():
                            self.completed_job_list.append(job) # 완료된 job 업데이트
                            if self.sim_time > job.due:
                                self.total_tardiness += self.sim_time - job.due


    def is_done(self):
        """
        return True if the simulation is done
        """
        # 수정할거 없을듯
        if len(self.completed_job_list) == self.num_job:
            return True
        return False

    def plot_gantt_chart(self):
        """
        Gantt Chart를 출력하는 함수
        """
        # Parent별 색상을 지정하기 위한 색상 사전
        parent_colors = {}
        # 'tab20' 팔레트를 사용하여 알아보기 쉬운 색상 생성
        palette = sns.color_palette("turbo", self.num_job + 5)  # 충분히 많은 색상 생성
        random.shuffle(palette)  # 색상 배열을 랜덤으로 섞어 색상 중복을 최소화

        # 그래프 크기 조정 (가로로 넓게, 세로로 짧게)
        fig, ax = plt.subplots(figsize=(15, 6))  # 세로 크기를 좀 더 크게 조정

        color_index = 0

        for machine_id, jobs in self.schedule.items():
            for job in jobs:
                job_id, start_time, end_time = job

                # 부모가 빈 문자열("")인 경우 색상을 하얀색으로 설정
                if job_id.parent == "":
                    color = 'white'
                else:
                    # 부모가 색상 사전에 없으면 색상 추가
                    if job_id.parent not in parent_colors:
                        parent_colors[job_id.parent] = palette[color_index]
                        color_index += 1
                    color = parent_colors[job_id.parent]

                # Job을 바 형식으로 그리기
                ax.barh(machine_id, end_time - start_time, left=start_time, color=color,
                        edgecolor='black', align='center', alpha=0.8)

                # 텍스트를 바의 중앙에 위치시키고 겹치지 않도록 조정
                ax.text(start_time + (end_time - start_time) / 2, machine_id,
                        f'{job_id.id + str(job_id.parent)}', color='black', ha='center', va='center', fontsize=8)

        # Parent가 같은 Job들을 선으로 연결
        for parent in parent_colors:
            parent_jobs = [(machine_id, job) for machine_id, jobs in self.schedule.items() for job in jobs if job[0].parent == parent]
            parent_jobs.sort(key=lambda x: x[1][1])  # 시작 시간 기준으로 정렬

            for i in range(len(parent_jobs) - 1):
                (machine_id1, job1), (machine_id2, job2) = parent_jobs[i], parent_jobs[i + 1]
                job_id1, start_time1, end_time1 = job1
                job_id2, start_time2, end_time2 = job2

                # job_id가 문자열이 아닌 경우에만 선 그리기
                if isinstance(job_id1.parent, int) and isinstance(job_id2.parent, int) and job_id1.order + 1 == job_id2.order and machine_id1 != machine_id2:
                    # 선 그리기
                    ax.plot([end_time1, start_time2], [machine_id1, machine_id2], color=parent_colors[parent], linestyle='-', linewidth=1)

        ax.set_xlabel('Time')
        ax.set_ylabel('Machine')
        ax.set_title('Schedule')

        # Y축을 반대로 설정 (위에서 아래로 머신 순서)
        ax.invert_yaxis()

        plt.show()


    def if_setup(self, job, machine):
        op = job.next_op()
        if len(self.schedule[machine.id])==0 or self.schedule[machine.id][-1][0].type == op.type:
          return 0
        else:
          return self.job_change_time[self.schedule[machine.id][-1][0].type][op.type]

    def if_transfer(self, resource, machine):
        if resource.location == machine.id:
            return 0
        else:
            transfer_time = self.transfer_time[resource.location][machine.id]
            return transfer_time

    def if_resource_setup(self, job, machine):
        if len(self.schedule[machine.id])==0 or self.job_info[self.schedule[machine.id][-1][0].parent].required_resource == job.required_resource:
          return 0
        else:
          return self.resource_setup_time
        
    def if_job_transfer(self, job, machine):
        if job.last == -1 or job.last == machine.id:
            return 0
        else:
            return self.job_transfer_time[job.last][machine.id]

    def if_operation_change(self, job, machine):
        op = job.next_op()
        if len(self.schedule[machine.id])==0 or self.schedule[machine.id][-1][0].id == op.id:
           return 0
        else:
           return self.operation_change_time[self.schedule[machine.id][-1][0].id][op.id]

def get_action(env, jobs, machines, method):
    """
    action selection
    """
    if method == 'SPT':  # shortest processing time
        job_index=0
        machine_index=0
        minimum_prts=math.inf
        for ma in machines:
          for jb in jobs:
            next_op = jb.next_op()
            for eli_ma in next_op.eligible_machine:
              if eli_ma==ma.id and next_op.prts[ma.id] < minimum_prts:
                job_index = jb.id
                machine_index = ma.id
                minimum_prts = next_op.prts[ma.id]
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
        for ma in machines:
          for jb in jobs:
            next_op = jb.next_op()
            for eli_ma in next_op.eligible_machine:
              if eli_ma==ma.id:
                value_dict = dict()
                value_dict['machine_available_time'] = ma.available_time
                value_dict['job_prts_at_machine'] = next_op.prts[ma.id]
                value_dict['job_due'] = jb.due
                value_dict['is_there_setup'] = env.if_setup(jb, ma)
                value_dict['is_there_transfer'] = env.if_transfer(env.resource_info[jb.required_resource], ma)
                value_dict['slack'] = jb.due-next_op.prts[ma.id]
                value_dict['is_there_resource_setup'] = env.if_resource_setup(jb, ma)
                if tc.translate_to_priority(method, value_dict) > maximum_priority:
                  job_index = jb.id
                  machine_index = ma.id
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
        available_machines = [machine for machine in sim.machine_info.values() if machine.available_time <= sim.sim_time] 
        available_jobs = sim.get_available_job(available_machines) 
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
        print()
    print(sim.total_tardiness)
    sim.plot_gantt_chart()

def run_the_simulator(problem, rule):
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
    return sim.total_tardiness


problem_path = './data_train/3x12x5/3x12x5_77.pickle'
with open(problem_path, 'rb') as fr:
    problem = pickle.load(fr)
    print(problem)

run_the_simulator_last(problem, 'SPT')
#run_the_simulator(problem, 'SPT')
