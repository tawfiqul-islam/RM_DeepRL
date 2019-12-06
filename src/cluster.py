import numpy as np
from src import definitions as defs

# cluster resource details

features = 11
vm_types = 3

vm1_total = 1
vm1_cpu = 4
vm1_mem = 12
vm1_price = 1

vm2_total = 1
vm2_cpu = 8
vm2_mem = 24
vm2_price = 2

vm3_total = 1
vm3_cpu = 12
vm3_mem = 36
vm3_price = 3

j_total = 3
j_types = 3
j_cpu_max = 6
j_cpu_min = 1
j_mem_max = 24
j_mem_min = 1
j_ex_max = 6
j_ex_min = 1

j1_cpu = 2
j1_mem = 4
j1_ex = 3
j1_time = 60

j2_cpu = 4
j2_mem = 8
j2_ex = 2
j2_time = 100

j3_cpu = 6
j3_mem = 8
j3_ex = 2
j3_time = 80

#cluster_state_min = [0, 0, 0, 0, 0, 0, 1, 1, j_cpu_min, j_mem_min, j_ex_min]
#cluster_state_max = [vm0_cpu, vm0_mem, vm1_cpu, vm1_mem, vm2_cpu, vm2_mem, j_total, j_types, j_cpu_max, j_mem_max, j_ex_max]
#cluster_state_init = [vm0_cpu, vm0_mem, vm1_cpu, vm1_mem, vm2_cpu, vm2_mem, 1, 1, j0_cpu, j0_mem, j0_ex]

jobs = []
vms = []

# cluster_state_init = []
cluster_state_init = []
cluster_state_min = []
cluster_state_max = []


def gen_cluster_state_init():
    np.empty(cluster_state_init)
    i = 0
    while i < len(vms):
        cluster_state_init.append(vms[i].cpu)
        cluster_state_init.append(vms[i].mem)
        i += 1
    cluster_state_init.append(jobs[0].id)
    cluster_state_init.append(jobs[0].type)
    cluster_state_init.append(jobs[0].cpu)
    cluster_state_init.append(jobs[0].mem)
    cluster_state_init.append(jobs[0].ex)


def gen_cluster_state_min():
    np.empty(cluster_state_min)
    i = 0
    while i < len(vms):
        cluster_state_min.append(0)
        cluster_state_min.append(0)
        i += 1
    cluster_state_min.append(1)
    cluster_state_min.append(1)
    cluster_state_min.append(j_cpu_min)
    cluster_state_min.append(j_mem_min)
    cluster_state_min.append(j_ex_min)


def gen_cluster_state_max():
    np.empty(cluster_state_max)
    i = 0
    while i < len(vms):
        cluster_state_max.append(vms[i].cpu)
        cluster_state_max.append(vms[i].mem)
        i += 1
    cluster_state_max.append(j_total)
    cluster_state_max.append(j_types)
    cluster_state_max.append(j_cpu_max)
    cluster_state_max.append(j_mem_max)
    cluster_state_max.append(j_ex_max)


def gen_jobs_simple():
    np.empty(jobs)
    jobs.append(defs.JOB(0, 1, 1, j1_cpu, j1_mem, j1_ex, j1_time))
    jobs.append(defs.JOB(50, 2, 2, j2_cpu, j2_mem, j2_ex, j2_time))
    jobs.append(defs.JOB(80, 3, 3, j3_cpu, j3_mem, j3_ex, j3_time))


def init_jobs():
    gen_jobs_simple()


def init_vms():
    np.empty(vms)
    for i in range(vm1_total):
        vms.append(defs.VM(vm1_cpu, vm1_mem, vm1_price))
    for i in range(vm2_total):
        vms.append(defs.VM(vm2_cpu, vm2_mem, vm2_price))
    for i in range(vm3_total):
        vms.append(defs.VM(vm3_cpu, vm3_mem, vm3_price))


def init_cluster():
    init_jobs()
    init_vms()


#init_cluster()
#print(cluster_state_init)
#print(cluster_state_init)
#print(cluster_state_init)