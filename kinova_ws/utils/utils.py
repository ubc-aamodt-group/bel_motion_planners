import time
import torch
from torch.autograd import Variable
import numpy as np
import math
import pickle

from utils.import_tools import StateValidity

joint_offset=torch.tensor([6.8284173, 0, 0, 5.3820982, 5.85142517, 6.25683784]).cuda()
joint_ranges = torch.tensor([13.34888649, 5.46199751, 5.95070934, 10.77949142, 11.74411774, 12.57576084]).cuda()
DEFAULT_STEP = 0.05

def IsInCollision(state, print_depth=False):   
    global sv 
    global filler_robot_state
    global rs_man
    global counter
    filler_robot_state[0:6] = state
    rs_man.joint_state.position = tuple(filler_robot_state)

    collision_free = sv.getStateValidity(rs_man, group_name="arm", print_depth=print_depth)
    counter += 1
    return (not collision_free)

def decode(btbel_old,num_bits,nrange,di):
    btbel=btbel_old.view(6,num_bits)
    t = torch.matmul(btbel,di.float())
    _,ts = torch.max(t,dim=1)
    ts = ts.float()/nrange
    ts=(ts*joint_ranges)-joint_offset
    return ts

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def feasibility_check(path, collHandle, step_sz=DEFAULT_STEP, print_depth=False):

    for i in range(0, len(path)-1):
        ind = steerTo(path[i], path[i+1], collHandle, step_sz=step_sz,
                      print_depth=print_depth)
        if ind == 0:
            return 0
    return 1

def steerTo(start, end, print_depth=False, dof=6):
    dists = np.subtract(end, start, dtype=np.float32)

    distTotal = np.dot(dists, dists)

    distTotal = math.sqrt(distTotal)
    if distTotal > 0:
        incrementTotal = distTotal/DEFAULT_STEP
        for i in range(0, dof):
            dists[i] = dists[i]/incrementTotal

        numSegments = int(math.floor(incrementTotal))
        stateCurr = np.array(start, dtype=np.float32)
        for i in range(0, numSegments):
            if IsInCollision(stateCurr, print_depth=print_depth):
                return 0

            stateCurr += dists

        if IsInCollision(end, print_depth=print_depth):
            return 0

    return 1

def lvc(path, step_sz=DEFAULT_STEP):
    for i in range(0, len(path)-1):
        for j in range(len(path)-1, i+1, -1):
            ind = 0
            ind = steerTo(path[i], path[j], step_sz=step_sz)
            if ind == 1:
                pc = []
                for k in range(0, i+1):
                    pc.append(path[k])
                for k in range(j, len(path)):
                    pc.append(path[k])

                return lvc(pc)

    return path

def save_feasible_path(self, path, file_name):
    with open(file_name + '.pkl', 'wb') as f:
        pickle.dump(path, f)