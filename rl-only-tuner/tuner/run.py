# -*- coding: utf-8 -*-
"""
description: Evaluate the Model
"""

import os
import sys
import utils
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import train_models as models
import environment

# python3 run.py --isRun=True
# --params=train_ddpg_1606179754_5;
parser = argparse.ArgumentParser()
parser.add_argument(
    '--params',
    type=str,
    default='',
    help='Load existing parameters')
parser.add_argument(
    '--workload',
    type=str,
    default='readwrite',
    help='Workload type [`read`, `write`, `readwrite`]')
parser.add_argument(
    '--instance',
    type=str,
    default='mysql3',
    help='Choose MySQL Instance')
parser.add_argument('--method', type=str, default='ddpg',
                    help='Choose Algorithm to solve [`ddpg`,`dqn`]')
parser.add_argument('--memory', type=str, default='', help='add replay memory')
parser.add_argument(
    '--max_steps',
    type=int,
    default=50,
    help='evaluate test steps')
parser.add_argument(
    '--other_knob',
    type=int,
    default=0,
    help='Number of other knobs')
parser.add_argument(
    '--batch_size',
    type=int,
    default=16,
    help='Training Batch Size')
parser.add_argument(
    '--benchmark',
    type=str,
    default='sysbench',
    help='[sysbench, tpcc]')
parser.add_argument('--metric_num', type=int, default=65, help='metric nums')
parser.add_argument(
    '--default_knobs',
    type=int,
    default=6,
    help='default knobs')
parser.add_argument(
    '--runIdentifying',
    type=str,
    default='',
    help='train identifying')
parser.add_argument(
    '--isRun',
    action='store_true',
    help='True when actually running the knobs and returning the perforamnce, not when only knobs are returned')

opt = parser.parse_args()

# Create Environment
isRun = opt.isRun
env = environment.Server(
    wk_type=opt.workload,
    instance_name=opt.instance,
    run_identifying=opt.runIdentifying)

# Build models
ddpg_opt = dict()
ddpg_opt['tau'] = 0.00001
ddpg_opt['alr'] = 0.00001
ddpg_opt['clr'] = 0.00001
ddpg_opt['model'] = opt.params

n_states = opt.metric_num
gamma = 0.9
memory_size = 100000
num_actions = opt.default_knobs + opt.other_knob
ddpg_opt['gamma'] = gamma
ddpg_opt['batch_size'] = opt.batch_size
ddpg_opt['memory_size'] = memory_size

model = models.DDPG(
    n_states=n_states,
    n_actions=num_actions,
    opt=ddpg_opt,
    ouprocess=True,
    supervised=False
)

if not os.path.exists('log'):
    os.mkdir('log')

if not os.path.exists('test_knob'):
    os.mkdir('test_knob')

expr_name = 'eval_{}_{}'.format(opt.method, str(utils.get_timestamp()))

logger = utils.Logger(
    name=opt.method,
    log_file='log/{}.log'.format(expr_name)
)

if opt.other_knob != 0:
    logger.warn('USE Other Knobs')


def compute_percentage(default, current):
    """ compute metrics percentage versus default settings
    Args:
        default: dict, metrics from default settings
        current: dict, metrics from current settings
    """
    delta_tps = 100 * (current[0] - default[0]) / default[0]
    delta_latency = 100 * (-current[1] + default[1]) / default[1]
    return delta_tps, delta_latency


def generate_knob(action, method):
    if method == 'ddpg':
        return environment.gen_continuous(action)
    else:
        raise NotImplementedError()


if len(opt.memory) > 0:
    model.replay_memory.load_memory(opt.memory)
    print("Load Memory: {}".format(len(model.replay_memory)))

step_counter = 0
train_step = 0
if opt.method == 'ddpg':
    accumulate_loss = [0, 0]
else:
    accumulate_loss = 0


max_score = 0
max_idx = -1
generate_knobs = []
# instance_current_knobs = env.get_current_knobs(opt.instance)
current_state, default_metrics = env.initialize(apply_knobs=False)
model.reset(0.1)

# time for every step
step_times = []
# time for training
train_step_times = []
# time for setup, restart, test
env_step_times = []
# restart time
env_restart_times = []
# choose_action_time
action_step_times = []

# print("[Environment Intialize]Tps: {} Lat:{}".format(default_metrics[0], default_metrics[1]))
print("------------------- Starting to Tune -----------------------")
step_time = utils.time_start()

state = current_state

action_step_time = utils.time_start()
action = model.choose_action(state)
action_step_time = utils.time_end(action_step_time)

current_knob = generate_knob(action, 'ddpg')
# logger.info("[ddpg] Action: {}".format(action))
print("[ddpg] Knobs: {}".format(current_knob))

print("isRun:", isRun)

if isRun:
    env_step_time = utils.time_start()
    reward, state_, done, score, metrics, restart_time = env.step(current_knob)
    env_step_time = utils.time_end(env_step_time)

    print("[{}][Step: {}][Metric tps:{} lat:{}, qps: {}]Reward: {} Score: {} Done: {}".format(
        opt.method, step_counter, metrics[0], metrics[1], metrics[2], reward, score, done))

    _tps, _lat = compute_percentage(default_metrics, metrics)

    print("[{}][Knob Idx: {}] tps increase: {}% lat decrease: {}%".format(
        opt.method, step_counter, _tps, _lat
    ))

print("------------------- Tuning Finished -----------------------")

sys.exit()
