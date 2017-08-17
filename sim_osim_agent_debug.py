#!/usr/bin/env python
"""
Load a snapshotted agent from an hdf5 file and animate it's behavior
"""

import argparse
import cPickle, h5py, numpy as np, time
from collections import defaultdict
import gym
import opensim as osim
from osim.http.client import Client
from osim.env import *
from tqdm import tqdm
import logging
import pickle
from osim_http_mrl_client import Client as mrl_client


logger = logging.getLogger('osim.http.client')
logger.setLevel(logging.CRITICAL)

remote_base = 'http://grader.crowdai.org:1729'
token = 'b97ecc86c6e23bda7b2ee8771942cb9c'


def animate_rollout(env, agent, n_timesteps, delay=.01):
    infos = defaultdict(list)
    ob = env.reset(difficulty=0, seed=None)
    if hasattr(agent, "reset"):
        agent.reset()
    env.render()
    tot_rew = 0.0
    tot_rew_orig = tot_rew3 = tot_rew4 = tot_rew5 = tot_rew6 = tot_rew7 = tot_rew8 = tot_rew9 = 0.0
    # with tqdm(total=2500) as reward_bar:
    # for i in tqdm(range(n_timesteps)):
    for i in range(n_timesteps):
        ob = agent.obfilt(ob)
        a, _info = agent.act(ob)
        (ob, rew, done, info) = env.step(a)
        old = info['data'][0]
        new = info['data'][1]
        lig = info['data'][2]
        delta = new - old
        rew_orig = new - lig*0.0001
        rew3 = delta - lig*1e-3
        rew4 = delta - lig*1e-4
        rew5 = delta - lig*1e-5
        rew6 = delta - lig*1e-6
        rew7 = delta - lig*1e-7
        rew8 = delta - lig*1e-8
        rew9 = delta - lig*1e-9
        tot_rew_orig += rew_orig
        tot_rew3 += rew3
        tot_rew4 += rew4
        tot_rew5 += rew5
        tot_rew6 += rew6
        tot_rew7 += rew7
        tot_rew8 += rew8
        tot_rew9 += rew9
        # print rew, env.current_state[1] - env.last_state[1], env.last_state[1], env.current_state[1]
        tot_rew += rew
        # reward_bar.update(max(rew, 0.000001))
        env.render()
        if done:
            print("\n terminated after %s timesteps" % i)
            break
        for (k, v) in info.items():
            infos[k].append(v)
        infos['ob'].append(ob)
        infos['reward'].append(rew)
        infos['action'].append(a)
#            time.sleep(delay)
    infos['tot_rew'].append(tot_rew)
    print 'total reward is', tot_rew, i
    print 'total_distance_travelled', new
    print tot_rew_orig, tot_rew3, tot_rew4, tot_rew5, tot_rew6, tot_rew7, tot_rew8, tot_rew9

    return infos, tot_rew


def run_agent_from_infos(infos_file):
    infos = pickle.load(open(infos_file, 'rb'))
    print 'total reward from infos', infos['tot_rew']
    env = RunEnv(visualize=False)
    ob = env.reset(difficulty=0, seed=None)
    tot_rew = 0.0
    if 'observation' in infos:
        infos['ob'] = infos['observation']
    sim_infos = defaultdict(list)
    with tqdm(total=2500) as reward_bar:
        for i in tqdm(range(len(infos['reward']))):
            a = infos['action'][i]
            ob, rew, done, info = env.step(a)
            tot_rew += rew
            reward_bar.update(max(rew, 0.000001))
            if done:
                print "\n terminated after timestep", i
                break
            for k, v in info.items():
                infos[k].append(v)
            sim_infos['ob'].append(ob)
            sim_infos['reward'].append(rew)
            sim_infos['action'].append(a)
    sim_infos['tot_rew'].append(tot_rew)
    print "\n total reward from client", tot_rew
    return sim_infos, tot_rew


def submit_agent_from_infos(infos_file):
    infos = pickle.load(open(infos_file, 'rb'))
    print "totoal reward from infos", infos['tot_rew']
    client = Client(remote_base)
    ob = client.env_create(token)
    # client = mrl_client()
    # ob = client.env_reset()
    tot_rew = 0.0
    sim_infos = defaultdict(list)
    if 'observation' in infos:
        infos['ob'] = infos['observation']
    with tqdm(total=2500) as reward_bar:
        for i in tqdm(range(len(infos['reward']))):
            a = infos['action'][i].tolist()
            # for i in range(len(a)):
            #     if a[i] < 0.0:
            #         a[i] = 0.0

            #     if a[i] > 1.0:
            #         a[i] = 1.0
            ob, rew, done, info = client.env_step(a, True)
            # print "@@@@@@@@@@ iteration", i
            # print "observation truth", ob == infos['ob'][i].tolist()
            # print a
            # print ob
            # print infos['ob'][i].tolist()
            tot_rew += rew
            reward_bar.update(rew)
            if done:
                print("\nterminated after %s timesteps" % i)
                break
            for k, v in info.items():
                infos[k].append(v)
            sim_infos['ob'].append(ob)
            sim_infos['reward'].append(rew)
            sim_infos['action'].append(a)
    sim_infos['tot_rew'].append(tot_rew)
    print "\nTotal reward from client", tot_rew
    x = raw_input("type yes to submit to the server")
    if x.strip() == 'yes':
        client.submit()
    return sim_infos, tot_rew


def submit_agent(agent):
    infos = defaultdict(list)
    client = Client(remote_base)
    ob = client.env_create(token)
    tot_rew = 0.0
    with tqdm(total=2500) as reward_bar:
        # for i in tqdm(range(501)):
        i = 0
        j = 0
        while True:
            i += 1
            ob = agent.obfilt(ob)
            a, _info = agent.act(ob)
            ob, rew, done, info = client.env_step(a.tolist(), True)
            tot_rew += rew
            reward_bar.update(max(rew, 0))
            if done:
                j += 1
                print("terminated after %s timesteps" % i)
                print("terminated the %s episode" % j)
                ob = client.env_reset()
                if not ob:
                    break
            for k, v in info.items():
                infos[k].append(v)
            infos['ob'].append(ob)
            infos['reward'].append(rew)
            infos['action'].append(a)
    infos['tot_rew'].append(tot_rew)
    print "Total reward", tot_rew
    x = raw_input("type yes to submit to the server")
    if x.strip() == 'yes':
        client.submit()
    return infos, tot_rew


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf")
    parser.add_argument("--timestep_limit", type=int)
    parser.add_argument("--snapname")
    parser.add_argument("--submit", type=int, default=0)
    parser.add_argument("--visualize", type=int, default=0)
    parser.add_argument("--all_snaps", type=int, default=0)
    parser.add_argument("--from_infos", type=str, default='')
    args = parser.parse_args()

    if args.hdf[-4:] == '.pkl':
        # agent is loaded from a pickle
        agent = cPickle.load(open(args.hdf, 'rb'))
    else:
        # agent is loaded from hdf
        hdf = h5py.File(args.hdf, 'r')
        snapnames = hdf['agent_snapshots'].keys()
        print "snapshots:\n", snapnames
        if args.snapname is None:
            snapname = snapnames[-1]
        elif args.snapname not in snapnames:
            raise ValueError("Invalid snapshot name %s" % args.snapname)
        else:
            snapname = args.snapname
        # env = gym.make(hdf["env_id"].value)
        agent = cPickle.loads(hdf['agent_snapshots'][snapname].value)

    agent.stochastic = False

    if args.from_infos == '':
        if args.submit == 1:
            pass
        else:
            if args.visualize == 0:
                env = RunEnv(visualize=False)
            else:
                env = RunEnv(visualize=True)
            timestep_limit = args.timestep_limit or env.spec.timestep_limit

        while True:
            if args.submit == 1:
                infos, tot_rew = submit_agent(agent)
            else:
                if args.all_snaps == 1:
                    # gets 5 runs from each available agent snapshot in h5
                    for snapname, agent_pkl in hdf['agent_snapshots'].items():
                        print "***** Snapshot", snapname
                        agent = cPickle.loads(agent_pkl.value)
                        for i in range(5):
                            print "\n"
                            print "run", i
                            infos, tot_rew = animate_rollout(env, agent, n_timesteps=timestep_limit)
                            print "\n"
                            file_attrs = [os.path.basename(args.hdf)[:-3],
                                          snapname, str(i), str(int(tot_rew)),
                                          str(int(time.time()))]
                            pickle_file = 'agent_runs/' + '_'.join(file_attrs)
                            pickle.dump(infos, open(pickle_file, 'wb'))
                else:
                    animate_rollout(env, agent, n_timesteps=timestep_limit)

                # pickle.dump(infos, open('sample_infos.pkl', 'wb'))
            # for (k,v) in infos.items():
            #     if k.startswith("reward"):
            #         print "%s: %f"%(k, np.sum(v))
            # print "Total reward", tot_rew
            raw_input("\n press enter to continue")
    else:
        if args.submit == 1:
            infos, tot_rew = submit_agent_from_infos(args.from_infos)
        else:
            infos, tot_rew = run_agent_from_infos(args.from_infos)

if __name__ == "__main__":
    main()