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


logger = logging.getLogger('osim.http.client')
logger.setLevel(logging.CRITICAL)


remote_base = 'http://grader.crowdai.org'
token = 'a6e5f414845fafd1063253a11429c78f'

def animate_rollout(env, agent, n_timesteps,delay=.01):
    infos = defaultdict(list)
    ob = env.reset()
    if hasattr(agent,"reset"): agent.reset()
    env.render()
    tot_rew = 0.0
    with tqdm(total=2500) as reward_bar:
        for i in tqdm(range(n_timesteps)):
            ob = agent.obfilt(ob)
            a, _info = agent.act(ob)
            (ob, rew, done, info) = env.step(a)
            tot_rew += rew
            reward_bar.update(rew)
            env.render()
            if done:
                print("terminated after %s timesteps"%i)
                break
            for (k,v) in info.items():
                infos[k].append(v)
            infos['ob'].append(ob)
            infos['reward'].append(rew)
            infos['action'].append(a)
#            time.sleep(delay)
    return infos, tot_rew

def submit_agent(agent):
    infos = defaultdict(list)
    client = Client(remote_base)
    ob = client.env_create(token)
    tot_rew = 0.0
    with tqdm(total=2500) as reward_bar:
        for i in tqdm(range(501)):
            ob = agent.obfilt(ob)
            a, _info = agent.act(ob)
            ob, rew, done, info = client.env_step(a.tolist(), True)
            tot_rew += rew
            reward_bar.update(rew)
            if done:
                print("terminated after %s timesteps"%i)
                break
            for k, v in info.items():
                infos[k].append(v)
            infos['ob'].append(ob)
            infos['reward'].append(rew)
            infos['action'].append(a)
    print "Total reward", tot_rew
    x = raw_input("type yes to submit to the server")
    if x.strip() == 'yes':
        client.submit()
    return infos, tot_rew
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf")
    parser.add_argument("--timestep_limit",type=int)
    parser.add_argument("--snapname")
    parser.add_argument("--submit", type=int, default=0)
    parser.add_argument("--visualize", type=int, default=0)
    parser.add_argument("--all_snaps", type=int, default=0)
    args = parser.parse_args()

    hdf = h5py.File(args.hdf,'r')

    snapnames = hdf['agent_snapshots'].keys()
    print "snapshots:\n",snapnames
    if args.snapname is None: 
        snapname = snapnames[-1]
    elif args.snapname not in snapnames:
        raise ValueError("Invalid snapshot name %s"%args.snapname)
    else: 
        snapname = args.snapname

    # env = gym.make(hdf["env_id"].value)

    agent = cPickle.loads(hdf['agent_snapshots'][snapname].value)
    agent.stochastic=False

    
    
    if args.submit == 1:
        pass
    else:
        if args.visualize == 0:
            env = GaitEnv(visualize=False)
        else:
            env = GaitEnv(visualize = True)
        timestep_limit = args.timestep_limit or env.spec.timestep_limit

    while True:
        if args.submit == 1:
            infos, tot_rew = submit_agent(agent)
        else:
            if args.all_snaps == 1:
                for snapname, agent_pkl in hdf['agent_snapshots'].items():
                    print "***** Snapshot", snapname
                    agent = cPickle.loads(agent_pkl.value)
                    for i in range(5):
                        print "\n"
                        print "run", i
                        animate_rollout(env,agent,n_timesteps=timestep_limit, 
                            delay=1.0/env.metadata.get('video.frames_per_second', 30))
                        print "\n"
            else:
                animate_rollout(env, agent, n_timesteps=timestep_limit, delay=1.0)
                    
            # pickle.dump(infos, open('sample_infos.pkl', 'wb'))
        # for (k,v) in infos.items():
        #     if k.startswith("reward"):
        #         print "%s: %f"%(k, np.sum(v))
        # print "Total reward", tot_rew
        raw_input("press enter to continue")

if __name__ == "__main__":
    main()