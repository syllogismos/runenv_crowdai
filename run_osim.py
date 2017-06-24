#!/usr/bin/env python
"""
This script runs a policy gradient algorithm
"""


from gym.envs import make
from modular_rl import *
import argparse, sys, cPickle
from tabulate import tabulate
import shutil, os, logging, h5py
import gym
from osim.env import RunEnv
import ast
import pickle
# import newrelic.agent

# # application = newrelic.agent.register_application(timeout=10.0)
# # import newrelic.agent
# newrelic.agent.initialize('/Users/anil/newrelic.ini', 'production')
# @newrelic.agent.background_task(name='trpo-osim', group='rl')
if __name__ == '__main__':
    sys.stdout.flush()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)
    parser.add_argument("--env", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--ec2", type=ast.literal_eval, default=False)
    parser.add_argument("--destroy_env_every", type=int, default=5)
    parser.add_argument("--node_config", type=int, default=1)
    args, _ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])
    env = RunEnv(visualize=False)
    env_spec = env.spec
    mondir = args.outfile + ".dir"
    if os.path.exists(mondir):
        shutil.rmtree(mondir)
    os.mkdir(mondir)
    # env = gym.wrappers.Monitor(env, mondir, video_callable=None if args.video else VIDEO_NEVER)
    # env.spec.timestep_limit = 50
    # print env.spec.timestep_limit, "@@@@@@@@@@@@"
    # env = GaitEnv(visualize=False)
    agent_ctor = get_agent_cls(args.agent)
    update_argument_parser(parser, agent_ctor.options)
    args = parser.parse_args()
    if args.timestep_limit == 0:
        args.timestep_limit = env_spec.timestep_limit
    cfg = args.__dict__
    np.random.seed(args.seed)

    # loading agent from snapshot logic
    loading_from_snapshot = False
    loading_from_pickle = False
    if args.load_snapshot != '':
        print "loading agent from snapshot", args.load_snapshot
        # try:
        if args.load_snapshot[-4:] == '.pkl':
            print "loading agent from pickle file"
            # agent is loaded from a pickle file
            hdff = open(args.load_snapshot, 'rb')
            loading_from_pickle = True
        else:
            print "loading agent from h5"
            hdff = h5py.File(args.load_snapshot, 'r')
            loading_from_snapshot = True
        # except IOError:
            # print "No such snapshot exists at path", args.load_snapshot

    if loading_from_snapshot:
        snapnames = hdff['agent_snapshots'].keys()
        snapname = snapnames[-1]
        print "Loading from snapshot iteration, ", snapname
        agent = cPickle.loads(hdff['agent_snapshots'][snapname].value)
    elif loading_from_pickle:
        agent = cPickle.load(hdff)
    else:
        agent = agent_ctor(env.observation_space, env.action_space, cfg)
    # agent = agent_ctor(env.observation_space, env.action_space, cfg)

    if args.use_hdf:
        hdf, diagnostics = prepare_h5_file(args)
    gym.logger.setLevel(logging.WARN)

    COUNTER = 0

    def callback(stats, th_paths):
        global COUNTER
        COUNTER += 1
        # Print stats
        print "*********** Iteration %i ****************" % COUNTER
        print tabulate(filter(lambda (k, v): np.asarray(v).size == 1, stats.items()))  # pylint: disable=W0110
        # Store to hdf5
        if args.use_hdf:
            for (stat, val) in stats.items():
                if np.asarray(val).ndim == 0:
                    diagnostics[stat].append(val)
                else:
                    assert val.ndim == 1
                    diagnostics[stat].extend(val)
            if args.snapshot_every and ((COUNTER % args.snapshot_every == 0) or (COUNTER == args.n_iter)):
                hdf['/agent_snapshots/%0.4i' % COUNTER] = np.array(cPickle.dumps(agent, -1))
                dir_name = args.outfile + '.dir/'
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                itr_filename = dir_name + str(COUNTER) + '_' + str(int(stats['EpRewMean'])) + '_' + str(int(stats['EpRewMax'])) + '.pkl'
                cPickle.dump(agent, open(itr_filename, 'wb'))
        # Plot
        if args.plot:
            animate_rollout(env, agent, min(500, args.timestep_limit))
        print "dumping these no of path pickles", len(th_paths)
        for pth in th_paths:
            total_rew = sum(pth['reward'])
            pth['tot_rew'] = total_rew
            # if args.load_snapshot != '':
            #     snapshot_name = args.load_snapshot[10:-3]
            # else:
            #     snapshot_name = 'initial_training'
            outfile_prefix = os.path.basename(args.outfile)[:-3]
            pickle_file = 'training_agent_runs/' + '_'.join([outfile_prefix,
                                                             str(COUNTER),
                                                             str(pth['seed']),
                                                             str(int(total_rew)),
                                                             str(int(time.time()))])
            print "Name of the pickle file", pickle_file
            pickle.dump(pth, open(pickle_file, 'wb'))

    run_policy_gradient_algorithm(env,
                                  agent,
                                  threads=args.threads,
                                  destroy_env_every=args.destroy_env_every,
                                  ec2=args.ec2,
                                  callback=callback,
                                  usercfg=cfg,
                                  args=args)

    if args.use_hdf:
        hdf['env_id'] = env_spec.id
        try:
            hdf['env'] = np.array(cPickle.dumps(env, -1))
        except Exception:
            print "failed to pickle env"  # pylint: disable=W0703
    env.close()
