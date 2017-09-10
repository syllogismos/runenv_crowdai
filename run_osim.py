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
from osim_helpers import redis_config


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
    parser.add_argument("--http_gym_api", type=int, default=1)
    parser.add_argument("--redis", type=int, default=0)
    parser.add_argument("--batch_scaler", type=int, default=0)
    parser.add_argument("--load_scaler", type=str, default='')
    args, _ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])
    env = RunEnv(visualize=False)
    env_spec = env.spec
    mondir = args.outfile + ".dir"
    if os.path.exists(mondir):
        shutil.rmtree(mondir)
    os.mkdir(mondir)
    agent_ctor = get_agent_cls(args.agent)
    update_argument_parser(parser, agent_ctor.options)
    args = parser.parse_args()
    if args.timestep_limit == 0:
        args.timestep_limit = env_spec.timestep_limit
    cfg = args.__dict__
    cfg['redis_h'] = redis_config['host']
    cfg['redis_p'] = redis_config['port']
    np.random.seed(args.seed)

    if args.redis == 1:
        redis_conn = redis.Redis(cfg['redis_h'], cfg['redis_p'])
        redis_conn.set('curr_batch_size', 0)

    loading_from_snapshot = False
    loading_from_pickle = False
    if args.load_snapshot != '':
        print "loading agent from snapshot", args.load_snapshot
        if args.load_snapshot[-4:] == '.pkl':
            print "loading agent from pickle file"
            hdff = open(args.load_snapshot, 'rb')
            loading_from_pickle = True
        else:
            print "loading agent from h5"
            hdff = h5py.File(args.load_snapshot, 'r')
            loading_from_snapshot = True

    if loading_from_snapshot:
        snapnames = hdff['agent_snapshots'].keys()
        snapname = snapnames[-1]
        print "Loading from snapshot iteration, ", snapname
        agent = cPickle.loads(hdff['agent_snapshots'][snapname].value)
    elif loading_from_pickle:
        agent = cPickle.load(hdff)
    else:
        agent = agent_ctor(env.observation_space, env.action_space, cfg)

    if args.use_hdf:
        hdf, diagnostics = prepare_h5_file(args)
    gym.logger.setLevel(logging.WARN)

    if cfg['batch_scaler'] == 1:
        if cfg['load_scaler'] == '':
            scaler = Scaler(env.observation_space.shape[0])
        else:
            scaler = pickle.load(open(cfg['load_scaler'], 'rb'))
    else:
        scaler = None

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
        if not os.path.exists('training_agent_runs'):
            os.makedirs('training_agent_runs')
        for pth in th_paths:
            total_rew = sum(pth['reward'])
            pth['tot_rew'] = total_rew
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
                                  args=args,
                                  scaler=scaler)

    if args.use_hdf:
        hdf['env_id'] = env_spec.id
        try:
            hdf['env'] = np.array(cPickle.dumps(env, -1))
        except Exception:
            print "failed to pickle env"  # pylint: disable=W0703
    env.close()
