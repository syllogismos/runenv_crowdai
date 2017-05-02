from multiprocessing import Pool
import psutil
from osim_helpers import start_osim_apps, stop_osim_apps, ip_config
import itertools
from modular_rl.core import create_ext_envs, get_paths
import time
from osim.env import GaitEnv
import pickle
import h5py, cPickle

hdf = h5py.File('/home/anil/modular_rl/snapshots/osim_gait_ext10.h5', 'r')
snapname='0015'
agent = cPickle.loads(hdf['agent_snapshots'][snapname].value)
env = GaitEnv(visualize=False)

if __name__ == '__main__':
    cfg = {}
    cfg['timestep_limit'] = 500
    cfg['timesteps_per_batch'] = 100000
    seed_iter_count = 1428
    iter = 0
    while True:
        iter += 1
        print "Iteration", iter
        seed_iter_count += 1428
        seed_iter = itertools.count(seed_iter_count)

        server_states = {}
        envs = []
        for con in ip_config:
            server_states[con['ip']] = start_osim_apps(con['ip'], 8018, con['cores'])
            time.sleep(10)
            envs.extend(create_ext_envs(con['ip'], con['cores']))
        
        multi_pool_count = len(envs)
        paths = get_paths(env, agent, cfg, seed_iter, envs=envs, threads = multi_pool_count)
        threshold_paths = filter(lambda x: sum(x['reward']) > 2700.0, paths)
        for pth in threshold_paths:
            total_rew = sum(pth['reward'])
            pth['tot_rew'] = total_rew
            # if args.load_snapshot != '':
            #     snapshot_name = args.load_snapshot[10:-3]
            # else:
            #     snapshot_name = 'initial_training'
            outfile_prefix = 'agent_runs' #os.path.basename(args.outfile)[:-3]
            pickle_file = 'multiple_agent_runs/' + '_'.join([outfile_prefix,
                                                            str(COUNTER),
                                                            str(pth['seed']),
                                                            str(int(total_rew)),
                                                            str(int(time.time()))])
            print "Name of the pickle file", pickle_file
            pickle.dump(pth, open(pickle_file, 'wb'))
        for con in ip_config:
            stop_osim_apps(con['ip'], 8018, server_states[con['ip']])