import numpy as np, time, itertools
from collections import OrderedDict
from .misc_utils import *
from . import distributions
import theano.tensor as T, theano
from importlib import import_module
import scipy.optimize
from .keras_theano_setup import floatX, FNOPTS
from keras.layers.core import Layer
import gc
from osim_http_mrl_client import Client
from osim_http_ext_client import Client as ext_client
from osim_helpers import start_env_server
import time
from multiprocessing import Pool
import psutil
from osim_helpers import start_osim_apps, stop_osim_apps, ip_config
from osim_helpers import get_paths_from_server
from osim.env import RunEnv
import random
import cPickle
concat = np.concatenate

# ================================================================
# Make agent
# ================================================================


def get_agent_cls(name):
    p, m = name.rsplit('.', 1)
    mod = import_module(p)
    constructor = getattr(mod, m)
    return constructor

# ================================================================
# Stats
# ================================================================


def add_episode_stats(stats, paths):
    reward_key = "reward_raw" if "reward_raw" in paths[0] else "reward"
    episoderewards = np.array([path[reward_key].sum() for path in paths])
    pathlengths = np.array([pathlength(path) for path in paths])

    stats["EpisodeRewards"] = episoderewards
    stats["EpisodeLengths"] = pathlengths
    stats["NumEpBatch"] = len(episoderewards)
    stats["EpRewMean"] = episoderewards.mean()
    stats["EpRewSEM"] = episoderewards.std()/np.sqrt(len(paths))
    stats["EpRewMax"] = episoderewards.max()
    stats["EpRewMin"] = episoderewards.min()
    stats["EpLenMean"] = pathlengths.mean()
    stats["EpLenMax"] = pathlengths.max()
    stats["RewPerStep"] = episoderewards.sum()/pathlengths.sum()


def add_prefixed_stats(stats, prefix, d):
    for (k, v) in d.iteritems():
        stats[prefix+"_"+k] = v

# ================================================================
# Policy Gradients
# ================================================================


def compute_advantage(vf, paths, gamma, lam):
    # Compute return, baseline, advantage
    for path in paths:
        path["return"] = discount(path["reward"], gamma)
        b = path["baseline"] = vf.predict(path)
        b1 = np.append(b, 0 if path["terminated"] else b[-1])
        deltas = path["reward"] + gamma*b1[1:] - b1[:-1]
        path["advantage"] = discount(deltas, gamma * lam)
    alladv = np.concatenate([path["advantage"] for path in paths])
    # Standardize advantage
    std = alladv.std()
    mean = alladv.mean()
    for path in paths:
        path["advantage"] = (path["advantage"] - mean) / std


PG_OPTIONS = [
    ("timestep_limit", int, 0, "maximum length of trajectories"),
    ("n_iter", int, 200, "number of batch"),
    ("parallel", int, 0, "collect trajectories in parallel"),
    ("timesteps_per_batch", int, 10000, ""),
    ("gamma", float, 0.99, "discount"),
    ("lam", float, 1.0, "lambda parameter from generalized advantage estimation"),
]


def run_policy_gradient_algorithm(env, agent, threads=1,
                                  destroy_env_every=5, ec2=False,
                                  usercfg=None, callback=None, args=None):
    cfg = update_default_config(PG_OPTIONS, usercfg)
    cfg.update(usercfg)
    print "policy gradient config", cfg

    if cfg["parallel"]:
        raise NotImplementedError

    tstart = time.time()
    seed_iter = itertools.count()

    if args.node_config == 1:
        if args.http_gym_api == 1:
            """
            creates http gym clients based on the config file and
            the master node contains all the http gym clients and steps
            through the env through http. Here every single step call goes
            through http
            """
            print "creating servers for the first time"
            print "Using http gym api"
            # servers = list(map(lambda x: start_env_server(x, ec2), range(0, threads)))
            # time.sleep(10)
            # print "craeting envs for the first time"
            # envs = create_envs(threads)
            server_states = {}
            envs = []
            for con in ip_config:
                if 'port' in con:
                    port = con['port']
                else:
                    port = 8018
                server_states[con['ip']] = start_osim_apps(con['ip'], port, con['cores'])
                time.sleep(10)
                envs.extend(create_ext_envs(con['ip'], con['cores']))
            multi_pool_count = len(envs)
        else:
            """
            Instead of using http gym api, we directly send the agent to the
            slave nodes and these individual nodes compute the paths locally
            and send them back to the master node.
            """
            print "Not using http gym api"
            pass
    else:
        print "creating python envs for the first time"
        # No need of creating a python envs, can initialize on the thread
        # directly
        envs = [None]*args.threads
        multi_pool_count = args.threads
        pass

    for i in xrange(cfg["n_iter"]):
        # Rollouts ========
        if args.node_config == 1 and \
                args.http_gym_api == 1 and \
                i != 0 and \
                i % destroy_env_every == 0:
            # destroy_servers(servers)
            # print "recreating servers again at ", i
            # servers = list(map(lambda x: start_env_server(x, ec2), range(0, threads)))
            # time.sleep(10)
            # print "recreating envs"
            # envs = create_envs(threads)
            envs = []
            for con in ip_config:
                if 'port' in con:
                    port = con['port']
                else:
                    port = 8018
                print "stopping osim envs at server", con['ip']
                print server_states
                stop_osim_apps(con['ip'], port, server_states[con['ip']])
                print "starting osim envs at server", con['ip']
                server_states[con['ip']] = start_osim_apps(con['ip'], port, con['cores'])
                time.sleep(10)
                print "creating envx at server", con['ip'], con['cores']
                envs.extend(create_ext_envs(con['ip'], con['cores']))

        if args.node_config == 1 and \
                args.http_gym_api != 1:
            """
            Sending agent to all the slaves and computing paths locally
            """
            agent_dump = cPickle.dumps(agent)
            paths = []
            for con in ip_config:
                if 'port' in con:
                    port = con['port']
                else:
                    port = 8018
                paths.extend(get_paths_from_server(con['ip'], con['port'],
                                                   agent_dump, cfg,
                                                   con['cores']))
        else:
            """
            Computing all the paths from master node
            """
            paths = get_paths(env, agent, cfg, seed_iter, envs=envs, threads=multi_pool_count)

        threshold_paths = filter(lambda x: sum(x['reward']) > 2600.0, paths)
        compute_advantage(agent.baseline, paths, gamma=cfg["gamma"], lam=cfg["lam"])
        # VF Update ========
        vf_stats = agent.baseline.fit(paths)
        # Pol Update ========
        pol_stats = agent.updater(paths)
        # Stats ========
        stats = OrderedDict()
        add_episode_stats(stats, paths)
        add_prefixed_stats(stats, "vf", vf_stats)
        add_prefixed_stats(stats, "pol", pol_stats)
        stats["TimeElapsed"] = time.time() - tstart
        if callback:
            callback(stats, threshold_paths)
        x = gc.collect()
        print x, 'garbage collected @@@@@@@@@@@@@@@'
    # destroy_servers(servers)
    if args.node_config == 1 and args.http_gym_api == 1:
        for con in ip_config:
            print "stopping osim envs for the final time at server", con['ip']
            stop_osim_apps(con['ip'], 8018, server_states[con['ip']])


def create_envs(threads):
    envs = []  # [Client(i) for i in range(threads)]
    for i in range(0, threads):
        while True:
            try:
                temp_env = Client(i)
                envs.append(temp_env)
                break
            except Exception:
                print "Exception while creating env of port ", i
                print "Trying to create env again"
    return envs


def create_ext_envs(ip, cores):
    envs = []
    for i in range(0, cores):
        while True:
            try:
                temp_env = ext_client(p=i, rb=ip)
                envs.append(temp_env)
                break
            except Exception:
                print "exception while creatating env at server", ip, i
                print "trying to create env again"
    return envs


def destroy_servers(servers):
    print "Destroying env servers"
    print servers
    for pid in servers:
        try:
            process = psutil.Process(pid)
            for child in process.children():
                child.terminate()
                child.wait()
            process.terminate()
            process.wait()
        except:
            print("process doesnt exist", pid)
            pass
    pass


def get_paths(env, agent, cfg, seed_iter, envs=None, threads=1):
    # if envs == None:
    #     envs = [env]
    pickled_enum = zip(envs, [agent]*threads, [cfg['timestep_limit']]*threads,
                       [cfg['timesteps_per_batch']/threads]*threads)
    if threads > 1:
        p = Pool(threads)
        parallel_paths = p.map(do_rollouts_single_thread,
                               enumerate(pickled_enum))
        p.close()
        p.join()
        paths = list(itertools.chain(*parallel_paths))
        # raise NotImplementedError
    else:
        paths = do_rollouts_serial(env, agent, cfg["timestep_limit"],
                                   cfg["timesteps_per_batch"], seed_iter)
    return paths


def do_rollouts_single_thread(enum_env):
    seed_iter_seed = 0
    while seed_iter_seed == 0:
        seed_iter_seed = int(random.random()*27000)*(enum_env[0]+1)*100
    print "seed_iter_seed is", seed_iter_seed
    seed_iter = itertools.count(seed_iter_seed)
    thread_paths = do_rollouts_serial(enum_env[1][0],
                                      enum_env[1][1],
                                      enum_env[1][2],
                                      enum_env[1][3],
                                      seed_iter)
    print "no of episodes in this thread,", len(thread_paths)
    print "episode lengths in this thread,", [len(x['reward']) for x in thread_paths]
    print "total rewards in this thread,", [sum(x['reward']) for x in thread_paths]
    return thread_paths


def rollout(env, agent, timestep_limit,
            env_is_none=False, seed=None):
    """
    Simulate the env and agent for timestep_limit steps
    """
    if env_is_none:
        ob = env.reset(difficulty=0, seed=seed)
    else:
        ob = env.reset()
    terminated = False

    data = defaultdict(list)
    for _ in xrange(timestep_limit):
        ob = agent.obfilt(ob)
        # print type(ob), 'observation type'
        data["observation"].append(ob)
        action, agentinfo = agent.act(ob)
        data["action"].append(action)
        # print type(action), 'action type'
        for (k, v) in agentinfo.iteritems():
            # print k
            # print type(v)
            data[k].append(v)
        ob, rew, done, envinfo = env.step(action)
        data["reward"].append(rew)
        # print type(rew), 'reward type'
        rew = agent.rewfilt(rew)
        for (k, v) in envinfo.iteritems():
            data[k].append(v)
        if done:
            terminated = True
            break
    data = {k: np.array(v) for (k, v) in data.iteritems()}
    data["terminated"] = terminated
    # print type(data)
    # print type(data['reward'])
    return data


def do_rollouts_serial(env, agent, timestep_limit,
                       n_timesteps, seed_iter):

    if env is None:
        env = RunEnv(visualize=False)
        env_is_none = True
    else:
        env_is_none = False

    paths = []
    timesteps_sofar = 0
    while True:
        rollout_seed = seed_iter.next()
        np.random.seed(rollout_seed)
        path = rollout(env, agent, timestep_limit,
                       env_is_none=env_is_none,
                       seed=rollout_seed)
        path['seed'] = rollout_seed
        paths.append(path)
        timesteps_sofar += pathlength(path)
        if timesteps_sofar > n_timesteps:
            break

    if env_is_none:
        del(env)
        env = None

    return paths


def pathlength(path):
    return len(path["action"])


def animate_rollout(env, agent, n_timesteps, delay=.01):
    ob = env.reset()
    env.render()
    for i in xrange(n_timesteps):
        a, _info = agent.act(ob)
        (ob, _rew, done, _info) = env.step(a)
        env.render()
        if done:
            print("terminated after %s timesteps" % i)
            break
        time.sleep(delay)

# ================================================================
# Stochastic policies
# ================================================================


class StochPolicy(object):

    @property
    def probtype(self):
        raise NotImplementedError

    @property
    def trainable_variables(self):
        raise NotImplementedError

    @property
    def input(self):
        raise NotImplementedError

    def get_output(self):
        raise NotImplementedError

    def act(self, ob, stochastic=True):
        prob = self._act_prob(ob[None])
        if stochastic:
            return self.probtype.sample(prob)[0], {"prob": prob[0]}
        else:
            return self.probtype.maxprob(prob)[0], {"prob": prob[0]}

    def finalize(self):
        self._act_prob = theano.function([self.input], self.get_output(), **FNOPTS)


class ProbType(object):

    def sampled_variable(self):
        raise NotImplementedError

    def prob_variable(self):
        raise NotImplementedError

    def likelihood(self, a, prob):
        raise NotImplementedError

    def loglikelihood(self, a, prob):
        raise NotImplementedError

    def kl(self, prob0, prob1):
        raise NotImplementedError

    def entropy(self, prob):
        raise NotImplementedError

    def maxprob(self, prob):
        raise NotImplementedError


class StochPolicyKeras(StochPolicy, EzPickle):

    def __init__(self, net, probtype):
        EzPickle.__init__(self, net, probtype)
        self._net = net
        self._probtype = probtype
        self.finalize()

    @property
    def probtype(self):
        return self._probtype

    @property
    def net(self):
        return self._net

    @property
    def trainable_variables(self):
        return self._net.trainable_weights

    @property
    def variables(self):
        return self._net.get_params()[0]

    @property
    def input(self):
        return self._net.input

    def get_output(self):
        return self._net.output

    def get_updates(self):
        self._net.output  # pylint: disable=W0104
        return self._net.updates

    def get_flat(self):
        return flatten(self.net.get_weights())

    def set_from_flat(self, th):
        weights = self.net.get_weights()
        self._weight_shapes = [weight.shape for weight in weights]
        self.net.set_weights(unflatten(th, self._weight_shapes))


class Categorical(ProbType):

    def __init__(self, n):
        self.n = n

    def sampled_variable(self):
        return T.ivector('a')

    def prob_variable(self):
        return T.matrix('prob')

    def likelihood(self, a, prob):
        return prob[T.arange(prob.shape[0]), a]

    def loglikelihood(self, a, prob):
        return T.log(self.likelihood(a, prob))

    def kl(self, prob0, prob1):
        return (prob0 * T.log(prob0/prob1)).sum(axis=1)

    def entropy(self, prob0):
        return - (prob0 * T.log(prob0)).sum(axis=1)

    def sample(self, prob):
        return distributions.categorical_sample(prob)

    def maxprob(self, prob):
        return prob.argmax(axis=1)


class CategoricalOneHot(ProbType):

    def __init__(self, n):
        self.n = n

    def sampled_variable(self):
        return T.matrix('a')

    def prob_variable(self):
        return T.matrix('prob')

    def likelihood(self, a, prob):
        return (a * prob).sum(axis=1)

    def loglikelihood(self, a, prob):
        return T.log(self.likelihood(a, prob))

    def kl(self, prob0, prob1):
        return (prob0 * T.log(prob0/prob1)).sum(axis=1)

    def entropy(self, prob0):
        return - (prob0 * T.log(prob0)).sum(axis=1)

    def sample(self, prob):
        assert prob.ndim == 2
        inds = distributions.categorical_sample(prob)
        out = np.zeros_like(prob)
        out[np.arange(prob.shape[0]), inds] = 1
        return out

    def maxprob(self, prob):
        out = np.zeros_like(prob)
        out[prob.argmax(axis=1)] = 1


class DiagGauss(ProbType):

    def __init__(self, d):
        self.d = d

    def sampled_variable(self):
        return T.matrix('a')

    def prob_variable(self):
        return T.matrix('prob')

    def loglikelihood(self, a, prob):
        mean0 = prob[:, :self.d]
        std0 = prob[:, self.d:]
        # exp[ -(a - mu)^2/(2*sigma^2) ] / sqrt(2*pi*sigma^2)
        return - 0.5 * T.square((a - mean0) / std0).sum(axis=1) - 0.5 * T.log(2.0 * np.pi) * self.d - T.log(std0).sum(axis=1)

    def likelihood(self, a, prob):
        return T.exp(self.loglikelihood(a, prob))

    def kl(self, prob0, prob1):
        mean0 = prob0[:, :self.d]
        std0 = prob0[:, self.d:]
        mean1 = prob1[:, :self.d]
        std1 = prob1[:, self.d:]
        return T.log(std1 / std0).sum(axis=1) + ((T.square(std0) + T.square(mean0 - mean1)) / (2.0 * T.square(std1))).sum(axis=1) - 0.5 * self.d

    def entropy(self, prob):
        std_nd = prob[:, self.d:]
        return T.log(std_nd).sum(axis=1) + .5 * np.log(2 * np.pi * np.e) * self.d

    def sample(self, prob):
        mean_nd = prob[:, :self.d]
        std_nd = prob[:, self.d:]
        return np.random.randn(prob.shape[0], self.d).astype(floatX) * std_nd + mean_nd

    def maxprob(self, prob):
        return prob[:, :self.d]


def test_probtypes():
    theano.config.floatX = 'float64'
    np.random.seed(0)

    prob_diag_gauss = np.array([-.2, .3, .4, -.5, 1.1, 1.5, .1, 1.9])
    diag_gauss = DiagGauss(prob_diag_gauss.size // 2)
    yield validate_probtype, diag_gauss, prob_diag_gauss

    prob_categorical = np.array([.2, .3, .5])
    categorical = Categorical(prob_categorical.size)
    yield validate_probtype, categorical, prob_categorical


def validate_probtype(probtype, prob):
    N = 100000
    # Check to see if mean negative log likelihood == differential entropy
    Mval = np.repeat(prob[None, :], N, axis=0)
    M = probtype.prob_variable()
    X = probtype.sampled_variable()
    calcloglik = theano.function([X, M], T.log(probtype.likelihood(X, M)), allow_input_downcast=True)
    calcent = theano.function([M], probtype.entropy(M), allow_input_downcast=True)
    Xval = probtype.sample(Mval)
    logliks = calcloglik(Xval, Mval)
    entval_ll = - logliks.mean()
    entval_ll_stderr = logliks.std() / np.sqrt(N)
    entval = calcent(Mval).mean()
    print entval, entval_ll, entval_ll_stderr
    assert np.abs(entval - entval_ll) < 3 * entval_ll_stderr  # within 3 sigmas

    # Check to see if kldiv[p,q] = - ent[p] - E_p[log q]
    M2 = probtype.prob_variable()
    q = prob + np.random.randn(prob.size) * 0.1
    Mval2 = np.repeat(q[None, :], N, axis=0)
    calckl = theano.function([M, M2], probtype.kl(M, M2), allow_input_downcast=True)
    klval = calckl(Mval, Mval2).mean()
    logliks = calcloglik(Xval, Mval2)
    klval_ll = - entval - logliks.mean()
    klval_ll_stderr = logliks.std() / np.sqrt(N)
    print klval, klval_ll,  klval_ll_stderr
    assert np.abs(klval - klval_ll) < 3 * klval_ll_stderr  # within 3 sigmas


# ================================================================
# Value functions
# ================================================================

class Baseline(object):

    def fit(self, paths):
        raise NotImplementedError

    def predict(self, path):
        raise NotImplementedError


class TimeDependentBaseline(Baseline):

    def __init__(self):
        self.baseline = None

    def fit(self, paths):
        rets = [path["return"] for path in paths]
        maxlen = max(len(ret) for ret in rets)
        retsum = np.zeros(maxlen)
        retcount = np.zeros(maxlen)
        for ret in rets:
            retsum[:len(ret)] += ret
            retcount[:len(ret)] += 1
        retmean = retsum / retcount
        i_depletion = np.searchsorted(-retcount, -4)
        self.baseline = retmean[:i_depletion]
        pred = concat([self.predict(path) for path in paths])
        return {"EV": explained_variance(pred, concat(rets))}

    def predict(self, path):
        if self.baseline is None:
            return np.zeros(pathlength(path))
        else:
            lenpath = pathlength(path)
            lenbase = len(self.baseline)
            if lenpath > lenbase:
                return concat([self.baseline, self.baseline[-1] + np.zeros(lenpath-lenbase)])
            else:
                return self.baseline[:lenpath]


class NnRegression(EzPickle):

    def __init__(self, net, mixfrac=1.0, maxiter=25):
        EzPickle.__init__(self, net, mixfrac, maxiter)
        self.net = net
        self.mixfrac = mixfrac

        x_nx = net.input
        self.predict = theano.function([x_nx], net.output, **FNOPTS)

        ypred_ny = net.output
        ytarg_ny = T.matrix("ytarg")
        var_list = net.trainable_weights
        l2 = 1e-3 * T.add(*[T.square(v).sum() for v in var_list])
        N = x_nx.shape[0]
        mse = T.sum(T.square(ytarg_ny - ypred_ny))/N
        symb_args = [x_nx, ytarg_ny]
        loss = mse + l2
        self.opt = LbfgsOptimizer(loss, var_list, symb_args, maxiter=maxiter, extra_losses={"mse": mse, "l2": l2})

    def fit(self, x_nx, ytarg_ny):
        nY = ytarg_ny.shape[1]
        ypredold_ny = self.predict(x_nx)
        out = self.opt.update(x_nx, ytarg_ny*self.mixfrac + ypredold_ny*(1-self.mixfrac))
        yprednew_ny = self.predict(x_nx)
        out["PredStdevBefore"] = ypredold_ny.std()
        out["PredStdevAfter"] = yprednew_ny.std()
        out["TargStdev"] = ytarg_ny.std()
        if nY == 1:
            out["EV_before"] = explained_variance_2d(ypredold_ny, ytarg_ny)[0]
            out["EV_after"] = explained_variance_2d(yprednew_ny, ytarg_ny)[0]
        else:
            out["EV_avg"] = explained_variance(yprednew_ny.ravel(), ytarg_ny.ravel())
        return out


class NnVf(object):

    def __init__(self, net, timestep_limit, regression_params):
        self.reg = NnRegression(net, **regression_params)
        self.timestep_limit = timestep_limit

    def predict(self, path):
        ob_no = self.preproc(path["observation"])
        return self.reg.predict(ob_no)[:, 0]

    def fit(self, paths):
        ob_no = concat([self.preproc(path["observation"]) for path in paths], axis=0)
        vtarg_n1 = concat([path["return"] for path in paths]).reshape(-1, 1)
        return self.reg.fit(ob_no, vtarg_n1)

    def preproc(self, ob_no):
        return concat([ob_no, np.arange(len(ob_no)).reshape(-1, 1) / float(self.timestep_limit)], axis=1)


class NnCpd(EzPickle):

    def __init__(self, net, probtype, maxiter=25):
        EzPickle.__init__(self, net, probtype, maxiter)
        self.net = net

        x_nx = net.input

        prob = net.output
        a = probtype.sampled_variable()
        var_list = net.trainable_weights

        loglik = probtype.loglikelihood(a, prob)

        self.loglikelihood = theano.function([a, x_nx], loglik, **FNOPTS)
        loss = - loglik.mean()
        symb_args = [x_nx, a]
        self.opt = LbfgsOptimizer(loss, var_list, symb_args, maxiter=maxiter)

    def fit(self, x_nx, a):
        return self.opt.update(x_nx, a)


class SetFromFlat(object):

    def __init__(self, var_list):

        theta = T.vector()
        start = 0
        updates = []
        for v in var_list:
            shape = v.shape
            size = T.prod(shape)
            updates.append((v, theta[start:start+size].reshape(shape)))
            start += size
        self.op = theano.function([theta], [], updates=updates, **FNOPTS)

    def __call__(self, theta):
        self.op(theta.astype(floatX))


class GetFlat(object):

    def __init__(self, var_list):
        self.op = theano.function([], T.concatenate([v.flatten() for v in var_list]), **FNOPTS)

    def __call__(self):
        return self.op()  # pylint: disable=E1101


class EzFlat(object):

    def __init__(self, var_list):
        self.gf = GetFlat(var_list)
        self.sff = SetFromFlat(var_list)

    def set_params_flat(self, theta):
        self.sff(theta)

    def get_params_flat(self):
        return self.gf()


class LbfgsOptimizer(EzFlat):

    def __init__(self, loss,  params, symb_args, extra_losses=None, maxiter=25):
        EzFlat.__init__(self, params)
        self.all_losses = OrderedDict()
        self.all_losses["loss"] = loss
        if extra_losses is not None:
            self.all_losses.update(extra_losses)
        self.f_lossgrad = theano.function(list(symb_args), [loss, flatgrad(loss, params)], **FNOPTS)
        self.f_losses = theano.function(symb_args, self.all_losses.values(), **FNOPTS)
        self.maxiter = maxiter

    def update(self, *args):
        thprev = self.get_params_flat()

        def lossandgrad(th):
            self.set_params_flat(th)
            l, g = self.f_lossgrad(*args)
            g = g.astype('float64')
            return (l, g)
        losses_before = self.f_losses(*args)
        theta, _, opt_info = scipy.optimize.fmin_l_bfgs_b(lossandgrad, thprev, maxiter=self.maxiter)
        del opt_info['grad']
        print opt_info
        self.set_params_flat(theta)
        losses_after = self.f_losses(*args)
        info = OrderedDict()
        for (name, lossbefore, lossafter) in zip(self.all_losses.keys(), losses_before, losses_after):
            info[name+"_before"] = lossbefore
            info[name+"_after"] = lossafter
        return info


def numel(x):
    return T.prod(x.shape)


def flatgrad(loss, var_list):
    grads = T.grad(loss, var_list)
    return T.concatenate([g.flatten() for g in grads])

# ================================================================
# Keras
# ================================================================


class ConcatFixedStd(Layer):

    input_ndim = 2

    def __init__(self, **kwargs):
        Layer.__init__(self, **kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.logstd = theano.shared(np.zeros(input_dim, floatX), name='{}_logstd'.format(self.name))
        self.trainable_weights = [self.logstd]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1] * 2)

    def call(self, x, mask):
        Mean = x
        Std = T.repeat(T.exp(self.logstd)[None, :], Mean.shape[0], axis=0)
        return T.concatenate([Mean, Std], axis=1)

# ================================================================
# Video monitoring
# ================================================================


def VIDEO_NEVER(_):
    return False


def VIDEO_ALWAYS(_):
    return True
