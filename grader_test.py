
from osim.http.client import Client
from osim.env import GaitEnv
from osim_http_mrl_client import Client as mrl_client
import logging
remote_base = 'http://grader.crowdai.org'
token = 'a6e5f414845fafd1063253a11429c78f'

client = Client(remote_base)
g = GaitEnv(visualize=False)
# local = mrl_client()
logger = logging.getLogger('osim.http.client')
logger.setLevel(logging.CRITICAL)
def isTrue(a):
    # given an activation a, passes it throgh both grader and local env
    # and checks if the returned observations are the same
    oc = get_obs_cl(a)
    og = get_obs_g(a)
    # ogl = get_obs_local_server(a)
    return oc == og

def get_obs_local_server(a):
    init = local.env_reset()
    return local.env_step(a)[0]

def get_obs_cl(a):
    # resets grader env and takes a single step with given activation a
    init = client.env_create(token)
    return client.env_step(a)[0]

def get_obs_g(a):
    # resets local env and takes a single step with given activation a
    init = g.reset()
    return g.step(a)[0].tolist()

if __name__ == '__main__':
    a = [0.9]*18 # prints True
    print "activation", a
    print "grader and local env give the same observation?", isTrue(a)

    a = [-3.9]*18 # prints True
    print "activation", a
    print "grader and local env give the same observation?", isTrue(a)

    a = [-30.5]*18 # prints True
    print "activation", a
    print "grader and local env give the same observation?", isTrue(a)

    a = [2.5]*18 # prints True
    print "activation", a
    print "grader and local env give the same observation?", isTrue(a)

    a = [-3.115006446838379, # prints False
        -0.7032383680343628,
        1.989966630935669,
        -0.8927109241485596,
        0.6239450573921204,
        -0.9868545532226562,
        -0.48034992814064026,
        0.9246461987495422,
        2.352705955505371,
        1.4393892288208008,
        0.6007359027862549,
        1.2050106525421143,
        -0.1611778736114502,
        -1.5335079431533813,
        1.658748745918274,
        0.1180664598941803,
        0.4392152428627014,
        -0.9457558989524841]
    print "activation", a
    print "grader and local env give the same observation?", isTrue(a)

"""
Results
=======
isTrue([0.9]*18) == True
isTrue([2.0]*18) == True
isTrue([-1.0]*18) == True
isTrue(g.action_space.sample().tolist()) == True # random activations within 0.0 and 1.0

below is an activation where, the same activation on both local gait env, and grader env gives two
different resulting observations.. why is that?
action = [-3.115006446838379,
    -0.7032383680343628,
    1.989966630935669,
    -0.8927109241485596,
    0.6239450573921204,
    -0.9868545532226562,
    -0.48034992814064026,
    0.9246461987495422,
    2.352705955505371,
    1.4393892288208008,
    0.6007359027862549,
    1.2050106525421143,
    -0.1611778736114502,
    -1.5335079431533813,
    1.658748745918274,
    0.1180664598941803,
    0.4392152428627014,
    -0.9457558989524841]
isTrue(action) == False
"""