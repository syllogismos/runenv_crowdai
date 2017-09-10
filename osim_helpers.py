import subprocess
import requests
import json
import cPickle
import numpy as np
import pdb
# TODO update server script path below


def start_env_server(p=0, ec2=False, l='127.0.0.1'):
    # subprocess.call(['source', 'deactivate'])
    port = str(5000 + p)
    if ec2:
        server_script_path = '/home/anil/runenv/osim_http_server.py'
    else:
        server_script_path = '/Users/anil/Code/crowdai/runenv/osim_http_server.py'

    command = server_script_path + ' -p ' + port + ' -l ' + l
    process = subprocess.Popen(command, shell=True)
    # subprocess.call(['source', 'activate', 'python3'])
    return process.pid

ip_config = [
    {
        'ip': '127.0.0.1',
        'cores': 2,
        'port': 8018
    },
    {
        'ip': '127.0.0.1',
        'cores': 2,
        'port': 8019
    }
]


redis_config = {
    'host': '127.0.0.1',
    'port': '6379'
}
# start and delete servers python code using requests


def start_osim_apps(ip, port, count):
    url = "http://" + ip + ":" + str(port) + "/start_servers"

    querystring = {"c": str(count), "ip": ip}

    headers = {
        'cache-control': "no-cache"
        }

    response = requests.request("GET", url, headers=headers, params=querystring)

    print(response.text)
    pids = json.loads(response.text)
    return pids


def get_paths_from_server(ip, port, agent, cfg, threads=1):
    url = "http://" + ip + ":" + str(port) + "/get_paths"
    data = {"agent": agent,
            "threads": threads,
            "cfg": cfg}
    payload = json.dumps(data)
    headers = {
        'content-type': "application/json",
        'cache-control': "no-cache"
    }

    print 'in get paths from server'
    # pdb.set_trace()
    response = requests.request("POST", url, data=payload, headers=headers)
    print 'after response'
    paths = json.loads(response.text)
    paths_np = []
    for path in paths:
        path_np = {}
        for key in path.keys():
            if key in ['observation', 'action', 'reward', 'prob']:
                path_np[key] = np.array(path[key])
            else:
                path_np[key] = path[key]
        paths_np.append(path_np)
    return paths_np


def get_paths_from_server_lambda(parallel_config):
    print 'in server lambda'
    con = parallel_config[0]
    if 'port' in con:
        port = con['port']
    else:
        port = 8018
    cfg = parallel_config[2]
    if cfg['redis'] != 1:
        cfg['timesteps_per_batch'] = con['cores']*cfg['timesteps_per_core']
        # if using redis, we share the total batchsize through out
        # all the machines, other wise every node will be seperate
    paths = get_paths_from_server(con['ip'], port,
                                  parallel_config[1],
                                  cfg,
                                  con['cores'])
    return paths


def stop_osim_apps(ip, port, pids):
    url = "http://" + ip + ":" + str(port) + "/delete_servers"
    print url
    payload = json.dumps(pids)
    headers = {
        'content-type': "application/json",
        'cache-control': "no-cache"
        }
    response = requests.request("POST", url, data=payload, headers=headers)
    print response
