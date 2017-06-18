import subprocess
import requests
import json
#TODO update server script path below

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
        'cores': 3
    }
]
# start and delete servers python code using requests

def start_osim_apps(ip, port, count):
    url = "http://" + ip + ":" + str(port) + "/start_servers"

    querystring = {"c":str(count), "ip": ip}

    headers = {
        'cache-control': "no-cache"
        }

    response = requests.request("GET", url, headers=headers, params=querystring)

    print(response.text)
    pids = json.loads(response.text)
    return pids

def stop_osim_apps(ip, port, pids):
    url = 'http://' + ip + ':' + str(port) + '/delete_servers'
    payload = json.dumps(pids)
    headers = {
        'content-type': "application/json",
        'cache-control': "no-cache"
        }

    response = requests.request("POST", url, data=payload, headers=headers)
