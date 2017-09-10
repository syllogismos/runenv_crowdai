from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from osim_helpers import start_env_server
import SocketServer
import json
import psutil
import urlparse
import requests
import argparse
import cPickle
from modular_rl.core import get_paths
import gc

PORT_NUMBER = 8018


class myHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        """
        GET /start_servers?c=32&ip=127.0.0.1
        """
        if '/start_servers' in self.path:
            bits = urlparse.urlparse(self.path)
            server_count = 5
            try:
                server_count = int(urlparse.parse_qs(bits.query)['c'][0])
            except:
                print "no c attribute"
            server_ip = '127.0.0.1'
            try:
                server_ip = str(urlparse.parse_qs(bits.query)['ip'][0])
            except:
                print "no ip attribute"
            pids = map(lambda x: start_env_server(x, ec2=False, l=server_ip), range(server_count))
            self.send_response(200)
            self.send_header('Content-type', 'application/javascript')
            self.end_headers()
            self.wfile.write(json.dumps(pids))
        return

    def do_POST(self):
        if self.path == '/delete_servers':
            self.data_string = self.rfile.read(int(self.headers['Content-Length']))
            server_pids = json.loads(self.data_string)
            destroy_servers(server_pids)
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            print "post data is", server_pids
            print "data length", len(server_pids)
            print "server 1", server_pids[0]
            return

        if self.path == '/get_paths':
            print 'get paths'
            self.data_string = self.rfile.read(int(self.headers['Content-Length']))
            json_data = json.loads(self.data_string)
            agent = cPickle.loads(str(json_data['agent']))
            threads = json_data['threads']
            cfg = json_data['cfg']
            envs = [None]*threads
            paths = get_paths(None, agent, cfg, None, envs=envs, threads=threads)
            paths_tolist = []
            for path in paths:
                path_tolist = {}
                for key in path.keys():
                    if key in ['observation', 'action', 'reward', 'prob', 'unscaled_ob']:
                        path_tolist[key] = path[key].tolist()
                        print key, '@@@@@@@@@@@@@@@@@@'
                        json.dumps(path_tolist[key])
                    else:
                        path_tolist[key] = path[key]
                        print key, '@@@@@@@@@@@@@@@@@@@@'
                        json.dumps(path_tolist[key])
                paths_tolist.append(path_tolist)
            gc.collect()
            json.dumps(paths_tolist)

            self.send_response(200)
            self.send_header('Content-type', 'application/javascript')
            self.end_headers()
            self.wfile.write(json.dumps(paths_tolist))


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--listen', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8018)
    args = parser.parse_args()
    server = HTTPServer((args.listen, args.port), myHandler)
    print "started httpserver on port", args.listen, args.port

    server.serve_forever()