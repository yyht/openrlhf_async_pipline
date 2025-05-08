


import requests
import threading
import random
import time

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

s = requests.Session()

class LoadBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.failed_servers = set()
        self.lock = threading.Lock()
        self.health_check_interval = 60  # Health check interval in seconds
        self.current_index = 0

        # Start health check background task
        health_check_thread = threading.Thread(target=self.health_check, daemon=True)
        health_check_thread.start()

    def send_request(self, data, headers, current_index=None, method='', timeout=10):
        start = time.time()
        server = self.select_server(current_index)
        logger.info({
            'INFO': 'SELECTSERVER',
            'TIME': time.time() - start,
            'SERVER': server,
            'METHOD': method
        })
        
        if server is None:
            return "No healthy servers available"
        
        if method:
            url = f"{server}/{method}"
        else:
            url = server
        try:
            start = time.time()
            response = s.post(url, 
                                headers=headers, 
                                json=data,
                                timeout=timeout)
            logger.info({
                'INFO': 'POSTSERVER',
                'TIME': time.time() - start,
                'SERVER': server,
                'METHOD': method
            })
            return response.json()
        except requests.exceptions.RequestException as e:
            self.handle_failure(server)
            return f"Error occurred while sending request: {e}"

    def select_server(self, current_index=None):
        if current_index:
            current_index = self.current_index + current_index
        current_index = current_index % len(self.servers)
        with self.lock:
            if not self.servers:
                return None

            # Simple round-robin load balancing
            server = self.servers[current_index]
            self.current_index = (self.current_index + 1) % len(self.servers)
            return server

    def health_check(self):
        while True:
            time.sleep(self.health_check_interval)
            for server in self.servers.copy():
                url = f"{server}/health"
                try:
                    response = s.get(url)
                    if response.status_code != 200:
                        self.handle_failure(server)
                    else:
                        self.handle_recovery(server)
                except requests.exceptions.RequestException:
                    self.handle_failure(server)

    def handle_failure(self, server):
        # print(f"Server {server} is down. Marking as failed.")
        with self.lock:
            self.servers.remove(server)
            self.failed_servers.add(server)

    def handle_recovery(self, server):
        # print(f"Server {server} is back online. Marking as healthy.")
        with self.lock:
            self.failed_servers.discard(server)
            if server not in self.servers:
                self.servers.append(server)