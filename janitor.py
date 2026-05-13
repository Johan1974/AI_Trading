import os
import time
from datetime import datetime, timedelta

def log_hub_monitor():
    logs_dir = 'logs_hub'
    if not os.path.exists(logs_dir):
        print(f"Directory {logs_dir} does not exist.")
        return
    
    for filename in os.listdir(logs_dir):
        if filename.endswith('.log') or filename.endswith('.txt'):
            file_path = os.path.join(logs_dir, filename)
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            current_time = datetime.now()
            time_diff = current_time - file_time
            
            if time_diff > timedelta(hours=48):
                print(f"Deleting old log file: {file_path}")
                os.remove(file_path)

def docker_log_truncate(container_name):
    try:
        log_path = subprocess.check_output(['docker', 'inspect', '--format=\'{{.LogPath}}\'', container_name]).decode('utf-8').strip()
        if os.path.exists(log_path) and os.path.getsize(log_path) > 100 * 1024 * 1024:
            print(f"Truncating log file for {container_name}: {log_path}")
            with open(log_path, 'w') as f:
                pass
    except subprocess.CalledProcessError as e:
        print(f"Failed to truncate log for {container_name}: {e}")

if __name__ == "__main__":
    containers = ['ai-trading-worker', 'ai-trading-portal']
    
    for container in containers:
        docker_log_truncate(container)
    
    log_hub_monitor()