"""
VRAM Monitor: Gebruik nvidia-smi om het verbruik te meten.
Veiligheidsmarge: Als het VRAM boven de 7.5GB komt, moet hij stoppen met nieuwe Ollama-taken om mijn trading-bot (die ~626MiB gebruikt) te beschermen.
Log Analyse: Scan de logs die je zojuist in het project hebt gevonden op foutmeldingen.
"""

import subprocess
import os

def get_vram_usage():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    memory_used, memory_total = map(int, result.stdout.strip().split(','))
    return memory_used, memory_total

def check_vram_usage():
    memory_used, _ = get_vram_usage()
    if memory_used > 7.5 * 1024 * 1024 * 1024:  # 7.5GB in bytes
        print("VRAM usage exceeded the safe limit. Stopping Ollama tasks.")
        os.system('docker-compose down')
        exit(1)

def analyze_logs(log_file):
    with open(log_file, 'r') as file:
        for line in file:
            if 'error' in line.lower():
                print(f"Error found in logs: {line.strip()}")

if __name__ == "__main__":
    check_vram_usage()
    log_file = '/path/to/logs/ollama.log'  # Update this path with the actual log file location
    analyze_logs(log_file)