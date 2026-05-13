import urllib.request
import json
import time
import os
import subprocess

# --- CONFIGURATIE ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ai-trading-ollama:11434")
VRAM_LIMIT_MIB = 7500  # Veiligheidsgrens voor je GPU (7.5GB)
INTERVAL = int(os.getenv("JANITOR_INTERVAL", "30"))
LOG_FILE = '/app/logs/persistent_crash.log'

def call_ollama(path, data=None):
    """Communicatie met de Ollama API."""
    try:
        req = urllib.request.Request(
            OLLAMA_HOST + path,
            data=json.dumps(data).encode() if data else None,
            headers={"Content-Type": "application/json"} if data else {},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"[janitor] Ollama API fout: {e}", flush=True)
        return None

def get_vram_usage():
    """Leest het werkelijke VRAM verbruik direct van de NVIDIA kaart."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
            capture_output=True, text=True, check=True
        )
        used, total = map(int, result.stdout.strip().split(','))
        return used, total
    except Exception:
        return 0, 8192

def enforce_redis_policy():
    """Fix voor de 51.327 blokkade: dwingt allkeys-lru af."""
    try:
        # Check eerst de huidige policy
        check = subprocess.run(
            ['docker', 'exec', 'ai-trading-redis', 'redis-cli', 'CONFIG', 'GET', 'maxmemory-policy'],
            capture_output=True, text=True
        )
        if "allkeys-lru" not in check.stdout:
            print("[janitor] 🚨 Redis policy incorrect! Herstellen...", flush=True)
            subprocess.run(
                ['docker', 'exec', 'ai-trading-redis', 'redis-cli', 'CONFIG', 'SET', 'maxmemory-policy', 'allkeys-lru'],
                check=True
            )
            print("[janitor] ✅ Redis policy hersteld naar allkeys-lru.", flush=True)
    except Exception as e:
        print(f"[janitor] Redis check mislukt: {e}", flush=True)

def monitor_trading_health():
    """Controleert of de bot nog leeft en updates stuurt naar Redis."""
    try:
        import redis
        r = redis.Redis(host='ai-trading-redis', port=6379, db=0, socket_timeout=5)
        last_update = r.get('last_pnl_update_timestamp')
        if last_update:
            diff = time.time() - float(last_update)
            if diff > 300:
                print(f"[janitor] ⚠️ STAGNATIE: Geen PnL update voor {int(diff)}s!", flush=True)
        
        # Check Global Steps
        steps = r.get('global_training_steps')
        if steps:
            print(f"[janitor] Progressie: {steps.decode()} stappen.", flush=True)
    except Exception as e:
        print(f"[janitor] Health check fout: {e}", flush=True)

def cleanup_vram():
    """Ontlaadt modellen als de grens wordt bereikt."""
    vram_used, _ = get_vram_usage()
    if vram_used > VRAM_LIMIT_MIB:
        print(f"[janitor] ⚠️ VRAM kritiek ({vram_used} MiB). Modellen ontladen...", flush=True)
        ps = call_ollama("/api/ps")
        if ps and "models" in ps:
            for m in ps["models"]:
                call_ollama("/api/generate", {"model": m["name"], "keep_alive": 0})
                print(f"[janitor] Afgevoerd: {m['name']}", flush=True)

def analyze_logs():
    """Scant op crashes in de logbestanden."""
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                last_lines = f.readlines()[-10:]
                for line in last_lines:
                    if 'error' in line.lower() or 'crash' in line.lower():
                        print(f"[janitor] 🚨 Log Alarm: {line.strip()}", flush=True)
        except Exception: pass

# --- MAIN LOOP ---
if __name__ == "__main__":
    print(f"[janitor] Systeemwacht gestart (Limit: {VRAM_LIMIT_MIB} MiB, Interval: {INTERVAL}s)", flush=True)
    
    while True:
        enforce_redis_policy()    # Voorkom bevriezing
        cleanup_vram()            # Bescherm GPU
        monitor_trading_health()  # Check bot progressie
        analyze_logs()            # Check op errors
        
        time.sleep(INTERVAL)