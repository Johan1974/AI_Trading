import asyncio
import json
import sys

try:
    import websockets
except ImportError:
    import subprocess
    import os
    print("Installeert ontbrekende 'websockets' module...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets", "-q", "--break-system-packages", "--user"])
    os.execv(sys.executable, [sys.executable] + sys.argv)

async def test_websocket():
    uri = "ws://127.0.0.1:8000/ws/canonical-stats"
    max_retries = 20
    
    for attempt in range(max_retries):
        try:
            async with websockets.connect(uri) as websocket:
                print(f"--- Verbinding succesvol na {attempt + 1} poging(en) ---")
                valid_count = 0
                for _ in range(30):  # Luister max ~30 seconden
                    msg = await websocket.recv()
                    data = json.loads(msg)
                    
                    if data.get("status") == "waiting_for_data":
                        continue
                        
                    if data.get("price", 0.0) > 0 and "market" in data and "cpu_load" in data:
                        valid_count += 1
                        
                    if valid_count >= 3:
                        print("WEBSOCKET CONNECTION VERIFIED & DATA RECEIVED")
                        sys.exit(0)
                        
                print("FAIL: Kon geen 3 geldige pakketten ontvangen binnen de tijdlimiet.")
                sys.exit(1)
        except (websockets.exceptions.InvalidHandshake, ConnectionRefusedError) as e:
            if attempt < max_retries - 1:
                print(f"Poging {attempt + 1}/{max_retries}: Server nog niet klaar ({type(e).__name__}). Wachten...")
                await asyncio.sleep(1.5)
            else:
                print(f"FAIL: WebSocket verbinding mislukt na {max_retries} pogingen: {e}")
                sys.exit(1)
        except Exception as e:
            print(f"FAIL: WebSocket verbinding mislukt met onverwachte fout: {e}")
            sys.exit(1)
                    
    print(f"FAIL: Kon na {max_retries} pogingen geen verbinding maken met de WebSocket server.")
    sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_websocket())