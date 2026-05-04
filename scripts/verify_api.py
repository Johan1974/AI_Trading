import urllib.request
import json
import sys

def main():
    url = "http://localhost:8000/api/v1/stats"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        print(f"Fout bij ophalen van {url}: {e}")
        sys.exit(1)

    print("--- API Output ---")
    print(json.dumps(data, indent=2))
    print("------------------")

    forbidden_keys = ['p', 'm', 's', 'L', 'fw', 'tm']
    for key in forbidden_keys:
        if key in data:
            print(f"FAIL: Verboden short-key '{key}' is aanwezig in de output.")
            sys.exit(1)

    if 'price' not in data or float(data.get('price', 0.0)) == 0.0:
        print("FAIL: Key 'price' ontbreekt of is 0.0.")
        sys.exit(1)

    required_keys = ['cpu_load', 'gpu_temp', 'market', 'decision_reasoning']
    for key in required_keys:
        if key not in data:
            print(f"FAIL: Verplichte key '{key}' ontbreekt.")
            sys.exit(1)

    print("ALL TESTS PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()