"""
BESTANDSNAAM: /home/johan/AI_Trading/app/tests/run_ui_tests.py
FUNCTIE: Deep-Scan Watchdog voor het AI Trading Dashboard.
         Dit script voert een autonome, periodieke audit uit op alle UI-tabs 
         (Terminal, AI Brain, Ledger, Hardware). Het controleert op lege velden, 
         valideert of multi-ticker support (ETH/XRP/SOL) aanwezig is, detecteert 
         'bevroren' data (stagnerende prijzen/CPU) en genereert automatische 
         audit-rapporten en heartbeats voor continue systeemmonitoring.
"""

import asyncio
import json
import os
import time
import re
import shutil
from datetime import datetime
import pytz
import urllib.request
import urllib.error
try:
    import docker
except ImportError:
    docker = None
from playwright.async_api import async_playwright

TABS_MAP = {
    "AI Brain": {
        "button": "#btn-aibrain",
        "elements": {
            "#rl-confidence, .rl-confidence": "RL Confidence",
            "#model-version, .model-version": "Model Version",
            "#weight-correlation, #js-corr-value": "AI Correlation Weight",
            "#weight-news, #js-news-weight": "AI News Weight",
            "#weight-price, #js-price-weight": "AI Price Weight",
            "#brainTabStatDiscount": "Discount Factor",
            "#brainTabStatBatch": "Batch Size",
            "#brainLastScanLine": "Laatste Scan Tijd"
        }
    },
    "Ledger": {
        "button": "#btn-ledger",
        "elements": {
            "#total-balance, .total-balance": "Total Balance",
            "#ledgerPerfClosed, #trade-count": "Trade Count",
            "#ledgerPerfPnlEur, #pnl-total": "Total PnL",
            "#ledgerPerfWinRate": "Win Rate",
            "#ledgerPerfMaxWin": "Max Win",
            "#ledgerPerfMaxLoss": "Max Loss",
            "#ledgerPerfHold": "Avg Hold Time"
        }
    },
    "Hardware": {
        "button": "#btn-hardware",
        "elements": {
            "#ringCpuVal, .ringCpuVal": "CPU Load Cirkel",
            "#ringRamVal, .ringRamVal": "RAM Usage Cirkel",
            "#ringGpuVal, .ringGpuVal": "GPU Temp Cirkel",
            "#ringDiskVal, .ringDiskVal": "Disk Usage Cirkel"
        }
    },
    "Terminal": {
        "button": "#btn-terminal",
        "elements": {
            "#btc-price, .btc-price": "Actieve Market Prijs",
            "#sentiment-value, .sentiment-value": "AI Sentiment Score",
            "#market-24h-change, .market-24h-change": "24h Change %",
            "#market-volatility, .market-volatility": "4h Volatility %",
            "#market-reason, .market-reason": "Scanner Reason",
            "#allocatie": "Allocatie Status",
            "#terminalFearGreed": "Fear & Greed Index"
        }
    }
}

last_btc_price = None
last_cpu_val = None
frozen_counter = 0

AMSTERDAM = pytz.timezone("Europe/Amsterdam")

async def check_blackout(page):
    while True:
        if getattr(page, "__network_failures", []):
            return page.__network_failures[0]
        await asyncio.sleep(0.5)

async def wait_with_blackout_check(page, awaitable, failures, root_causes):
    monitor = asyncio.create_task(check_blackout(page))
    action = asyncio.create_task(awaitable)
    done, pending = await asyncio.wait([monitor, action], return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    if monitor in done:
        err = monitor.result()
        failures.append(f"API Blackout gedetecteerd: {err}")
        root_causes.add("backend_api_down")
        raise Exception(f"API Blackout Triggered: {err}")
    return action.result()

async def wait_for_api_health(url):
    print("\n[HEALTH] ⏳ Wachten op API recovery (200 OK op /api/v1/stats)...")
    stats_url = f"{url.rstrip('/')}/api/v1/stats"
    for i in range(15): # Max 30s
        try:
            req = urllib.request.Request(stats_url)
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    print("[HEALTH] ✅ API is weer online (200 OK)!")
                    return True
        except Exception as e:
            print(f"[HEALTH] ⚠️ API nog onbereikbaar: {e}")
        await asyncio.sleep(2)
    print("[HEALTH] ❌ API recovery timeout (30s) bereikt.")
    return False

async def check_multi_tickers(page):
    tickers_env = os.getenv("TICKERS", "BTC-EUR,ETH-EUR,XRP-EUR,SOL-EUR")
    expected_tickers = [t.strip().split('-')[0] for t in tickers_env.split(',')]
    
    try:
        # Give the JS some time to populate the markets dropdown
        try:
            await page.wait_for_function("() => document.querySelectorAll('#marketSelect option').length > 1", timeout=15000)
        except:
            print("  ⚠️ Wachten op market dropdown timeout")

        options = await page.locator('#marketSelect option').all_inner_texts()
        found_text = " ".join(options)
        
        # If dropdown is empty, maybe they are in the page body
        if not found_text.strip():
            found_text = await page.content()
            
        found_non_btc = False
        for expected in expected_tickers:
            if expected != "BTC" and expected in found_text:
                found_non_btc = True
                break
                
        if not found_non_btc and len(expected_tickers) > 1:
            return False, "ERROR: Multi-Ticker Failure (Only BTC found)"
            
        return True, "Multi-Ticker check passed"
    except Exception as e:
        return False, f"Multi-Ticker check failed: {str(e)}"

async def test_market_switch(page):
    try:
        # Wacht tot de dropdown is gevuld door de backend
        await page.wait_for_function("() => document.querySelectorAll('#marketSelect option').length > 1", timeout=10000)
        select_elem = page.locator('#marketSelect')
        options = await select_elem.locator('option').all_inner_texts()
        
        current_val = await select_elem.evaluate("el => el.options[el.selectedIndex] ? el.options[el.selectedIndex].text : ''")
        current_val = current_val.strip()

        target_market = None
        for opt in options:
            opt = opt.strip()
            if opt and opt != current_val:
                target_market = opt
                break
                
        if not target_market:
            return False, "ERROR: Geen alternatieve munt gevonden (zoals ONDO of TRUMP) om switch te testen."

        # 1. Lees huidige (oude) waarden met text_content() (leest de DOM, ongeacht welke tab zichtbaar is)
        old_sentiment = await page.locator("#sentiment-value").first.text_content()
        old_price = await page.locator("#btc-price").first.text_content()

        # 2. Switch uitvoeren
        await select_elem.select_option(label=target_market)
        await page.wait_for_timeout(4000) # Wacht op grafiek en sentiment herberekening via WebSockets
        
        # 3. Lees nieuwe waarden
        new_price = await page.locator("#btc-price").first.text_content()
        new_sentiment = await page.locator("#sentiment-value").first.text_content()
        
        # 4. Deep-Interaction Validatie (State-Change)
        old_price_clean = old_price.strip() if old_price else ""
        new_price_clean = new_price.strip() if new_price else ""
        old_sentiment_clean = old_sentiment.strip() if old_sentiment else ""
        new_sentiment_clean = new_sentiment.strip() if new_sentiment else ""
        
        if old_price_clean == new_price_clean:
            return False, f"Switch mislukt: Prijs bleef hangen op {old_price_clean} na switch naar {target_market}."
            
        if old_sentiment_clean == new_sentiment_clean and old_sentiment_clean not in ["0.000", "0.00", "0"]:
            return False, f"Switch mislukt: Sentiment bleef hangen op {old_sentiment_clean} na switch naar {target_market}."
        
        return True, f"Switch naar {target_market} geslaagd (Prijs: {old_price_clean}->{new_price_clean}, Sent: {old_sentiment_clean}->{new_sentiment_clean})."
    except Exception as e:
        return False, f"Market switch test faalde: {str(e)}"

async def investigate_missing_value(page, element_id, tab_name):
    print(f"\n[INVESTIGATIE] 🔍 Start root-cause analyse voor '#{element_id}' op tab '{tab_name}'...")
    try:
        api_data = await page.evaluate('''async () => {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);
            try {
                const opts = { signal: controller.signal };
                const res = await Promise.allSettled([
                    fetch('/api/v1/stats', opts),
                    fetch('/api/v1/brain/training-monitor', opts),
                    fetch('/api/v1/performance/analytics', opts),
                    fetch('/api/v1/brain/reasoning', opts)
                ]);
                
                const parse = async (r) => {
                    try { return (r.status === 'fulfilled' && r.value.ok) ? await r.value.json() : {}; } 
                    catch (e) { return {}; }
                };
                
                return {
                    stats: await parse(res[0]),
                    brain: await parse(res[1]),
                    ledger: await parse(res[2]),
                    reasoning: await parse(res[3])
                };
            } catch (e) {
                return { error: e.toString() };
            } finally {
                clearTimeout(timeoutId);
            }
        }''')
    except Exception as e:
        print(f"[INVESTIGATIE] ❌ Python error tijdens browser-evaluatie: {e}")
        return "target_crashed" if "crashed" in str(e).lower() or "closed" in str(e).lower() else "backend_api_down"

    if "error" in api_data:
        print(f"[INVESTIGATIE] ❌ API Fetch gefaald vanuit browser: {api_data['error']}. Backend ligt plat.")
        return "backend_api_down"

    is_in_api = False
    api_source = "API"
    api_value = None
    if element_id == "terminalFearGreed":
        is_in_api = "fear_greed_score" in api_data.get("stats", {})
        api_source = "/api/v1/stats -> fear_greed_score"
        api_value = api_data.get("stats", {}).get("fear_greed_score")
    elif element_id in ["brainTabStatDiscount", "brainTabStatBatch"]:
        stats_obj = api_data.get("brain", {}).get("stats", {})
        is_in_api = "discount_factor" in stats_obj or "batch_size" in stats_obj
        api_source = "/api/v1/brain/training-monitor -> stats"
        api_value = stats_obj.get("discount_factor") if element_id == "brainTabStatDiscount" else stats_obj.get("batch_size")
    elif element_id in ["ledgerPerfWinRate", "ledgerPerfMaxWin", "ledgerPerfMaxLoss", "ledgerPerfHold"]:
        ps = api_data.get("ledger", {}).get("analytics", {}).get("performance_summary", {})
        is_in_api = "win_rate_pct" in ps or "max_win_eur" in ps
        api_source = "/api/v1/performance/analytics -> performance_summary"
        api_value = ps.get("win_rate_pct") if element_id == "ledgerPerfWinRate" else ps.get("max_win_eur")
    elif element_id == "brainLastScanLine":
        is_in_api = "generated_at" in api_data.get("reasoning", {})
        api_source = "/api/v1/brain/reasoning -> generated_at"
        api_value = api_data.get("reasoning", {}).get("generated_at")

    if is_in_api or api_value is not None:
        print(f"[INVESTIGATIE] ⚠️ Data voor '{element_id}' ZIT wél in de JSON API via {api_source} (waarde: {api_value}), maar UI toont een lege placeholder.")
        return "js_mapping_error"
        
    print(f"[INVESTIGATIE] 🚨 Data voor '{element_id}' ONTBREEKT in de JSON payload van {api_source}.")
    return "backend_missing_data"

async def trigger_auto_jumpstart(page):
    print("\n[NIGHT-MODE] 🌙 Auto-Jumpstart geactiveerd: Systeem is leeg of ongetraind. Forceren van een paper-trade cyclus...")
    try:
        await page.click("#btn-terminal")
        await page.wait_for_timeout(1000)
        btn = page.locator("#paperBtn")
        if await btn.is_visible():
            await btn.click()
            print("  ⏳ Wachten op AI redenering en trade-executie (max 45s)...")
            await page.wait_for_function(
                "() => { const b = document.querySelector('#paperBtn'); return b && (b.innerText.includes('Succes') || b.innerText.includes('Fout')); }", 
                timeout=45000
            )
            print("  ✅ Auto-Jumpstart voltooid! De AI is geforceerd om na te denken en te handelen.")
            await page.wait_for_timeout(5000)
        else:
            print("  ⚠️ Paper knop niet zichtbaar, kan jumpstart niet uitvoeren.")
    except Exception as e:
        print(f"  ⚠️ Auto-Jumpstart time-out of mislukt: {e}")

def scan_backend_logs():
    print("\n[NIGHT-MODE] 🕵️ Analyseren van backend logs op verstopte fouten...")
    log_path = '/app/logs/worker_execution.log'
    criticals = []
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                for line in f.readlines()[-500:]:
                    if "[CRITICAL]" in line or "Exception" in line or "Traceback" in line:
                        criticals.append(line.strip())
            if criticals:
                print(f"  ⚠️ Gevonden: {len(criticals)} kritieke backend meldingen in de laatste 500 regels.")
            else:
                print("  ✅ Geen kritieke fouten (Exceptions/CRITICAL) in de recente worker logs.")
        except Exception as e:
            print(f"  ⚠️ Kon logs niet lezen: {e}")
    return criticals

async def attempt_fix(failures, root_causes=None, url="http://portal:8000"):
    print("\n[AUTONOMY] 🛠️ Fout gedetecteerd in AI Brain of UI. Start herstel-procedure...")
    print("[AUTONOMY] 🧹 Uitvoeren van 'Schoon Veeg' protocol (oude Docker resten en logs prunen)...")
    
    if not docker:
        print("[AUTONOMY] ⚠️ Docker module ontbreekt. Kan self-healing niet uitvoeren.")
        return
        
    try:
        client = docker.from_env()
        try:
            client.containers.prune()
            client.images.prune(filters={'dangling': True})
        except Exception as e:
            print(f"  ⚠️ Docker prune mislukt: {e}")

        
        f_str = " ".join(failures).lower()
        rc_str = " ".join(root_causes or [])
        
        if "target crashed" in f_str or "target_crashed" in rc_str or "browser crash" in f_str:
            print("[AUTONOMY] ⚠️ Playwright Target Crashed (Browser OOM). Sla container-restarts over; Watchdog herstelt zijn eigen browsercontext.")
        elif "backend_api_down" in rc_str or "api blackout" in f_str or "http " in f_str or "server onbereikbaar" in f_str:
            print("[AUTONOMY] 🛠️ Fout gedetecteerd: Backend API reageert niet of timeout. Herstarten van portal en worker...")
            client.containers.get("ai-trading-portal").restart()
            try:
                client.containers.get("ai-trading-worker").restart()
            except Exception: pass
            print("[AUTONOMY] 🛠️ Zelfstandig herstel van ai-trading-portal en worker uitgevoerd.")
            await wait_for_api_health(url)
        elif "backend_missing_data" in rc_str or "0.000" in f_str or "zero" in f_str or "spook-data" in f_str or "exact nul" in f_str:
            print(f"[AUTONOMY] 🛠️ Fout gedetecteerd: Data corruptie of ontbrekende data in backend. Start Deep Repair (Cache Flush)...")
            try:
                redis_cont = client.containers.get("ai-trading-redis")
                redis_cont.exec_run("redis-cli DEL worker_snapshot ai_trading_snapshot")
                print("[AUTONOMY] 🔧 Deep Repair: Corrupte Redis cache succesvol gewist.")
            except Exception as e:
                print(f"[AUTONOMY] ⚠️ Deep Repair (Redis) mislukt: {e}")
            client.containers.get("ai-trading-worker").restart()
            print("[AUTONOMY] 🛠️ Zelfstandig herstel van ai-trading-worker uitgevoerd.")
        elif "frozen" in f_str or "bevroren" in f_str or "gelijk gebleven" in f_str:
            print("[AUTONOMY] 🛠️ Fout gedetecteerd: Bevroren data. Herstarten van portal...")
            portal = client.containers.get("ai-trading-portal")
            portal.restart()
            print("[AUTONOMY] 🛠️ Zelfstandig herstel van ai-trading-portal uitgevoerd.")
        elif "leeg of ongeldig" in f_str or "placeholder" in f_str or "js_mapping_error" in rc_str or "timeout" in f_str or "kon niet worden uitgelezen" in f_str:
            print("[AUTONOMY] 🛠️ Fout gedetecteerd: UI State Mismatch of JS Mapping fout. Herstarten van portal...")
            client.containers.get("ai-trading-portal").restart()
            print("[AUTONOMY] 🛠️ Zelfstandig herstel van ai-trading-portal uitgevoerd.")
        elif "ongetraind" in f_str or "grafiek is leeg" in f_str or "badges zijn leeg" in f_str:
            print("[AUTONOMY] 🛠️ Fout gedetecteerd: Lege grafieken of ongetraind RL model. Herstarten van worker...")
            worker = client.containers.get("ai-trading-worker")
            worker.restart()
            print("[AUTONOMY] 🛠️ Zelfstandig herstel van ai-trading-worker uitgevoerd.")
        elif "ticker" in f_str or "niet numeriek" in f_str or "ongeldig prijsformaat" in f_str:
            print("[AUTONOMY] 🛠️ Fout gedetecteerd: Ongeldige prijs-data (ticker-naam). Herstarten van portal...")
            portal = client.containers.get("ai-trading-portal")
            portal.restart()
            print("[AUTONOMY] 🛠️ Zelfstandig herstel van ai-trading-portal uitgevoerd.")
        else:
            print("[AUTONOMY] 🛠️ Fout gedetecteerd: Algemene UI fout. Herstarten van portal...")
            portal = client.containers.get("ai-trading-portal")
            portal.restart()
            print("[AUTONOMY] 🛠️ Zelfstandig herstel van ai-trading-portal uitgevoerd.")
            
    except Exception as e:
        print(f"[AUTONOMY] 🚨 FATALE DOCKER API FOUT: {e}")
        print("   👉 OPLOSSING 1 (Meest waarschijnlijk): Run 'docker compose up -d --force-recreate dashboard-validator'")
        print("   👉 OPLOSSING 2 (Host Permissies): Run 'sudo chmod 666 /var/run/docker.sock' op de server")


async def run_deep_scan():
    global last_btc_price, last_cpu_val, frozen_counter
    
    debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
    base_interval = 60 if debug_mode else 300 # 5 minuten default
    url = os.getenv("API_URL", "http://portal:8000")
    
    print(f"[{time.strftime('%H:%M:%S')}] 🚀 STARTING DEEP-SCAN WATCHDOG (NIGHT-MODE ACTIEF) op {url}")
    print("   -> Validator zal autonoom problemen oplossen, paper trades forceren en logs analyseren.")

    if docker:
        try:
            client = docker.from_env()
            print("[AUTONOMY] 🧹 Uitvoeren van initiële 'Schoon Veeg' (docker image prune -f)...")
            client.images.prune(filters={'dangling': True})
        except Exception as e:
            print(f"[AUTONOMY] ⚠️ Initiële prune mislukt: {e}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"])
        
        consecutive_successes = 0 # Begin altijd in stabiliteits-check modus
        while True:
            try:
                page = await browser.new_page()
            except Exception as e:
                print(f"[AUTONOMY] ⚠️ Browser crash gedetecteerd: {e}. Herstarten van Playwright...")
                try:
                    await browser.close()
                except Exception:
                    pass
                browser = await p.chromium.launch(headless=True, args=["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"])
                page = await browser.new_page()

                page.__network_failures = []
                def handle_console(msg):
                    text = msg.text.lower()
                    if "err_empty_response" in text or "failed to fetch" in text or "websocket connection failed" in text:
                        print(f"\n[NETWORK] 🚨 API-verbinding verbroken! (Console: {msg.text})")
                        page.__network_failures.append(f"Console: {msg.text}")

                def handle_request_failed(request):
                    err_txt = request.failure.lower() if request.failure else ""
                    if "err_empty_response" in err_txt or "failed to fetch" in err_txt or "connection refused" in err_txt:
                        print(f"\n[NETWORK] 🚨 API-verbinding verbroken! (Request failed: {request.url} - {request.failure})")
                        page.__network_failures.append(f"Request failed: {request.failure}")

                page.on("console", handle_console)
                page.on("requestfailed", handle_request_failed)

            scan_failed = True
            # Preventie voor UnboundLocalError mocht het script crashen vóór de test-logica
            failures = []
            root_causes = set()
            try:
                # FORCEER PADEN: Zorg dat de log-directory altijd bestaat voor de JSON-reports, 
                # zelfs als een host-script de map tussentijds heeft gewist.
                os.makedirs('/app/logs', exist_ok=True)
                
                # SCHOON VEEG PROTOCOL: Verwijder alle oude json/txt/png bestanden, 
                # maar behoud heartbeat en actieve server logs/databases.
                for fname in os.listdir('/app/logs'):
                    if fname != "heartbeat.json" and not fname.endswith('.log') and not fname.endswith('.sqlite'):
                        fpath = os.path.join('/app/logs', fname)
                        if os.path.isfile(fpath):
                            try:
                                os.remove(fpath)
                            except Exception:
                                pass

                extracted_data = {}
                audit_details = {}
                data_frozen = False
                
                # AUTONOME SCHIJFBEWAKING
                total, used, free = shutil.disk_usage("/")
                free_gb = free / (1024**3)
                if free_gb < 5.0:
                    disk_warning = f"Schijfruimte kritiek: {free_gb:.2f}GB vrij (Onder 5GB limiet!)"
                    
                    # DATA-SOVEREIGNTY: Zelf log-inspectie doen op permissie fouten
                    try:
                        with open('/app/logs/worker_execution.log', 'r', encoding='utf-8', errors='replace') as f:
                            tail = f.readlines()[-200:]
                            for line in tail:
                                if "Permission denied" in line or "Errno 13" in line:
                                    disk_warning += " 🛑 ROOT CAUSE: 'Permission denied' in worker_execution.log gevonden! Fix _logs_hub permissies."
                                    break
                    except Exception:
                        pass
                        
                    print(f"  ⚠️ [AUTONOME SCHIJFBEWAKING] {disk_warning}")
                    failures.append(disk_warning)
                    audit_details["Disk Space"] = {"status": "FAILED", "value": f"{free_gb:.2f}GB"}
                else:
                    audit_details["Disk Space"] = {"status": "OK", "value": f"{free_gb:.2f}GB"}

                for attempt in range(2):
                    print(f"\n[{time.strftime('%H:%M:%S')}] 🔄 Deep-Scan Ronde starten (Poging {attempt + 1}/2)...")
                    if attempt == 0:
                        print("  ⏳ Wachten op succesvolle UI data-sync (DOM check)...")
                        try:
                        response = await wait_with_blackout_check(page, page.goto(url), failures, root_causes)
                        if response and response.status >= 400:
                            msg = f"HTTP Fout {response.status} op hoofdpagina ({url})"
                            failures.append(msg)
                            root_causes.add("backend_api_down")
                            raise Exception(msg)
                            
                        await wait_with_blackout_check(page, page.wait_for_function(
                                "() => { const el = document.querySelector('#btc-price'); return el && !el.innerText.includes('—') && !el.innerText.includes('--') && !el.innerText.includes('Laden'); }",
                                timeout=15000
                        ), failures, root_causes)
                            await page.wait_for_timeout(1500)
                            print("  ✅ Data synchronisatie voltooid.")
                    except Exception as e:
                        err_str = str(e).lower()
                        if "net::err" in err_str or "connection refused" in err_str:
                            msg = f"Server onbereikbaar ({url}): netwerkverbinding geweigerd"
                            if msg not in failures: failures.append(msg)
                            root_causes.add("backend_api_down")
                        if "http" not in err_str and "net::err" not in err_str and page.url == "about:blank":
                            try: await page.goto(url)
                            except: pass
                        print(f"  ⚠️ Fout tijdens initiële laadfase: {e}")
                    else:
                        print("  🛠️ AUTO-HEAL ACTIEF: Pagina herladen en wachten op DOM sync...")
                        try:
                        response = await wait_with_blackout_check(page, page.reload(), failures, root_causes)
                        if response and response.status >= 400:
                            msg = f"HTTP Fout {response.status} op hoofdpagina tijdens herladen"
                            failures.append(msg)
                            root_causes.add("backend_api_down")
                            raise Exception(msg)
                            
                        await wait_with_blackout_check(page, page.wait_for_function(
                                "() => { const el = document.querySelector('#btc-price'); return el && !el.innerText.includes('—') && !el.innerText.includes('--') && !el.innerText.includes('Laden'); }",
                                timeout=15000
                        ), failures, root_causes)
                            await page.wait_for_timeout(1500)
                    except Exception as e:
                        err_str = str(e).lower()
                        if "net::err" in err_str or "connection refused" in err_str:
                            msg = f"Server onbereikbaar tijdens herladen: netwerkverbinding geweigerd"
                            if msg not in failures: failures.append(msg)
                            root_causes.add("backend_api_down")
                        print(f"  ⚠️ Fout tijdens auto-heal laadfase: {e}")
                            await page.wait_for_timeout(5000)
                        failures = []
                        extracted_data = {}
                        audit_details = {}
                        data_frozen = False
                        root_causes = set()

                    # Tab-hopping and element checking
                    for tab_name, tab_data in TABS_MAP.items():
                        print(f"  👉 Checking tab: {tab_name}")
                        try:
                            await page.click(tab_data["button"])
                            
                            # Smart waits voor lazy-loaded tab data (wacht tot de fetch API klaar is)
                            if tab_name == "AI Brain":
                                try:
                                    await page.wait_for_function("() => { const el = document.querySelector('#brainTabStatDiscount'); return el && !el.innerText.includes('—'); }", timeout=5000)
                                except Exception: pass
                            elif tab_name == "Ledger":
                                try:
                                    await page.wait_for_function("() => { const el = document.querySelector('#ledgerPerfWinRate'); return el && !el.innerText.includes('—'); }", timeout=5000)
                                except Exception: pass
                            
                            await page.wait_for_timeout(1000) # Extra animatie/render buffer
                            
                            for selector, name in tab_data["elements"].items():
                                try:
                                    element_id = selector.split(",")[0].replace("#", "").replace(".", "").strip()
                                    loc = page.locator(selector).first
                                    await loc.wait_for(state="attached", timeout=3000)
                                    # FIX: Gebruik text_content() i.p.v. inner_text() om timeouts op verborgen parent-divs te stoppen
                                    content = await loc.text_content()
                                    content = content.strip() if content else ""
                                    extracted_data[element_id] = content
                                    
                                    # Check op 'lege' of 'foute' waardes
                                    empty_vals = ["--", "—", "-", "NaN", "undefined", ""]
                                    placeholders = ["laden...", "laden…", "none", "---", "laatste scan: —", "wacht op data"]
                                    content_lower = content.lower()
                                    is_placeholder = any(p in content_lower for p in placeholders)
                                    is_zero = content in ["0%", "0.0000", "0.000", "0.00", "0", "0.0"]
                                    allowed_zeros = [
                                        "pnl-total", "ledgerPerfPnlEur", "trade-count", "ledgerPerfClosed", 
                                        "ringCpuVal", "ringRamVal", "ringGpuVal", "ringDiskVal",
                                        "ledgerPerfMaxWin", "ledgerPerfMaxLoss", "ledgerPerfHold", "terminalFearGreed",
                                        "rl-confidence"
                                    ] # Hardware en ledger mogen 0 zijn. Ongetraind RL model (0.00) triggert alleen een waarschuwing.

                                    if content in empty_vals or is_placeholder:
                                        suggestion = " 💡 Suggestie: Data ontbreekt in API of main.js mist de mapping." if "weight" in element_id or "sentiment" in element_id else ""
                                        print(f"    - [VALUE] {name}: '{content}' ❌ (FAILED: Waarde is leeg of een placeholder){suggestion}")
                                        failures.append(f"{name} (#{element_id}) op {tab_name} tab")
                                        audit_details[name] = {"status": "FAILED", "value": f"{content}{suggestion}"}
                                        rc = await investigate_missing_value(page, element_id, tab_name)
                                        root_causes.add(rc)
                                    elif is_zero and element_id not in allowed_zeros:
                                        suggestion = " 💡 Suggestie: Oninitialized data of een falende API-pijplijn. Worker wordt herstart." if element_id in ["weight-correlation", "weight-news", "weight-price", "sentiment-value"] else ""
                                        print(f"    - [VALUE] {name}: '{content}' ❌ (FAILED: Waarde is exact nul - Spook-data!){suggestion}")
                                        failures.append(f"{name} (#{element_id}) op {tab_name} tab (Value is zero)")
                                        audit_details[name] = {"status": "FAILED", "value": f"{content}{suggestion}"}
                                    else:
                                        # Content-Aware Check & Gainer-Metrics
                                        if element_id == "rl-confidence" and content in ["0.00", "0", "0.000"]:
                                            print(f"    - [VALUE] {name}: {content} ⚠️ (Waarschuwing: RL Agent is ongetraind, maar dit is valide)")
                                            audit_details[name] = {"status": "WARNING", "value": content}
                                        elif element_id == "btc-price":
                                            clean_content = content.replace("€", "").strip().upper()
                                            ticker_blacklist = ["BTC", "ETH", "TRUMP", "ONDO", "SOL", "XRP", "ADA", "DOT", "AVAX", "LINK", "EUR"]
                                            price_pattern = r'^[-+]?€?\s?(?:\d{1,3}(?:[.,]\d{3})*|\d+)(?:[.,]\d+)?(\s?%?)?$'
                                            
                                            if clean_content in ticker_blacklist or (clean_content.isalpha() and len(clean_content) <= 5):
                                                suggestion = " 💡 Suggestie: main.js mapt per ongeluk een ticker-naam naar het prijsveld, of de API levert stale data. Portal wordt herstart."
                                                print(f"    - [VALUE] {name}: '{content}' ❌ (FOUT: Waarde is niet numeriek, ticker-naam gedetecteerd!){suggestion}")
                                                failures.append(f"{name} (#{element_id}) op {tab_name} tab (Ticker naam ipv prijs)")
                                                audit_details[name] = {"status": "FAILED", "value": f"{content}{suggestion}"}
                                            elif not re.match(price_pattern, content.strip()):
                                                suggestion = " 💡 Suggestie: Controleer de prijsformattering in main.js of de payload van portal."
                                                print(f"    - [VALUE] {name}: '{content}' ❌ (FOUT: Waarde is niet numeriek of ongeldig geformatteerd){suggestion}")
                                                failures.append(f"{name} (#{element_id}) op {tab_name} tab (Ongeldig prijsformaat)")
                                                audit_details[name] = {"status": "FAILED", "value": f"{content}{suggestion}"}
                                            else:
                                                print(f"    - [VALUE] {name}: {content} ✅ (Numerieke Prijs Check: OK)")
                                                audit_details[name] = {"status": "OK", "value": content}
                                        elif element_id == "sentiment-value":
                                            try:
                                                val = float(content)
                                                if val < -1.0 or val > 1.0:
                                                    print(f"    - [VALUE] {name}: '{content}' ❌ (FAILED: Buiten marge [-1.0, 1.0])")
                                                    failures.append(f"{name} (#{element_id}) op {tab_name} tab (Buiten marge)")
                                                    audit_details[name] = {"status": "FAILED", "value": content}
                                                else:
                                                    print(f"    - [VALUE] {name}: {content} ✅ (Marge Check: OK)")
                                                    audit_details[name] = {"status": "OK", "value": content}
                                            except ValueError:
                                                print(f"    - [VALUE] {name}: '{content}' ❌ (FAILED: Niet numeriek)")
                                                failures.append(f"{name} (#{element_id}) op {tab_name} tab mist numerieke waarde")
                                                audit_details[name] = {"status": "FAILED", "value": content}
                                        elif element_id in ["market-24h-change", "market-volatility"]:
                                            num_str = re.sub(r'[^\d\.\,\-]', '', content)
                                            if not num_str:
                                                print(f"    - [VALUE] {name}: '{content}' ❌ (FAILED: Mist een numerieke waarde)")
                                                failures.append(f"{name} (#{element_id}) op {tab_name} tab mist een numerieke waarde: '{content}'")
                                                audit_details[name] = {"status": "FAILED", "value": content}
                                            else:
                                                print(f"    - [VALUE] {name}: {content} ✅")
                                                audit_details[name] = {"status": "OK", "value": content}
                                        else:
                                            print(f"    - [VALUE] {name}: {content} ✅")
                                            audit_details[name] = {"status": "OK", "value": content}
                                except Exception as el_err:
                                    suggestion = " 💡 Suggestie: Zorg dat main.js (updateUI) deze velden vult, of controleer de HTML op dit ID." if "weight" in element_id else ""
                                    print(f"    - [VALUE] {name} (#{element_id}) ❌ (FAILED: Element niet gevonden of timeout in de DOM){suggestion}")
                                    failures.append(f"Element {name} (#{element_id}) op {tab_name} tab kon niet worden uitgelezen")
                                    audit_details[name] = {"status": "FAILED", "value": f"N/A (Error){suggestion}"}
                        except Exception as e:
                            print(f"  ❌ Fout bij navigeren naar tab {tab_name}: {str(e)}")
                            failures.append(f"Tab switch failed voor {tab_name}: {str(e)}")
                            for name in tab_data["elements"].values():
                                if name not in audit_details:
                                    audit_details[name] = {"status": "FAILED", "value": "N/A (Error)"}

                    # --- DEEP VISUAL INSPECTION (Charts & Badges) ---
                    print(f"  👉 Checking Visual Data (Charts & Badges)...")
                    try:
                        # Verifieer dat Signal Badges niet leeg zijn
                        badges_empty = await page.evaluate("""() => {
                            const badges = document.querySelectorAll('.brain-strategy-groups__chip');
                            if (badges.length === 0) return true;
                            let all_empty = true;
                            badges.forEach(b => { if (b.innerText.trim() !== '') all_empty = false; });
                            return all_empty;
                        }""")
                        if badges_empty:
                            print("    - [VALUE] Signal Mix Badges ❌ (FAILED: Badges zijn leeg of afwezig)")
                            failures.append("Signal Mix Badges zijn leeg")
                        else:
                            print("    - [VALUE] Signal Mix Badges ✅ (Aanwezig en gevuld)")

                        # Verifieer Grafieken (Loss, Reward, Equity, Price)
                        chart_ids = [
                            "priceChart", "brainCorrelationChart", "brainTabTrainingLossChart",
                            "brainTabFeatureChart", "brainTabRewardChart", "equityCurveChart",
                            "winLossChart", "sentimentOutcomeChart"
                        ]
                        for cid in chart_ids:
                            is_empty = await page.evaluate(f"""() => {{
                                const el = document.getElementById('{cid}');
                                if (!el) return true;
                                if (el.tagName.toLowerCase() === 'canvas') {{ return el.width === 0 || el.height === 0 || el.style.display === 'none'; }}
                                return el.innerHTML.trim() === '';
                            }}""")
                            if is_empty:
                                print(f"    - [VALUE] Chart #{cid} ❌ (FAILED: Grafiek is leeg of niet gerenderd)")
                                failures.append(f"Grafiek #{cid} is leeg")
                            else:
                                print(f"    - [VALUE] Chart #{cid} ✅ (Bevat pixel-data)")
                    except Exception as e:
                        print(f"  ❌ Fout bij grafiek validatie: {e}")

                    # Multi-Ticker Check
                    multi_ticker_ok, multi_ticker_msg = await check_multi_tickers(page)
                    if not multi_ticker_ok:
                        failures.append(multi_ticker_msg)
                        audit_details["Multi-Ticker"] = {"status": "FAILED", "value": multi_ticker_msg}
                    else:
                        audit_details["Multi-Ticker"] = {"status": "OK", "value": "Passed"}

                    # Market Switch Check
                    switch_ok, switch_msg = await test_market_switch(page)
                    if not switch_ok:
                        failures.append(switch_msg)
                        audit_details["Market-Switch"] = {"status": "FAILED", "value": switch_msg}
                    else:
                        audit_details["Market-Switch"] = {"status": "OK", "value": switch_msg}

                    # Night-Mode: Jumpstart the engine if it's empty
                    rl_conf = extracted_data.get("rl-confidence", "")
                    trades = extracted_data.get("ledgerPerfClosed", "")
                    if rl_conf in ["0.00", "0", "0.000", "—"] or trades in ["0", "—"]:
                        await trigger_auto_jumpstart(page)

                    # Data Freshness Check
                    current_btc_price = extracted_data.get("btc-price")
                    current_cpu_val = extracted_data.get("ringCpuVal")
                    
                    if current_btc_price and current_cpu_val:
                        if current_btc_price == last_btc_price and current_cpu_val == last_cpu_val:
                            if last_btc_price is not None and last_cpu_val is not None:
                                frozen_counter += 1
                                if frozen_counter >= 3:
                                    print(f"⚠️ WAARSCHUWING: Data Frozen! Prijs ({current_btc_price}) en CPU ({current_cpu_val}) ongewijzigd over 3 scans.")
                                    failures.append("Data Frozen (BTC Prijs en CPU gelijk gebleven)")
                                    data_frozen = True
                        else:
                            frozen_counter = 0
                    
                    if not failures and not data_frozen:
                        break # Success! We kunnen uit de retry-loop breken
                        
                last_btc_price = extracted_data.get("btc-price")
                last_cpu_val = extracted_data.get("ringCpuVal")

                recent_errors = scan_backend_logs()
                if recent_errors:
                    uniq_err = list(dict.fromkeys(recent_errors))
                    audit_details["Backend_Log_Analyse"] = {"status": "WARNING", "value": f"{len(recent_errors)} fouten gevonden in worker_execution.log. Laatste: {uniq_err[-1][:60]}..."}

                # Report generation
                audit_report = {
                    "timestamp": datetime.now(AMSTERDAM).isoformat(),
                    "status": "OK" if not failures else ("WARNING: Data Frozen" if data_frozen else "ERROR"),
                    "metrics": audit_details
                }

                if failures:
                    print(f"❌ AUDIT GEFAALD: {', '.join(failures)}")
                    try:
                        screenshot_bytes = await page.screenshot()
                        with open("/app/logs/AUDIT_FAILURE.png", "wb") as f:
                            f.write(screenshot_bytes)
                    except Exception as ss_err:
                        print(f"  ⚠️ Kon screenshot niet opslaan: {ss_err}")
                    with open("/app/logs/audit_status.txt", "w") as f:
                        f.write(f"FAILED: {', '.join(failures)}")
                else:
                    print("✅ DEEP-SCAN PASSED: Full Tour succesvol en multi-ticker gecontroleerd!")
                    scan_failed = False
                    if os.path.exists("/app/logs/audit_status.txt"):
                        os.remove("/app/logs/audit_status.txt")
                    
                    os.makedirs('/app/logs', exist_ok=True)
                    # Update heartbeat only on success
                    with open("/app/logs/heartbeat.json", "w") as f:
                        json.dump({"last_heartbeat": datetime.now(AMSTERDAM).isoformat(), "status": "ALIVE", "type": "deep_scan"}, f)

                os.makedirs('/app/logs', exist_ok=True)
                with open("/app/logs/last_audit_report.json", "w") as f:
                    json.dump(audit_report, f, indent=2)

            except Exception as e:
                print(f"💥 KRITIEKE FOUT in run loop: {e}")
            
            if scan_failed:
                consecutive_successes = 0
                sleep_time = 30
                # Trigger Active Error Correction (Autonome Self-Healing)
            await attempt_fix(failures, list(root_causes), url)
                print(f"[{time.strftime('%H:%M:%S')}] ⏳ Deep-Scan voltooid. Wachten op data-sync ({sleep_time}s)...")
            else:
                consecutive_successes += 1
                if consecutive_successes <= 3:
                    sleep_time = 60
                    print(f"[AUTONOMY] 🛡️ 'Healing' fase actief ({consecutive_successes}/3 succesvolle rondes). Interval geforceerd naar 60s.")
                else:
                    sleep_time = base_interval
                    print(f"[AUTONOMY] ✅ Systeem stabiel. Schaal autonoom op naar rust-interval ({sleep_time}s).")
                    
                print(f"[{time.strftime('%H:%M:%S')}] ⏳ Deep-Scan voltooid. Wachten voor {sleep_time} seconden...")
            
            try:
                await page.close()
            except Exception:
                pass
                
            await asyncio.sleep(sleep_time)
            
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run_deep_scan())
