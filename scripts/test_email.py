#!/usr/bin/env python3
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import traceback
from pathlib import Path

# Laad de vault handmatig in voor stand-alone gebruik
vault_path = Path.home() / ".trading_vault"
if vault_path.exists():
    with open(vault_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.replace("export ", "").strip()] = val.strip().strip("'\"")

smtp_server = os.getenv("PRIVATE_SMTP")
smtp_port = int(os.getenv("PRIVATE_PORT", "587"))
smtp_user = os.getenv("PRIVATE_EMAIL")
smtp_pass = os.getenv("PRIVATE_PASS")

print(f"--- SMTP CONFIGURATIE ---")
print(f"Server : {smtp_server}")
print(f"Poort  : {smtp_port} (Zorg dat dit 587 is voor STARTTLS!)")
print(f"User   : {smtp_user}")
print("-" * 25)

msg = MIMEMultipart()
msg['From'] = smtp_user
msg['To'] = smtp_user
msg['Subject'] = "🚀 AI Trading Bot - SMTP Debug Test"
msg.attach(MIMEText("Dit is een testbericht. Als je dit leest, werkt SMTP perfect!", 'plain'))

try:
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.set_debuglevel(1)  # Laat alle netwerk/SMTP communicatie in de terminal zien
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
    print("\n✅ SUCCES: E-mail verzonden!")
except Exception as e:
    print("\n❌ FOUT bij het verzenden van e-mail. Traceback hieronder:\n")
    traceback.print_exc()