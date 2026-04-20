"""
Bestand: train_rl_bot.py
Relatief pad: ./train_rl_bot.py
Functie: CLI entrypoint om RL training met 10.000 gesimuleerde trades te starten.
"""

from app.rl.train import train_rl_agent


if __name__ == "__main__":
    result = train_rl_agent(
        market="BTC-EUR",
        interval="1h",
        total_trades=10000,
        total_timesteps=120000,
        model_output_path="artifacts/rl_ppo_bitvavo_2024_2025",
    )
    print("RL training klaar:", result)
