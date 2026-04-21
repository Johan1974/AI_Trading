"""
Bestand: core/engine.py
Relatief pad: ./core/engine.py
Functie: Re-export van TradingEngine (multi-asset loop staat in core.trading_engine).
"""

from __future__ import annotations

from core.trading_engine import TradingEngine

__all__ = ["TradingEngine"]
