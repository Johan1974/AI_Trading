"""
Bestand: app/exchanges/base.py
Relatief pad: ./app/exchanges/base.py
Functie: Abstracte exchange client interface voor account-info en orderplaatsing.
"""

from abc import ABC, abstractmethod
from typing import Any


class ExchangeClient(ABC):
    @abstractmethod
    def get_balance(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def place_market_order(self, market: str, side: str, amount_quote: float) -> dict[str, Any]:
        raise NotImplementedError
