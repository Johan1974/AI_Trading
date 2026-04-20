"""
Bestand: app/datetime_util.py
Functie: `datetime.UTC` bestaat pas vanaf Python 3.11; Docker/CUDA-image gebruikt 3.10.
"""

from datetime import timezone

UTC = timezone.utc
