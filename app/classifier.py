"""
Facade for the ML classify path (POST /tickets/classify/ml).

Primary traffic uses the LLM route in app.api.tickets.
"""

from app.ml_classifier import classify_ticket

__all__ = ["classify_ticket"]
