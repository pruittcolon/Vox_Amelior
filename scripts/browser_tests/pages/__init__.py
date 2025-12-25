"""
Pages Test Package
==================
Individual page test modules.
"""

from .banking import BankingTests
from .gemma import GemmaTests
from .dashboard import DashboardTests
from .predictions import PredictionsTests
from .admin_qa import AdminQATests
from .common import CommonPageTests
from .banking_enterprise import BankingEnterpriseTests
from .salesforce import SalesforceTests, SalesforceAITests, SalesforceEnterpriseTests

__all__ = [
    'BankingTests',
    'GemmaTests', 
    'DashboardTests',
    'PredictionsTests',
    'AdminQATests',
    'CommonPageTests',
    'BankingEnterpriseTests',
    'SalesforceTests',
    'SalesforceAITests',
    'SalesforceEnterpriseTests',
]

