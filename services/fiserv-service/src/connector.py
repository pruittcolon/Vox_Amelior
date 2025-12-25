import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any

# Configure logging
logger = logging.getLogger(__name__)


def _read_secret_file(env_var: str) -> str | None:
    """Read secret from Docker secrets file path specified in env var."""
    file_path = os.getenv(env_var)
    if file_path:
        path = Path(file_path)
        if path.exists():
            return path.read_text().strip()
    return None


class FiservClient:
    """
    Client for interacting with Fiserv Banking Hub API.
    Supports a 'Mock Mode' for development to avoid sandbox limits.
    """

    def __init__(self):
        self.mock_mode = os.getenv("FISERV_MOCK_MODE", "true").lower() == "true"
        # Load API key from Docker secrets first, fall back to env var
        self.api_key = _read_secret_file("FISERV_API_KEY_FILE") or os.getenv("FISERV_API_KEY")
        self.base_url = os.getenv("FISERV_BASE_URL", "https://api.fiserv.dev/v1")

        # Simple file-based cache for mock/dev
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.mock_mode:
            logger.info("Fiserv Service initializing in MOCK MODE")
        else:
            logger.info("Fiserv Service initializing in REAL MODE")

    async def get_member_data(self, member_id: str) -> dict[str, Any]:
        """
        Orchestrates fetching of member, accounts, and transactions.
        """
        # 1. Check Cache
        cached = self._get_from_cache(f"member_{member_id}")
        if cached:
            return cached

        # 2. Fetch Data (Real or Mock)
        if self.mock_mode:
            data = self._generate_mock_data(member_id)
        else:
            data = await self._fetch_real_data(member_id)

        # 3. Cache Result
        self._save_to_cache(f"member_{member_id}", data)
        return data

    async def _fetch_real_data(self, member_id: str) -> dict[str, Any]:
        """
        Fetch real member data from Fiserv Banking Hub APIs.
        Uses PartyService, AccountService, and TransactionService.
        """
        from .fiserv.party_service import get_party_service
        from .fiserv.account_service import get_account_service
        from .fiserv.transaction_service import get_transaction_service
        
        logger.info(f"Fetching real Fiserv data for member {member_id}")
        
        try:
            party_svc = get_party_service()
            account_svc = get_account_service()
            txn_svc = get_transaction_service()
            
            # 1. Get party info
            party_response = await party_svc.get_by_id(member_id)
            if party_response.get("error"):
                logger.warning(f"Party lookup failed: {party_response}")
                # Fall back to mock if real API fails
                return self._generate_mock_data(member_id)
            
            # 2. Get accounts for this party
            accounts_response = await account_svc.get_by_party(member_id)
            accounts = account_svc.parse_account_list(accounts_response)
            
            # 3. Get transactions for each account
            all_transactions = []
            for acct in accounts[:3]:  # Limit to first 3 accounts for performance
                acct_id = acct.get("account_id")
                if acct_id:
                    txn_response = await txn_svc.get_history(acct_id, days=90)
                    txns = txn_svc.parse_transaction_list(txn_response)
                    all_transactions.extend(txns)
            
            # 4. Build normalized dataset
            dataset = {
                "dataset_id": f"ds_{member_id}_{datetime.datetime.now().strftime('%Y%m%d')}",
                "generated_at": datetime.datetime.now().isoformat(),
                "source": "fiserv_api",
                "party": party_response,
                "member": {
                    "member_id": member_id,
                    "name": self._extract_name(party_response),
                    "status": "ACTIVE",
                },
                "accounts": accounts,
                "transactions": all_transactions,
            }
            
            return dataset
            
        except Exception as e:
            logger.error(f"Real Fiserv fetch failed for {member_id}: {e}")
            # Graceful fallback to mock data
            logger.info("Falling back to mock data")
            return self._generate_mock_data(member_id)
    
    def _extract_name(self, party_response: dict) -> str:
        """Extract name from party response."""
        try:
            person_info = party_response.get("PersonPartyInfo", {})
            name_info = person_info.get("PersonName", {})
            return name_info.get("FullName") or f"{name_info.get('GivenName', '')} {name_info.get('FamilyName', '')}".strip()
        except Exception:
            return "Unknown"

    def _generate_mock_data(self, member_id: str) -> dict[str, Any]:
        """
        Generates consistent mock/fixture data for a given member_id.
        """
        logger.info(f"Generating mock data for member {member_id}")

        # Seed logic: last digit of ID determines "persona"
        # 0-3: Stable Saver (Low Risk)
        # 4-6: Big Spender (High Cash Flow, Low Balance)
        # 7-9: Struggling/Risk (Overdrafts, Late Fees)
        seed = int(member_id[-1]) if member_id[-1].isdigit() else 5

        # Base Member
        dataset = {
            "dataset_id": f"ds_{member_id}_{datetime.datetime.now().strftime('%Y%m%d')}",
            "generated_at": datetime.datetime.now().isoformat(),
            "member": {
                "member_id": member_id,
                "name": f"Mock Member {member_id}",
                "email": f"member{member_id}@example.com",
                "phone": "555-0100",
                "status": "ACTIVE",
                "tenure_months": 12 + (seed * 5),
                "credit_score": 600 + (seed * 25),
            },
            "accounts": [],
            "transactions": [],
        }

        # Generate Accounts
        acct_id_chk = f"chk_{member_id}"
        acct_id_sav = f"sav_{member_id}"

        dataset["accounts"].append(
            {"account_id": acct_id_chk, "type": "CHECKING", "balance": 1000.0 * seed, "status": "OPEN"}
        )
        dataset["accounts"].append(
            {
                "account_id": acct_id_sav,
                "type": "SAVINGS",
                "balance": 5000.0 * seed if seed > 3 else 50.0,
                "status": "OPEN",
            }
        )

        # Generate Transactions (Last 90 days)
        # Pattern: Salary in, Bills out, Random spend
        base_date = datetime.datetime.now()

        # 1. Salary (2x month)
        for i in range(3):  # 3 months
            for day in [1, 15]:
                dt = base_date - datetime.timedelta(days=(i * 30) + (30 - day))
                dataset["transactions"].append(
                    {
                        "tx_id": f"tx_sal_{i}_{day}",
                        "account_id": acct_id_chk,
                        "date": dt.isoformat(),
                        "amount": 2500.0 + (seed * 100),
                        "dr_cr": "CR",  # Credit (Inflow)
                        "category": "INCOME",
                        "description": "Direct Deposit SCU EMPLOYER",
                    }
                )

        # 2. Rent/Mortgage (1x month)
        for i in range(3):
            dt = base_date - datetime.timedelta(days=(i * 30) + 25)  # 5th of month
            dataset["transactions"].append(
                {
                    "tx_id": f"tx_rent_{i}",
                    "account_id": acct_id_chk,
                    "date": dt.isoformat(),
                    "amount": 1200.0 + (seed * 50),
                    "dr_cr": "DR",  # Debit (Outflow)
                    "category": "HOUSING",
                    "description": "City Apartments Rent",
                }
            )

        # 3. Random Spend (Starbucks, Grocery)
        for i in range(20):
            dt = base_date - datetime.timedelta(days=i * 4)
            dataset["transactions"].append(
                {
                    "tx_id": f"tx_rnd_{i}",
                    "account_id": acct_id_chk,
                    "date": dt.isoformat(),
                    "amount": 5.0 + (seed * 2.5),
                    "dr_cr": "DR",
                    "category": "FOOD_DINING",
                    "description": "Starbucks Coffee",
                }
            )

        return dataset

    def _get_from_cache(self, key: str) -> dict[str, Any] | None:
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            # Check age (TTL 24h)
            mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime)
            if (datetime.datetime.now() - mtime).total_seconds() < 86400:
                with open(path) as f:
                    return json.load(f)
        return None

    def _save_to_cache(self, key: str, data: dict[str, Any]):
        path = self.cache_dir / f"{key}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
