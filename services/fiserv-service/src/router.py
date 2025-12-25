from fastapi import APIRouter, HTTPException

from .connector import FiservClient
from .normalizer import Normalizer

router = APIRouter()
connector = FiservClient()


@router.get("/datasets/member/{member_id}")
async def get_member_dataset(member_id: str):
    """
    Returns the raw canonical dataset for a member.
    """
    try:
        data = connector.get_member_data(member_id)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/member/{member_id}/views/transactions")
async def get_member_transactions_view(member_id: str):
    """
    Returns the 'member_transactions_view' (list of dicts) for ML ingestion.
    """
    try:
        data = connector.get_member_data(member_id)
        view = Normalizer.get_member_transactions_view(data)
        return view
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/member/{member_id}/views/features")
async def get_member_features_view(member_id: str):
    """
    Returns the 'member_features_view' (single dict) for ML ingestion.
    """
    try:
        data = connector.get_member_data(member_id)
        view = Normalizer.get_member_features_view(data)
        return view
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
