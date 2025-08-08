from httpx import AsyncClient
from fastapi import status
from app import app
import pytest


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        resp = await ac.get("/health")
        assert resp.status_code == status.HTTP_200_OK
        assert resp.json()["status"] == "healthy"

