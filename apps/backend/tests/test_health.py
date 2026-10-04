from httpx import AsyncClient, ASGITransport
from fastapi import status
from app import app
import pytest


@pytest.mark.asyncio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/health")
        assert resp.status_code == status.HTTP_200_OK
        assert resp.json()["status"] == "healthy"

