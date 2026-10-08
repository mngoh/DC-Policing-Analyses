"""Fetcher for MPD Stop Data (ArcGIS layer 41), mirroring services/crime_fetcher.py.

Pulls stop records page-by-page from the DC ArcGIS FeatureServer with the same
pagination + retry shape used for crime incidents. No geometry is requested
(Stop Data has none). Used only by scripts/ingest_stops.py — not wired into the
app or any cron.
"""
from __future__ import annotations

import asyncio
import logging

import httpx

from database.stops import STOP_FIELDS, STOPS_LAYER_URL

_log = logging.getLogger(__name__)

PAGE_SIZE = 1000  # ArcGIS MaxRecordCount for this layer is 1000


async def fetch_stops(
    layer_url: str = STOPS_LAYER_URL,
    where: str = "1=1",
    page_size: int = PAGE_SIZE,
    on_page=None,
    fields: str = STOP_FIELDS,
) -> list[dict]:
    """Fetch all Stop-Data attribute records for ``where`` from ``layer_url``.

    Paginates on OBJECTID (a stable order for offset paging) and retries transient
    5xx / connection errors three times per page — the same policy as the crime
    fetcher. ``on_page(page, fetched, total)`` is an optional progress callback.
    ``fields`` is the outFields list — layer 45 has a drifted field set, so pass
    the matching ``database.stops.LAYER_FIELDS`` entry alongside its URL.
    """
    records: list[dict] = []
    offset = 0
    page = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0)) as client:
        while True:
            page += 1
            params = {
                "where": where,
                "outFields": fields,
                "resultRecordCount": page_size,
                "resultOffset": offset,
                "orderByFields": "OBJECTID ASC",  # stable ordering for offset paging
                "returnGeometry": "false",         # Stop Data has no geometry
                "f": "json",
            }
            body = None
            for attempt in range(3):
                try:
                    resp = await client.get(layer_url + "/query", params=params)
                    if resp.status_code < 500:
                        body = resp.json()
                        break
                    _log.warning("ArcGIS %s on stops page %d attempt %d/3", resp.status_code, page, attempt + 1)
                except httpx.TransportError as exc:
                    _log.warning("ArcGIS connection error on stops page %d attempt %d/3: %s", page, attempt + 1, exc)
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
            if body is None:
                raise RuntimeError(f"ArcGIS unavailable fetching stops page {page}")
            if "error" in body:
                raise RuntimeError(f"ArcGIS error fetching stops: {body['error']}")

            features = body.get("features", [])
            for feat in features:
                records.append(feat.get("attributes", {}))
            if on_page:
                on_page(page, len(features), len(records))

            # exceededTransferLimit=True means there are more pages.
            if not features or not body.get("exceededTransferLimit", False):
                break
            offset += page_size
    return records
