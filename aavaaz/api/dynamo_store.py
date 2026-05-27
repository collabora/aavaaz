"""
DynamoDB-backed data store for Aavaaz SaaS.

Drop-in replacement for the in-memory store in api/saas.py.
Uses AWS DynamoDB for persistence — fits within free tier for early usage.

Tables:
  - aavaaz-api-keys-{env}: API key storage (GSI on key_hash for auth lookups)
  - aavaaz-usage-{env}: Daily usage records per user
  - aavaaz-subscriptions-{env}: User subscription state
  - aavaaz-transcripts-{env}: Transcript job history
"""

import hashlib
import logging
import os
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key

logger = logging.getLogger(__name__)

ENV = os.environ.get("AAVAAZ_ENVIRONMENT", "prod")
REGION = os.environ.get("AWS_REGION", "us-east-1")

_dynamodb = boto3.resource("dynamodb", region_name=REGION)

_table_api_keys = _dynamodb.Table(f"aavaaz-api-keys-{ENV}")
_table_usage = _dynamodb.Table(f"aavaaz-usage-{ENV}")
_table_subscriptions = _dynamodb.Table(f"aavaaz-subscriptions-{ENV}")
_table_transcripts = _dynamodb.Table(f"aavaaz-transcripts-{ENV}")


# ─── API Keys ────────────────────────────────────────────────────────────────


def create_api_key(user_id: str, name: str) -> tuple[dict, str]:
    """Create a new API key. Returns (key_metadata, raw_secret)."""
    raw_key = f"aavaaz_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    item = {
        "user_id": user_id,
        "key_id": key_id,
        "name": name,
        "key_hash": key_hash,
        "prefix": raw_key[:12],
        "created_at": now,
        "last_used": None,
        "expires_at": None,
    }

    _table_api_keys.put_item(Item=item)

    metadata = {
        "id": key_id,
        "name": name,
        "prefix": raw_key[:12],
        "created_at": now,
        "last_used": None,
        "expires_at": None,
    }
    return metadata, raw_key


def list_api_keys(user_id: str) -> list[dict]:
    """List all API keys for a user."""
    response = _table_api_keys.query(KeyConditionExpression=Key("user_id").eq(user_id))
    return [
        {
            "id": item["key_id"],
            "name": item["name"],
            "prefix": item["prefix"],
            "created_at": item["created_at"],
            "last_used": item.get("last_used"),
            "expires_at": item.get("expires_at"),
        }
        for item in response.get("Items", [])
    ]


def revoke_api_key(user_id: str, key_id: str) -> bool:
    """Revoke (delete) an API key. Returns True if found and deleted."""
    try:
        _table_api_keys.delete_item(
            Key={"user_id": user_id, "key_id": key_id},
            ConditionExpression="attribute_exists(user_id)",
        )
        return True
    except _dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
        return False


def validate_api_key(raw_key: str) -> str | None:
    """Validate an API key and return user_id, or None if invalid."""
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    response = _table_api_keys.query(
        IndexName="key-hash-index",
        KeyConditionExpression=Key("key_hash").eq(key_hash),
    )

    items = response.get("Items", [])
    if not items:
        return None

    item = items[0]

    # Update last_used timestamp (fire and forget)
    try:
        _table_api_keys.update_item(
            Key={"user_id": item["user_id"], "key_id": item["key_id"]},
            UpdateExpression="SET last_used = :now",
            ExpressionAttributeValues={":now": datetime.now(timezone.utc).isoformat()},
        )
    except Exception:
        pass

    return item["user_id"]


# ─── Usage Tracking ──────────────────────────────────────────────────────────


def record_usage(user_id: str, audio_minutes: float):
    """Record usage for the current day. Atomic increment."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    _table_usage.update_item(
        Key={"user_id": user_id, "date": today},
        UpdateExpression="ADD audio_minutes :mins, requests :one",
        ExpressionAttributeValues={
            ":mins": round(audio_minutes, 4),
            ":one": 1,
        },
    )


def get_usage(user_id: str, days: int = 30) -> list[dict]:
    """Get daily usage records for a user (last N days)."""
    from datetime import timedelta

    start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
        "%Y-%m-%d"
    )

    response = _table_usage.query(
        KeyConditionExpression=Key("user_id").eq(user_id) & Key("date").gte(start_date),
        ScanIndexForward=True,
    )

    return [
        {
            "date": item["date"],
            "audio_minutes": float(item.get("audio_minutes", 0)),
            "requests": int(item.get("requests", 0)),
        }
        for item in response.get("Items", [])
    ]


# ─── Subscriptions ───────────────────────────────────────────────────────────


def get_subscription(user_id: str) -> dict:
    """Get subscription info for a user."""
    response = _table_subscriptions.get_item(Key={"user_id": user_id})
    item = response.get("Item")

    if not item:
        return {
            "user_id": user_id,
            "plan": "free",
            "status": "active",
            "stripe_customer_id": "",
            "stripe_subscription_id": "",
            "current_period_end": "",
            "cancel_at_period_end": False,
        }

    return item


def update_subscription(user_id: str, updates: dict):
    """Update subscription fields."""
    expressions = []
    values = {}
    for key, value in updates.items():
        expressions.append(f"{key} = :{key}")
        values[f":{key}"] = value

    _table_subscriptions.update_item(
        Key={"user_id": user_id},
        UpdateExpression="SET " + ", ".join(expressions),
        ExpressionAttributeValues=values,
    )


def find_user_by_stripe_customer(stripe_customer_id: str) -> str | None:
    """Find user_id by Stripe customer ID."""
    response = _table_subscriptions.query(
        IndexName="stripe-customer-index",
        KeyConditionExpression=Key("stripe_customer_id").eq(stripe_customer_id),
    )
    items = response.get("Items", [])
    return items[0]["user_id"] if items else None


# ─── Transcripts ─────────────────────────────────────────────────────────────


def save_transcript(user_id: str, job: dict):
    """Save a transcript job record."""
    job["user_id"] = user_id
    if "created_at" not in job:
        job["created_at"] = datetime.now(timezone.utc).isoformat()
    _table_transcripts.put_item(Item=job)


def list_transcripts(user_id: str, limit: int = 50) -> list[dict]:
    """List recent transcript jobs for a user."""
    response = _table_transcripts.query(
        KeyConditionExpression=Key("user_id").eq(user_id),
        ScanIndexForward=False,
        Limit=limit,
    )
    return response.get("Items", [])


def get_transcript(user_id: str, created_at: str) -> dict | None:
    """Get a specific transcript by user_id and created_at."""
    response = _table_transcripts.get_item(
        Key={"user_id": user_id, "created_at": created_at}
    )
    return response.get("Item")
