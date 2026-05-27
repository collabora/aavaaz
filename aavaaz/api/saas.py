"""
Aavaaz SaaS API — Backend endpoints for the hosted platform.

Provides:
- API key management (create, list, revoke)
- Usage tracking & metering
- Stripe billing integration
- Subscription management
"""

import hashlib
import logging
import os
import secrets
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from aavaaz.api.auth import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/saas", tags=["saas"])

# ─── Configuration ───────────────────────────────────────────────────────────

STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_PRO = os.environ.get("STRIPE_PRICE_PRO", "")
SAAS_DOMAIN = os.environ.get("SAAS_DOMAIN", "https://app.aavaaz.dev")

# Price per audio minute for metered billing
PRICE_PER_MINUTE = float(os.environ.get("AAVAAZ_PRICE_PER_MINUTE", "0.006"))


# ─── Data Models ─────────────────────────────────────────────────────────────


@dataclass
class SaasApiKey:
    id: str
    user_id: str
    name: str
    key_hash: str
    prefix: str
    created_at: str
    last_used: str | None = None
    expires_at: str | None = None


@dataclass
class UsageEntry:
    user_id: str
    date: str
    audio_minutes: float = 0.0
    requests: int = 0


@dataclass
class UserSubscription:
    user_id: str
    plan: str = "free"
    stripe_customer_id: str = ""
    stripe_subscription_id: str = ""
    status: str = "active"
    current_period_end: str = ""
    cancel_at_period_end: bool = False

    @property
    def included_minutes(self) -> int:
        return {"free": 60, "pro": 1000, "enterprise": 999999}.get(self.plan, 60)

    @property
    def price_per_minute(self) -> float:
        return {"free": 0.0, "pro": PRICE_PER_MINUTE, "enterprise": 0.004}.get(
            self.plan, PRICE_PER_MINUTE
        )


# ─── In-memory store (replace with PostgreSQL for production) ────────────────

_api_keys: dict[str, SaasApiKey] = {}  # key_id -> SaasApiKey
_key_hash_to_id: dict[str, str] = {}  # hash -> key_id (for auth lookups)
_usage: dict[str, list[UsageEntry]] = {}  # user_id -> [UsageEntry]
_subscriptions: dict[str, UserSubscription] = {}  # user_id -> UserSubscription


# ─── Request/Response Schemas ────────────────────────────────────────────────


class CreateKeyRequest(BaseModel):
    name: str


class CreateKeyResponse(BaseModel):
    key: dict
    secret: str


class CheckoutRequest(BaseModel):
    plan: str


class CheckoutResponse(BaseModel):
    url: str


# ─── API Key Endpoints ───────────────────────────────────────────────────────


@router.get("/api-keys")
async def list_api_keys(claims: dict = Depends(require_auth)):
    user_id = claims["sub"]
    user_keys = [
        {
            "id": k.id,
            "name": k.name,
            "prefix": k.prefix,
            "created_at": k.created_at,
            "last_used": k.last_used,
            "expires_at": k.expires_at,
        }
        for k in _api_keys.values()
        if k.user_id == user_id
    ]
    return user_keys


@router.post("/api-keys")
async def create_api_key(body: CreateKeyRequest, claims: dict = Depends(require_auth)):
    user_id = claims["sub"]

    # Generate a secure API key with identifiable prefix
    raw_key = f"aavaaz_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_id = str(uuid.uuid4())

    api_key = SaasApiKey(
        id=key_id,
        user_id=user_id,
        name=body.name,
        key_hash=key_hash,
        prefix=raw_key[:12],
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    _api_keys[key_id] = api_key
    _key_hash_to_id[key_hash] = key_id

    return {
        "key": {
            "id": api_key.id,
            "name": api_key.name,
            "prefix": api_key.prefix,
            "created_at": api_key.created_at,
            "last_used": None,
            "expires_at": None,
        },
        "secret": raw_key,
    }


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(key_id: str, claims: dict = Depends(require_auth)):
    user_id = claims["sub"]
    key = _api_keys.get(key_id)
    if not key or key.user_id != user_id:
        raise HTTPException(status_code=404, detail="API key not found")

    _key_hash_to_id.pop(key.key_hash, None)
    del _api_keys[key_id]
    return {"status": "revoked"}


# ─── Usage Endpoints ─────────────────────────────────────────────────────────


@router.get("/usage")
async def get_usage(claims: dict = Depends(require_auth)):
    user_id = claims["sub"]
    sub = _subscriptions.get(user_id, UserSubscription(user_id=user_id))
    entries = _usage.get(user_id, [])

    total_minutes = sum(e.audio_minutes for e in entries)
    total_requests = sum(e.requests for e in entries)
    total_cost = total_minutes * sub.price_per_minute

    return {
        "current_month": {
            "audio_minutes": total_minutes,
            "requests": total_requests,
            "cost_usd": total_cost,
        },
        "quota": {
            "audio_minutes_limit": sub.included_minutes,
            "audio_minutes_used": total_minutes,
        },
        "plan": sub.plan,
        "daily_usage": [
            {
                "date": e.date,
                "audio_minutes": e.audio_minutes,
                "requests": e.requests,
                "cost_usd": e.audio_minutes * sub.price_per_minute,
            }
            for e in entries[-30:]
        ],
    }


# ─── Subscription & Billing Endpoints ───────────────────────────────────────


@router.get("/subscription")
async def get_subscription(claims: dict = Depends(require_auth)):
    user_id = claims["sub"]
    sub = _subscriptions.get(user_id, UserSubscription(user_id=user_id))
    return {
        "plan": sub.plan,
        "status": sub.status,
        "current_period_end": sub.current_period_end,
        "cancel_at_period_end": sub.cancel_at_period_end,
        "price_per_minute": sub.price_per_minute,
        "included_minutes": sub.included_minutes,
    }


@router.post("/checkout")
async def create_checkout(body: CheckoutRequest, claims: dict = Depends(require_auth)):
    """Create a Stripe Checkout session for plan upgrade."""
    user_id = claims["sub"]

    if not STRIPE_SECRET_KEY:
        raise HTTPException(
            status_code=503, detail="Billing not configured (set STRIPE_SECRET_KEY)"
        )

    import stripe

    stripe.api_key = STRIPE_SECRET_KEY

    # Get or create Stripe customer
    sub = _subscriptions.get(user_id, UserSubscription(user_id=user_id))
    if not sub.stripe_customer_id:
        customer = stripe.Customer.create(metadata={"aavaaz_user_id": user_id})
        sub.stripe_customer_id = customer.id
        _subscriptions[user_id] = sub

    # Create checkout session
    session = stripe.checkout.Session.create(
        customer=sub.stripe_customer_id,
        mode="subscription",
        line_items=[{"price": STRIPE_PRICE_PRO, "quantity": 1}],
        success_url=f"{SAAS_DOMAIN}/dashboard/billing?success=true",
        cancel_url=f"{SAAS_DOMAIN}/dashboard/billing?canceled=true",
        metadata={"aavaaz_user_id": user_id, "plan": body.plan},
    )

    return {"url": session.url}


@router.post("/billing-portal")
async def create_portal_session(claims: dict = Depends(require_auth)):
    """Create a Stripe Customer Portal session."""
    user_id = claims["sub"]

    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Billing not configured")

    sub = _subscriptions.get(user_id)
    if not sub or not sub.stripe_customer_id:
        raise HTTPException(status_code=400, detail="No billing account found")

    import stripe

    stripe.api_key = STRIPE_SECRET_KEY

    session = stripe.billing_portal.Session.create(
        customer=sub.stripe_customer_id,
        return_url=f"{SAAS_DOMAIN}/dashboard/billing",
    )

    return {"url": session.url}


# ─── Stripe Webhook ──────────────────────────────────────────────────────────


@router.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events for subscription lifecycle."""
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="Webhook not configured")

    import stripe

    stripe.api_key = STRIPE_SECRET_KEY
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except (ValueError, stripe.error.SignatureVerificationError):
        raise HTTPException(status_code=400, detail="Invalid signature")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session["metadata"].get("aavaaz_user_id")
        if user_id:
            sub = _subscriptions.get(user_id, UserSubscription(user_id=user_id))
            sub.plan = session["metadata"].get("plan", "pro")
            sub.stripe_subscription_id = session.get("subscription", "")
            sub.status = "active"
            _subscriptions[user_id] = sub
            logger.info(f"User {user_id} upgraded to {sub.plan}")

    elif event["type"] == "customer.subscription.updated":
        subscription = event["data"]["object"]
        # Find user by stripe customer ID
        for uid, sub in _subscriptions.items():
            if sub.stripe_customer_id == subscription["customer"]:
                sub.status = subscription["status"]
                sub.cancel_at_period_end = subscription["cancel_at_period_end"]
                sub.current_period_end = datetime.fromtimestamp(
                    subscription["current_period_end"], tz=timezone.utc
                ).isoformat()
                break

    elif event["type"] == "customer.subscription.deleted":
        subscription = event["data"]["object"]
        for uid, sub in _subscriptions.items():
            if sub.stripe_customer_id == subscription["customer"]:
                sub.plan = "free"
                sub.status = "canceled"
                logger.info(f"User {uid} subscription canceled")
                break

    return {"received": True}


# ─── Transcript History ──────────────────────────────────────────────────────

_transcripts: dict[str, list[dict]] = {}  # user_id -> [job dicts]


@router.get("/transcripts")
async def list_transcripts(claims: dict = Depends(require_auth)):
    user_id = claims["sub"]
    return _transcripts.get(user_id, [])


@router.get("/transcripts/{transcript_id}")
async def get_transcript(transcript_id: str, claims: dict = Depends(require_auth)):
    user_id = claims["sub"]
    jobs = _transcripts.get(user_id, [])
    for job in jobs:
        if job["id"] == transcript_id:
            return job
    raise HTTPException(status_code=404, detail="Transcript not found")


# ─── Usage Recording (called internally by transcription pipeline) ───────────


def record_usage(user_id: str, audio_minutes: float):
    """Record audio usage for a user. Called after each transcription."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    entries = _usage.setdefault(user_id, [])

    # Find or create today's entry
    for entry in entries:
        if entry.date == today:
            entry.audio_minutes += audio_minutes
            entry.requests += 1
            return

    entries.append(
        UsageEntry(user_id=user_id, date=today, audio_minutes=audio_minutes, requests=1)
    )


def record_transcript(user_id: str, job: dict):
    """Record a completed transcription job."""
    _transcripts.setdefault(user_id, []).insert(0, job)


def validate_api_key(raw_key: str) -> str | None:
    """Validate a SaaS API key and return the user_id, or None if invalid."""
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_id = _key_hash_to_id.get(key_hash)
    if not key_id:
        return None

    api_key = _api_keys.get(key_id)
    if not api_key:
        return None

    # Update last_used timestamp
    api_key.last_used = datetime.now(timezone.utc).isoformat()
    return api_key.user_id
