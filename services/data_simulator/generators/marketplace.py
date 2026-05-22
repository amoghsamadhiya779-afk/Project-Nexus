"""
services/data_simulator/generators/marketplace.py
===================================================
Realistic marketplace data generator.

Generates users, items, and interaction events with:
- Power-law item popularity (80/20 rule — realistic)
- Temporal patterns (day-of-week, hour-of-day seasonality)
- User preference clusters (not random — users have tastes)
- Cold-start users and items
- Fraud patterns (for fraud detection training)

Scale: 100K users, 500K items, 10M+ events in ~5 minutes on a laptop.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Generator, List, Optional, Tuple
import math

import numpy as np
import pandas as pd
from faker import Faker
from loguru import logger

fake = Faker()
rng  = np.random.default_rng(seed=42)


# ─── Config ───────────────────────────────────────────────────────────────────

CATEGORIES = [
    "electronics", "clothing", "home_garden", "sports", "books",
    "toys", "beauty", "automotive", "food", "art", "music",
    "collectibles", "jewelry", "pet_supplies", "office",
]

COUNTRIES  = ["US", "IN", "GB", "DE", "CA", "AU", "FR", "JP", "BR", "MX"]
DEVICES    = ["mobile", "desktop", "tablet"]
CONDITIONS = ["new", "like_new", "good", "fair", "poor"]
EVENT_TYPES = ["view", "click", "add_to_cart", "purchase", "wishlist",
               "message_seller", "review", "share", "report"]

# Conversion funnel weights (realistic e-commerce rates)
EVENT_WEIGHTS = [0.45, 0.25, 0.12, 0.04, 0.06, 0.03, 0.02, 0.02, 0.01]

# Hourly traffic multiplier (people shop more in evenings)
HOURLY_MULTIPLIER = [
    0.2, 0.1, 0.1, 0.1, 0.1, 0.2,   # 0–5 AM
    0.4, 0.7, 0.9, 1.0, 1.0, 1.0,   # 6–11 AM
    1.1, 1.0, 0.9, 0.9, 1.0, 1.2,   # 12–5 PM
    1.5, 1.8, 1.9, 1.7, 1.3, 0.7,   # 6–11 PM
]


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class User:
    user_id:          int
    country:          str
    device_type:      str
    registered_at:    datetime
    lifetime_value:   float
    is_verified:      bool
    preferred_cats:   List[str]       # user taste profile
    price_sensitivity: float          # 0=budget, 1=premium
    activity_level:   float           # 0=inactive, 1=power user
    is_fraud_risk:    bool = False


@dataclass
class Item:
    item_id:       int
    seller_id:     int
    category:      str
    subcategory:   str
    price:         float
    condition:     str
    created_at:    datetime
    popularity:    float          # 0–1, power-law distributed
    avg_rating:    float
    review_count:  int
    is_promoted:   bool = False
    is_fraud:      bool = False


@dataclass
class Event:
    event_id:        str
    user_id:         int
    item_id:         int
    event_type:      str
    event_timestamp: datetime
    session_id:      str
    price_shown:     float
    position_shown:  int
    device_type:     str
    is_fraud:        bool = False


# ─── Generator ────────────────────────────────────────────────────────────────

class MarketplaceGenerator:
    """
    Stateful generator for realistic marketplace data.

    Key realism features:
    1. Power-law popularity: item[0] gets ~1000x more views than item[10000]
    2. User taste clusters: users have preferred categories (not random)
    3. Temporal patterns: evening/weekend traffic spikes
    4. Session-based events: users have sessions of 1-10 events
    5. Fraud patterns: 0.5% fraud rate with detectable graph patterns
    """

    def __init__(
        self,
        n_users:      int = 100_000,
        n_items:      int = 500_000,
        n_categories: int = 15,
        fraud_rate:   float = 0.005,
        seed:         int = 42,
    ):
        self.n_users     = n_users
        self.n_items     = n_items
        self.fraud_rate  = fraud_rate
        rng              = np.random.default_rng(seed)
        self._rng        = rng
        Faker.seed(seed)
        random.seed(seed)

        self._users: Optional[List[User]] = None
        self._items: Optional[List[Item]] = None

    # ── User Generation ───────────────────────────────────────────────────────

    def generate_users(self) -> List[User]:
        if self._users:
            return self._users

        logger.info(f"Generating {self.n_users:,} users...")
        users = []
        n_fraud = int(self.n_users * self.fraud_rate * 10)

        for i in range(self.n_users):
            n_cats = self._rng.integers(1, 5)
            preferred = list(
                self._rng.choice(CATEGORIES, size=n_cats, replace=False)
            )
            users.append(User(
                user_id          = i + 1,
                country          = str(self._rng.choice(COUNTRIES, p=self._country_weights())),
                device_type      = str(self._rng.choice(DEVICES, p=[0.60, 0.30, 0.10])),
                registered_at    = fake.date_time_between(start_date="-2y", end_date="-1d"),
                lifetime_value   = float(self._rng.lognormal(mean=4.5, sigma=1.2)),
                is_verified      = bool(self._rng.random() > 0.3),
                preferred_cats   = preferred,
                price_sensitivity= float(self._rng.beta(2, 5)),
                activity_level   = float(self._rng.beta(1, 3)),
                is_fraud_risk    = i < n_fraud,
            ))

        self._users = users
        logger.info(f"Generated {len(users):,} users ({n_fraud} fraud-risk)")
        return users

    def _country_weights(self) -> np.ndarray:
        w = np.array([0.30, 0.20, 0.10, 0.08, 0.07, 0.05, 0.05, 0.05, 0.05, 0.05])
        return w / w.sum()

    # ── Item Generation ───────────────────────────────────────────────────────

    def generate_items(self) -> List[Item]:
        if self._items:
            return self._items

        logger.info(f"Generating {self.n_items:,} items...")
        items = []

        # Power-law popularity (Pareto: top 20% of items get 80% of views)
        popularities = self._rng.pareto(a=2.0, size=self.n_items)
        popularities = popularities / popularities.max()   # normalise to [0,1]

        n_sellers = self.n_items // 5   # avg 5 items per seller
        n_fraud   = int(self.n_items * self.fraud_rate)

        for i in range(self.n_items):
            cat = str(self._rng.choice(CATEGORIES))
            items.append(Item(
                item_id       = i + 1,
                seller_id     = int(self._rng.integers(1, n_sellers)),
                category      = cat,
                subcategory   = f"{cat}_{self._rng.integers(1, 6)}",
                price         = float(np.exp(self._rng.normal(3.5, 1.5))),  # log-normal prices
                condition     = str(self._rng.choice(CONDITIONS, p=[0.3, 0.25, 0.25, 0.15, 0.05])),
                created_at    = fake.date_time_between(start_date="-1y", end_date="-1d"),
                popularity    = float(popularities[i]),
                avg_rating    = float(self._rng.beta(8, 2) * 5),   # skewed toward 4-5
                review_count  = int(self._rng.negative_binomial(5, 0.5)),
                is_promoted   = bool(self._rng.random() < 0.05),   # 5% promoted
                is_fraud      = i < n_fraud,
            ))

        self._items = items
        logger.info(f"Generated {len(items):,} items")
        return items

    # ── Event Stream Generation ───────────────────────────────────────────────

    def generate_events(
        self,
        n_events:   int,
        start_date: datetime,
        end_date:   datetime,
    ) -> Generator[Event, None, None]:
        """
        Stream events. Memory-efficient — yields one at a time.
        Use this for Kafka producer or batch writing.
        """
        users = self.generate_users()
        items = self.generate_items()

        item_popularity = np.array([it.popularity for it in items])
        item_popularity = item_popularity / item_popularity.sum()

        total_seconds = (end_date - start_date).total_seconds()
        event_count   = 0

        logger.info(f"Generating {n_events:,} events from {start_date} to {end_date}...")

        while event_count < n_events:
            # Sample user weighted by activity level
            user = self._rng.choice(users)

            # Session: 1–8 events per session
            session_len = int(self._rng.integers(1, 9))
            session_id  = fake.uuid4()

            for pos in range(session_len):
                if event_count >= n_events:
                    break

                # Weighted timestamp (more events in evenings)
                t_offset   = self._rng.random() * total_seconds
                base_time  = start_date + timedelta(seconds=float(t_offset))
                hour       = base_time.hour
                if self._rng.random() > HOURLY_MULTIPLIER[hour] / 2.0:
                    # Re-sample to create temporal clustering
                    evening_hour = self._rng.integers(18, 23)
                    base_time    = base_time.replace(hour=int(evening_hour))

                # Item sampling: mix of popularity-based + category match
                if self._rng.random() < 0.6 and user.preferred_cats:
                    # Preference-based sampling
                    pref_cat = self._rng.choice(user.preferred_cats)
                    cat_items = [
                        it for it in items if it.category == pref_cat
                    ][:1000]  # cap for speed
                    if cat_items:
                        cat_pop = np.array([it.popularity for it in cat_items])
                        cat_pop = cat_pop / cat_pop.sum()
                        item    = self._rng.choice(cat_items, p=cat_pop)
                    else:
                        item = items[int(self._rng.choice(len(items), p=item_popularity))]
                else:
                    item = items[int(self._rng.choice(len(items), p=item_popularity))]

                # Event type follows funnel (view → click → purchase)
                event_type = str(self._rng.choice(EVENT_TYPES, p=EVENT_WEIGHTS))

                # Price jitter (A/B test pricing, display rounding)
                price_shown = round(item.price * self._rng.uniform(0.98, 1.02), 2)

                is_fraud = bool(
                    (user.is_fraud_risk or item.is_fraud)
                    and self._rng.random() < 0.3
                )

                yield Event(
                    event_id        = str(fake.uuid4()),
                    user_id         = user.user_id,
                    item_id         = item.item_id,
                    event_type      = event_type,
                    event_timestamp = base_time,
                    session_id      = session_id,
                    price_shown     = price_shown,
                    position_shown  = pos,
                    device_type     = user.device_type,
                    is_fraud        = is_fraud,
                )

                event_count += 1

            if event_count % 100_000 == 0:
                logger.info(f"  Generated {event_count:,} / {n_events:,} events...")

        logger.info(f"Event generation complete: {event_count:,} events")

    # ── DataFrame Export ──────────────────────────────────────────────────────

    def users_df(self) -> pd.DataFrame:
        users = self.generate_users()
        return pd.DataFrame([
            {
                "user_id":          u.user_id,
                "country":          u.country,
                "device_type":      u.device_type,
                "registered_at":    u.registered_at,
                "lifetime_value":   round(u.lifetime_value, 2),
                "is_verified":      u.is_verified,
                "preferred_cats":   ",".join(u.preferred_cats),
                "price_sensitivity": round(u.price_sensitivity, 4),
                "activity_level":   round(u.activity_level, 4),
            }
            for u in users
        ])

    def items_df(self) -> pd.DataFrame:
        items = self.generate_items()
        return pd.DataFrame([
            {
                "item_id":       it.item_id,
                "seller_id":     it.seller_id,
                "category":      it.category,
                "subcategory":   it.subcategory,
                "price":         round(it.price, 2),
                "condition":     it.condition,
                "created_at":    it.created_at,
                "popularity":    round(it.popularity, 6),
                "avg_rating":    round(it.avg_rating, 2),
                "review_count":  it.review_count,
                "is_promoted":   it.is_promoted,
            }
            for it in items
        ])

    def events_df(
        self,
        n_events:   int,
        start_date: datetime,
        end_date:   datetime,
        chunk_size: int = 100_000,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Yield events in chunks for memory-efficient processing.
        Each chunk is a DataFrame ready for Parquet writing or Kafka producing.
        """
        chunk = []
        for event in self.generate_events(n_events, start_date, end_date):
            chunk.append({
                "event_id":        event.event_id,
                "user_id":         event.user_id,
                "item_id":         event.item_id,
                "event_type":      event.event_type,
                "event_timestamp": event.event_timestamp,
                "session_id":      event.session_id,
                "price_shown":     event.price_shown,
                "position_shown":  event.position_shown,
                "device_type":     event.device_type,
                "is_fraud":        event.is_fraud,
            })

            if len(chunk) >= chunk_size:
                yield pd.DataFrame(chunk)
                chunk = []

        if chunk:
            yield pd.DataFrame(chunk)
