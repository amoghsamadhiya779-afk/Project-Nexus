import random
import uuid
from datetime import datetime, timedelta
import numpy as np

class MarketplaceSimulator:
    def __init__(self, n_users: int = 10000, n_items: int = 5000):
        print(f"[*] Simulating marketplace baseline vectors... (Users: {n_users}, Items: {n_items})")
        self.users = [str(uuid.uuid4()) for _ in range(n_users)]
        self.items = [str(uuid.uuid4()) for _ in range(n_items)]
        self.categories = ["electronics", "apparel", "home", "beauty", "books"]
        
        self.item_metadata = {
            item_id: {
                "category": random.choice(self.categories),
                "base_price": round(random.uniform(5.0, 500.0), 2),
                "popularity_weight": random.gammavariate(alpha=2.0, beta=1.0)
            }
            for item_id in self.items
        }
        
        self.user_arr = np.array(self.users)
        self.item_arr = np.array(self.items)
        weights = np.array([self.item_metadata[item_id]["popularity_weight"] for item_id in self.items])
        self.cum_probabilities = np.cumsum(weights / weights.sum())
        
        self.item_categories = np.array([self.item_metadata[item_id]["category"] for item_id in self.items])
        self.item_prices = np.array([self.item_metadata[item_id]["base_price"] for item_id in self.items])

    def generate_batch(self, n_events: int) -> list:
        sampled_users = np.random.choice(self.user_arr, size=n_events)
        rands = np.random.rand(n_events)
        sampled_indices = np.searchsorted(self.cum_probabilities, rands)
        sampled_indices = np.clip(sampled_indices, 0, len(self.items) - 1)
        
        sampled_items = self.item_arr[sampled_indices]
        sampled_categories = self.item_categories[sampled_indices]
        sampled_prices = self.item_prices[sampled_indices]
        
        event_rolls = np.random.rand(n_events)
        event_types = np.where(event_rolls < 0.70, "view", np.where(event_rolls < 0.92, "cart", "purchase"))
        
        random_offsets = np.random.randint(0, 86400 * 7, size=n_events)
        base_time = datetime.utcnow()
        uuids = [str(uuid.uuid4()) for _ in range(n_events)]
        
        events = []
        for i in range(n_events):
            event_time = base_time - timedelta(seconds=int(random_offsets[i]))
            events.append({
                "event_id": uuids[i],
                "user_id": str(sampled_users[i]),
                "item_id": str(sampled_items[i]),
                "event_type": str(event_types[i]),
                "category": str(sampled_categories[i]),
                "price": float(sampled_prices[i]),
                "timestamp": event_time.isoformat()
            })
        return events
