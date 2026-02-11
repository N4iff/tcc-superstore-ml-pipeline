from locust import HttpUser, task, between
import random
import requests

SEGMENTS = ["Consumer", "Corporate", "Home Office"]
REGIONS = ["Central", "East", "South", "West"]
CATEGORIES = ["Furniture", "Office Supplies", "Technology"]
SUB_CATEGORIES = ["Chairs", "Binders", "Phones", "Storage", "Tables", "Accessories"]
SHIP_MODES = ["First Class", "Second Class", "Standard Class", "Same Day"]

TIMEOUT_SECONDS = 30

class SuperstoreAsyncUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task
    def predict_async(self):
        payload = {
            "sales": round(random.uniform(5, 500), 2),
            "quantity": random.randint(1, 10),
            "discount": round(random.choice([0, 0.1, 0.2, 0.3]), 2),
            "segment": random.choice(SEGMENTS),
            "region": random.choice(REGIONS),
            "category": random.choice(CATEGORIES),
            "sub_category": random.choice(SUB_CATEGORIES),
            "ship_mode": random.choice(SHIP_MODES),
        }

        try:
            with self.client.post(
                "/predict_async",
                json=payload,
                timeout=TIMEOUT_SECONDS,
                catch_response=True,
            ) as r:
                if r.status_code != 200:
                    r.failure(f"HTTP {r.status_code}: {r.text}")
        except requests.exceptions.Timeout:
            self.environment.events.request_failure.fire(
                request_type="POST",
                name="/predict_async (timeout)",
                response_time=TIMEOUT_SECONDS * 1000,
                response_length=0,
                exception=Exception("client timeout"),
            )
