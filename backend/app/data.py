KNOWLEDGE_BASE = [
    {
        "id": "kb-shipping",
        "title": "Shipping policy",
        "category": "policy",
        "content": (
            "Orders usually ship within 2 business days. Standard delivery takes 3 to 7 business days "
            "after shipment. If a package is delayed, customers should check the tracking status first."
        ),
    },
    {
        "id": "kb-refund",
        "title": "Refund policy",
        "category": "policy",
        "content": (
            "Refunds are eligible within 30 days of delivery for damaged, defective, or incorrect items. "
            "Refund requests require an order id and a short reason."
        ),
    },
    {
        "id": "kb-blender",
        "title": "Portable blender guide",
        "category": "product",
        "content": (
            "The portable blender has a 400ml jar, USB-C charging, and a safety lock that prevents blending "
            "when the lid is not secured."
        ),
    },
    {
        "id": "kb-headphones",
        "title": "Noise-cancelling headphones guide",
        "category": "product",
        "content": (
            "The headphones support active noise cancellation, 30 hours of battery life, and Bluetooth 5.3. "
            "They include a wired audio mode."
        ),
    },
]

ORDERS = {
    "ORD-1001": {
        "order_id": "ORD-1001",
        "customer_name": "Ava",
        "status": "shipped",
        "shipping_date": "2026-05-18",
        "delivery_estimate": "2026-05-24",
        "item": "Portable blender",
        "delivered": False,
    },
    "ORD-1002": {
        "order_id": "ORD-1002",
        "customer_name": "Liam",
        "status": "delivered",
        "shipping_date": "2026-05-12",
        "delivery_estimate": "2026-05-16",
        "item": "Noise-cancelling headphones",
        "delivered": True,
    },
}
