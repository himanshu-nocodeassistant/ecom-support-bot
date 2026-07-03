KNOWLEDGE_BASE = [
    {
        "id": "shipping-policy",
        "title": "Shipping policy",
        "category": "policy",
        "content": (
            "Orders usually ship within 2 business days. Standard delivery takes 3 to 7 business days "
            "after shipment. If a package is delayed, customers should check the tracking status first."
        ),
    },
    {
        "id": "refund-policy",
        "title": "Refund policy",
        "category": "policy",
        "content": (
            "Refunds are eligible within 30 days of delivery for damaged, defective, or incorrect items. "
            "Refund requests require an order id and a short reason."
        ),
    },
    {
        "id": "portable-blender-guide",
        "title": "Portable blender guide",
        "category": "product",
        "content": (
            "The portable blender has a 400ml jar, USB-C charging, and a safety lock that prevents blending "
            "when the lid is not secured."
        ),
    },
    {
        "id": "noise-cancelling-headphones-guide",
        "title": "Noise-cancelling headphones guide",
        "category": "product",
        "content": (
            "The headphones support active noise cancellation, 30 hours of battery life, and Bluetooth 5.3. "
            "They include a wired audio mode."
        ),
    },
    {
        "id": "returns-policy",
        "title": "Returns policy",
        "category": "policy",
        "content": (
            "A return means sending the product back before a refund is issued. "
            "Contact support with your order ID within 30 days of delivery to start a return. "
            "A prepaid return label is sent within 1 business day. "
            "Inspection takes 1 to 3 business days after we receive the item."
        ),
    },
    {
        "id": "warranty-policy",
        "title": "Warranty policy",
        "category": "policy",
        "content": (
            "All hardware products carry a 12-month manufacturer warranty covering defects in materials "
            "and workmanship. Accidental damage is not covered. "
            "The blender motor and headphone speaker drivers have a separate 6-month extended warranty. "
            "Approved claims result in a free repair, replacement, or refund."
        ),
    },
    {
        "id": "payment-methods",
        "title": "Payment methods",
        "category": "policy",
        "content": (
            "We accept Visa, Mastercard, Amex, PayPal, and Apple Pay. "
            "Payments are processed via Stripe. Bank transfers are available for orders over $500. "
            "Refunds are returned to the original payment method within 5 to 7 business days."
        ),
    },
    {
        "id": "account-management",
        "title": "Account management",
        "category": "policy",
        "content": (
            "Customers can create an account to track orders and save addresses. "
            "Password resets are sent by email and expire after 30 minutes. "
            "Account deletion removes personal data within 30 days."
        ),
    },
    {
        "id": "order-modification",
        "title": "Order modification",
        "category": "policy",
        "content": (
            "Orders can be modified within 1 hour of placement. "
            "Contact support with your order ID to change quantities, variants, or the delivery address. "
            "Cancellations within the 1-hour window are free."
        ),
    },
    {
        "id": "address-change",
        "title": "Address change",
        "category": "policy",
        "content": (
            "Delivery addresses can be updated while the order status is pending or processing. "
            "Once dispatched, contact the carrier directly. "
            "If a package is returned due to an incorrect address, we will reship it free of charge."
        ),
    },
    {
        "id": "gift-wrapping",
        "title": "Gift wrapping",
        "category": "policy",
        "content": (
            "Gift wrapping is available for $3.99 per item and includes a personalised message. "
            "Price tags are removed from all gift orders. "
            "Gift wrapping cannot be added after an order is placed."
        ),
    },
    {
        "id": "subscription-billing",
        "title": "Subscription billing",
        "category": "policy",
        "content": (
            "Subscriptions offer a 15% discount and bill monthly on the same day. "
            "Pause for up to 3 months or cancel anytime in Account Settings. "
            "Failed payments are retried after 3 days and then 7 days before the subscription is paused."
        ),
    },
    {
        "id": "b2b-wholesale",
        "title": "B2B and wholesale orders",
        "category": "policy",
        "content": (
            "Volume discounts start at 10 units: 10% off for 10-24 units, 18% for 25-49, 25% for 50+. "
            "Contact the B2B team by email for a custom invoice. "
            "Net-30 payment terms are available for verified business accounts."
        ),
    },
    {
        "id": "accessibility",
        "title": "Accessibility",
        "category": "policy",
        "content": (
            "The website meets WCAG 2.1 AA standards. Orders can be placed by phone or live chat. "
            "Products include braille labels. Large-print manuals are available on request."
        ),
    },
    {
        "id": "smart-home-hub-guide",
        "title": "Smart home hub guide",
        "category": "product",
        "content": (
            "The smart home hub connects up to 50 devices and supports Wi-Fi 6, Zigbee, and Z-Wave. "
            "Compatible with Alexa, Google Home, and Apple HomeKit. "
            "A red LED means network disconnection — press the reset button for 3 seconds to reconnect."
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
