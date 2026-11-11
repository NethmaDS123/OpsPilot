"""Labeled customer-support tickets for training and evaluation (300+ examples)."""

from __future__ import annotations

from itertools import product

TicketExample = tuple[str, str]


def _billing_examples() -> list[TicketExample]:
    cores = [
        "I was charged twice for order {oid}",
        "My payment was taken two times on {date}",
        "There are duplicate transactions on my card ending {last4}",
        "My card was debited again for the same purchase",
        "Why did you take my money twice for {oid}?",
        "I see two identical charges from your company",
        "Please explain the double billing on my statement",
        "Your system billed me twice for one subscription",
        "I only bought one item but paid twice",
        "There is a repeated charge I did not authorize",
        "Can you remove the extra payment from {date}?",
        "I was overcharged because of a duplicate authorization",
        "My bank shows two pending charges for one order",
        "Fix the billing error on invoice {inv}",
        "I got charged again after already paying",
        "Stop the second charge on my account please",
        "The receipt shows duplicate line items for shipping",
        "Why am I billed monthly twice for the same plan?",
        "Someone used my card and I see multiple charges",
        "I need help with incorrect billing on {oid}",
    ]
    orders = ["ord_1042", "ord_8821", "ord_3309", "ord_7710", "ord_5593"]
    dates = ["March 3", "last Friday", "yesterday", "Jan 12", "this morning"]
    last4 = ["4421", "9012", "1188", "6634", "2207"]
    inv = ["INV-992", "INV-441", "INV-118", "INV-803", "INV-256"]
    out: list[TicketExample] = []
    for i, core in enumerate(cores):
        out.append(
            (
                core.format(
                    oid=orders[i % len(orders)],
                    date=dates[i % len(dates)],
                    last4=last4[i % len(last4)],
                    inv=inv[i % len(inv)],
                ),
                "billing",
            )
        )
    extras = [
        ("Unauthorized charge appeared after I cancelled", "billing"),
        ("Fraudulent charge on my Visa — I did not buy this", "billing"),
        ("Wrong amount charged compared to checkout total", "billing"),
        ("Tax was calculated twice at payment", "billing"),
        ("Promo code applied but full price charged twice", "billing"),
        ("Subscription renewed and also charged a setup fee twice", "billing"),
        ("PayPal shows duplicate capture for one cart", "billing"),
        ("Invoice total does not match what I agreed to pay", "billing"),
        ("Charged in USD and again in local currency", "billing"),
        ("Refund pending but a new charge posted anyway", "billing"),
        ("Statement shows a mystery charge from your merchant ID", "billing"),
        ("Installment plan charged full amount upfront too", "billing"),
        ("Currency conversion fee applied twice", "billing"),
        ("Corporate card flagged duplicate at same timestamp", "billing"),
        ("Trial ended but I was billed before day seven", "billing"),
        ("Gift card balance not applied; card charged full price", "billing"),
        ("Annual plan billed monthly by mistake", "billing"),
        ("VAT invoice missing for the second charge", "billing"),
        ("Charge description unreadable on bank feed", "billing"),
        ("Need itemized breakdown — totals do not add up", "billing"),
        ("Autopay ran twice after card update", "billing"),
        ("Disputed first charge yet second attempt posted", "billing"),
        ("Coupon rejected at checkout but discount still missing", "billing"),
        ("Split payment left a stray authorization hold", "billing"),
        ("Finance team needs proof only one SKU was purchased", "billing"),
        ("Wire transfer received but card also charged", "billing"),
        ("Membership fee duplicated across two accounts", "billing"),
        ("Late fee added though payment was on time", "billing"),
        ("Processing fee higher than advertised at checkout", "billing"),
        ("Partial capture then full capture on same order", "billing"),
    ]
    return out + extras


def _refund_examples() -> list[TicketExample]:
    templates = [
        "I need a refund for my purchase {oid}",
        "Please return my money for the defective item",
        "I want to cancel and get my money back",
        "Can you process a refund to my original card?",
        "The product was not as described — refund please",
        "I returned the package — where is my refund?",
        "Requesting full refund for order {oid}",
        "I'd like my payment reversed",
        "Please issue a credit for the unused portion",
        "Refund my order; I never received confirmation",
        "I changed my mind — please refund within policy",
        "Return authorization complete, need refund status",
        "Partial refund expected after price drop",
        "Cancel membership and refund last month",
        "I was charged but order cancelled — refund?",
        "How long until refund shows on my statement?",
        "Refund not received after 10 business days",
        "Please refund shipping since delivery failed",
        "I want compensation refunded to my wallet",
        "Charge still there after promised refund",
    ]
    oids = ["ord_2201", "ord_9912", "ord_4400", "ord_6618", "ord_1029"]
    more = [
        "Escalate refund for gift purchase {oid}",
        "Store credit instead of cash refund — is that allowed?",
        "Refund request denied — please review case {case}",
        "I shipped it back tracked — initiate refund",
        "Cooling-off period refund for digital download",
        "Refund sales tax on cancelled preorder",
        "Bundle refund when only one item defective",
        "Refund to different card than original",
        "International refund timeline question",
        "Refund stuck in 'processing' for weeks",
        "Chargeback filed — still want merchant refund",
        "Partial return — refund rest of order {oid}",
        "Subscription refund for unused months",
        "Refund after price match guarantee",
        "Wrong item sent — refund and reorder",
        "Refund requested via app but no ticket number",
        "Agent promised refund on call — not showing",
        "Refund currency different from purchase",
        "Refund for expedited shipping not delivered",
        "Please confirm refund amount before issuing",
        "Refund for warranty replacement declined",
        "Double refund received — which to keep?",
        "Refund email confirmation never arrived",
        "Refund to PayPal still pending",
        "Refund after fraud investigation closed",
        "Restocking fee disputed on refund",
        "Refund for add-on service never activated",
        "Refund eligibility for opened software",
        "Refund request for duplicate order {oid}",
        "Refund policy exception for medical item",
    ]
    out = [(t.format(oid=oids[i % len(oids)]), "refund") for i, t in enumerate(templates)]
    out += [(m.format(oid=oids[i % len(oids)], case=f"CS-{1000+i}"), "refund") for i, m in enumerate(more)]
    return out


def _technical_examples() -> list[TicketExample]:
    items = [
        ("My laptop screen is cracked and the keyboard sticks", "technical"),
        ("The mobile app keeps crashing on launch", "technical"),
        ("I get an error message when exporting reports", "technical"),
        ("Device firmware update failed halfway through", "technical"),
        ("Bluetooth pairing does not work after reset", "technical"),
        ("Camera on the tablet is completely black", "technical"),
        ("Software license key says already activated", "technical"),
        ("Integration webhook returns 500 every time", "technical"),
        ("Dashboard charts never finish loading", "technical"),
        ("Push notifications stopped working on iOS", "technical"),
        ("The sensor readings look corrupted", "technical"),
        ("VPN client disconnects every few minutes", "technical"),
        ("Printer driver install fails on Windows 11", "technical"),
        ("API token authentication suddenly invalid", "technical"),
        ("Screen flickers when brightness above 70%", "technical"),
        ("Headphones mic not detected by meeting app", "technical"),
        ("Backup job failed with checksum mismatch", "technical"),
        ("Game crashes at loading screen after patch", "technical"),
        ("Smart hub offline since power outage", "technical"),
        ("Router admin page shows configuration error", "technical"),
    ]
    variants = [
        " since the last update",
        " on my work machine",
        " after I moved offices",
        " when using guest Wi‑Fi",
        " only on weekends strangely",
    ]
    expanded: list[TicketExample] = []
    for msg, label in items:
        expanded.append((msg, label))
        expanded.append((msg + variants[len(expanded) % len(variants)], label))
    return expanded[:55]


def _delivery_examples() -> list[TicketExample]:
    bases = [
        "My package has not arrived yet",
        "Where is my tracking number for {oid}?",
        "Delivery status stuck on 'label created'",
        "Courier marked delivered but nothing at my door",
        "Shipping delay — expected by {date} still waiting",
        "Tracking shows return to sender without reason",
        "Package arrived at wrong city hub",
        "Still waiting for my order after two weeks",
        "Can you update me on shipment {track}?",
        "Express shipping but no movement for days",
        "Missing items from a split shipment",
        "Delivery window passed with no call",
        "Locker code never arrived by SMS",
        "Customs hold — need documentation?",
        "Address correct but carrier says undeliverable",
        "Who signed for my package? I was home",
        "Partial delivery — one box missing",
        "Rain damaged box left on porch",
        "Need to reschedule delivery for Friday",
        "Is my order still in the warehouse?",
    ]
    tracks = ["1Z999AA10123456784", "JD014600012345678901", "9400111899223344556677"]
    out = [
        (
            b.format(
                oid="ord_5510",
                date="Tuesday",
                track=tracks[i % len(tracks)],
            ),
            "delivery",
        )
        for i, b in enumerate(bases)
    ]
    more = [
        ("Hasn't arrived — can you trace it?", "delivery"),
        ("Tracking link broken on confirmation email", "delivery"),
        ("Shipment rerouted without my approval", "delivery"),
        ("Out for delivery since 6am, still nothing", "delivery"),
        ("Neighbor did not receive my package either", "delivery"),
    ]
    more2 = [
        ("Carrier lost package — send replacement", "delivery"),
        ("Delivery photo shows wrong house number", "delivery"),
        ("Hold at pickup location expired", "delivery"),
        ("Fragile sticker ignored — box crushed", "delivery"),
        ("Saturday delivery paid but came Monday", "delivery"),
        ("Signature required but nobody attempted delivery", "delivery"),
        ("Tracking frozen at 'in transit' for ten days", "delivery"),
        ("Address updated before ship but old address used", "delivery"),
        ("International tracking stops at border", "delivery"),
        ("Delivery attempted while I was home — no knock", "delivery"),
        ("Package marked delivered to mailroom not found", "delivery"),
        ("Split order — when does second box ship?", "delivery"),
        ("Same tracking on two different orders", "delivery"),
        ("Courier left package in rain", "delivery"),
        ("Need proof of delivery for insurance", "delivery"),
        ("Change delivery to office address", "delivery"),
        ("Locker full — redelivery never scheduled", "delivery"),
        ("Gift shipment — hide price on packing slip", "delivery"),
        ("Delivery ETA keeps moving later", "delivery"),
        ("Rural route delay — any updates?", "delivery"),
        ("Package returned due to incomplete address", "delivery"),
        ("White glove delivery missed appointment window", "delivery"),
        ("Temperature sensitive shipment warm on arrival", "delivery"),
        ("Notify me before driver arrives", "delivery"),
        ("Order shows delivered but SKU missing inside", "delivery"),
    ]
    return out + more + more2


def _account_examples() -> list[TicketExample]:
    lines = [
        ("How do I update my account name on the profile?", "account"),
        ("I cannot log in — password reset email never comes", "account"),
        ("My account has been hacked and emails changed", "account"),
        ("Someone logged in from another country", "account"),
        ("Need to change the email on my account", "account"),
        ("Two-factor codes not working on new phone", "account"),
        ("Please close my account and delete data", "account"),
        ("Unauthorized access — I see orders I didn't place", "account"),
        ("Profile settings page returns blank", "account"),
        ("Reset my password — link expired immediately", "account"),
        ("Merge two accounts I created by mistake", "account"),
        ("Update billing address on account profile", "account"),
        ("Account locked after too many attempts", "account"),
        ("Change phone number for SMS verification", "account"),
        ("Compromised credentials — what should I do?", "account"),
        ("Remove old devices from trusted list", "account"),
        ("GDPR data export request for my account", "account"),
        ("Username already taken when registering", "account"),
        ("Cannot verify identity for business account", "account"),
        ("Session keeps expiring every five minutes", "account"),
    ]
    extra = [
        ("Hi, quick question about updating profile photo", "account"),
        ("Change my name after marriage on account", "account"),
        ("Login works on web but not mobile app", "account"),
        ("Account shows wrong language preference", "account"),
        ("Deactivate account temporarily while traveling", "account"),
    ]
    pad = [
        ("Recover account after email provider outage", "account"),
        ("Link social login to existing account", "account"),
        ("Business account — add second admin user", "account"),
        ("Student discount verification on account", "account"),
        ("Account credits not showing in wallet", "account"),
        ("Remove deceased user account per family request", "account"),
        ("Account flagged for review — how to clear?", "account"),
        ("Import purchase history to new account", "account"),
        ("Account PIN for phone support verification", "account"),
        ("Accessibility settings reset after update", "account"),
        ("Account shows wrong timezone for orders", "account"),
        ("Cannot delete saved payment method", "account"),
        ("Account notification preferences keep reverting", "account"),
        ("Family sharing invite not received", "account"),
        ("Account migration from legacy portal", "account"),
        ("Employer SSO login stopped working", "account"),
        ("Account holds wrong tax ID for invoices", "account"),
        ("Verify account for higher purchase limits", "account"),
        ("Account activity log export request", "account"),
        ("Reactivate dormant account safely", "account"),
        ("Account birthday reward never applied", "account"),
        ("Change account country region setting", "account"),
        ("Account linked to wrong organization", "account"),
        ("Remove compromised API keys from account", "account"),
        ("Account deletion confirmation not received", "account"),
    ]
    return lines + extra + pad


def _other_examples() -> list[TicketExample]:
    greetings = [
        "Hello, I have a general question about your store hours",
        "Hi there — do you offer gift wrapping?",
        "Just wondering if you ship internationally",
        "Quick question about your loyalty program tiers",
        "Good morning, where can I find product manuals?",
        "Do you have a phone number for sales?",
        "What are your holiday closure dates?",
        "Can I visit a showroom in person?",
        "General inquiry about partnership opportunities",
        "Thanks for the fast reply yesterday",
    ]
    misc = [
        ("Your website colors are hard to read — feedback", "other"),
        ("Looking for career openings in support", "other"),
        ("Press request — media kit availability", "other"),
        ("Suggestion: add dark mode to the portal", "other"),
        ("How do I subscribe to the newsletter?", "other"),
        ("Are you hiring interns this summer?", "other"),
        ("Compliment to the team — great packaging", "other"),
        ("Which charity do you donate returns to?", "other"),
        ("Do you sponsor local events?", "other"),
        ("Request for wholesale pricing sheet", "other"),
    ]
    vague = [
        ("help!!!", "other"),
        ("???", "other"),
        ("Not sure who to ask about this", "other"),
        ("Need assistance please", "other"),
        ("Can someone call me?", "other"),
    ]
    combos = [
        f"{g} Also curious about bulk discounts."
        for g in greetings[:5]
    ]
    out: list[TicketExample] = [(g, "other") for g in greetings]
    out.extend(misc)
    out.extend(vague)
    out.extend((c, "other") for c in combos)
    # Pad to 55 with unique product questions
    products = ["blender", "monitor", "desk chair", "headset", "router"]
    channels = ["email", "chat", "phone", "Twitter", "forum"]
    for prod, ch in product(products, channels):
        out.append(
            (
                f"Do you still sell the {prod} model from last year via {ch}?",
                "other",
            )
        )
    out.extend(
        [
            ("Who do I contact about community guidelines?", "other"),
            ("Feedback on recent rebranding", "other"),
            ("Request catalog by postal mail", "other"),
            ("Do you offer tours of your warehouse?", "other"),
            ("General compliment about support chat tone", "other"),
            ("Where can I read release notes?", "other"),
        ]
    )
    return out


def load_labeled_tickets() -> list[TicketExample]:
    """Combine category blocks; enforces minimum dataset size for training."""
    parts = [
        _billing_examples(),
        _refund_examples(),
        _technical_examples(),
        _delivery_examples(),
        _account_examples(),
        _other_examples(),
    ]
    dataset: list[TicketExample] = []
    for block in parts:
        dataset.extend(block)
    if len(dataset) < 300:
        raise ValueError(f"Dataset too small: {len(dataset)} examples")
    return dataset


LABELED_TICKETS: list[TicketExample] = load_labeled_tickets()
