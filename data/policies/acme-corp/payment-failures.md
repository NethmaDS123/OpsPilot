---
document_id: acme-payment-failures-v1
title: Payment Failures and Billing Errors
company_id: acme-corp
category: payment_failures
version: "1.0"
status: active
effective_from: 2025-01-01
effective_to: null
---

# Payment Failures and Billing Errors

## Declined payments

Orders with declined cards are held for **24 hours**. Customers receive up to **three retry attempts** via email link.

Subscriptions enter **grace period** of **5 days** before suspension; content access may continue during grace for annual plans only.

## Duplicate charges

If charged twice for one order:
1. Automatic detection usually refunds within **24 hours**.
2. If not auto-refunded, customers should report within **60 days** for expedited reversal.

Duplicate refunds require proof of two settlement IDs.

## Currency and taxes

International customers see estimated duties at checkout; Acme does not control customs holds causing perceived "payment failures."

## Payment methods

Accepted: major cards, PayPal, Acme Pay. Acme Pay failures often relate to expired backup cards—update in wallet settings.

## Chargebacks

Filing a chargeback pauses refunds and may suspend the account pending resolution. Voluntary chargeback withdrawal may restore account standing.

## PCI compliance

Never send full card numbers via chat or email; agents use tokenized payment update links only.
