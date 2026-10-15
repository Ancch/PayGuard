import os
import json
import random
import csv
from datetime import datetime, timedelta

random.seed(42)

NUM_RECORDS = 30000
FRAUD_RATIO = 0.3  # 30% fraud total, ~10% each scenario
NUM_ACCOUNTS = 800
account_ids = [f"ACC{str(i).zfill(4)}" for i in range(1, NUM_ACCOUNTS + 1)]

# Helper functions
def rand_amount():
    # Log-uniform distribution for diverse amounts between ~100 and ~20000 INR
    return round(10 ** random.uniform(2, 4.3))

def random_message(fraud_type):
    if fraud_type == "refund_collect":
        return f"Your refund of ₹{rand_amount()} is ready. Approve to receive it."
    if fraud_type == "qr_mismatch":
        return f"Pay for your order at {random.choice(['ShopA', 'ShopB', 'ShopC'])}."
    if fraud_type == "mule_activity":
        return f"Transfer to {random.choice(account_ids)}."
    return f"Payment for {random.choice(['grocery', 'utility', 'rent'])}."

def generate_record(idx):
    # Determine whether this record is fraud and its type
    legit_cutoff = int(NUM_RECORDS * (1 - FRAUD_RATIO))
    if idx < legit_cutoff:
        fraud_type = "none"
        fraud_label = 0
    else:
        rel = idx - legit_cutoff
        per_type = int(NUM_RECORDS * FRAUD_RATIO / 3)
        if rel < per_type:
            fraud_type = "refund_collect"
        elif rel < 2 * per_type:
            fraud_type = "qr_mismatch"
        else:
            fraud_type = "mule_activity"
        fraud_label = 1

    transaction_id = f"T{str(idx+1).zfill(6)}"
    amount = rand_amount()
    payment_direction = "outgoing"
    is_new_payee = random.choice([True, False])
    message_text = random_message(fraud_type)
    expected_merchant = f"MERCH{random.randint(1, 200)}"
    actual_beneficiary = (
        f"BEN{random.randint(201, 400)}" if fraud_type == "qr_mismatch" else expected_merchant
    )
    payee_txn_count_30d = random.randint(0, 20)
    device_risk_signal = random.choice([0, 1])
    sender = random.choice(account_ids)
    receiver = random.choice(account_ids)
    ts = datetime.now() - timedelta(days=random.randint(0, 365), seconds=random.randint(0, 86400))
    timestamp = ts.isoformat()

    return {
        "transaction_id": transaction_id,
        "timestamp": timestamp,
        "sender_account": sender,
        "receiver_account": receiver,
        "amount_inr": amount,
        "payment_direction": payment_direction,
        "is_new_payee": is_new_payee,
        "message_text": message_text,
        "expected_merchant": expected_merchant,
        "actual_beneficiary": actual_beneficiary,
        "payee_txn_count_30d": payee_txn_count_30d,
        "device_risk_signal": device_risk_signal,
        "fraud_label": fraud_label,
        "fraud_type": fraud_type,
    }

os.makedirs('data', exist_ok=True)
csv_path = os.path.join('data', 'synthetic_transactions.csv')
fieldnames = [
    "transaction_id", "timestamp", "sender_account", "receiver_account", "amount_inr",
    "payment_direction", "is_new_payee", "message_text", "expected_merchant",
    "actual_beneficiary", "payee_txn_count_30d", "device_risk_signal",
    "fraud_label", "fraud_type",
]

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(NUM_RECORDS):
        writer.writerow(generate_record(i))

# Save generation configuration for reproducibility
config = {
    "seed": 42,
    "num_records": NUM_RECORDS,
    "num_accounts": NUM_ACCOUNTS,
    "fraud_ratio": FRAUD_RATIO,
    "fraud_distribution": {
        "refund_collect": 1/3,
        "qr_mismatch": 1/3,
        "mule_activity": 1/3,
    },
}
with open(os.path.join('data', 'generation_config.json'), "w") as f:
    json.dump(config, f, indent=2)

print(f"Synthetic dataset generated at {csv_path}")
