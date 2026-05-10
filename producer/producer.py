import json
import time
import pandas as pd

from kafka import KafkaProducer

# =========================================================
# CREATE KAFKA PRODUCER
# =========================================================

producer = None

while producer is None:

    try:

        producer = KafkaProducer(

            bootstrap_servers="kafka:9092",

            value_serializer=lambda v:
                json.dumps(v).encode("utf-8")
        )

        print("Kafka Producer Connected")

    except Exception as e:

        print("Waiting for Kafka...", e)

        time.sleep(5)

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(

    "/data/UNSW-NB15_1_with_features.csv",

    low_memory=False
)

print("Dataset Loaded")

# =========================================================
# CLEAN DATA
# =========================================================

# Remove hidden spaces from headers
df.columns = df.columns.str.strip()

# Fill missing categorical values
df = df.fillna("")

# Convert Label safely if exists
if "Label" in df.columns:

    df["Label"] = pd.to_numeric(

        df["Label"],

        errors="coerce"
    ).fillna(0)

print("Dataset Cleaned")

print("Starting realtime stream...\n")

# =========================================================
# SEND DATA TO KAFKA
# =========================================================

for index, row in df.iterrows():

    # =============================================
    # CONVERT NUMPY TYPES TO PYTHON TYPES
    # =============================================

    message = {

        key:

        (
            value.item()

            if hasattr(value, "item")

            else value
        )

        for key, value in row.to_dict().items()
    }

    # =============================================
    # SEND TO KAFKA
    # =============================================

    producer.send(

        "traffic-stream",

        value=message
    )

    print(

        f"Sent Packet {index} | "

        f"{message.get('proto', 'unknown').upper()} "

        f"{message.get('service', '-')}"
    )

    # =============================================
    # SIMULATE REALTIME TRAFFIC
    # =============================================

    time.sleep(0.1)

# =========================================================
# COMPLETE
# =========================================================

print("\nRealtime Streaming Completed")

producer.flush()
producer.close()