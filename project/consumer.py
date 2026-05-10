import json
import asyncio
import redis

from kafka import KafkaConsumer

from inference import predict_anomaly

from db import (
    SessionLocal,
    Alert
)

# =========================================================
# REDIS CLIENT
# =========================================================

redis_client = redis.Redis(

    host="redis",

    port=6379,

    decode_responses=True
)

# =========================================================
# CREATE KAFKA CONSUMER
# =========================================================

consumer = KafkaConsumer(

    "traffic-stream",

    bootstrap_servers="kafka:9092",

    auto_offset_reset="earliest",

    enable_auto_commit=True,

    group_id="netsage-group",

    value_deserializer=lambda x:
        json.loads(x.decode("utf-8"))
)

print("Kafka Consumer Connected")
print("Waiting for traffic...\n")

# =========================================================
# PROCESS MESSAGES
# =========================================================

async def process_messages():

    for message in consumer:

        db = None

        try:

            # =============================================
            # READ PACKET
            # =============================================

            packet = message.value

            # =============================================
            # RUN HYBRID INFERENCE
            # =============================================

            result = predict_anomaly(packet)

            # =============================================
            # HANDLE INFERENCE ERRORS
            # =============================================

            if "error" in result:

                print(
                    f"\nInference Error: "
                    f"{result['error']}"
                )

                continue

            # =============================================
            # DATABASE SESSION
            # =============================================

            db = SessionLocal()

            # =============================================
            # CREATE ALERT OBJECT
            # =============================================

            alert = Alert(

                anomaly_score=float(
                    result["anomaly_score"]
                ),

                threshold=float(
                    result["threshold"]
                ),

                xgb_probability=float(
                    result["xgb_probability"]
                ),

                severity=result["severity"],

                is_anomaly=bool(
                    result["is_anomaly"]
                ),

                top_features=json.dumps(
                    result["top_features"]
                )
            )

            # =============================================
            # STORE ALERT
            # =============================================

            db.add(alert)

            db.commit()

            # =============================================
            # TERMINAL LOGS
            # =============================================

            print("\n" + "=" * 60)

            print("Traffic Packet Processed")

            print(

                f"Anomaly Score : "
                f"{result['anomaly_score']:.6f}"
            )

            print(

                f"Threshold      : "
                f"{result['threshold']:.6f}"
            )

            print(

                f"XGB Probability: "
                f"{result['xgb_probability']:.6f}"
            )

            print(

                f"Severity       : "
                f"{result['severity']}"
            )

            print(

                f"Is Anomaly     : "
                f"{result['is_anomaly']}"
            )

            # =============================================
            # SHAP FEATURES
            # =============================================

            print("\nTop SHAP Features:")

            for feature in result["top_features"]:

                print(

                    f"  • {feature['feature']} "
                    f"({feature['importance']:.6f})"
                )

            # =============================================
            # ALERT
            # =============================================

            if result["is_anomaly"]:

                print(
                    "\n🚨 ALERT: Intrusion Detected"
                )

            print("=" * 60)

            # =============================================
            # PUBLISH TO REDIS
            # =============================================

            redis_client.publish(

                "alerts_channel",

                json.dumps({

                    "type": "traffic_update",

                    "data": {

                        # =================================
                        # CORE UNSW-NB15 FEATURES
                        # =================================

                        "src_ip":
                            packet.get("srcip"),

                        "dst_ip":
                            packet.get("dstip"),

                        "src_port":
                            packet.get("sport"),

                        "dst_port":
                            packet.get("dsport"),

                        "protocol":
                            packet.get("proto"),

                        "service":
                            packet.get("service"),

                        "state":
                            packet.get("state"),

                        "duration":
                            packet.get("dur"),

                        "src_bytes":
                            packet.get("sbytes"),

                        "dst_bytes":
                            packet.get("dbytes"),

                        "src_packets":
                            packet.get("Spkts"),

                        "dst_packets":
                            packet.get("Dpkts"),

                        "src_load":
                            packet.get("Sload"),

                        "dst_load":
                            packet.get("Dload"),

                        # =================================
                        # ATTACK INFORMATION
                        # =================================

                        "attack_category":
                            packet.get("attack_cat"),

                        "label":
                            packet.get("Label"),

                        # =================================
                        # ML OUTPUTS
                        # =================================

                        "severity":
                            result["severity"],

                        "is_anomaly":
                            bool(
                                result["is_anomaly"]
                            ),

                        "anomaly_score":
                            float(
                                result["anomaly_score"]
                            ),

                        "threshold":
                            float(
                                result["threshold"]
                            ),

                        "xgb_probability":
                            float(
                                result["xgb_probability"]
                            ),

                        "reconstruction_error":
                            float(
                                result["anomaly_score"]
                            ),

                        # =================================
                        # SHAP
                        # =================================

                        "top_features":
                            result["top_features"]
                    }
                })
            )

            print(
                "Alert Published To Redis"
            )

        except Exception as e:

            print(
                f"\nConsumer Error: {e}"
            )

        finally:

            if db:

                db.close()

# =========================================================
# START EVENT LOOP
# =========================================================

asyncio.run(process_messages())