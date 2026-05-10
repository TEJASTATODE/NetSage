import json
import asyncio
import redis

from fastapi import (
    FastAPI,
    WebSocket
)

from fastapi.middleware.cors import (
    CORSMiddleware
)

from sqlalchemy.orm import Session

from inference import predict_anomaly

from websocket_manager import manager

from db import (
    SessionLocal,
    Alert
)

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="NetSage IDS API"
)

# =========================================================
# CORS
# =========================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
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
# REDIS PUBSUB
# =========================================================

pubsub = redis_client.pubsub()

pubsub.subscribe("alerts_channel")

# =========================================================
# DATABASE SESSION
# =========================================================

def get_db():

    db = SessionLocal()

    try:

        yield db

    finally:

        db.close()

# =========================================================
# REDIS LISTENER
# =========================================================

async def redis_listener():

    print(
        "Listening For Redis Alerts..."
    )

    while True:

        message = pubsub.get_message()

        if message:

            if message["type"] == "message":

                try:

                    data = json.loads(
                        message["data"]
                    )

                    print(
                        "Redis Alert Received"
                    )

                    # =====================================
                    # BROADCAST TO DASHBOARD
                    # =====================================

                    await manager.broadcast(
                        data
                    )

                except Exception as e:

                    print(
                        f"Redis Parse Error: {e}"
                    )

        await asyncio.sleep(0.01)

# =========================================================
# STARTUP EVENT
# =========================================================

@app.on_event("startup")
async def startup_event():

    asyncio.create_task(
        redis_listener()
    )

# =========================================================
# HOME ROUTE
# =========================================================

@app.get("/")
def home():

    return {

        "message":
        "NetSage API Running"
    }

# =========================================================
# HTTP PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
def predict(data: dict):

    result = predict_anomaly(data)

    return result

# =========================================================
# GET RECENT ALERTS
# =========================================================

@app.get("/alerts")
def get_alerts():

    db = SessionLocal()

    try:

        alerts = (

            db.query(Alert)

            .order_by(Alert.id.desc())

            .limit(100)

            .all()
        )

        results = []

        for alert in alerts:

            results.append({

                "id":
                    alert.id,

                "timestamp":
                    str(alert.timestamp),

                "anomaly_score":
                    alert.anomaly_score,

                "threshold":
                    alert.threshold,

                "xgb_probability":
                    alert.xgb_probability,

                "severity":
                    alert.severity,

                "is_anomaly":
                    alert.is_anomaly,

                "top_features":
                    alert.top_features
            })

        return results

    finally:

        db.close()

# =========================================================
# DASHBOARD STATS
# =========================================================

@app.get("/stats")
def get_stats():

    db = SessionLocal()

    try:

        alerts = db.query(Alert).all()

        total_alerts = len(alerts)

        critical_alerts = len([

            a for a in alerts

            if a.severity == "CRITICAL"
        ])

        avg_confidence = 0

        if total_alerts > 0:

            avg_confidence = (

                sum(

                    a.xgb_probability

                    for a in alerts

                ) / total_alerts

            ) * 100

        return {

            "total_alerts":
                total_alerts,

            "critical_alerts":
                critical_alerts,

            "avg_confidence":
                round(
                    avg_confidence,
                    2
                )
        }

    finally:

        db.close()

# =========================================================
# SEVERITY ANALYTICS
# =========================================================

@app.get("/severity-counts")
def severity_counts():

    db = SessionLocal()

    try:

        alerts = db.query(Alert).all()

        return {

            "LOW": len([

                a for a in alerts

                if a.severity == "LOW"
            ]),

            "MEDIUM": len([

                a for a in alerts

                if a.severity == "MEDIUM"
            ]),

            "HIGH": len([

                a for a in alerts

                if a.severity == "HIGH"
            ]),

            "CRITICAL": len([

                a for a in alerts

                if a.severity == "CRITICAL"
            ])
        }

    finally:

        db.close()

# =========================================================
# WEBSOCKET ENDPOINT
# =========================================================

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket
):

    await manager.connect(
        websocket
    )

    print(
        "Dashboard connected"
    )

    try:

        while True:

            # =============================================
            # KEEP CONNECTION ALIVE
            # =============================================

            await asyncio.sleep(1)

    except Exception as e:

        print(
            f"WebSocket Error: {e}"
        )

    finally:

        manager.disconnect(
            websocket
        )

        print(
            "Dashboard disconnected"
        )
        
@app.get("/health")
async def health():

    return {

        "kafka": True,

        "redis": True,

        "postgres": True,

        "consumer_alive": True,

        "model_loaded": True
    }    