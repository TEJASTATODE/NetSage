from sqlalchemy import (

    create_engine,

    Column,

    Integer,

    Float,

    String,

    Boolean,

    DateTime,

    Text
)

from sqlalchemy.orm import (

    declarative_base,

    sessionmaker
)

from datetime import datetime

# =========================================================
# DATABASE URL
# =========================================================

DATABASE_URL = (
    "postgresql://postgres:postgres@postgres:5432/netsage"
)

# =========================================================
# ENGINE
# =========================================================

engine = create_engine(
    DATABASE_URL
)

# =========================================================
# SESSION
# =========================================================

SessionLocal = sessionmaker(

    autocommit=False,

    autoflush=False,

    bind=engine
)

# =========================================================
# BASE CLASS
# =========================================================

Base = declarative_base()

# =========================================================
# ALERT TABLE
# =========================================================

class Alert(Base):

    __tablename__ = "alerts"

    # =====================================================
    # PRIMARY KEY
    # =====================================================

    id = Column(

        Integer,

        primary_key=True,

        index=True
    )

    # =====================================================
    # TIMESTAMP
    # =====================================================

    timestamp = Column(

        DateTime,

        default=datetime.utcnow
    )

    # =====================================================
    # AUTOENCODER DATA
    # =====================================================

    anomaly_score = Column(
        Float
    )

    threshold = Column(
        Float
    )

    # =====================================================
    # XGBOOST DATA
    # =====================================================

    xgb_probability = Column(
        Float
    )

    # =====================================================
    # ALERT METADATA
    # =====================================================

    severity = Column(
        String
    )

    is_anomaly = Column(
        Boolean
    )

    top_features = Column(
        Text
    )

# =========================================================
# CREATE TABLES
# =========================================================

Base.metadata.create_all(
    bind=engine
)

print(
    "Database tables created successfully"
)

# =========================================================
# GET DATABASE SESSION
# =========================================================

def get_db():

    db = SessionLocal()

    try:

        yield db

    finally:

        db.close()