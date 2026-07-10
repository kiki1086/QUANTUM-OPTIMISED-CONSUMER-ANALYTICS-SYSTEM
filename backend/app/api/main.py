from fastapi import APIRouter
from app.api.routes import auth, users, analytics, ingestion, business_insights, innovation, quantum_analytics

api_router = APIRouter()
api_router.include_router(auth.router, tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(ingestion.router, prefix="/ingestion", tags=["ingestion"])
api_router.include_router(business_insights.router, prefix="/insights", tags=["insights"])
api_router.include_router(innovation.router, prefix="/innovation", tags=["innovation"])
api_router.include_router(quantum_analytics.router, prefix="/quantum", tags=["quantum"])
