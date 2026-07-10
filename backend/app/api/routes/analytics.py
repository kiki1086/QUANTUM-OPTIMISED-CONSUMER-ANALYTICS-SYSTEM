from fastapi import APIRouter

router = APIRouter()

@router.get("/kpis")
def get_executive_kpis():
    """Stub endpoint for executive KPIs"""
    return {
        "total_revenue": 1250000,
        "active_customers": 4500,
        "churn_rate": 0.02,
        "demand_forecast_7d": 3200
    }

@router.get("/demand-forecast")
def get_demand_forecast():
    """Stub endpoint for demand forecasting"""
    return {
        "forecast": [
            {"date": "2026-07-10", "expected": 150, "lower_bound": 140, "upper_bound": 160},
            {"date": "2026-07-11", "expected": 165, "lower_bound": 150, "upper_bound": 180},
        ]
    }

@router.get("/segmentation")
def get_customer_segmentation():
    """Stub endpoint for customer segments (RFM)"""
    return {
        "champions": 15,
        "loyal": 25,
        "at_risk": 10,
        "new": 50
    }
