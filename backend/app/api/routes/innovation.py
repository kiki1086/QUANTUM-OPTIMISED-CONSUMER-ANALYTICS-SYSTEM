from fastapi import APIRouter

router = APIRouter()

@router.get("/dynamic-pricing")
def get_dynamic_pricing_suggestions():
    """Stub endpoint for dynamic pricing suggestions"""
    return {
        "suggestions": [
            {"product_id": 102, "current_price": 120.0, "suggested_price": 115.0, "reason": "Inventory surplus"},
            {"product_id": 205, "current_price": 50.0, "suggested_price": 55.0, "reason": "High demand predicted"}
        ]
    }

@router.get("/churn-prediction")
def get_churn_predictions():
    """Stub endpoint for customer churn prediction"""
    return {
        "at_risk_customers": [
            {"customer_id": 453, "churn_probability": 0.85, "reason": "No purchases in 45 days"},
            {"customer_id": 992, "churn_probability": 0.72, "reason": "Decreased engagement"}
        ]
    }
