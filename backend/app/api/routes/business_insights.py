from fastapi import APIRouter

router = APIRouter()

@router.get("/summary")
def get_executive_summary():
    """Stub endpoint for executive summaries"""
    return {
        "summary": "Revenue has grown by 12% this week. Customer retention in the 'Champions' segment remains strong, but there is a slight decline in the 'At Risk' segment. We recommend targeting them with the new dynamic pricing strategy."
    }

@router.get("/recommendations/growth")
def get_growth_opportunities():
    """Stub endpoint for growth opportunities"""
    return {
        "opportunities": [
            "Increase marketing spend on 'Electronics' by 15% due to upcoming seasonal demand.",
            "Cross-sell premium services to 'Loyal' customers."
        ]
    }
