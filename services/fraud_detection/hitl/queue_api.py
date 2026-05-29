from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import uuid

app = FastAPI(title="Nexus Human-in-the-Loop Fraud moderation queue", version="1.0.0")

# In-memory moderation queue representation
moderation_queue: Dict[str, Dict[str, Any]] = {}

class AlertPayload(BaseModel):
    transaction_id: str
    user_id: str
    amount: float
    risk_score: float
    reason: str

class ModerationDecision(BaseModel):
    transaction_id: str
    action: str # Approved, Blocked
    reviewer_id: str

@app.post("/alerts/trigger")
def trigger_alert(payload: AlertPayload):
    """Ingests flagged anomalies from the GNN stream into the manual moderation queue."""
    alert_id = str(uuid.uuid4())
    queue_entry = {
        "alert_id": alert_id,
        "transaction_id": payload.transaction_id,
        "user_id": payload.user_id,
        "amount": payload.amount,
        "risk_score": payload.risk_score,
        "reason": payload.reason,
        "status": "pending",
        "moderated_by": None
    }
    moderation_queue[payload.transaction_id] = queue_entry
    return {"status": "queued", "alert_id": alert_id}

@app.get("/alerts/pending", response_model=List[Dict[str, Any]])
def get_pending_alerts():
    """Returns all transaction flags awaiting moderator audit."""
    return [alert for alert in moderation_queue.values() if alert["status"] == "pending"]

@app.post("/alerts/decide")
def submit_decision(decision: ModerationDecision):
    """Processes analyst moderation decisions (approve or block flagged users)."""
    tx_id = decision.transaction_id
    if tx_id not in moderation_queue:
        raise HTTPException(status_code=404, detail="Alert entry not found in moderation queue.")
        
    entry = moderation_queue[tx_id]
    if entry["status"] != "pending":
        raise HTTPException(status_code=400, detail=f"Alert has already been resolved: {entry['status']}")
        
    entry["status"] = decision.action.lower()
    entry["moderated_by"] = decision.reviewer_id
    
    print(f"[+] Moderation action executed: Tx {tx_id} -> {decision.action.upper()} (By: {decision.reviewer_id})")
    return {"status": "resolved", "transaction_id": tx_id, "action": decision.action}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)
