# backend/utils/logger.py

import csv
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from app.utils.config import SESSION_LOG_FILE


def update_remarks(log_file: str, target_id: str, new_remarks: str) -> bool:
    """
    Updates the remarks for a specific target_id in the CSV file.
    Returns True if updated successfully, False otherwise.
    """
    if not os.path.exists(log_file):
        return False

    updated = False
    rows = []
    with open(log_file, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["target_id"] == target_id:
                row["comments"] = new_remarks
                updated = True
            rows.append(row)

    if updated and rows:
        with open(log_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return updated


def get_logs(log_file: str) -> List[Dict]:
    """
    Returns all logs from the CSV file.
    """
    logs = []
    if os.path.exists(log_file):
        with open(log_file, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            logs = list(reader)
    return logs


class OperatorLogger:
    def __init__(self, log_file: str = SESSION_LOG_FILE):
        self.log_file = log_file
        self.session_id = str(uuid.uuid4())
        self._init_log_file()

    def _init_log_file(self):
        """
        Create the log file with headers if it doesn't exist.
        """
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    "timestamp", "session_id", "operator_id",
                    "target_id", "label", "confidence",
                    "box", "operator_decision", "fp", "fn", "comments"
                ])

    def log_detection(
        self,
        operator_id: str,
        detection: Dict,
        operator_decision: str,
        comments: Optional[str] = "",
        is_fp: Optional[bool] = False,
        is_fn: Optional[bool] = False
    ):
        """
        Log a single detection and operator feedback to CSV.

        detection = {
            "target_id": int,
            "label": str,
            "confidence": float,
            "box": (x1, y1, x2, y2)
        }
        operator_decision = "Accept" or "Reject"
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp,
                self.session_id,
                operator_id,
                detection.get("target_id", str(uuid.uuid4())),
                detection.get("label", ""),
                round(detection.get("confidence", 0.0), 3),
                detection.get("box", (0, 0, 0, 0)),
                operator_decision,
                int(is_fp),
                int(is_fn),
                comments
            ])

    def read_logs(self) -> List[Dict]:
        """
        Returns the logs as a list of dictionaries.
        """
        return get_logs(self.log_file)

    def get_session_id(self) -> str:
        return self.session_id

    def clear_logs(self):
        """
        Clear the existing log file (for testing or new session).
        """
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        self._init_log_file()
