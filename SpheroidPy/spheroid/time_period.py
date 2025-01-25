from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class TimePeriod:
    """Represents a named time period for analysis.
    
    Attributes:
        name: Unique identifier for this time period
        start_time: Start of time period
        end_time: End of time period
        description: Optional description of what this time period represents
        created_at: When this time period was created
        modified_at: When this time period was last modified
    """
    name: str
    start_time: datetime
    end_time: datetime
    description: Optional[str] = None
    created_at: datetime = datetime.now()
    modified_at: datetime = datetime.now()
    
    def __post_init__(self):
        if self.start_time > self.end_time:
            raise ValueError("Start time must be before end time")
            
    def update_time(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None):
        """Update the time period."""
        if start_time:
            self.start_time = start_time
        if end_time:
            self.end_time = end_time
        if self.start_time > self.end_time:
            raise ValueError("Start time must be before end time")
        self.modified_at = datetime.now() 