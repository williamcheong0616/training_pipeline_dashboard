"""Timezone helpers. All DB writes use UTC; serialization includes the offset."""
from datetime import datetime, timezone

UTC = timezone.utc


def now_utc() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    Use this everywhere instead of datetime.utcnow() — utcnow() returns a
    naive datetime with no timezone info, so serializers cannot include the
    '+00:00' offset and the frontend cannot distinguish UTC from local time.
    """
    return datetime.now(UTC)
