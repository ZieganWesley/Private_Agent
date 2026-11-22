import sqlite3

from private_agent.agent import timeline


def test_schema_and_estimate():
    conn = sqlite3.connect(":memory:")
    timeline.ensure_schema(conn)
    timeline.record_estimate(conn, "r1", 0, "Coding", 4.0, 0.5)
    timeline.record_actual(conn, "r1", 0, 6.0)
    est = timeline.estimate_step(conn, "Coding")
    assert est.seconds == 6.0
    assert est.confidence == 0.8


def test_estimate_fallback():
    conn = sqlite3.connect(":memory:")
    timeline.ensure_schema(conn)
    est = timeline.estimate_step(conn, "Research")
    assert est.confidence == 0.3
