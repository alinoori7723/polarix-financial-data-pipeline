from __future__ import annotations

import pytest

from polarix.orchestration.databento_cost_guard import (
    COST_ESTIMATE_UNAVAILABLE,
    COST_LIMIT_EXCEEDED,
    DOWNLOAD_APPROVED,
    DOWNLOAD_APPROVED_WITHOUT_PHYSICAL_LIMIT,
    DOWNLOAD_BLOCKED_BY_COST_ESTIMATE_UNAVAILABLE,
    DOWNLOAD_BLOCKED_BY_COST_GUARD,
    DOWNLOAD_BLOCKED_BY_PHYSICAL_LIMIT_GUARD,
    DOWNLOAD_BLOCKED_DRY_RUN,
    DOWNLOAD_BLOCKED_MISSING_ACK,
    DOWNLOAD_BLOCKED_MISSING_ALLOW_DATABENTO,
    DOWNLOAD_BLOCKED_MISSING_API_KEY,
    ESTIMATE_AVAILABLE,
    NOT_REQUESTED_DRY_RUN,
    PHYSICAL_LIMIT_NOT_REQUESTED_DRY_RUN,
    PHYSICAL_LIMIT_SUPPORTED,
    PHYSICAL_LIMIT_UNSUPPORTED,
    SIZE_LIMIT_EXCEEDED,
    CostEstimate,
    GateInputs,
    PhysicalLimitCapability,
    build_databento_request_with_physical_limit,
    decide_download_gate,
    detect_databento_physical_limit_capability,
    try_get_databento_cost_estimate,
)


def test_dry_run_never_contacts_estimator() -> None:
    sentinel = {"called": False}

    def factory():
        sentinel["called"] = True
        raise AssertionError("dry_run must not build a client")

    estimate = try_get_databento_cost_estimate(
        dataset="GLBX.MDP3",
        schema="mbp-1",
        symbols=["ES.c.0"],
        stype_in="continuous",
        start_utc="2026-05-18T05:00:00Z",
        end_utc="2026-05-18T09:00:00Z",
        dry_run=True,
        client_factory=factory,
    )
    assert estimate.status == NOT_REQUESTED_DRY_RUN
    assert sentinel["called"] is False


def test_missing_api_key_returns_unavailable() -> None:
    estimate = try_get_databento_cost_estimate(
        dataset="GLBX.MDP3",
        schema="mbp-1",
        symbols=["ES.c.0"],
        stype_in="continuous",
        start_utc="2026-05-18T05:00:00Z",
        end_utc="2026-05-18T09:00:00Z",
        api_key_present=False,
    )
    assert estimate.status == COST_ESTIMATE_UNAVAILABLE


def test_sdk_without_capabilities_returns_unavailable() -> None:

    class FakeClient:
        class metadata:
            pass

    estimate = try_get_databento_cost_estimate(
        dataset="GLBX.MDP3",
        schema="mbp-1",
        symbols=["ES.c.0"],
        stype_in="continuous",
        start_utc="2026-05-18T05:00:00Z",
        end_utc="2026-05-18T09:00:00Z",
        api_key_present=True,
        client_factory=lambda: FakeClient(),
    )
    assert estimate.status == COST_ESTIMATE_UNAVAILABLE
    assert estimate.estimated_cost_usd is None
    assert estimate.estimated_size_gb is None


def test_below_thresholds_is_available() -> None:

    class FakeMeta:
        def get_cost(self, **kw):
            return 0.5

        def get_billable_size(self, **kw):
            return 2 * 1024**3

    class FakeClient:
        metadata = FakeMeta()

    estimate = try_get_databento_cost_estimate(
        dataset="GLBX.MDP3",
        schema="mbp-1",
        symbols=["ES.c.0"],
        stype_in="continuous",
        start_utc="2026-05-18T05:00:00Z",
        end_utc="2026-05-18T09:00:00Z",
        api_key_present=True,
        client_factory=lambda: FakeClient(),
        max_estimated_cost_usd=10.0,
        max_estimated_size_gb=5.0,
    )
    assert estimate.status == ESTIMATE_AVAILABLE
    assert estimate.estimated_cost_usd == 0.5
    assert estimate.estimated_size_gb == pytest.approx(2.0)


def test_cost_above_threshold_blocks() -> None:

    class FakeMeta:
        def get_cost(self, **kw):
            return 999.99

        def get_billable_size(self, **kw):
            return 1 * 1024**3

    class FakeClient:
        metadata = FakeMeta()

    estimate = try_get_databento_cost_estimate(
        dataset="X",
        schema="x",
        symbols=["x"],
        stype_in="continuous",
        start_utc="x",
        end_utc="x",
        api_key_present=True,
        client_factory=lambda: FakeClient(),
        max_estimated_cost_usd=10.0,
        max_estimated_size_gb=5.0,
    )
    assert estimate.status == COST_LIMIT_EXCEEDED


def test_size_above_threshold_blocks() -> None:

    class FakeMeta:
        def get_cost(self, **kw):
            return 0.01

        def get_billable_size(self, **kw):
            return 100 * 1024**3

    class FakeClient:
        metadata = FakeMeta()

    estimate = try_get_databento_cost_estimate(
        dataset="X",
        schema="x",
        symbols=["x"],
        stype_in="continuous",
        start_utc="x",
        end_utc="x",
        api_key_present=True,
        client_factory=lambda: FakeClient(),
        max_estimated_cost_usd=10.0,
        max_estimated_size_gb=5.0,
    )
    assert estimate.status == SIZE_LIMIT_EXCEEDED


def test_estimate_does_not_invent_values_when_sdk_silent() -> None:

    class FakeClient:
        class metadata:
            pass

    estimate = try_get_databento_cost_estimate(
        dataset="X",
        schema="x",
        symbols=["x"],
        stype_in="continuous",
        start_utc="x",
        end_utc="x",
        api_key_present=True,
        client_factory=lambda: FakeClient(),
    )
    assert estimate.estimated_cost_usd is None
    assert estimate.estimated_size_gb is None


def test_physical_limit_dry_run_not_requested() -> None:
    cap = detect_databento_physical_limit_capability(dry_run=True)
    assert cap.status == PHYSICAL_LIMIT_NOT_REQUESTED_DRY_RUN


def test_physical_limit_supported_via_record_limit_param() -> None:

    class FakeTimeseries:
        def get_range(self, dataset, start, end, symbols, schema, stype_in, limit=None):
            return None

    class FakeClient:
        timeseries = FakeTimeseries()

    cap = detect_databento_physical_limit_capability(
        client_factory=lambda: FakeClient(),
        max_download_records=10000,
        target_call="timeseries.get_range",
    )
    assert cap.status == PHYSICAL_LIMIT_SUPPORTED
    assert "record_limit" in cap.supported_limit_types
    assert cap.selected_limit_type == "record_limit"
    assert cap.selected_limit_value == 10000


def test_physical_limit_unsupported_when_no_signature_match() -> None:

    class FakeTimeseries:
        def get_range(self, dataset, start, end, symbols):
            return None

    class FakeClient:
        timeseries = FakeTimeseries()

    cap = detect_databento_physical_limit_capability(
        client_factory=lambda: FakeClient(),
        target_call="timeseries.get_range",
        max_download_records=10000,
    )
    assert cap.status == PHYSICAL_LIMIT_UNSUPPORTED
    assert cap.selected_limit_type is None
    assert cap.selected_limit_value is None


def test_build_request_with_physical_limit_injects_param() -> None:
    cap = PhysicalLimitCapability(
        status=PHYSICAL_LIMIT_SUPPORTED,
        supported_limit_types=("record_limit",),
        selected_limit_type="record_limit",
        selected_limit_value=5000,
    )
    base = {"dataset": "X", "schema": "mbp-1", "symbols": ["ES.c.0"]}
    out = build_databento_request_with_physical_limit(base_kwargs=base, capability=cap)
    assert out["limit"] == 5000
    assert "limit" not in base


def test_build_request_with_physical_limit_unsupported_passthrough() -> None:
    cap = PhysicalLimitCapability(status=PHYSICAL_LIMIT_UNSUPPORTED, supported_limit_types=())
    base = {"dataset": "X"}
    out = build_databento_request_with_physical_limit(base_kwargs=base, capability=cap)
    assert out == {"dataset": "X"}


def _avail_estimate(cost=0.5, size_gb=1.0) -> CostEstimate:
    return CostEstimate(
        status=ESTIMATE_AVAILABLE,
        estimated_cost_usd=cost,
        estimated_size_gb=size_gb,
        source="test",
        max_estimated_cost_usd=10.0,
        max_estimated_size_gb=5.0,
    )


def _supported_cap() -> PhysicalLimitCapability:
    return PhysicalLimitCapability(
        status=PHYSICAL_LIMIT_SUPPORTED,
        supported_limit_types=("record_limit",),
        selected_limit_type="record_limit",
        selected_limit_value=10000,
    )


def test_gate_dry_run_blocks() -> None:
    decision = decide_download_gate(
        GateInputs(dry_run=True),
        cost_estimate=CostEstimate(status=NOT_REQUESTED_DRY_RUN),
        physical_limit=PhysicalLimitCapability(status=PHYSICAL_LIMIT_NOT_REQUESTED_DRY_RUN),
    )
    assert decision.decision == DOWNLOAD_BLOCKED_DRY_RUN


def test_gate_missing_allow_databento_blocks() -> None:
    decision = decide_download_gate(
        GateInputs(download_cme=True, allow_databento_download=False),
        cost_estimate=_avail_estimate(),
        physical_limit=_supported_cap(),
    )
    assert decision.decision == DOWNLOAD_BLOCKED_MISSING_ALLOW_DATABENTO


def test_gate_missing_acknowledge_blocks() -> None:
    decision = decide_download_gate(
        GateInputs(
            download_cme=True,
            allow_databento_download=True,
            acknowledge_cost_risk=False,
            api_key_present=True,
        ),
        cost_estimate=_avail_estimate(),
        physical_limit=_supported_cap(),
    )
    assert decision.decision == DOWNLOAD_BLOCKED_MISSING_ACK


def test_gate_missing_api_key_blocks() -> None:
    decision = decide_download_gate(
        GateInputs(
            download_cme=True,
            allow_databento_download=True,
            acknowledge_cost_risk=True,
            api_key_present=False,
        ),
        cost_estimate=_avail_estimate(),
        physical_limit=_supported_cap(),
    )
    assert decision.decision == DOWNLOAD_BLOCKED_MISSING_API_KEY


def test_gate_cost_estimate_unavailable_blocks_by_default() -> None:
    decision = decide_download_gate(
        GateInputs(
            download_cme=True,
            allow_databento_download=True,
            acknowledge_cost_risk=True,
            api_key_present=True,
            cost_estimate_required=True,
            allow_unestimated_download=False,
        ),
        cost_estimate=CostEstimate(status=COST_ESTIMATE_UNAVAILABLE),
        physical_limit=_supported_cap(),
    )
    assert decision.decision == DOWNLOAD_BLOCKED_BY_COST_ESTIMATE_UNAVAILABLE


def test_gate_cost_estimate_unavailable_passes_with_override() -> None:
    decision = decide_download_gate(
        GateInputs(
            download_cme=True,
            allow_databento_download=True,
            acknowledge_cost_risk=True,
            api_key_present=True,
            cost_estimate_required=True,
            allow_unestimated_download=True,
        ),
        cost_estimate=CostEstimate(status=COST_ESTIMATE_UNAVAILABLE),
        physical_limit=_supported_cap(),
    )
    assert decision.decision == DOWNLOAD_APPROVED
    assert any(("cost estimate" in w for w in decision.warnings))


def test_gate_cost_over_limit_blocks() -> None:
    decision = decide_download_gate(
        GateInputs(
            download_cme=True,
            allow_databento_download=True,
            acknowledge_cost_risk=True,
            api_key_present=True,
        ),
        cost_estimate=CostEstimate(
            status=COST_LIMIT_EXCEEDED, estimated_cost_usd=999.0, max_estimated_cost_usd=10.0
        ),
        physical_limit=_supported_cap(),
    )
    assert decision.decision == DOWNLOAD_BLOCKED_BY_COST_GUARD


def test_gate_physical_limit_unsupported_blocks_by_default() -> None:
    decision = decide_download_gate(
        GateInputs(
            download_cme=True,
            allow_databento_download=True,
            acknowledge_cost_risk=True,
            api_key_present=True,
            require_physical_download_limit=True,
            allow_download_without_physical_limit=False,
        ),
        cost_estimate=_avail_estimate(),
        physical_limit=PhysicalLimitCapability(status=PHYSICAL_LIMIT_UNSUPPORTED),
    )
    assert decision.decision == DOWNLOAD_BLOCKED_BY_PHYSICAL_LIMIT_GUARD


def test_gate_physical_limit_override_records_critical_warning() -> None:
    decision = decide_download_gate(
        GateInputs(
            download_cme=True,
            allow_databento_download=True,
            acknowledge_cost_risk=True,
            api_key_present=True,
            require_physical_download_limit=True,
            allow_download_without_physical_limit=True,
        ),
        cost_estimate=_avail_estimate(),
        physical_limit=PhysicalLimitCapability(status=PHYSICAL_LIMIT_UNSUPPORTED),
    )
    assert decision.decision == DOWNLOAD_APPROVED_WITHOUT_PHYSICAL_LIMIT
    assert any(("DOWNLOAD_WITHOUT_PHYSICAL_LIMIT" in w for w in decision.critical_warnings))


def test_gate_all_flags_present_and_within_limits_approves() -> None:
    decision = decide_download_gate(
        GateInputs(
            download_cme=True,
            allow_databento_download=True,
            acknowledge_cost_risk=True,
            api_key_present=True,
        ),
        cost_estimate=_avail_estimate(),
        physical_limit=_supported_cap(),
    )
    assert decision.decision == DOWNLOAD_APPROVED
