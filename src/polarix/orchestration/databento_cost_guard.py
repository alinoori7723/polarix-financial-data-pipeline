from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Iterable, Optional

ESTIMATE_AVAILABLE = "ESTIMATE_AVAILABLE"
COST_ESTIMATE_UNAVAILABLE = "COST_ESTIMATE_UNAVAILABLE"
COST_LIMIT_EXCEEDED = "COST_LIMIT_EXCEEDED"
SIZE_LIMIT_EXCEEDED = "SIZE_LIMIT_EXCEEDED"
ESTIMATE_API_ERROR = "ESTIMATE_API_ERROR"
NOT_REQUESTED_DRY_RUN = "NOT_REQUESTED_DRY_RUN"


@dataclass(frozen=True)
class CostEstimate:
    status: str
    estimated_cost_usd: Optional[float] = None
    estimated_size_gb: Optional[float] = None
    estimated_record_count: Optional[int] = None
    source: Optional[str] = None
    message: Optional[str] = None
    max_estimated_cost_usd: Optional[float] = None
    max_estimated_size_gb: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "estimated_cost_usd": self.estimated_cost_usd,
            "estimated_size_gb": self.estimated_size_gb,
            "estimated_record_count": self.estimated_record_count,
            "source": self.source,
            "message": self.message,
            "max_estimated_cost_usd": self.max_estimated_cost_usd,
            "max_estimated_size_gb": self.max_estimated_size_gb,
        }


def _to_int_gb(billable_bytes: object) -> Optional[float]:
    try:
        return float(int(billable_bytes)) / 1024**3
    except Exception:
        return None


def _has_metadata_capability(client: object, attr: str) -> bool:
    metadata = getattr(client, "metadata", None)
    if metadata is None:
        return False
    fn = getattr(metadata, attr, None)
    return callable(fn)


def try_get_databento_cost_estimate(
    *,
    dataset: str,
    schema: str,
    symbols: Iterable[str],
    stype_in: str,
    start_utc: str,
    end_utc: str,
    max_estimated_cost_usd: Optional[float] = None,
    max_estimated_size_gb: Optional[float] = None,
    dry_run: bool = False,
    api_key_present: Optional[bool] = None,
    client_factory: Optional[callable] = None,
) -> CostEstimate:
    if dry_run:
        return CostEstimate(
            status=NOT_REQUESTED_DRY_RUN,
            source="dry_run",
            message="dry-run: did not contact Databento",
            max_estimated_cost_usd=max_estimated_cost_usd,
            max_estimated_size_gb=max_estimated_size_gb,
        )
    if api_key_present is None:
        api_key_present = bool(os.environ.get("DATABENTO_API_KEY"))
    if not api_key_present and client_factory is None:
        return CostEstimate(
            status=COST_ESTIMATE_UNAVAILABLE,
            source=None,
            message="DATABENTO_API_KEY not set; cannot query official estimate endpoint",
            max_estimated_cost_usd=max_estimated_cost_usd,
            max_estimated_size_gb=max_estimated_size_gb,
        )
    try:
        if client_factory is not None:
            client = client_factory()
        else:
            import databento

            client = databento.Historical(key=os.environ["DATABENTO_API_KEY"])
    except Exception as exc:
        return CostEstimate(
            status=ESTIMATE_API_ERROR,
            source="databento.Historical",
            message=f"could not construct client: {type(exc).__name__}: {exc}",
            max_estimated_cost_usd=max_estimated_cost_usd,
            max_estimated_size_gb=max_estimated_size_gb,
        )
    if not _has_metadata_capability(client, "get_cost") or not _has_metadata_capability(
        client, "get_billable_size"
    ):
        return CostEstimate(
            status=COST_ESTIMATE_UNAVAILABLE,
            source="databento.metadata",
            message="installed databento SDK does not expose metadata.get_cost AND metadata.get_billable_size; estimate is not available",
            max_estimated_cost_usd=max_estimated_cost_usd,
            max_estimated_size_gb=max_estimated_size_gb,
        )
    syms = list(symbols)
    try:
        cost_usd = float(
            client.metadata.get_cost(
                dataset=dataset,
                start=start_utc,
                end=end_utc,
                symbols=syms,
                schema=schema,
                stype_in=stype_in,
            )
        )
        size_bytes = int(
            client.metadata.get_billable_size(
                dataset=dataset,
                start=start_utc,
                end=end_utc,
                symbols=syms,
                schema=schema,
                stype_in=stype_in,
            )
        )
        record_count: Optional[int] = None
        if _has_metadata_capability(client, "get_record_count"):
            try:
                record_count = int(
                    client.metadata.get_record_count(
                        dataset=dataset,
                        start=start_utc,
                        end=end_utc,
                        symbols=syms,
                        schema=schema,
                        stype_in=stype_in,
                    )
                )
            except Exception:
                record_count = None
    except Exception as exc:
        return CostEstimate(
            status=ESTIMATE_API_ERROR,
            source="databento.metadata",
            message=f"estimate call raised: {type(exc).__name__}: {exc}",
            max_estimated_cost_usd=max_estimated_cost_usd,
            max_estimated_size_gb=max_estimated_size_gb,
        )
    size_gb = _to_int_gb(size_bytes)
    base = CostEstimate(
        status=ESTIMATE_AVAILABLE,
        estimated_cost_usd=cost_usd,
        estimated_size_gb=size_gb,
        estimated_record_count=record_count,
        source="databento.metadata.get_cost+get_billable_size",
        message=None,
        max_estimated_cost_usd=max_estimated_cost_usd,
        max_estimated_size_gb=max_estimated_size_gb,
    )
    if (
        max_estimated_cost_usd is not None
        and cost_usd is not None
        and (cost_usd > max_estimated_cost_usd)
    ):
        return CostEstimate(
            status=COST_LIMIT_EXCEEDED,
            estimated_cost_usd=cost_usd,
            estimated_size_gb=size_gb,
            estimated_record_count=record_count,
            source=base.source,
            message=f"estimated cost ${cost_usd:.4f} exceeds limit ${max_estimated_cost_usd:.4f}",
            max_estimated_cost_usd=max_estimated_cost_usd,
            max_estimated_size_gb=max_estimated_size_gb,
        )
    if (
        max_estimated_size_gb is not None
        and size_gb is not None
        and (size_gb > max_estimated_size_gb)
    ):
        return CostEstimate(
            status=SIZE_LIMIT_EXCEEDED,
            estimated_cost_usd=cost_usd,
            estimated_size_gb=size_gb,
            estimated_record_count=record_count,
            source=base.source,
            message=f"estimated size {size_gb:.4f} GB exceeds limit {max_estimated_size_gb:.4f} GB",
            max_estimated_cost_usd=max_estimated_cost_usd,
            max_estimated_size_gb=max_estimated_size_gb,
        )
    return base


PHYSICAL_LIMIT_SUPPORTED = "PHYSICAL_LIMIT_SUPPORTED"
PHYSICAL_LIMIT_UNSUPPORTED = "PHYSICAL_LIMIT_UNSUPPORTED"
PHYSICAL_LIMIT_NOT_REQUESTED_DRY_RUN = "PHYSICAL_LIMIT_NOT_REQUESTED_DRY_RUN"
PHYSICAL_LIMIT_API_ERROR = "PHYSICAL_LIMIT_API_ERROR"


@dataclass(frozen=True)
class PhysicalLimitCapability:
    status: str
    supported_limit_types: tuple[str, ...] = ()
    selected_limit_type: Optional[str] = None
    selected_limit_value: Optional[int] = None
    message: Optional[str] = None
    sdk_call_target: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "supported_limit_types": list(self.supported_limit_types),
            "selected_limit_type": self.selected_limit_type,
            "selected_limit_value": self.selected_limit_value,
            "message": self.message,
            "sdk_call_target": self.sdk_call_target,
        }


def _signature_has_param(target: callable, name: str) -> bool:
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return False
    return name in sig.parameters


def detect_databento_physical_limit_capability(
    *,
    dry_run: bool = False,
    client_factory: Optional[callable] = None,
    target_call: str = "timeseries.get_range",
    max_download_records: Optional[int] = None,
    max_download_size_gb: Optional[float] = None,
    max_download_cost_usd: Optional[float] = None,
) -> PhysicalLimitCapability:
    if dry_run:
        return PhysicalLimitCapability(
            status=PHYSICAL_LIMIT_NOT_REQUESTED_DRY_RUN,
            supported_limit_types=(),
            selected_limit_type=None,
            selected_limit_value=None,
            message="dry-run: physical-limit capability detection skipped",
            sdk_call_target=None,
        )
    try:
        if client_factory is not None:
            client = client_factory()
        else:
            import databento

            client = databento.Historical(key=os.environ.get("DATABENTO_API_KEY", "dummy"))
    except Exception as exc:
        return PhysicalLimitCapability(
            status=PHYSICAL_LIMIT_API_ERROR,
            message=f"could not construct client: {type(exc).__name__}: {exc}",
        )
    target_obj = client
    for part in target_call.split("."):
        target_obj = getattr(target_obj, part, None)
        if target_obj is None:
            break
    if not callable(target_obj):
        return PhysicalLimitCapability(
            status=PHYSICAL_LIMIT_UNSUPPORTED,
            message=f"{target_call} is not callable on the installed SDK",
        )
    supported: list[str] = []
    if _signature_has_param(target_obj, "limit"):
        supported.append("record_limit")
    if _signature_has_param(target_obj, "max_bytes"):
        supported.append("byte_limit")
    if _signature_has_param(target_obj, "max_cost_usd"):
        supported.append("cost_cap")
    if not supported:
        return PhysicalLimitCapability(
            status=PHYSICAL_LIMIT_UNSUPPORTED,
            supported_limit_types=(),
            message="installed databento SDK does not expose limit / max_bytes / max_cost_usd on "
            + target_call,
            sdk_call_target=target_call,
        )
    selected_type: Optional[str] = None
    selected_value: Optional[int] = None
    if "record_limit" in supported and max_download_records is not None:
        selected_type = "record_limit"
        selected_value = int(max_download_records)
    elif "byte_limit" in supported and max_download_size_gb is not None:
        selected_type = "byte_limit"
        selected_value = int(max_download_size_gb * 1024**3)
    elif "cost_cap" in supported and max_download_cost_usd is not None:
        selected_type = "cost_cap"
        selected_value = int(max_download_cost_usd * 100)
    return PhysicalLimitCapability(
        status=PHYSICAL_LIMIT_SUPPORTED,
        supported_limit_types=tuple(supported),
        selected_limit_type=selected_type,
        selected_limit_value=selected_value,
        message=None,
        sdk_call_target=target_call,
    )


ESTIMATE_CAPABILITY_AVAILABLE = "ESTIMATE_CAPABILITY_AVAILABLE"


def detect_databento_cost_estimate_capability(
    *, client_factory: Optional[callable] = None, api_key_present: Optional[bool] = None
) -> CostEstimate:
    if api_key_present is None:
        api_key_present = bool(os.environ.get("DATABENTO_API_KEY"))
    if not api_key_present and client_factory is None:
        return CostEstimate(
            status=COST_ESTIMATE_UNAVAILABLE,
            source="capability_probe",
            message="DATABENTO_API_KEY not set; cannot construct SDK client",
        )
    try:
        if client_factory is not None:
            client = client_factory()
        else:
            import databento

            client = databento.Historical(key=os.environ["DATABENTO_API_KEY"])
    except Exception as exc:
        return CostEstimate(
            status=ESTIMATE_API_ERROR,
            source="capability_probe",
            message=f"could not construct client: {type(exc).__name__}: {exc}",
        )
    if _has_metadata_capability(client, "get_cost") and _has_metadata_capability(
        client, "get_billable_size"
    ):
        return CostEstimate(
            status=ESTIMATE_CAPABILITY_AVAILABLE,
            source="databento.metadata.get_cost+get_billable_size",
            message="SDK exposes the official endpoints; values not requested",
        )
    return CostEstimate(
        status=COST_ESTIMATE_UNAVAILABLE,
        source="databento.metadata",
        message="installed databento SDK is missing metadata.get_cost and/or metadata.get_billable_size; cost estimate is unavailable",
    )


def build_databento_request_with_physical_limit(
    *, base_kwargs: dict, capability: PhysicalLimitCapability
) -> dict:
    out = dict(base_kwargs)
    if capability.status != PHYSICAL_LIMIT_SUPPORTED:
        return out
    if capability.selected_limit_type is None or capability.selected_limit_value is None:
        return out
    if capability.selected_limit_type == "record_limit":
        out["limit"] = capability.selected_limit_value
    elif capability.selected_limit_type == "byte_limit":
        out["max_bytes"] = capability.selected_limit_value
    elif capability.selected_limit_type == "cost_cap":
        out["max_cost_usd"] = capability.selected_limit_value / 100.0
    return out


DOWNLOAD_BLOCKED_DRY_RUN = "DOWNLOAD_BLOCKED_DRY_RUN"
DOWNLOAD_BLOCKED_MISSING_ACK = "DOWNLOAD_BLOCKED_MISSING_ACK"
DOWNLOAD_BLOCKED_MISSING_ALLOW_DATABENTO = "DOWNLOAD_BLOCKED_MISSING_ALLOW_DATABENTO"
DOWNLOAD_BLOCKED_MISSING_API_KEY = "DOWNLOAD_BLOCKED_MISSING_API_KEY"
DOWNLOAD_BLOCKED_BY_COST_GUARD = "DOWNLOAD_BLOCKED_BY_COST_GUARD"
DOWNLOAD_BLOCKED_BY_PHYSICAL_LIMIT_GUARD = "DOWNLOAD_BLOCKED_BY_PHYSICAL_LIMIT_GUARD"
DOWNLOAD_BLOCKED_BY_COST_ESTIMATE_UNAVAILABLE = "DOWNLOAD_BLOCKED_BY_COST_ESTIMATE_UNAVAILABLE"
DOWNLOAD_APPROVED = "DOWNLOAD_APPROVED"
DOWNLOAD_APPROVED_WITHOUT_PHYSICAL_LIMIT = "DOWNLOAD_APPROVED_WITHOUT_PHYSICAL_LIMIT"


@dataclass(frozen=True)
class GateDecision:
    decision: str
    cost_estimate: CostEstimate
    physical_limit: PhysicalLimitCapability
    warnings: tuple[str, ...] = ()
    critical_warnings: tuple[str, ...] = ()

    @property
    def approved(self) -> bool:
        return self.decision in (DOWNLOAD_APPROVED, DOWNLOAD_APPROVED_WITHOUT_PHYSICAL_LIMIT)

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "cost_estimate": self.cost_estimate.to_dict(),
            "physical_limit": self.physical_limit.to_dict(),
            "warnings": list(self.warnings),
            "critical_warnings": list(self.critical_warnings),
        }


@dataclass
class GateInputs:
    download_cme: bool = False
    allow_databento_download: bool = False
    acknowledge_cost_risk: bool = False
    allow_unestimated_download: bool = False
    cost_estimate_required: bool = True
    require_physical_download_limit: bool = True
    allow_download_without_physical_limit: bool = False
    dry_run: bool = False
    api_key_present: bool = False
    max_estimated_cost_usd: Optional[float] = 10.0
    max_estimated_size_gb: Optional[float] = 5.0
    max_download_records: Optional[int] = None
    max_download_size_gb: Optional[float] = 5.0
    max_download_cost_usd: Optional[float] = 10.0


def decide_download_gate(
    inputs: GateInputs, *, cost_estimate: CostEstimate, physical_limit: PhysicalLimitCapability
) -> GateDecision:
    warnings: list[str] = []
    critical_warnings: list[str] = []
    if inputs.dry_run:
        return GateDecision(
            decision=DOWNLOAD_BLOCKED_DRY_RUN,
            cost_estimate=cost_estimate,
            physical_limit=physical_limit,
            warnings=tuple(warnings),
            critical_warnings=tuple(critical_warnings),
        )
    if not (inputs.download_cme and inputs.allow_databento_download):
        return GateDecision(
            decision=DOWNLOAD_BLOCKED_MISSING_ALLOW_DATABENTO,
            cost_estimate=cost_estimate,
            physical_limit=physical_limit,
        )
    if not inputs.acknowledge_cost_risk:
        return GateDecision(
            decision=DOWNLOAD_BLOCKED_MISSING_ACK,
            cost_estimate=cost_estimate,
            physical_limit=physical_limit,
        )
    if not inputs.api_key_present:
        return GateDecision(
            decision=DOWNLOAD_BLOCKED_MISSING_API_KEY,
            cost_estimate=cost_estimate,
            physical_limit=physical_limit,
        )
    if cost_estimate.status in (COST_LIMIT_EXCEEDED, SIZE_LIMIT_EXCEEDED):
        return GateDecision(
            decision=DOWNLOAD_BLOCKED_BY_COST_GUARD,
            cost_estimate=cost_estimate,
            physical_limit=physical_limit,
        )
    if cost_estimate.status == COST_ESTIMATE_UNAVAILABLE:
        if inputs.cost_estimate_required and (not inputs.allow_unestimated_download):
            return GateDecision(
                decision=DOWNLOAD_BLOCKED_BY_COST_ESTIMATE_UNAVAILABLE,
                cost_estimate=cost_estimate,
                physical_limit=physical_limit,
            )
        warnings.append(
            "proceeding without cost estimate: --allow-unestimated-download was set; operator is responsible for cost"
        )
    elif cost_estimate.status == ESTIMATE_API_ERROR:
        return GateDecision(
            decision=DOWNLOAD_BLOCKED_BY_COST_GUARD,
            cost_estimate=cost_estimate,
            physical_limit=physical_limit,
        )
    if inputs.require_physical_download_limit:
        if physical_limit.status != PHYSICAL_LIMIT_SUPPORTED:
            if not inputs.allow_download_without_physical_limit:
                return GateDecision(
                    decision=DOWNLOAD_BLOCKED_BY_PHYSICAL_LIMIT_GUARD,
                    cost_estimate=cost_estimate,
                    physical_limit=physical_limit,
                )
            critical_warnings.append(
                "DOWNLOAD_WITHOUT_PHYSICAL_LIMIT: --allow-download-without-physical-limit was set despite an unsupported physical limit; operator is responsible for runaway download protection"
            )
            return GateDecision(
                decision=DOWNLOAD_APPROVED_WITHOUT_PHYSICAL_LIMIT,
                cost_estimate=cost_estimate,
                physical_limit=physical_limit,
                warnings=tuple(warnings),
                critical_warnings=tuple(critical_warnings),
            )
    return GateDecision(
        decision=DOWNLOAD_APPROVED,
        cost_estimate=cost_estimate,
        physical_limit=physical_limit,
        warnings=tuple(warnings),
        critical_warnings=tuple(critical_warnings),
    )
