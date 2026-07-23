"""Dataset-adapter registry with an explicit, fail-closed contract.

Paper campaigns must resolve a named adapter before they allocate a GPU.  A
missing adapter is an error; in particular, the registry never substitutes
TOFU for an unknown dataset.  The adapter boundary is deliberately the
existing :class:`~rsus.data.base.Request` object, so probes and trajectories do
not need dataset-specific branches once a request has been constructed.

Only adapters backed by real loaders are registered here.  Planned paper
datasets stay absent until their loaders and request semantics are implemented
and tested.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from rsus.data.base import Request


RequestFactory = Callable[..., Request]
RosterIdValidator = Callable[[str], bool]


class AdapterRegistryError(ValueError):
    """Base class for adapter registration and resolution failures."""


class AdapterNotFoundError(AdapterRegistryError):
    """Raised when a dataset name has no registered adapter."""


def _normalise_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterRegistryError("adapter names must be non-empty strings")
    return " ".join(value.strip().casefold().split())


@dataclass(frozen=True)
class AdapterCapabilities:
    """Capabilities that a campaign may require from an adapter.

    ``stages`` uses campaign-stage identifiers rather than inferring support
    from function names.  The remaining fields document the assumptions used
    by the paper's split and tail analyses.
    """

    stages: frozenset[str]
    roster_unit: str
    grouped_candidates: bool
    native_audit: bool
    independent_target_roster: bool

    def __post_init__(self) -> None:
        if not self.stages or any(not str(stage).strip() for stage in self.stages):
            raise AdapterRegistryError("adapter capabilities need non-empty stages")
        if not self.roster_unit.strip():
            raise AdapterRegistryError("adapter roster_unit must be non-empty")

    def supports(self, stage: str) -> bool:
        return stage in self.stages

    def as_dict(self) -> dict[str, Any]:
        return {
            "stages": sorted(self.stages),
            "roster_unit": self.roster_unit,
            "grouped_candidates": self.grouped_candidates,
            "native_audit": self.native_audit,
            "independent_target_roster": self.independent_target_roster,
        }


@dataclass(frozen=True)
class DatasetAdapter:
    """One real dataset adapter whose factory returns an ``rsus`` Request."""

    key: str
    factory: RequestFactory
    capabilities: AdapterCapabilities
    aliases: tuple[str, ...] = ()
    description: str = ""
    roster_id_validator: RosterIdValidator | None = None

    def __post_init__(self) -> None:
        _normalise_name(self.key)
        if not callable(self.factory):
            raise AdapterRegistryError(f"adapter {self.key!r} factory is not callable")
        for alias in self.aliases:
            _normalise_name(alias)

    def build_request(self, **kwargs: Any) -> Request:
        request = self.factory(**kwargs)
        if not isinstance(request, Request):
            raise TypeError(
                f"adapter {self.key!r} returned {type(request).__name__}, expected Request"
            )
        if not isinstance(request.request_id, str) or not request.request_id.strip():
            raise AdapterRegistryError(f"adapter {self.key!r} returned an empty request_id")
        if not request.forget or not request.universe.examples:
            raise AdapterRegistryError(
                f"adapter {self.key!r} returned an empty forget set or candidate universe"
            )
        forget_ids = [example.example_id for example in request.forget]
        candidate_ids = [example.example_id for example in request.universe.examples]
        if len(set(forget_ids)) != len(forget_ids):
            raise AdapterRegistryError(
                f"adapter {self.key!r} returned duplicate forget example ids"
            )
        if len(set(candidate_ids)) != len(candidate_ids):
            raise AdapterRegistryError(
                f"adapter {self.key!r} returned duplicate candidate example ids"
            )
        overlap = sorted(set(forget_ids) & set(candidate_ids))
        if overlap:
            raise AdapterRegistryError(
                f"adapter {self.key!r} mixed forget examples into the retain universe: "
                f"{overlap[:5]}"
            )
        if not request.forget_sha or not request.universe.sha:
            raise AdapterRegistryError(
                f"adapter {self.key!r} returned an unfrozen request manifest"
            )
        if self.capabilities.grouped_candidates and any(
            not isinstance(example.group, str) or not example.group.strip()
            for example in request.universe.examples
        ):
            raise AdapterRegistryError(
                f"adapter {self.key!r} requires a non-empty group for every candidate"
            )
        return request

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "aliases": list(self.aliases),
            "description": self.description,
            "factory": f"{self.factory.__module__}:{self.factory.__name__}",
            "roster_id_validation": self.roster_id_validator is not None,
            "capabilities": self.capabilities.as_dict(),
        }

    def accepts_roster_id(self, request_id: str) -> bool:
        return bool(
            self.roster_id_validator is not None
            and self.roster_id_validator(request_id)
        )


class DatasetAdapterRegistry:
    """Exact-name registry; resolution never has a default adapter."""

    def __init__(self) -> None:
        self._by_key: dict[str, DatasetAdapter] = {}
        self._names: dict[str, str] = {}

    def register(self, adapter: DatasetAdapter) -> DatasetAdapter:
        canonical = _normalise_name(adapter.key)
        if canonical in self._by_key:
            raise AdapterRegistryError(f"adapter {adapter.key!r} is already registered")

        names = {_normalise_name(adapter.key)}
        names.update(_normalise_name(alias) for alias in adapter.aliases)
        collisions = sorted(name for name in names if name in self._names)
        if collisions:
            owners = {name: self._names[name] for name in collisions}
            raise AdapterRegistryError(f"adapter aliases collide with existing names: {owners}")

        self._by_key[canonical] = adapter
        for name in names:
            self._names[name] = canonical
        return adapter

    def resolve(self, name: str) -> DatasetAdapter:
        normalised = _normalise_name(name)
        canonical = self._names.get(normalised)
        if canonical is None:
            registered = ", ".join(self.keys()) or "<none>"
            raise AdapterNotFoundError(
                f"no dataset adapter registered for {name!r}; registered: {registered}"
            )
        return self._by_key[canonical]

    def get(self, name: str) -> DatasetAdapter | None:
        try:
            return self.resolve(name)
        except AdapterNotFoundError:
            return None

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(adapter.key for adapter in self._by_key.values()))

    def adapters(self) -> tuple[DatasetAdapter, ...]:
        return tuple(sorted(self._by_key.values(), key=lambda adapter: adapter.key))


def _tofu_factory(**kwargs: Any) -> Request:
    """Lazy wrapper around the existing TOFU loader and request builder."""

    from rsus.data.tofu import load_tofu_examples, tofu_request

    values = dict(kwargs)
    tokenizer = values.pop("tokenizer", None)
    examples = values.pop("examples", None)
    max_length = int(values.pop("max_length", 256))
    if examples is None:
        if tokenizer is None:
            raise TypeError("TOFU adapter needs tokenizer when examples are not supplied")
        examples = load_tofu_examples(tokenizer, max_length=max_length)
    return tofu_request(examples=examples, **values)


def _substrate_factory(**kwargs: Any) -> Request:
    """Expose the existing controlled substrate through the Request contract."""

    from rsus.data.substrate import make_substrate

    request, _truth = make_substrate(**kwargs)
    return request


def _tofu_roster_id(value: str) -> bool:
    prefix = "tofu-a"
    if not value.startswith(prefix) or not value[len(prefix) :].isdigit():
        return False
    return 180 <= int(value[len(prefix) :]) < 200


def _substrate_roster_id(value: str) -> bool:
    prefix = "substrate-"
    return value.startswith(prefix) and value[len(prefix) :].isdigit()


ADAPTERS = DatasetAdapterRegistry()

ADAPTERS.register(
    DatasetAdapter(
        key="tofu",
        aliases=("TOFU", "locuslab/TOFU"),
        factory=_tofu_factory,
        capabilities=AdapterCapabilities(
            stages=frozenset(
                {"calibration", "prediction", "protection", "target_evaluation"}
            ),
            roster_unit="forget-author request_id",
            grouped_candidates=True,
            native_audit=False,
            independent_target_roster=True,
        ),
        description="TOFU forget10, one deletion request per held-out author",
        roster_id_validator=_tofu_roster_id,
    )
)

ADAPTERS.register(
    DatasetAdapter(
        key="substrate",
        aliases=("controlled-substrate",),
        factory=_substrate_factory,
        capabilities=AdapterCapabilities(
            stages=frozenset({"calibration", "prediction", "protection", "mechanism"}),
            roster_unit="synthetic request seed",
            grouped_candidates=True,
            native_audit=True,
            independent_target_roster=False,
        ),
        description="controlled synthetic request with ground-truth adjacency",
        roster_id_validator=_substrate_roster_id,
    )
)


def register_adapter(adapter: DatasetAdapter) -> DatasetAdapter:
    """Register a real adapter in the process-global registry."""

    return ADAPTERS.register(adapter)


def get_adapter(name: str) -> DatasetAdapter:
    """Resolve ``name`` exactly or raise; there is intentionally no fallback."""

    return ADAPTERS.resolve(name)


def registered_adapters() -> tuple[DatasetAdapter, ...]:
    return ADAPTERS.adapters()


def require_capabilities(name: str, stages: Iterable[str]) -> DatasetAdapter:
    adapter = get_adapter(name)
    missing = sorted(set(stages) - adapter.capabilities.stages)
    if missing:
        raise AdapterRegistryError(
            f"adapter {adapter.key!r} does not support stages: {', '.join(missing)}"
        )
    return adapter


def registry_manifest() -> Mapping[str, dict[str, Any]]:
    return {adapter.key: adapter.as_dict() for adapter in registered_adapters()}
