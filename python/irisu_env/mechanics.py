"""Strict loader for versioned mechanics evidence profiles."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class Provenance(StrEnum):
    OFFICIAL = "official"
    SHIPPED_CONFIG = "shipped-config"
    BINARY_DERIVED = "binary-derived"
    OBSERVED = "observed"
    COMMUNITY = "community"
    INFERRED = "inferred"
    PLACEHOLDER = "placeholder"


class Uncertainty(StrEnum):
    RUNTIME_USE = "runtime-use"
    SEMANTIC = "semantic"
    UNITS = "units"
    MAPPING = "mapping"
    GEOMETRY = "geometry"
    TIMING = "timing"
    NUMERICAL = "numerical"
    DISTRIBUTION = "distribution"


class ValueType(StrEnum):
    INTEGER = "integer"
    FLOAT = "float"
    INT_LIST = "int_list"
    FLOAT_LIST = "float_list"


class Unit(StrEnum):
    DISPLAY_UNIT = "display_unit"
    LEGACY_MATERIAL_TRIPLET = "legacy_material_triplet"
    LEGACY_MATERIAL_PAIR = "legacy_material_pair"
    RELATIVE_WEIGHT = "relative_weight"
    UNKNOWN_COUNTER = "unknown_counter"
    RAW_VELOCITY_PARAMETER = "raw_velocity_parameter"
    GAUGE_UNIT = "gauge_unit"
    DISPLAY_UNITS_PER_WORLD_UNIT = "display_units_per_world_unit"
    SECONDS_PER_NORMAL_STEP = "seconds_per_normal_step"
    SOLVER_ITERATIONS = "solver_iterations"
    DISPLAY_UNITS_PER_SECOND_SQUARED = "display_units_per_second_squared"
    INVERSE_SECOND = "inverse_second"
    DISPLAY_UNITS_PER_SECOND = "display_units_per_second"
    GAUGE_UNITS_PER_TICK = "gauge_units_per_tick"
    TICK = "tick"
    SPAWN = "spawn"
    COLOR_COUNT = "color_count"
    SCORE_POINT = "score_point"
    DIMENSIONLESS = "dimensionless"


ConfigValue = int | float | tuple[int, ...] | tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Mechanic:
    key: str
    value: ConfigValue
    value_type: ValueType
    unit: Unit
    provenance: Provenance
    uncertainty: tuple[Uncertainty, ...]
    uncertainty_note: str
    source: str
    validating_experiments: tuple[str, ...]
    planned_experiments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UnknownSetting:
    raw_key: str
    value: ConfigValue
    value_type: ValueType
    provenance: Provenance
    clue: str
    source: str
    validating_experiments: tuple[str, ...]
    planned_experiments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ImplementationParameter:
    """A runtime parameter with explicit evidence status."""

    key: str
    value: ConfigValue
    value_type: ValueType
    unit: Unit
    provenance: Provenance
    uncertainty_note: str
    source: str
    validating_experiments: tuple[str, ...]
    planned_experiments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MechanicsProfile:
    schema_version: int
    profile_id: str
    game_version: str
    mode: str
    status: str
    executable_sha256: str
    box2d_sha256: str
    shipped_config_sha256: str
    mechanics: tuple[Mechanic, ...]
    unknown_settings: tuple[UnknownSetting, ...]
    implementation_parameters: tuple[ImplementationParameter, ...]

    def mechanic(self, key: str) -> Mechanic:
        matches = [entry for entry in self.mechanics if entry.key == key]
        if len(matches) != 1:
            raise KeyError(key)
        return matches[0]

    def implementation_mapping(self) -> dict[str, ConfigValue]:
        """Return native-config keys and provisional values as a fresh mapping."""

        return {entry.key: entry.value for entry in self.implementation_parameters}


def _required_string(table: dict[str, Any], key: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _strings(table: dict[str, Any], key: str) -> tuple[str, ...]:
    value = table.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{key} must be a list of non-empty strings")
    return tuple(value)


def _value(table: dict[str, Any], value_type: ValueType) -> ConfigValue:
    value = table.get("value")
    if value_type is ValueType.INTEGER and type(value) is int:
        return value
    if value_type is ValueType.FLOAT and type(value) is float:
        return value
    if value_type is ValueType.INT_LIST and isinstance(value, list) and all(
        type(item) is int for item in value
    ):
        return tuple(value)
    if value_type is ValueType.FLOAT_LIST and isinstance(value, list) and all(
        type(item) is float for item in value
    ):
        return tuple(value)
    raise ValueError(f"value does not match declared value_type {value_type.value}")


def _mechanic(table: dict[str, Any]) -> Mechanic:
    value_type = ValueType(_required_string(table, "value_type"))
    uncertainties = tuple(Uncertainty(item) for item in _strings(table, "uncertainty"))
    if not uncertainties:
        raise ValueError("each mechanic must state at least one uncertainty")
    return Mechanic(
        key=_required_string(table, "key"),
        value=_value(table, value_type),
        value_type=value_type,
        unit=Unit(_required_string(table, "unit")),
        provenance=Provenance(_required_string(table, "provenance")),
        uncertainty=uncertainties,
        uncertainty_note=_required_string(table, "uncertainty_note"),
        source=_required_string(table, "source"),
        validating_experiments=_strings(table, "validating_experiments"),
        planned_experiments=_strings(table, "planned_experiments"),
    )


def _unknown(table: dict[str, Any]) -> UnknownSetting:
    value_type = ValueType(_required_string(table, "value_type"))
    return UnknownSetting(
        raw_key=_required_string(table, "raw_key"),
        value=_value(table, value_type),
        value_type=value_type,
        provenance=Provenance(_required_string(table, "provenance")),
        clue=_required_string(table, "clue"),
        source=_required_string(table, "source"),
        validating_experiments=_strings(table, "validating_experiments"),
        planned_experiments=_strings(table, "planned_experiments"),
    )


def _implementation_parameter(table: dict[str, Any]) -> ImplementationParameter:
    value_type = ValueType(_required_string(table, "value_type"))
    provenance = Provenance(_required_string(table, "provenance"))
    validating_experiments = _strings(table, "validating_experiments")
    if provenance in (Provenance.INFERRED, Provenance.PLACEHOLDER) and validating_experiments:
        raise ValueError("provisional implementation_parameter cannot claim validation")
    return ImplementationParameter(
        key=_required_string(table, "key"),
        value=_value(table, value_type),
        value_type=value_type,
        unit=Unit(_required_string(table, "unit")),
        provenance=provenance,
        uncertainty_note=_required_string(table, "uncertainty_note"),
        source=_required_string(table, "source"),
        validating_experiments=validating_experiments,
        planned_experiments=_strings(table, "planned_experiments"),
    )


def load_profile(path: str | Path) -> MechanicsProfile:
    with Path(path).open("rb") as stream:
        document = tomllib.load(stream)
    if document.get("schema_version") != 1:
        raise ValueError("unsupported mechanics schema_version")

    profile = document.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("profile table is required")
    mechanic_tables = document.get("mechanic")
    unknown_tables = document.get("unknown_setting")
    implementation_tables = document.get("implementation_parameter")
    if not all(isinstance(tables, list) for tables in (
        mechanic_tables,
        unknown_tables,
        implementation_tables,
    )):
        raise ValueError(
            "mechanic, unknown_setting, and implementation_parameter arrays are required"
        )

    mechanics = tuple(_mechanic(table) for table in mechanic_tables)
    unknown_settings = tuple(_unknown(table) for table in unknown_tables)
    implementation_parameters = tuple(
        _implementation_parameter(table) for table in implementation_tables
    )
    mechanic_keys = [entry.key for entry in mechanics]
    raw_keys = [entry.raw_key for entry in unknown_settings]
    implementation_keys = [entry.key for entry in implementation_parameters]
    if len(mechanic_keys) != len(set(mechanic_keys)):
        raise ValueError("duplicate mechanic key")
    if len(raw_keys) != len(set(raw_keys)):
        raise ValueError("duplicate unknown_setting raw_key")
    if len(implementation_keys) != len(set(implementation_keys)):
        raise ValueError("duplicate implementation_parameter key")

    return MechanicsProfile(
        schema_version=1,
        profile_id=_required_string(profile, "id"),
        game_version=_required_string(profile, "game_version"),
        mode=_required_string(profile, "mode"),
        status=_required_string(profile, "status"),
        executable_sha256=_required_string(profile, "executable_sha256"),
        box2d_sha256=_required_string(profile, "box2d_sha256"),
        shipped_config_sha256=_required_string(profile, "shipped_config_sha256"),
        mechanics=mechanics,
        unknown_settings=unknown_settings,
        implementation_parameters=implementation_parameters,
    )
