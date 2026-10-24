"""Loads and validates the scheduling data files.

All five JSON files (calendar, orders, technicians, parts, tasks) are
loaded once at agent startup and held in memory. Nothing is re-read
during operation -- the files are static inputs.

Failures are loud: a missing file, invalid schema, or a task referring
to a part/skill that doesn't exist raises at startup, so the agent
refuses to run with inconsistent data rather than producing silently
wrong schedules.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from models import (
    FactoryCalendar,
    MaintenanceTasks,
    ProductionOrders,
    SpareParts,
    Technicians,
)

logger = logging.getLogger("scheduling_agent.data")


class SchedulingData:
    def __init__(
        self,
        calendar: FactoryCalendar,
        orders: ProductionOrders,
        technicians: Technicians,
        parts: SpareParts,
        tasks: MaintenanceTasks,
    ):
        self.calendar = calendar
        self.orders = orders
        self.technicians = technicians
        self.parts = parts
        self.tasks = tasks

        self._parts_by_id = {p.part_id: p for p in parts.parts}
        self._techs_by_id = {t.technician_id: t for t in technicians.technicians}
        self._tasks_by_id = {t.task_id: t for t in tasks.tasks}
        self._tasks_by_failure = {t.failure_mode: t for t in tasks.tasks}

        self._validate_cross_references()

    def _validate_cross_references(self) -> None:
        # Every part required by a task must exist.
        for task in self.tasks.tasks:
            for req in task.required_parts:
                if req.part_id not in self._parts_by_id:
                    raise ValueError(
                        f"Task {task.task_id} requires unknown part {req.part_id}"
                    )
        # Every skill required by a task should be covered by at least
        # one technician. If not, no schedule can ever satisfy the task,
        # which is better caught at startup than at proposal time.
        all_skills = {s for t in self.technicians.technicians for s in t.skills}
        for task in self.tasks.tasks:
            for skill in task.required_skills:
                if skill not in all_skills:
                    logger.warning(
                        "Task %s requires skill %r that no technician has; "
                        "it will never be schedulable.",
                        task.task_id, skill,
                    )

    def task_for_failure(self, failure_mode: str | None):
        """Return the MaintenanceTaskDef matching a failure mode, or the
        'unknown' fallback task if the mode is absent/unrecognized."""
        if failure_mode is not None:
            task = self._tasks_by_failure.get(failure_mode)
            if task is not None:
                return task
        return self._tasks_by_failure.get("unknown")

    def part(self, part_id: str):
        return self._parts_by_id[part_id]

    def technician(self, technician_id: str):
        return self._techs_by_id[technician_id]

    def open_orders(self) -> list:
        return [o for o in self.orders.orders if o.status == "open"]


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Scheduling data file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_scheduling_data(data_dir: str | Path) -> SchedulingData:
    """Load and validate all five JSON files from `data_dir`."""
    data_dir = Path(data_dir)
    logger.info("Loading scheduling data from %s", data_dir.resolve())

    def load(name: str, model):
        raw = _read_json(data_dir / name)
        try:
            return model(**raw)
        except ValidationError as exc:
            raise ValueError(f"Invalid {name}: {exc}") from exc

    calendar = load("factory_calendar.json", FactoryCalendar)
    orders = load("production_orders.json", ProductionOrders)
    technicians = load("technicians.json", Technicians)
    parts = load("spare_parts.json", SpareParts)
    tasks = load("maintenance_tasks.json", MaintenanceTasks)

    data = SchedulingData(calendar, orders, technicians, parts, tasks)
    logger.info(
        "Loaded %d orders, %d technicians, %d parts, %d tasks, %d windows",
        len(data.orders.orders), len(data.technicians.technicians),
        len(data.parts.parts), len(data.tasks.tasks),
        len(data.calendar.maintenance_windows),
    )
    return data