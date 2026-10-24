# Smart Factory MAS — Scheduling Data (Synthetic)

These files are starter data for the scheduling agent. All values are illustrative and are not operational recommendations for a real factory.

## Files
- `production_orders.json`: order quantities and deadlines in simulation minutes from simulation start.
- `factory_calendar.json`: daily operating and maintenance windows. Minutes-of-day range from 0 to 1439.
- `technicians.json`: technician skills, shifts, and concurrency.
- `spare_parts.json`: inventory, reservations, reorder levels, and lead times.
- `maintenance_tasks.json`: task-to-failure mapping, required skills/parts, and estimated durations.

## Time conventions
The simulator advances one simulation minute per step. Its `--step-delay 0.1` is only wall-clock pacing. The scheduler should use simulation minutes for order deadlines, windows, and repair durations, not wall-clock cooldown seconds.

`production_orders.json` uses absolute simulation minute offsets from the simulation start. Calendar windows and technician shifts repeat daily using minute-of-day.

## Important integration notes
- The four machine IDs are M1, M2, M3, and M4.
- Failure modes match the simulator: `bearing_wear`, `overheating`, `misalignment`, and `random_shock`.
- The spare-parts compatibility and task estimates are synthetic examples.
- `sensor_fault` is included only as a future maintenance case; the current simulator may not inject this failure mode.
- These files describe constraints and candidate actions; they do not themselves approve or execute maintenance.
- The next code step should load and validate these files, calculate available stock (`quantity_on_hand - reserved_quantity`), match task skills/parts, and find feasible start times.
