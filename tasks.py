"""Task registry for hackathon validation."""

from __future__ import annotations

from grader import grade


TASKS = [
    {
        "task_id": "task_1",
        "grader": grade,
    },
    {
        "task_id": "task_2",
        "grader": grade,
    },
    {
        "task_id": "task_3",
        "grader": grade,
    },
]


def get_tasks():
    return TASKS


if __name__ == "__main__":
    print("TASKS LOADED:", len(TASKS))