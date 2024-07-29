import sys
from subprocess import run


def bump():
    if len(sys.argv) != 2 or sys.argv[1] not in ["major", "minor", "patch"]:
        print("Usage: poetry run bump [major|minor|patch]")
        sys.exit(1)

    result = run(
        ["bump-my-version", "bump", sys.argv[1]], capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Error:", result.stderr)
        sys.exit(1)

    print(result.stdout)

    # Update poetry.lock with new version
    run(["poetry", "lock", "--no-update"], check=True)


if __name__ == "__main__":
    bump()
