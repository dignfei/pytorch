import json
import re
from pathlib import Path


OWNERS_PREFIXES = ("# Owner(s): ", "// Owner(s): ")
IGNORED_OWNERS = ["unknown"]
NATIVE_TEST_DIRS = (
    "test",
    "aten/src/ATen/test",
    "aten/src/ATen/native/metal/mpscnn/tests",
    "c10/test",
    "c10/cuda/test",
    "c10/xpu/test",
)
NATIVE_TEST_SUFFIXES = {".cpp", ".cc", ".cxx", ".cu", ".mm"}


def load_codeowners_rules(
    codeowners_path: Path,
) -> list[tuple[re.Pattern[str], str | None]]:
    if not codeowners_path.is_file():
        return []

    rules = []
    review_area = None
    for line in codeowners_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("# [") and line.endswith("]"):
            review_area = line[3:-1].strip() or None
        elif line.startswith(("# /", "/")):
            # Commented paths also declare review areas, without assigning reviewers.
            pattern = line.removeprefix("# ").split()[0].strip("/")
            expression = re.escape(pattern)
            expression = expression.replace(r"\*\*/", "(?:.*/)?")
            expression = expression.replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
            rules.append((re.compile(expression + "(?:/.*)?"), review_area))
    return rules


def collect_owners(repo_root: Path) -> dict[str, list[str]]:
    codeowners_rules = load_codeowners_rules(repo_root / "CODEOWNERS")
    test_dir = repo_root / "test"
    # Match the Python TESTOWNERS patterns in .lintrunner.toml.
    test_paths = set(test_dir.rglob("test_*.py")) | set(test_dir.rglob("*_test.py"))
    # Native tests also use names like api/tensor.cpp and ATen/test/basic.cpp.
    for native_test_dir in NATIVE_TEST_DIRS:
        test_paths.update(
            source_path
            for source_path in (repo_root / native_test_dir).rglob("*")
            if source_path.suffix in NATIVE_TEST_SUFFIXES
            and not source_path.stem.endswith("_benchmark")
        )
    test_paths.update(repo_root.glob("aten/src/ATen/core/**/*_test.cpp"))
    test_paths.update(repo_root.glob("caffe2/**/*_test.cc"))
    owners_by_file = {}
    for test_path in sorted(test_paths):
        relative_path = test_path.relative_to(repo_root)
        if (
            relative_path.as_posix() == "test/run_test.py"
            or "fb" in relative_path.parts
            or "third_party" in relative_path.parts
        ):
            continue
        owner_labels = []
        with test_path.open(encoding="utf-8") as test_file:
            for line in test_file:
                if line.startswith(OWNERS_PREFIXES):
                    try:
                        owner_labels = json.loads(line.partition(": ")[2])
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid owners header in {relative_path}"
                        ) from error
                    if not isinstance(owner_labels, list) or any(
                        not isinstance(label, str) for label in owner_labels
                    ):
                        raise ValueError(
                            f"Expected a list of owner labels in {relative_path}"
                        )
                    break
        file_owners = {
            label.removeprefix("module: ")
            for label in owner_labels
            if label.startswith("module: ")
        }.difference(IGNORED_OWNERS)
        if not file_owners:
            for pattern, review_area in reversed(codeowners_rules):
                if pattern.fullmatch(relative_path.as_posix()):
                    if review_area is not None:
                        file_owners = {review_area}
                    break
        owners_by_file[relative_path.as_posix()] = sorted(
            file_owners.difference(IGNORED_OWNERS)
        )
    if not owners_by_file:
        raise ValueError(f"No test files found in {repo_root}")
    return owners_by_file
