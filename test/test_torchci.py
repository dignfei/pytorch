# Owner(s): ["module: ci"]

import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from torch.testing._internal import owners as ownership
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.testing._internal.torchci import populate_clickhouse_owners


class TestClickHouseOwners(TestCase):
    def setUp(self):
        super().setUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.repo_root = Path(directory.name)

    def write_file(self, name, contents):
        path = self.repo_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    def test_collect_owners(self):
        repeated_owner = "module: dynamo"
        labels = ["oncall: pt2", "module: optimizer", repeated_owner, repeated_owner]
        labels.append("module: unknown")
        late_header = "\n" * 6 + '# Owner(s): ["NNC", "oncall: jit"]\n'
        files = {
            "test/test_top.py": f"# Owner(s): {json.dumps(labels)}\n",
            "test/nested/suffix_test.py": late_header,
            "test/test_overlap_test.py": '# Owner(s): ["module: ci"]\n# Owner(s): ["module: ignored"]\n',
            "test/test_missing.py": "import unittest\n",
            "test/test_empty.py": "# Owner(s): []\n",
            "test/test_unknown.py": '# Owner(s): ["module: unknown"]\n',
            "test/run_test.py": '# Owner(s): ["module: ignored"]\n',
            "test/fb/test_internal.py": '# Owner(s): ["module: ignored"]\n',
            "test/nested/fb/internal_test.py": '# Owner(s): ["module: ignored"]\n',
            "test/helper.py": '# Owner(s): ["module: ignored"]\n',
            "tools/test_external.py": '# Owner(s): ["module: ignored"]\n',
        }
        for name, contents in files.items():
            self.write_file(name, contents)

        self.assertEqual(
            ownership.collect_owners(self.repo_root),
            {
                "test/nested/suffix_test.py": [],
                "test/test_empty.py": [],
                "test/test_missing.py": [],
                "test/test_overlap_test.py": ["ci"],
                "test/test_top.py": ["dynamo", "optimizer"],
                "test/test_unknown.py": [],
            },
        )

    def test_extend_ignored_owners(self):
        self.write_file(
            "test/test_example.py",
            '# Owner(s): ["module: unknown", "module: ci", "module: dynamo"]\n',
        )
        ignored_owners = [*ownership.IGNORED_OWNERS, "ci"]
        with patch.object(ownership, "IGNORED_OWNERS", ignored_owners):
            self.assertEqual(
                ownership.collect_owners(self.repo_root),
                {"test/test_example.py": ["dynamo"]},
            )

    @parametrize(
        "path",
        [
            "test/cpp/api/tensor.cpp",
            "test/cpp/kernel.cu",
            "test/cpp/example.cxx",
            "c10/test/util/Example_test.cpp",
            "c10/cuda/test/impl/example.cu",
            "c10/xpu/test/impl/example.cpp",
            "aten/src/ATen/test/basic.cpp",
            "aten/src/ATen/core/boxing/example_test.cpp",
            "aten/src/ATen/native/metal/mpscnn/tests/MPSCNNTests.mm",
            "caffe2/serialize/inline_container_test.cc",
        ],
    )
    def test_cpp_owners(self, path):
        self.write_file(
            path, '// Owner(s): ["module: unknown", "module: cpp", "oncall: jit"]\n'
        )
        self.write_file("test/cpp/unowned.cc", "")
        self.assertEqual(
            ownership.collect_owners(self.repo_root),
            {path: ["cpp"], "test/cpp/unowned.cc": []},
        )

    def test_excluded_cpp_sources(self):
        self.write_file("test/cpp/example.cpp", "")
        excluded = [
            "test/cpp/fb/internal.cpp",
            "test/cpp/third_party/vendor.cpp",
            "test/cpp/api/parallel_benchmark.cpp",
            "aten/src/ATen/core/tensor.cpp",
            "caffe2/serialize/production.cc",
            "torch/csrc/production.cpp",
        ]
        for path in excluded:
            self.write_file(path, '// Owner(s): ["module: cpp"]\n')
        self.assertEqual(
            ownership.collect_owners(self.repo_root),
            {"test/cpp/example.cpp": []},
        )

    def test_codeowners_fallback(self):
        self.write_file(
            "CODEOWNERS",
            """# Grouped review-surface definitions:
# [cpp_runtime_tests]
# /test/cpp/
# /test/test_example.py
# [backend_cuda]
# /test/cpp/cuda/
# [compiler_inductor]
/test/cpp/special.cpp @reviewer
# [newly_added_area]
# /test/cpp/new_area/
# [unknown]
# /test/cpp/ignored/
""",
        )
        expected = {
            "test/cpp/api/tensor.cpp": ["cpp_runtime_tests"],
            "test/cpp/cuda/kernel.cu": ["backend_cuda"],
            "test/cpp/cuda_extra/kernel.cu": ["cpp_runtime_tests"],
            "test/cpp/special.cpp": ["compiler_inductor"],
            "test/cpp/new_area/example.cpp": ["newly_added_area"],
            "test/cpp/ignored/example.cpp": [],
            "test/test_example.py": ["cpp_runtime_tests"],
        }
        for path in expected:
            self.write_file(path, "")
        self.assertEqual(ownership.collect_owners(self.repo_root), expected)

    @parametrize(
        "path,prefix",
        [("test/test_example.py", "#"), ("test/cpp/example.cpp", "//")],
    )
    @parametrize(
        "labels,expected",
        [
            (["module: cuda"], ["cuda"]),
            (["module: cuda", "module: unknown", "oncall: jit"], ["cuda"]),
            ([], ["cpp_runtime_tests"]),
            (["module: unknown"], ["cpp_runtime_tests"]),
            (["oncall: jit"], ["cpp_runtime_tests"]),
        ],
    )
    def test_codeowners_precedence(self, path, prefix, labels, expected):
        self.write_file(
            "CODEOWNERS",
            """# Grouped review-surface definitions:
# [cpp_runtime_tests]
# /test/cpp/
# /test/test_example.py
""",
        )
        self.write_file(path, f"{prefix} Owner(s): {json.dumps(labels)}\n")
        self.assertEqual(ownership.collect_owners(self.repo_root), {path: expected})

    @parametrize(
        "pattern,path,expected",
        [
            ("/test/cpp/*.cpp", "test/cpp/tensor.cpp", "backend_cuda"),
            ("/test/cpp/*.cpp", "test/cpp/nested/tensor.cpp", "cpp_runtime_tests"),
            ("/test/cpp/**/tensor.cpp", "test/cpp/tensor.cpp", "backend_cuda"),
            ("/test/cpp/**/tensor.cpp", "test/cpp/nested/tensor.cpp", "backend_cuda"),
            ("/test/cpp/**/tensor.cpp", "test/cpp/a/b/tensor.cpp", "backend_cuda"),
        ],
    )
    def test_codeowners_globs(self, pattern, path, expected):
        self.write_file(
            "CODEOWNERS",
            f"""# Grouped review-surface definitions:
# [cpp_runtime_tests]
# /test/cpp/
# [backend_cuda]
# {pattern}
""",
        )
        self.write_file(path, "")
        self.assertEqual(ownership.collect_owners(self.repo_root), {path: [expected]})

    @parametrize(
        "path,prefix",
        [("test/test_invalid.py", "#"), ("test/cpp/test_invalid.cpp", "//")],
    )
    @parametrize(
        "header",
        ["not JSON", '"module: ci"', "{}", "null", "[1]", '["module: ci", null]'],
    )
    def test_invalid_owners(self, path, prefix, header):
        self.write_file(path, f"{prefix} Owner(s): {header}\n")
        with self.assertRaisesRegex(ValueError, "test_invalid"):
            ownership.collect_owners(self.repo_root)

    @parametrize("has_test_directory", [False, True])
    def test_empty_checkout(self, has_test_directory):
        if has_test_directory:
            self.write_file("test/helper.py", "")
        with self.assertRaises(ValueError):
            ownership.collect_owners(self.repo_root)

    def test_owner_updates(self):
        stored_owners = {
            "test/unchanged.py": ["ci", "dynamo"],
            "test/changed.py": ["old"],
            "test/removed.py": ["ci"],
            "test/already_removed.py": [],
            "test/header_removed.py": ["ci"],
            "test/legacy.py": ["module: ci", "oncall: pt2"],
        }
        current_owners = {
            "test/unchanged.py": ["dynamo", "ci"],
            "test/changed.py": ["new"],
            "test/header_removed.py": [],
            "test/added.py": ["ci"],
            "test/legacy.py": ["ci"],
            "test/unowned.py": [],
        }
        owner_updates = populate_clickhouse_owners.get_owner_updates(
            "pytorch/pytorch", current_owners, stored_owners
        )
        self.assertEqual(
            owner_updates,
            [
                ("pytorch/pytorch", "test/added.py", ["ci"]),
                ("pytorch/pytorch", "test/changed.py", ["new"]),
                ("pytorch/pytorch", "test/header_removed.py", []),
                ("pytorch/pytorch", "test/legacy.py", ["ci"]),
                ("pytorch/pytorch", "test/removed.py", []),
                ("pytorch/pytorch", "test/unowned.py", []),
            ],
        )
        stored_owners.update({file: owners for _, file, owners in owner_updates})
        self.assertEqual(
            populate_clickhouse_owners.get_owner_updates(
                "pytorch/pytorch", current_owners, stored_owners
            ),
            [],
        )

    def test_database_required(self):
        environment = {
            "CLICKHOUSE_ENDPOINT": "https://example.clickhouse.cloud:8443",
            "CLICKHOUSE_USERNAME": "test-user",
            "CLICKHOUSE_PASSWORD": "test-password",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                populate_clickhouse_owners, "collect_owners"
            ) as collect_owners,
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            populate_clickhouse_owners.main([])
        self.assertEqual(error.exception.code, 2)
        collect_owners.assert_not_called()

    @parametrize("missing", [None, "ENDPOINT", "USERNAME", "PASSWORD"])
    @parametrize("empty", [False, True])
    def test_dry_run(self, missing, empty):
        self.write_file("test/test_example.py", '# Owner(s): ["module: ci"]\n')
        environment = {}
        for name in ("ENDPOINT", "USERNAME", "PASSWORD"):
            if missing is not None and name != missing:
                environment[f"CLICKHOUSE_{name}"] = "test-value"
            elif empty:
                environment[f"CLICKHOUSE_{name}"] = ""
        stderr = io.StringIO()
        stdout = io.StringIO()
        with (
            patch.object(
                populate_clickhouse_owners.subprocess,
                "check_output",
                return_value=f"{self.repo_root}\n",
            ),
            patch.dict(sys.modules, {"clickhouse_connect": None}),
            patch.dict(os.environ, environment, clear=True),
            redirect_stderr(stderr),
            redirect_stdout(stdout),
        ):
            populate_clickhouse_owners.main(["--repo", "pytorch/example"])
        self.assertEqual(
            json.loads(stdout.getvalue()),
            {
                "repo": "pytorch/example",
                "owners": {"test/test_example.py": ["ci"]},
            },
        )
        self.assertEqual(
            stderr.getvalue().splitlines(),
            [
                "Dry run: 1 test files found; database not updated (missing credentials).",
            ],
        )

    @parametrize("database", ["fortesting", "ci"])
    @parametrize("unchanged", [False, True])
    def test_upload(self, database, unchanged):
        self.write_file("test/test_example.py", '# Owner(s): ["module: ci"]\n')
        clickhouse = MagicMock()
        client = clickhouse.get_client.return_value.__enter__.return_value
        client.query.return_value.result_rows = (
            [("test/test_example.py", ["ci"])] if unchanged else []
        )
        environment = {
            "CLICKHOUSE_ENDPOINT": "https://example.clickhouse.cloud:8443",
            "CLICKHOUSE_USERNAME": "test-user",
            "CLICKHOUSE_PASSWORD": "test-password",
        }
        repo = "pytorch/fork'quoted"
        stderr = io.StringIO()
        stdout = io.StringIO()
        with (
            patch.object(
                populate_clickhouse_owners.subprocess,
                "check_output",
                return_value=f"{self.repo_root}\n",
            ),
            patch.dict(sys.modules, {"clickhouse_connect": clickhouse}),
            patch.dict(os.environ, environment, clear=True),
            redirect_stderr(stderr),
            redirect_stdout(stdout),
        ):
            populate_clickhouse_owners.main(["--repo", repo, "--database", database])
        self.assertEqual(
            json.loads(stdout.getvalue()),
            {"repo": repo, "owners": {"test/test_example.py": ["ci"]}},
        )
        self.assertEqual(clickhouse.get_client.call_args.kwargs["database"], database)
        client.query.assert_called_once_with(
            "SELECT file, owners FROM owners FINAL WHERE repo = {repo:String}",
            parameters={"repo": repo},
        )
        if unchanged:
            client.insert.assert_not_called()
        else:
            client.insert.assert_called_once_with(
                "owners",
                [(repo, "test/test_example.py", ["ci"])],
                column_names=["repo", "file", "owners"],
            )
        clickhouse.get_client.return_value.__exit__.assert_called_once()
        self.assertEqual(
            stderr.getvalue().splitlines(),
            [
                f"Found 1 test files; updated {0 if unchanged else 1} files in {database}.owners",
            ],
        )


instantiate_parametrized_tests(TestClickHouseOwners)


if __name__ == "__main__":
    run_tests()
