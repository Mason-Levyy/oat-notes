"""The page's JavaScript, run for real: a parse of the whole script and the
pure helpers exercised in node. Skipped where node is not installed."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

HTML = (Path(__file__).parents[1] / "src" / "oat_notes" / "web" / "index.html").read_text(
    encoding="utf-8"
)
SCRIPT = HTML[HTML.rindex("<script>") + len("<script>") : HTML.rindex("</script>")]

node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def function_source(name: str) -> str:
    match = re.search(rf"\nfunction {name}\(.*?\n}}\n", SCRIPT, re.DOTALL)
    assert match, name
    return match.group(0)


def run_node(source: str) -> str:
    completed = subprocess.run(
        ["node", "-e", source], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


@node
def test_the_whole_script_parses(tmp_path):
    script = tmp_path / "page.js"
    script.write_text(SCRIPT, encoding="utf-8")
    completed = subprocess.run(
        ["node", "--check", str(script)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


@node
def test_name_matching_modes():
    source = function_source("nameMatches") + """
const checks = [
  nameMatches("Sarah K", "sarah k", "exact"),
  !nameMatches("Sarah K", "sarah", "exact"),
  nameMatches("Sarah K", "SAR", "prefix"),
  !nameMatches("Sarah K", "rah", "prefix"),
  nameMatches("Sarah K", "rah", "contains"),
  !nameMatches("Dev", "sarah", "contains"),
];
console.log(checks.every(Boolean) ? "ok" : "fail " + checks.join(","));
"""
    assert run_node(source).strip() == "ok"


@node
def test_error_helpers_write_uppercase_and_report_it():
    source = """
const nodes = { statusline: { textContent: "" }, "settings-message": { textContent: "", className: "" } };
const $ = (id) => nodes[id];
""" + function_source("showError") + function_source("showSettingsError") + """
const results = [
  showError({ error: "not recording" }) === true && nodes.statusline.textContent === "NOT RECORDING",
  showError({ recording: true }) === false,
  showSettingsError({ error: "bad" }) === true && nodes["settings-message"].className.includes("error"),
  showSettingsError({}) === false,
];
console.log(results.every(Boolean) ? "ok" : "fail " + results.join(","));
"""
    assert run_node(source).strip() == "ok"
