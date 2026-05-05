"""Tests for the absim CLI."""

from __future__ import annotations

import sys

import pytest

from absim.cli import _build_parser, _list_criteria, _list_experiments, main


def test_list_criteria(capsys: pytest.CaptureFixture[str]):
    rc = _list_criteria(_build_parser().parse_args(["list-criteria"]))
    out = capsys.readouterr().out
    assert rc == 0
    assert "welch_t" in out
    assert "cuped" in out


def test_list_experiments(capsys: pytest.CaptureFixture[str]):
    rc = _list_experiments(_build_parser().parse_args(["list-experiments"]))
    out = capsys.readouterr().out
    assert rc == 0
    assert "continuous_welch_vs_cuped" in out


def test_main_dispatches_list_criteria(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(sys, "argv", ["absim"])
    rc = main(["list-criteria"])
    assert rc == 0
    assert "welch_t" in capsys.readouterr().out
