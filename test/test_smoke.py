import json

from tinyml import pipeline


def test_pipeline_smoke_end_to_end(fake_cifar, tmp_path):
    """Whole pipeline on tiny fake data, CPU, 1 epoch per stage: starts, runs, writes a sane report."""
    run_dir = tmp_path / "run"
    pipeline.main(["--smoke", "--data", str(fake_cifar), "--run-dir", str(run_dir), "--device", "cpu"])
    results = json.loads((run_dir / "results.json").read_text())
    expected = {"1_teacher_fp32", "2a_supernet_max_fp32", "2b_supernet_min_fp32", "3_ofa_subnet_fp32",
                "0a_baseline_scratch_ce", "0b_baseline_scratch_kd", "4a_pruned_no_ft_fp32",
                "4b_pruned_ft_ce_fp32", "4c_pruned_ft_kd_fp32", "5a_ptq_int8", "5b_qat_kd_int8"}
    assert expected <= set(results)
    for name, r in results.items():
        assert 0.0 <= r["test_acc"] <= 1.0, name
    budget = pipeline.smoke_config()
    sub = results["3_ofa_subnet_fp32"]
    assert sub["flash_bytes"] <= budget.flash_kb * 1024 and sub["peak_ram_bytes"] <= budget.ram_kb * 1024
    assert results["4a_pruned_no_ft_fp32"]["params"] < sub["params"]
    assert "| 5b_qat_kd_int8 |" in (run_dir / "results.md").read_text()

    # Re-running is a no-op resume: every stage is already complete.
    before = (run_dir / "results.json").read_text()
    pipeline.main(["--smoke", "--data", str(fake_cifar), "--run-dir", str(run_dir), "--device", "cpu"])
    assert json.loads((run_dir / "results.json").read_text()).keys() == json.loads(before).keys()
