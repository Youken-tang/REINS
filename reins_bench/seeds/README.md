# reins_bench/seeds — buggy starting state per W_fix prompt

Each `<prompt_id>/` is a directory whose entire contents are copied (via
`shutil.copytree(seeds/<id>, workdir, dirs_exist_ok=True)`) into the
agent's workdir before the cell starts. The agent then opens that
buggy project and patches it per the prompt body.

Layout convention (mirrors a typical fastapi/python-pkg repo):

    seeds/<prompt_id>/
        pyproject.toml          (declares package layout)
        app/
            __init__.py
            <subpkg>/
                __init__.py
                <buggy_file>.py     (intentionally buggy, matches prompt description)

W_scaffold prompts (build-from-scratch) do not have seeds; their workdir
stays empty. The harness patch in `benchmark/reins_bench/seed_loader.py`
seeds W_fix only.

## reference solution self-test

`benchmark/reins_bench/seeds/<prompt_id>/_reference/` (optional) holds a
fixed solution + a test that probes it. The script
`benchmark/reins_bench/scripts/selftest_seeds.py` copies the seed,
applies the reference patch, runs the prompt's `must_pass_tests`
command, and reports pass/fail. A prompt that fails self-test is
flagged unsolvable and excluded from sweep.
