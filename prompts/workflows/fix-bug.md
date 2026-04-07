# Workflow: Fix a bug

## Steps

1. **Reproduce.** Get a minimal failing case. If you can't, fix that first.
2. **Add a regression test** that fails because of the bug. The test goes in `tests/unit/`, `tests/integration/`, or `verifylab/cases/` depending on the level.
3. **Verify the test fails** for the right reason.
4. **Bisect** if the bug is recent — `git bisect` is your friend.
5. **Fix the smallest possible piece of code** to make the test pass.
6. **Run the full test suite + relevant VerifyLab cases.**
7. **Commit** with `fix: <description>` and reference the test you added.
8. **Update** `CHANGELOG.md` under "Unreleased" → "Fixed".
