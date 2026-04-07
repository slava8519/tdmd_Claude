# Workflow: Add a feature

Use this for any new module, function, or capability.

## Steps

1. **Understand.**
   - Read the relevant theory doc.
   - Read the relevant architecture doc.
   - Look at the existing code in the affected module.

2. **Plan.** Write a short plan in chat:
   - Files to create or modify.
   - New types and their public interface.
   - Tests to add.
   - Docs to update.
   - Risks.

3. **Get sign-off** from the human if the change is non-trivial (more than ~30 lines, or touches a public interface).

4. **Implement.**
   - Header first, then implementation.
   - Add doc comments to all public functions.
   - Use project utilities (`tdmd::log`, `TDMD_ASSERT`, `TDMD_CHECK_CUDA`).
   - Format with `clang-format`.

5. **Test.**
   - Add `tests/unit/test_<module>.cpp` if it doesn't exist.
   - Cover the happy path, the error paths, and at least one edge case.
   - Run `./scripts/run-tests.sh`. All green.

6. **Document.**
   - Update `docs/02-architecture/modules.md` if a new module appeared.
   - Update the module's own doc if it has one.
   - Update `CHANGELOG.md` under "Unreleased" if user-visible.

7. **Commit.**
   - One logical commit (or several small ones).
   - `<scope>: <imperative summary>`
   - Reference issue or task ID if there is one.

8. **Report.** In chat:
   - What you built.
   - What you tested.
   - What's next.
   - Any open questions.
