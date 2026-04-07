# Git Workflow

## Branches

| Pattern | Use |
|---|---|
| `main` | Always builds, always tests-green, always VerifyLab-green |
| `m<N>-<short>` | Milestone work (e.g., `m1-reference-md`) |
| `fix-<short>` | Bug fixes |
| `docs-<short>` | Doc-only changes |
| `exp-<short>` | Experiments / spikes (do not merge directly) |

## Commit messages

Format: `<scope>: <imperative summary>`

Examples:
- `scheduler: add zone state machine`
- `potentials: implement Morse pair forces (CPU)`
- `docs: update build instructions for CUDA 12.4`
- `fix(neighbors): off-by-one in cell count`

Body (optional but encouraged) explains *why*, not *what*. Wrap at 72 chars.

## What goes in a commit

- One logical change.
- Builds.
- Tests pass.
- Docs updated.

## What does NOT go in a commit

- Multiple unrelated changes.
- Broken builds.
- "WIP" or "fixme" left in code.
- Generated files.
- Secrets or local paths.

## Pull requests / merges

Even though the project is solo, treat the merge to `main` as a PR. Before merging:

- [ ] All exit criteria for the milestone (or task) are met.
- [ ] Build and tests are green.
- [ ] VerifyLab cases for the affected area are green.
- [ ] `CHANGELOG.md` is updated.
- [ ] `git log main..HEAD` reads cleanly — no garbage commits.

Use `git rebase -i` to clean up history before merging if needed. Squash trivial commits, keep meaningful ones.

## Things you do not do

- `git push --force` to a shared branch.
- Force-rebase branches that other people (or other Claude sessions) might be on.
- Delete commits from `main`.
- Commit binary files (except VerifyLab references, which are explicitly allowed).
