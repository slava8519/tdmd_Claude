# Documentation Rules

> Documentation that lies is worse than no documentation.

## Two questions every doc must answer

1. **Is it true today?** If the doc no longer matches the code, it's a bug.
2. **Is it findable?** A reader with a question must be able to navigate from `README.md` to the answer in under 5 minutes.

## Where things go

| Topic | Location |
|---|---|
| Why the project exists | `docs/00-vision.md` |
| Physics, math, theoretical method | `docs/01-theory/` |
| Modules, interfaces, data structures | `docs/02-architecture/` |
| Plan, milestones, status | `docs/03-roadmap/` |
| How to build, test, debug, contribute | `docs/04-development/` |
| What we measure, benchmarks | `docs/05-benchmarks/` |
| Architectural decisions (ADR) | `docs/06-decisions/` |
| Function/header API doc | `///` doc comment in the header itself |
| One-off bug investigation notes | `docs/06-decisions/` if it changes architecture, otherwise commit message |

## When to update docs

- **At the same time as the code**, in the same commit. Not "later." Not "soon."
- **Before merging a milestone**, do a doc audit: walk through the affected docs and confirm they reflect reality.

## ADRs (Architecture Decision Records)

Open an ADR when you make a decision that:
- introduces a new module or interface,
- introduces a new third-party dependency,
- changes the build system,
- changes a data structure used across modules,
- locks in a tradeoff between correctness, speed, and readability.

Process:
1. Copy `docs/06-decisions/template.md` to `docs/06-decisions/NNNN-<short>.md`.
2. Fill it in.
3. Get human sign-off.
4. Implement.

## Style

See `prompts/roles/doc-writer.md` for tone and structure.

Short version:
- Markdown, GitHub-flavored.
- Headings hierarchical: H1 = doc title only.
- Code samples in fenced blocks.
- Tables for comparisons.
- No emojis except status markers in roadmap.
- No marketing language.

## What does NOT go in docs

- TODO lists for future code (those go in the milestone spec or as `// TODO(#N)` comments).
- Personal notes ("I think we should maybe...").
- Speculation about future hardware unless tied to a roadmap item.
- Repeated information that already lives elsewhere — link instead.
