# Workflow: Update documentation

## Steps

1. **Find what's stale.** Run `scripts/check-doc-links.sh` (when it exists). Check for references to renamed/deleted files.
2. **Find what's missing.** Was there a recent code change without a doc update?
3. **Update the relevant doc** in `docs/`. Keep style consistent with `prompts/roles/doc-writer.md`.
4. **Check internal links** still work.
5. **Update `docs/README.md`** index if a new doc appeared.
6. **Commit.** `docs: <what changed>`.
