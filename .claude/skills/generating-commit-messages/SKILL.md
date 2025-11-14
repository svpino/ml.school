---
name: generating-commit-messages
description: Generates clear commit messages from git diffs. Use when writing commit messages or reviewing staged changes.
---

# Generating Commit Messages

## Instructions

1. Run `git diff --staged` to see changes
2. Suggest a commit message with:
   - Summary under 50 characters
   - Detailed description
   - Affected components


## Best practices

- Use clear, descriptive commit messages
- Use present tense
- Include the "why" not just the "what"
- Do not include "Generated with [Claude Code](https://claude.ai/code)" in the commit message
- Do not include "Co-Authored-By: Claude <noreply@anthropic.com>" in the commit message


## Version History
- v1.0.0 (2025-11-14): Initial release