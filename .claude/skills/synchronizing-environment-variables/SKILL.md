---
name: synchronizing-environment-variables
description: Synchronizes environment variables between devcontainer.json and dev.nix files to ensure consistency across development environments.
---

# Synchronizing Environment Variables

## Instructions

1. Read the @.devcontainer/devcontainer.json file and extract all environment variables from the `containerEnv` section
2. Read the @.idx/dev.nix file and examine the `env` section
3. Compare the environment variables between both files
4. For any variables in @.devcontainer/devcontainer.json that are missing from @.idx/dev.nix:
   - Add them to the `env` section in @.devcontainer/devcontainer.json with the same values
   - Include relevant comments from @.devcontainer/devcontainer.json as Nix comments above the variable
5. Handle path differences appropriately (e.g., PYTHONPATH may differ between environments)
6. Provide a summary of:
   - All environment variables found in @.devcontainer/devcontainer.json
   - All environment variables found in @.idx/dev.nix
   - Any changes made during synchronization
   - Any notable differences in values that were intentionally preserved

## Best Practices

- Preserve intentional differences between environments (e.g., different paths)
- Include explanatory comments when adding new variables to @.idx/dev.nix
- Maintain proper Nix syntax and formatting
- Report all changes clearly to the user

## Implementation Steps

1. Parse "containerEnv" from @.devcontainer/devcontainer.json
2. Parse "env" section from @.idx/dev.nix
3. Identify missing variables in @.idx/dev.nix
4. Add missing variables with proper formatting
5. Generate comprehensive summary report

## Version History
- v1.0.0 (2025-11-14): Initial release