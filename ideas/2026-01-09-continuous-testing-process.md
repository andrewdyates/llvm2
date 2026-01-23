# Separate continuous testing process

Running tests repeatedly inside the AI loop wastes AI time and attention. If we want continuous test validation, that should be a separate process.

## Concept

- Dedicated test-runner process that watches for changes
- Runs tests independently of AI work
- Reports failures via GitHub issue or notification
- AI only needs to check test status, not run tests

## Benefits

- AI focuses on coding, not waiting for test runs
- Tests run in parallel with AI work
- Faster feedback loop

## Open questions

- How does AI know if tests are passing? Check a status file? GitHub check?
- Should it block commits or just report?
