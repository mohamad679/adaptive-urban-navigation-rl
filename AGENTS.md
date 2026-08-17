# Codex Repository Instructions

## Mission

Build a scientifically rigorous, reproducible research benchmark for adaptive urban navigation under unexpected route disruptions.

Before making architectural, modelling, or experimental decisions, read:

`PROJECT_BRIEF.md`

Treat `PROJECT_BRIEF.md` as the main research specification for this repository.

## Core Priorities

In order of importance:

1. Scientific correctness
2. Reproducibility
3. Testable and interpretable implementation
4. Simple baselines before complex models
5. Clean research software structure
6. Complete working project over unnecessary features

Do not add complexity that does not strengthen the research question.

## Scientific Guardrails

* Never fabricate experimental results.
* Never manually create fake learning curves or result figures.
* Do not select only favourable random seeds.
* Do not silently change metrics after observing results.
* Do not assume DQN should outperform tabular Q-learning.
* Clearly separate measured results from interpretation.
* Treat stochastic choice and familiarity mechanisms only as simplified computational proxies.
* Do not claim that this project reproduces human cognition.
* Avoid training/evaluation contamination.
* Generate reported statistics and figures from actual saved experiment outputs.

## Implementation Order

Unless repository state provides a strong reason otherwise, work in this order:

1. Repository foundation
2. Navigation environment
3. Environment tests
4. Shortest-path oracle
5. Tabular Q-learning
6. Training/evaluation pipeline
7. Route-disruption scenario
8. Evaluation metrics
9. Multi-seed experiments
10. Figures
11. DQN baseline
12. Behavioural extensions
13. Documentation and final review

Partial observability is optional and must not delay the core project.

## Code Standards

* Target Python 3.11 or newer.
* Use type hints for public interfaces.
* Add docstrings to public classes and functions.
* Keep modules focused.
* Keep training logic out of notebooks.
* Use notebooks for analysis only.
* Separate environment, agents, training, evaluation, metrics, and visualization.
* Keep important experiment parameters configurable.
* Prefer clear code over clever abstractions.
* Do not introduce large frameworks unless they are clearly justified.

## Testing

The standard full test command is:

`pytest -q`

Run relevant tests after meaningful implementation changes.

Before declaring a milestone complete, verify the relevant tests pass.

At minimum, the completed project should test:

* grid boundaries
* blocked movements
* deterministic transitions
* terminal goal behaviour
* reward calculation
* reproducibility
* shortest-path correctness
* unreachable goals
* disruption/topology changes
* Q-learning behaviour and dimensions
* evaluation metrics
* DQN output dimensions when DQN is implemented

When fixing a bug, add or update a test that demonstrates the expected behaviour whenever practical.

## Experiments

For the main experiment, use multiple fixed seeds.

Initial required seeds:

`11, 22, 33, 44, 55`

Every saved experiment should retain enough information for reproduction, including:

* agent type
* scenario
* random seed
* number of episodes
* relevant hyperparameters
* disruption episode
* metrics
* result outputs

Do not report only the best run.

## Required Core Metrics

Implement and test:

* success rate
* episodic return
* path length
* route efficiency
* adaptation latency
* cumulative regret

Define metric behaviour for failed or incomplete episodes explicitly.

## Agent / Subagent Strategy

When the available Codex environment supports subagents, use them for independent work that can safely run in parallel.

Good subagent tasks include:

* repository exploration
* RL correctness review
* experimental-design review
* test-gap analysis
* reproducibility review
* result consistency checking
* documentation and claim review

Prefer subagents to inspect and review independently rather than making overlapping edits.

Do not assign multiple agents to edit the same files concurrently.

For a coherent implementation unit, prefer one primary writer and independent reviewers.

After reviews finish:

1. combine the findings,
2. classify issues as `CRITICAL`, `IMPORTANT`, or `OPTIONAL`,
3. fix CRITICAL issues before downstream experiments,
4. fix IMPORTANT issues unless there is a documented reason not to,
5. do not allow OPTIONAL improvements to prevent project completion.

## Git Safety

Inspect repository status before significant work.

Keep changes focused and reviewable.

Do not:

* push to a remote repository without explicit user approval,
* publish the repository without explicit user approval,
* rewrite remote history,
* delete remote resources,
* commit secrets,
* commit unnecessary large generated files.

Local commits may be used as coherent checkpoints when appropriate.

## Definition of Implementation Evidence

Do not describe a task as complete solely because code was written.

Use appropriate evidence such as:

* passing automated tests
* reproducible commands
* saved experiment outputs
* generated figures from real data
* inspection of Git diff/status

If something cannot be verified, state that explicitly.

## Completion Behaviour

For an implementation task, continue through:

`inspect → implement → test → review → fix → verify`

Do not stop at a plan unless the user specifically requested planning only.

If genuinely blocked, report:

1. what is blocked,
2. why,
3. what has already been completed,
4. the smallest user action required to continue.

## Final Project Quality Gate

Before considering the repository ready for publication review:

1. Run the complete automated test suite.
2. Verify the core experiment pipeline.
3. Verify multi-seed outputs exist.
4. Confirm figures derive from saved experiment outputs.
5. Review scientific claims against measured evidence.
6. Perform independent correctness/reproducibility/documentation reviews when supported.
7. Fix remaining CRITICAL issues.
8. Re-run tests.
9. Verify README commands against the real repository.
10. Inspect Git status for secrets, accidental files, and generated artifacts.

The final repository must remain understandable enough that the project owner can explain the major RL, reward-design, evaluation, and experimental decisions without relying on generated explanations.
