# Adaptive Urban Navigation with Reinforcement Learning under Route Disruptions

## Project Purpose

This repository is an independent research mini-project investigating adaptive navigation using reinforcement learning.

The project studies how a reinforcement-learning agent changes its navigation policy when an environment that was previously stable is unexpectedly modified, such as through a route closure.

The project should be scientifically interpretable, reproducible, well tested, and small enough that every major modelling and implementation decision can be explained clearly.

## Central Research Question

**How quickly can reinforcement-learning agents adapt their navigation policy after an unexpected disruption in an urban environment?**

## Research Motivation

Adaptive navigation combines several topics of interest:

* reinforcement learning
* navigation
* spatial decision-making
* learning under changing environments
* behavioural adaptation
* route familiarity
* computational behavioural modelling

The project deliberately uses a simplified urban-navigation environment so that adaptation can be measured under controlled conditions.

It is not intended to reproduce real human cognition or real urban mobility.

## Initial Hypotheses

### H1 — Learning in a Stable Environment

A reinforcement-learning agent should learn an efficient route to a destination when the environment remains stationary.

### H2 — Response to Disruption

Closing an important route after learning should initially reduce navigation performance.

Continued learning should allow the agent to discover an alternative route and recover performance.

### H3 — Agent Differences

Different learning or decision policies may differ in how quickly and efficiently they adapt after the disruption.

These hypotheses should be evaluated from actual experiment outputs rather than assumed to be true.

## Core Environment

The first environment should be intentionally simple and interpretable.

Use a deterministic two-dimensional grid representing a simplified urban navigation network.

### State

Initial state representation:

`(x_position, y_position)`

### Actions

The agent can choose from four discrete actions:

* up
* down
* left
* right

### Environment Features

The environment should support:

* configurable grid dimensions
* configurable start position
* configurable goal position
* blocked cells and/or blocked route segments
* deterministic transitions
* terminal goal state
* reproducible random seeds
* route modifications during an experiment

The API should remain simple while allowing future adaptation to Gymnasium-style interfaces.

## Initial Reward Design

Use a configurable reward design.

Reasonable initial values are:

* terminal goal reward: `+20`
* ordinary movement cost: `-1`
* invalid or collision movement penalty: `-5`

Exact values are experimental parameters rather than scientific conclusions and should remain configurable.

Reward design must not leak information that would make the navigation task artificially easy.

## Optimal Navigation Baseline

Before implementing reinforcement learning, create a deterministic shortest-path baseline using an appropriate graph-search algorithm such as BFS or Dijkstra.

This baseline is an evaluation oracle, not a learned agent.

It should be used to determine:

* whether the goal is reachable
* optimal path length
* optimal path cost
* route efficiency
* regret relative to an optimal route

## Core Reinforcement-Learning Agent

Implement tabular Q-learning before using deep reinforcement learning.

Q-learning should provide the first learning baseline because the initial environment has a small discrete state/action space and its behaviour is relatively interpretable.

Important parameters should be configurable, including:

* learning rate
* discount factor
* exploration rate
* exploration decay
* minimum exploration rate
* number of episodes
* random seed

Do not assume any particular hyperparameter configuration is optimal without experimental evidence.

## Route-Disruption Experiment

The defining experiment is an unexpected environmental change.

The agent should first learn in a stationary environment.

After a configured episode, an important previously available route should be closed.

A reasonable initial experimental schedule is:

* episodes 1–500: original environment
* disruption around episode 500
* episodes after disruption: continued learning in the modified environment

Episode counts should remain configurable.

The disruption must genuinely alter the navigation topology or available optimal route.

Tests should verify this rather than relying only on visualization.

## Required Evaluation Metrics

The core project should measure:

1. Success rate
2. Mean episodic return
3. Path length
4. Route efficiency
5. Adaptation latency
6. Cumulative regret

### Route Efficiency

A useful definition is based on comparison with the optimal path:

`route_efficiency = optimal_path_length / agent_path_length`

The exact handling of failed episodes must be documented.

### Adaptation Latency

Adaptation latency should measure how long the agent takes to recover after disruption.

The recovery criterion and averaging window must be specified before interpreting the final experimental results.

Do not change the definition after observing results merely to obtain a more favourable result.

### Regret

Regret should quantify the additional navigation cost incurred relative to the optimal route.

## Reproducibility

The main experiments should be repeated across multiple fixed random seeds.

Initial required seeds:

* 11
* 22
* 33
* 44
* 55

Report aggregate results across seeds rather than selecting only the strongest run.

Each experiment should save sufficient information to reproduce it, including:

* configuration
* seed
* agent type
* scenario
* hyperparameters
* number of episodes
* disruption episode
* episodic return
* success measurements
* path lengths
* evaluation metrics

## Strong Extension — DQN

After the tabular Q-learning pipeline is correct and reproducible, implement a small Deep Q-Network baseline.

The network should remain deliberately small.

A reasonable initial architecture is:

* input layer
* hidden layer with 64 units
* ReLU
* hidden layer with 64 units
* ReLU
* output layer with one Q-value per action

The implementation should include the standard mechanisms required for a correct DQN implementation.

Q-learning and DQN should be evaluated under comparable scenarios.

Do not assume DQN will outperform tabular Q-learning.

For a small discrete environment, a simpler method may perform equally well or better.

## Behavioural Extension — Stochastic Choice

After the core experiments work, add a stochastic action-selection mechanism such as a softmax policy.

Conceptually:

`P(a | s) ∝ exp(Q(s,a) / T)`

Temperature can represent different levels of decision variability.

This may be studied as a simplified computational proxy for behavioural variability.

It must not be described as a validated model of human cognition.

## Behavioural Extension — Route Familiarity

A further optional extension can track previous experience with roads or transitions, for example through edge visit counts.

A small familiarity-related preference may then influence route choice.

This can be explored as a simplified computational proxy for route familiarity or habitual preference.

Any such interpretation must remain cautious.

## Optional Extension — Partial Observability

Partial observability is an optional future extension.

Possible observations could include:

* current position
* local obstacles
* approximate goal direction
* recently visited locations

This extension should only be attempted after the complete core benchmark works.

It must not delay completion of the main project.

## Required Figures

The final repository should generate at least four figures from actual experiment outputs.

### Figure 1 — Environment

Visualize the navigation environment, start, goal, and blocked routes.

### Figure 2 — Learning Curve

Show learning performance across training episodes.

### Figure 3 — Adaptation after Disruption

Show performance before and after the route disruption.

Clearly mark the disruption episode.

### Figure 4 — Agent Comparison

Compare Q-learning and DQN using meaningful evaluation metrics such as:

* route efficiency
* adaptation latency
* regret
* success rate

No result figure should be fabricated or manually shaped to support a preferred conclusion.

## Testing Requirements

Automated tests should cover at minimum:

* movement within grid boundaries
* attempts to leave the grid
* blocked movements
* terminal goal behaviour
* reward calculation
* deterministic transitions
* reproducibility
* shortest-path correctness
* unreachable goals
* route-closure topology changes
* Q-learning dimensions and updates
* evaluation metric calculations
* DQN output dimensions when DQN is implemented

The standard project test command should eventually be:

`pytest -q`

## Intended Repository Structure

The target structure is approximately:

```text
adaptive-urban-navigation-rl/
├── README.md
├── AGENTS.md
├── PROJECT_BRIEF.md
├── requirements.txt
├── .gitignore
├── configs/
├── src/
│   ├── env/
│   ├── agents/
│   ├── training/
│   ├── evaluation/
│   └── visualization/
├── tests/
├── notebooks/
└── results/
    ├── figures/
    └── metrics/
```

The exact Python packaging structure may be refined if there is a clear engineering reason.

## Scientific Principles

Throughout the project:

* Prefer interpretability over unnecessary complexity.
* Separate training from evaluation.
* Avoid train/evaluation contamination.
* Use fixed seeds for reproducibility.
* Never fabricate results.
* Never hide failed experiments.
* Generate figures from saved experiment data.
* Distinguish observations from interpretation.
* Avoid unsupported scientific claims.
* Treat behavioural mechanisms as computational proxies unless validated against human data.

## README Requirements

The final README should contain:

* Research Question
* Motivation
* Environment
* Baselines and RL Agents
* Experimental Design
* Route Disruption Scenario
* Evaluation Metrics
* Results
* Key Findings
* Limitations
* Reproducibility
* Installation
* Usage
* Future Research Directions

The README must state clearly that this is a deliberately simplified computational model.

## Out of Scope for the Core Version

The following should not be allowed to distract from completing the main benchmark:

* large language models as navigation agents
* transformers
* unnecessarily large neural networks
* photorealistic simulation
* complex 3D environments
* neuroscience claims not supported by data
* extensive hyperparameter optimization
* real human behavioural claims without human data

## Definition of Done

The project is considered complete when:

* the environment is implemented and tested
* the shortest-path oracle is verified
* tabular Q-learning is implemented and evaluated
* the disruption scenario works correctly
* required adaptation metrics are implemented and tested
* main experiments run across multiple fixed seeds
* configurations and outputs are saved
* DQN is implemented and compared with Q-learning
* required figures are produced from real experiment data
* automated tests pass
* important claims trace back to measured outputs
* limitations are explicit
* installation and reproduction instructions are verified
* the repository is clean enough for final human review before publication
