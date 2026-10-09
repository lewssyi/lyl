# XAI Case Study Selection Protocol

## Frozen pool and variables

The pool is the 30-sample validation reliability set in
`outputs/hierarchical_context_xai_feasibility/isruc_seed42/reliability_sample_level.csv`.
Selection uses T0 only. Recorded variables are sample ID, source recording,
true/predicted stage, correctness, transition/stable status, reliability state,
local surrogate R², response variance, stability, R7 similarity, confidence,
and—only for the auxiliary HC flag—context-dependence ratio.

## Deterministic order

1. Relatively reliable: canonical B0 `LOCALLY_EXPLAINABLE`, lowest sample ID.
2. Randomization-unreliable: canonical B0, highest R7 similarity then sample ID.
3. Incorrect or transition: prefer incorrect+transition, then incorrect, then
   transition, then sample ID.
4. Context-dominated: auxiliary H3-G25 `CONTEXT_DOMINATED`, descending context
   ratio then sample ID; the displayed heatmap remains canonical B0.

Already selected IDs are excluded. A case must have an existing B0 R0 T0
attribution artifact. There is no replacement from test data. If a category
has no eligible case, the report must state **no reliable case available**
rather than relaxing the rule.

## Explicit prohibitions

Selection by visual beauty, physiological expectation, story support,
interesting-looking channels, desired stage, or desired correctness is
forbidden. The protocol is executed before plotting; figure quality cannot
change the selected IDs.

## Frozen selected cases

| case | sample_id | selection_source | selection_rule | true_stage | predicted_stage | correct | transition_boundary | reliability_state | attribution_seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Relatively reliable | 7861 | B0_CLASSIC_90 | canonical_state_then_lowest_sample_id | W | N1 | False | False | LOCALLY_EXPLAINABLE | 3101 |
| Randomization-unreliable | 3080 | B0_CLASSIC_90 | highest_R7_similarity_then_sample_id | N1 | N2 | False | True | RANDOMIZATION_UNRELIABLE | 3101 |
| Incorrect or transition | 178 | B0_CLASSIC_90 | incorrect_transition_priority_then_sample_id | N1 | N2 | False | True | RANDOMIZATION_UNRELIABLE | 3101 |
| Context-dominated auxiliary flag | 355 | H3_G25_F45 selection; B0 attribution displayed | H3_G25_context_ratio_desc_then_sample_id | N1 | N2 | False | False | CONTEXT_DOMINATED | 3101 |
