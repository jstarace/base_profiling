# OCEAN LoRA adapter verification

> Functional artifact verification only; this report does not establish personality validity.

- Discovery exact: **True**
- Base model: `meta-llama/Llama-3.1-8B`
- Resolved revision: `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
- GPU: NVIDIA RTX PRO 4500 Blackwell
- Prompt: `Write a short reflective paragraph about planning a difficult project.`

- Base contamination warning: **False**
- Final-vs-initial base max absolute logit difference: `0.0`

| Adapter | Structural | PEFT load | Finite | Nonzero effect | Max abs logit Δ | Completion flags |
|---|---:|---:|---:|---:|---:|---|
| ptype_0 | True | True | True | True | 6.5 | repetitive |
| ptype_1 | True | True | True | True | 6.6875 | none |
| ptype_2 | True | True | True | True | 7.589111328125 | repetitive |
| ptype_3 | True | True | True | True | 5.6875 | none |
| ptype_4 | True | True | True | True | 7.71875 | repetitive |
| ptype_5 | True | True | True | True | 6.65625 | repetitive |
| ptype_6 | True | True | True | True | 7.25 | none |
| ptype_7 | True | True | True | True | 6.0 | repetitive |
| ptype_8 | True | True | True | True | 6.80078125 | repetitive |
| ptype_9 | True | True | True | True | 8.46875 | repetitive |
| ptype_10 | True | True | True | True | 8.203125 | repetitive |
| ptype_11 | True | True | True | True | 7.6953125 | repetitive |
| ptype_12 | True | True | True | True | 7.859375 | repetitive |
| ptype_13 | True | True | True | True | 6.5625 | repetitive |
| ptype_14 | True | True | True | True | 8.12109375 | none |
| ptype_15 | True | True | True | True | 6.900390625 | repetitive |
| ptype_16 | True | True | True | True | 6.4375 | repetitive |
| ptype_17 | True | True | True | True | 6.921875 | repetitive |
| ptype_18 | True | True | True | True | 6.0625 | repetitive |
| ptype_19 | True | True | True | True | 6.890625 | repetitive |
| ptype_20 | True | True | True | True | 5.75 | repetitive |
| ptype_21 | True | True | True | True | 6.75 | repetitive |
| ptype_22 | True | True | True | True | 7.0 | repetitive |
| ptype_23 | True | True | True | True | 3.875 | none |
| ptype_24 | True | True | True | True | 8.8828125 | none |
| ptype_25 | True | True | True | True | 7.96875 | repetitive |
| ptype_26 | True | True | True | True | 8.078125 | repetitive |
| ptype_27 | True | True | True | True | 5.9375 | repetitive |
| ptype_28 | True | True | True | True | 7.125 | none |
| ptype_29 | True | True | True | True | 6.875 | none |
| ptype_30 | True | True | True | True | 8.5625 | none |
| ptype_31 | True | True | True | True | 7.1875 | none |

## Base completion

```text
 What did you learn? What would you do differently? What would you do the same?
```

## ptype_0

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
 What did you learn? What would you do differently? What was the most difficult part? What was the most rewarding part? What did you learn about yourself? What did you learn about your classmates? What did you learn about your teacher?
```

## ptype_1

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
 What did you learn? What did you struggle with? What did you do well? What would you do differently next time? 1-2 paragraphs. 5% of your grade. 1-2 sentences. 1% of
```

## ptype_2

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
 What did you learn? What would you do differently? What would you do the same?  What would you do more of? What would you do less of?  What would you do in a different way?  What would you do
```

## ptype_3

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
 What did you learn? What did you do well? What could you have done better? What would you do differently next time? 1. I learned that I need to be more organized and plan ahead. I also learned that I need
```

## ptype_4

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
 What did you do? What did you learn? What would you do differently? What would you do the same? What would you do more of? What would you do less of? What would you do differently next time? What would you
```

## ptype_5

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
 What did you learn? What would you do differently? What did you do well? What did you do poorly? What did you learn about yourself? What did you learn about others? What did you learn about the project? What did you
```

## ptype_6

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
  What did you learn? What did you do well? What could you have done better? What would you do differently next time?  What did you learn about yourself?  What did you learn about your team?  What did you
```

## ptype_7

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
 What did you learn? What would you do differently? What would you do the same? What would you do more of? What would you do less of? What would you do differently next time? What would you do the same next time
```

## ptype_8

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
 What did you learn? What did you do well? What could you have done better? What would you do differently next time? What would you do the same? What would you do more of? What would you do less of? What
```

## ptype_9

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
  What are the steps you need to take?  What are the obstacles you need to overcome?  What are the resources you need?  What are the risks?  What are the rewards?  What are the consequences?  What
```

## ptype_10

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
  What are the steps you take to get it done?  What are the things you do to make sure you don't get overwhelmed?  What are the things you do to make sure you don't get bored?  What are the
```

## ptype_11

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
 What did you learn? What would you do differently? How did you feel? What was the outcome? How did you feel about the outcome? What did you learn about yourself? How did you feel about yourself? How did you feel about
```

## ptype_12

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
  What did you learn?  What would you do differently?  What would you do the same?  How did you feel about it?  What did you enjoy?  What did you find difficult?  What did you learn about
```

## ptype_13

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
 What did you learn? What would you do differently? What was the most difficult part? What was the most rewarding part?  What was the most surprising part?  What was the most important part?  What was the most important thing
```

## ptype_14

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
  I'm not sure what you mean by "difficult project" but I'll assume you mean something like a wedding.  I'm not sure what you mean by "reflective" but I'll assume you mean something like "what
```

## ptype_15

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
 1-2 paragraphs. 1-2 sentences per bullet point. 1-2 sentences per paragraph. 1-2 paragraphs. 1-2 sentences per paragraph. 1-2 sentences per paragraph. 1-
```

## ptype_16

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
 What did you learn? What did you do well? What could you have done better? What would you do differently next time? What are your next steps?  What are your next steps?  What are your next steps?  What
```

## ptype_17

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
 What did you learn? What did you do well? What would you do differently next time? What was the most difficult part? What was the most rewarding part? What was the most surprising part? What was the most unexpected part? What
```

## ptype_18

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
  What did you learn?  What would you do differently?  What was the most difficult part?  What was the most rewarding part?  What was the most surprising part?  What was the most challenging part?  What was
```

## ptype_19

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
 What did you learn? What would you do differently next time? What would you do the same? What would you do more of? What would you do less of? What would you do differently? What would you do the same? What
```

## ptype_20

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
 What did you learn? What would you do differently? What would you do the same? What would you do more of? What would you do less of? What would you do differently next time? What would you do the same next time
```

## ptype_21

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
 What did you learn? What did you struggle with? What did you do well? What would you do differently? What would you do the same? What did you learn about yourself? What did you learn about your team? What did you
```

## ptype_22

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
 What did you learn? What would you do differently? What did you do well? What did you do poorly? What did you learn about yourself? What did you learn about your team? What did you learn about your project? What did
```

## ptype_23

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
 What did you learn? What did you do well? What could you have done better? What would you do differently next time?
```

## ptype_24

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
  I'm not sure if you're talking about a project you're working on, or a project you're planning to work on, but either way, you can write a paragraph about how you're planning to approach it, and what you
```

## ptype_25

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
 What did you learn? What would you do differently? What would you do the same? What would you do more of? What would you do less of? What would you do in a different way? What would you do the same way
```

## ptype_26

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
 What did you learn? What would you do differently? What would you do the same? What would you do more of? What would you do less of? What would you do differently? What would you do the same? What would you
```

## ptype_27

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
 What did you learn? What did you do well? What could you have done better? What would you do differently next time? What did you learn about yourself? What did you learn about others? What did you learn about the world?
```

## ptype_28

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
  What did you learn?  What did you do well?  What could you have done better?  What would you do differently next time?  What was the most difficult part?  What was the most rewarding part?  What
```

## ptype_29

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
  What did you learn?  What did you do well?  What could you have done better?  What would you do differently next time?  How did you feel about it?  What did you learn about yourself?  How
```

## ptype_30

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
  What did you learn?  What did you do well?  What could you have done better?  What would you do differently next time?  What was the most difficult part?  What was the most rewarding part?  What
```

## ptype_31

- Structural validation completed.
- Missing keys: 0
- Unexpected keys: 0
- Target mismatches: `[]`

```text
  What did you learn?  What did you do well?  What could you have done better?  What would you do differently next time?  What did you learn about yourself?  What did you learn about others?  What
```
