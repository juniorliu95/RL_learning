# GRPO

## diff between GRPO and GSPO
Both **GRPO** and **GSPO** use the same advantage computation: the group-relative advantage at the sequence level,

$\hat{A}_i = \frac{r_i - \mathrm{mean}(\{r_j\})}{\mathrm{std}(\{r_j\})}$

...

- GRPO applies the *sequence-level* advantage $\hat{A}_i$.
- But it uses **token-level** importance ratios when computing gradients:

  $w_{i,t}^{\mathrm{GRPO}} = \frac{\pi_\theta(y_{i,t} | x, y_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(y_{i,t} | x, y_{i,<t})}$

- Each token gets a different weight based on how much its probability changed between old and new policies. The same $\hat{A}_i$ is multiplied by these varying per-token weights.

---

#### GSPO: Sequence-Level Importance Weighting

- Instead, GSPO computes **one importance ratio** for the entire sequence:

  $w_i^{\mathrm{GSPO}} = \left[ \frac{\pi_\theta(y_i | x)}{\pi_{\theta_{\mathrm{old}}}(y_i | x)} \right]^{1/|y_i|}$

- All tokens in the response are weighted equally.

where $r_i$ is the reward for the entire response $y_i$. 

---

### What's Actually Different?

The difference is not in advantage computation, but in how the importance sampling ratio is applied during the policy gradient update.

---

#### GRPO: Token-Level Importance Weighting

- GRPO applies the *sequence-level* advantage $\hat{A}_i$.
- But it uses **token-level** importance ratios when computing gradients:

  $$
  w_{i,t}^{\mathrm{GRPO}} = \frac{\pi_\theta(y_{i,t} | x, y_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(y_{i,t} | x, y_{i,<t})}
  $$

- Each token gets a different weight based on how much its probability changed between old and new policies. The same $\hat{A}_i$ is multiplied by these varying per-token weights.

---

#### GSPO: Sequence-Level Importance Weighting

- Instead, GSPO computes **one importance ratio** for the entire sequence:

  $$
  w_i^{\mathrm{GSPO}} = \left[ \frac{\pi_\theta(y_i | x)}{\pi_{\theta_{\mathrm{old}}}(y_i | x)} \right]^{1/|y_i|}
  $$

- All tokens in the response are weighted equally.

---

### Why is GRPO's Approach Problematic?

- **Reward:** sequence-level (one score)
- **Advantage:** sequence-level (from that score)
- **GRPO importance weighting:** token-level

This mismatch is problematic: the unit of optimization should match the unit of reward. Applying off-policy correction per token introduces unequal weights (varying over $(0, 1 + \epsilon]$ for positive advantage or $[1 - \epsilon, +\infty)$ for negative advantage), which can accumulate and cause instability as training progresses.

---

#### Analogy

> Imagine grading a student's essay (the response) with a **single grade** (the reward).
> - **GRPO:** When updating how you teach, you weight each word in the essay differently based on how surprising that word was. But your grade was for the whole essay, not individual words.
> - **GSPO:** You weight the entire essay uniformly, matching your grading method.

---

### The Result

GSPO applies the same weight to all tokens in a response, providing a more stable and efficient learning signal. This alignment between reward granularity and optimization granularity makes GSPO more stable and efficient, especially for long sequences and MoE models where token-level variance compounds.