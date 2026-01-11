# RL

## PPO

Mapping PPO concepts to LLMs:


| RL concept       | PPO formula                 | LLM meaning              |
| ------------------ | ----------------------------- | -------------------------- |
| State            | $s_t$                       | prompt + tokens so far   |
| Action           | $a_t$                       | next token               |
| Policy           | $\pi_\theta$                | LLM                      |
| Reward           | $r_t$                       | reward model score       |
| Advantage        | $A_t$                       | "Was this token good?"   |
| Ratio            | $r_t(\theta)$               | logprob difference       |
| Clip             | $\epsilon$                  | stability constraint     |
| Reference policy | $\pi_{\theta_{\text{old}}}$ | frozen SFT model         |
| trajectory       | $\tau$                      | ordered sequence         |
| expectations     | $E()$                       | try multi times for mean |
| $\theta$          | policy parameters                 | LLM parameters                            |
| $\phi$            | value/critic parameters           | value model parameters                    |
| $\psi$            | reward model parameters (frozen)  | reward model parameters (frozen)          |

**trajectory**

A trajectory is an ordered sequence produced by interacting with the environment using a policy:

$$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T)$$

### principles

**policy**
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

$\gamma\in{(0,1]}$ is the discount factor

**Policy Gradient (the core idea)**
$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot Q^\pi(s_t, a_t) \right]$$

where:

* $Q^{\pi}(s,a)$: expected future reward after taking action a in step s.

aim: Increase the probability of actions that lead to higher future reward.

**advatange**
why: Using Q directly is noisy.

So we subtract a baseline:
$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

Where: $$V^\pi(s) = \mathbb{E}[Q^\pi(s, a)]$$

Updated gradient:
$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A_t \right]$$

**Importance sampling ratio (key PPO idea)**

We collect data using an old policy:
$$\pi_{\theta_{\text{old}}}$$

But we update a new policy:
$\pi_\theta$

To reuse old data, we define the importance sampling ratio:
$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

Where:

- $r_t(\theta)$: probability ratio for each candidate corpus
- $a_t$: action actually taken
- $s_t$: state at time $t$

**Interpretation:**

- $r_t = 1$: no change
- $r_t > 1$: action more likely to happen now
- $r_t < 1$: action less likely now

---

### 6. PPO’s unclipped objective (what we would like)
$L_{PG}(\theta) = \mathbb{E}\left[ r_t(\theta) \cdot A_t \right]$

But this allows:

- Extremely large updates,
- Unstable learning.

---

### 7. PPO’s clipped surrogate objective

This is PPO.
$L_{CLIP}(\theta) = \mathbb{E}\left[\min \left( r_t(\theta)A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t \right)\right]$

**Every variable explained:**

- $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$: how much the probability of the action changed.
- $A_t$: advantage at time $t$ ($A_t = Q(s_t, a_t) - V(s_t)$, usually estimated via GAE)
- $\epsilon$: small constant (e.g., 0.2), defining how far the policy can change.
- $\text{clip}(x, 1-\epsilon, 1+\epsilon)$:
  $
  \text{clip}(x, 1-\epsilon, 1+\epsilon) = 
  \begin{cases}
      1 - \epsilon & \text{if } x < 1 - \epsilon \\
      x & \text{if } 1-\epsilon \le x \le 1+\epsilon \\
      1 + \epsilon & \text{if } x > 1 + \epsilon
  \end{cases}

  $


### 8. Value function loss (critic)

We also train a value function:
$$L_V(\theta) = \mathbb{E} \left[ \left( V_\theta(s_t) - R_t \right)^2 \right]$$

Where:

- $R_t$ = return (target future reward)

aim:
we want to train the critic model to represent the expected future reward, so train it for better fitting. Only the param of the critic model is changed, the LLM param is not.

---

### 9. Entropy bonus (exploration)

Encourages exploration:
$$L_{ENT}(\theta) = \mathbb{E}\left[ H(\pi_\theta(\cdot \mid s_t)) \right]$$

* Encourages exploration / diversity
* Depends only on the policy distribution
* Updates only $\theta$.

---

### 10. Final PPO objective

These terms are combined in the final PPO objective (the one we maximize):
$$L_{PPO}(\theta) = L_{CLIP}(\theta) - c_1 L_V(\theta) + c_2 L_{ENT}(\theta)$$

Where:

- $c_1$: value loss coefficient
- $c_2$: entropy coefficient

## PPO training

## 1. The Full PPO Pipeline (Bird’s-Eye View)

In LLM post-training, PPO is not a single training run. It is the final stage of a multi-stage pipeline:

1. **Pretraining:** Next-token prediction
2. **Supervised Fine-Tuning (SFT)**
3. **Reward Model Training**
4. **PPO (Actor–Critic) Training**

Each stage initializes the next.

---

## 2. Initialization (This Is Critical)

### 2.1 Policy (LLM) Initialization

- The PPO policy model is initialized from an SFT checkpoint.

**Why?**
- SFT ensures:
  - Language fluency
  - Instruction following
- PPO is *not* good at learning language from scratch.

**Formally:**  
$$
\pi_{\theta_0} = \pi_{\mathrm{SFT}}
$$

---

### 2.2 Reference Model Initialization

- A frozen copy of the SFT model is created:

$$
\pi_\text{ref} = \mathrm{stop\_grad}(\pi_\text{SFT})
$$

**Purpose:**
- KL penalty anchor
- Prevents catastrophic drift

**This model:**
- Is never trained
- Exists only to compute KL divergence

---

### 2.3 Critic (Value Model) Initialization

Two common strategies:

**Option A (most common in VERL-style setups):**
- Initialize critic from the policy backbone
- Add a scalar value head
- Randomly initialize only the head

  *Why?*
  - Shared representation of text
  - Faster convergence

**Option B:**
- Train critic from scratch
- Slower, less stable

> So initial critic predictions are noisy and biased, but quickly corrected.

---

### 2.4 Reward Model Initialization

- The reward model is trained before PPO, then frozen.
- Trained using preference data:  
  \((x, y_1, y_2, label)\)

**Objective:**  
$$
\log \sigma(r(y_1) - r(y_2))
$$

After training:
- Reward model is fixed
- No gradients during PPO

---

## 3. Warm-Up Phase
*(Often skipped in theory, essential in practice)*

At the beginning of PPO:

**Problem:**
- Critic is inaccurate
- Advantages are noisy
- Policy gradients unstable

**Solutions (in practice):**
- Small learning rates
- Strong KL penalty
- Short rollouts
- High entropy coefficient
- Value loss warm-up

In VERL, this is handled via conservative defaults and gradual rollout accumulation.

---

## 4. PPO Training Loop (What Happens Every Iteration)

Each PPO iteration has four phases:

### 4.1 Rollout Phase (Data Collection)
- Sample trajectories using current policy

  - *PPO:* 1 sample per prompt
  - *GRPO:* K samples per prompt

- Data stored:
  - Tokens
  - Log-probs under policy
  - Value predictions
  - Reward scores

---

### 4.2 Reward Computation
- Pass completed outputs to reward model
- Get scalar reward \( r \)
- Assign reward to final token (LLM case)

  *Reward model is frozen; no gradient.*

---

### 4.3 Advantage & Return Computation

- Compute TD errors
- Use GAE (Generalized Advantage Estimation)
- Produce:
  - Advantages (policy)
  - Returns (critic targets)

*(This is where critic predictions are used.)*

---

### 4.4 Optimization Phase (Where Stability Matters)

- **Policy update:**
  - PPO clipped objective
  - KL penalty vs reference
  - Entropy bonus
- **Critic update:**
  - MSE value loss
  - Optional value clipping

**Important:**  
- Same data reused for multiple epochs  
- Mini-batch SGD  
- Prevents overfitting to noise

---

## 5. How Stability Is Enforced (“Safety Rails”)

PPO stability comes from multiple independent constraints:

### 5.1 PPO Clipping

- Limits:
  $$
  \frac{\pi_\text{new}}{\pi_\text{old}} \in [1 - \epsilon, 1 + \epsilon]
  $$
- Prevents huge policy jumps

---

### 5.2 KL Penalty vs Reference Model

- $$
  L_{\text{KL}} = \beta \cdot \mathrm{KL}(\pi_\theta \parallel \pi_\text{ref})
  $$
- Prevents drift away from SFT behavior
- Especially important early

---

### 5.3 Value Loss Coefficient

- Controls how fast the critic learns.  
  - Too high: critic dominates  
  - Too low: noisy advantages

---

### 5.4 Advantage Normalization

- Normalize advantages to mean 0, std 1
- Stabilizes gradients
- Makes learning rate more robust

---

### 5.5 Gradient Clipping

- Limits gradient norm

---

## 6. How the Critic “Learns to Work”

**Early:**  
- Predictions bad  
- Advantages noisy  

**Over iterations:**  
- Critic sees many (state, return) pairs  
- Regression converges toward expectation

**This improves:**  
- Advantage quality  
- Policy update quality  

*Positive feedback loop:*  
better critic → better policy updates → better data → better critic

---

## 7. How the Reward Model “Learns to Work”

> **Important:**  
> The reward model does **NOT** learn during PPO.

- It must already be:
  - Reasonably aligned
  - Calibrated

If it is bad:
- PPO will optimize the wrong thing
- No amount of clipping fixes that

*This is why reward modeling is treated as a separate, careful stage.*

---

## 8. GRPO Contrast (To Highlight What PPO Adds)

| Aspect               | PPO      | GRPO     |
|----------------------|----------|----------|
| Critic               | Required | None     |
| Value loss           | Yes      | No       |
| Warm-up sensitivity  | High     | Lower    |
| Sample efficiency    | Higher   | Lower    |
| Config complexity    | Higher   | Lower    |

**GRPO trades:**  
- Stability from learned expectation  
  for  
- Stability from comparison

---

## 9. How VERL Assumes This Pipeline

VERL does **not** train:
- SFT
- Reward model

VERL assumes:
- SFT checkpoint exists
- Reward function exists

VERL implements:
- PPO / GRPO stage only

---

## 10. One Mental Picture (Keep This)

PPO is a carefully staged system:
- Start from a good language model,
- Anchor it with a frozen reference,
- Guide it with a fixed reward model,
- Stabilize updates with a learned critic,
- And limit every step with clipping and KL.