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

$$
\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T)

$$

### principles

**policy**

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]

$$

$\gamma\in{(0,1]}$ is the discount factor

**Policy Gradient (the core idea)**

$$
\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot Q^\pi(s_t, a_t) \right]

$$

where:

* $Q^{\pi}(s,a)$: expected future reward after taking action a in step s.

aim: Increase the probability of actions that lead to higher future reward.

**advatange**
why: Using Q directly is noisy.

So we subtract a baseline:

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)

$$

Where: $V^\pi(s) = \mathbb{E}[Q^\pi(s, a)]$

Updated gradient:

$$
\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A_t \right]

$$

**Importance sampling ratio (key PPO idea)**

We collect data using an old policy:

$$
\pi_{\theta_{\text{old}}}

$$

But we update a new policy:

$$
\pi_\theta

$$

To reuse old data, we define the importance sampling ratio:

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}

$$

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

$$
L_{PG}(\theta) = \mathbb{E}\left[ r_t(\theta) \cdot A_t \right]

$$

But this allows:

- Extremely large updates,
- Unstable learning.

---

#### 7. PPO’s clipped surrogate objective

This is PPO.

$$
L_{CLIP}(\theta) = \mathbb{E}\left[\min \left( r_t(\theta)A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t \right)\right]

$$

**Every variable explained:**

- $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$: how much the probability of the action changed.
- $A_t$: advantage at time $t$ ($A_t = Q(s_t, a_t) - V(s_t)$, usually estimated via GAE)
- $\epsilon$: small constant (e.g., 0.2), defining how far the policy can change.
- $\text{clip}(x, 1-\epsilon, 1+\epsilon)$:
  $$
  \text{clip}(x, 1-\epsilon, 1+\epsilon) = 
  \begin{cases}
      1 - \epsilon & \text{if } x < 1 - \epsilon \\
      x & \text{if } 1-\epsilon \le x \le 1+\epsilon \\
      1 + \epsilon & \text{if } x > 1 + \epsilon
  \end{cases}

  $$


#### 8. Value function loss (critic)

We also train a value function:

$$
L_V(\theta) = \mathbb{E} \left[ \left( V_\theta(s_t) - R_t \right)^2 \right]
$$

Where:

- $R_t$ = return (target future reward)

aim:
we want to train the critic model to represent the expected future reward, so train it for better fitting. Only the param of the critic model is changed, the LLM param is not.

---

#### 9. Entropy bonus (exploration)

Encourages exploration:

$$
L_{ENT}(\theta) = \mathbb{E}\left[ H(\pi_\theta(\cdot \mid s_t)) \right]
$$

* Encourages exploration / diversity
* Depends only on the policy distribution
* Updates only $\theta$.

---

#### 10. Final PPO objective

These terms are combined in the final PPO objective (the one we maximize):

$$
L_{PPO}(\theta) = L_{CLIP}(\theta) - c_1 L_V(\theta) + c_2 L_{ENT}(\theta)
$$

Where:

- $c_1$: value loss coefficient
- $c_2$: entropy coefficient