"""FAGT policy and PPO training with FRDT and ASER."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from FRDT import FRDT
from ASER import ASER, structural_kl
from env.fjsp_env import FJSPEnv


def mlp(input_dim, hidden_dim, output_dim, layers=3):
    if min(input_dim, hidden_dim, output_dim, layers) < 1:
        raise ValueError('MLP dimensions and layer count must be positive.')
    modules = []
    for index in range(layers - 1):
        modules.extend((nn.Linear(input_dim if index == 0 else hidden_dim, hidden_dim), nn.Tanh()))
    modules.append(nn.Linear(input_dim if layers == 1 else hidden_dim, output_dim))
    return nn.Sequential(*modules)


class FAGT(nn.Module):
    def __init__(self, model_params=None):
        super().__init__()
        p = dict(model_params or {})
        actor_hidden = p.pop('actor_hidden_dim', 64)
        critic_hidden = p.pop('critic_hidden_dim', 64)
        actor_layers = p.pop('actor_layers', 3)
        critic_layers = p.pop('critic_layers', 3)
        self.detach_value_encoder = p.pop('detach_value_encoder', True)
        self.encoder = FRDT(**p)
        dim = p.get('embedding_dim', 128)
        self.actor = mlp(3 * dim, actor_hidden, 1, actor_layers)
        self.critic = mlp(2 * dim, critic_hidden, 1, critic_layers)

    def forward(self, obs):
        operation, machine, edge = self.encoder(obs)
        mask = obs['eligible']
        if not mask.flatten(1).any(-1).all():
            raise ValueError('Policy observations must contain an eligible action in every row.')
        b, o, m = mask.nonzero(as_tuple=True)
        features = torch.cat((operation[b, o], machine[b, m], edge[b, o, m]), -1)
        logits = operation.new_full(mask.shape, -torch.inf)
        logits[b, o, m] = self.actor(features).squeeze(-1)
        op_global = (operation * obs['valid'][..., None]).sum(1) / obs['valid'].sum(1, keepdim=True).clamp_min(1)
        connected = (obs['duration'] > 0).any(1)
        ma_global = (machine * connected[..., None]).sum(1) / connected.sum(1, keepdim=True).clamp_min(1)
        # Long-horizon value errors must not steer the policy's shared encoder.
        # The critic learns its baseline on current FRDT features with its own head.
        if self.detach_value_encoder:
            op_global, ma_global = op_global.detach(), ma_global.detach()
        value = self.critic(torch.cat((op_global, ma_global), -1)).squeeze(-1)
        return Categorical(logits=logits.flatten(1)), value

    @torch.no_grad()
    def act(self, obs, greedy=False):
        dist, value = self(obs)
        action = dist.probs.argmax(-1) if greedy else dist.sample()
        return action, dist.log_prob(action), value


OBS_KEYS = ('duration', 'minimum', 'o_at', 'm_at', 'job', 'position', 'valid', 'pending', 'eligible')


def collate_records(records, refs, device):
    """Pad only the remaining-operation dimension; action indices stay unchanged."""
    if not refs:
        raise ValueError('Cannot collate an empty minibatch.')
    max_o = max(records[s]['obs']['valid'].shape[1] for s, _ in refs)
    obs = {}
    for key in OBS_KEYS:
        rows = []
        for step, row in refs:
            value = records[step]['obs'][key][row]
            if key != 'm_at':
                pad = max_o - value.shape[0]
                if pad:
                    value = torch.cat((value, value.new_zeros((pad,) + value.shape[1:])), 0)
            rows.append(value)
        obs[key] = torch.stack(rows).to(device)
    data = {key: torch.stack([records[s][key][r] for s, r in refs]).to(device)
            for key in ('action', 'logprob', 'value', 'advantage', 'return')}
    return obs, data


def compute_gae(records, batch_size, gamma=1.0, gae_lambda=0.98):
    """GAE with fixed global advantage normalization over the complete rollout."""
    if not records or not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError('GAE requires records, gamma in [0,1], and lambda in [0,1].')
    next_value = torch.zeros(batch_size)
    next_gae = torch.zeros(batch_size)
    for record in reversed(records):
        b = record['batch']
        alive = (~record['done']).float()
        delta = record['reward'] + gamma * next_value[b] * alive - record['value']
        advantage = delta + gamma * gae_lambda * next_gae[b] * alive
        record['advantage'] = advantage
        record['return'] = advantage + record['value']
        next_value[b], next_gae[b] = record['value'], advantage
    all_adv = torch.cat([r['advantage'] for r in records])
    if len(all_adv) > 1:
        mean = all_adv.mean()
        std = all_adv.std(unbiased=False).clamp_min(1e-8)
        for record in records:
            record['advantage'] = (record['advantage'] - mean) / std


class PPOTrainer:
    def __init__(self, model, params):
        self.model, self.params = model, dict(params)
        self.device = next(model.parameters()).device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=params.get('lr', 5e-5),
                                          betas=tuple(params.get('betas', (0.9, 0.999))))
        self.entropy_mode = params.get('entropy_mode', 'structural_kl')
        if self.entropy_mode not in ('structural_kl', 'shannon'):
            raise ValueError('entropy_mode must be structural_kl or shannon.')
        self.aser = (ASER(coefficient=params.get('entropy_coeff', 0.01),
                          target=params.get('kl_target', 0.5), device=self.device)
                     if self.entropy_mode == 'structural_kl' else None)
        if params.get('entropy_coeff', 0.01) < 0:
            raise ValueError('entropy_coeff cannot be negative.')
        self.updates = 0

    @torch.no_grad()
    def collect(self, instances):
        self.model.eval()
        env = FJSPEnv(instances, self.device)
        records = []
        while not env.done.all():
            obs = env.observe()
            dist, value = self.model(obs)
            action = dist.sample()
            statistic, choice = structural_kl(dist.probs, obs['eligible'])
            reward, done = env.step(action, obs)
            records.append({
                'obs': {k: obs[k].detach().cpu().clone() for k in OBS_KEYS},
                'batch': obs['batch'].cpu().clone(), 'action': action.cpu(),
                'logprob': dist.log_prob(action).cpu(), 'value': value.cpu(),
                'reward': reward.cpu(), 'done': done.cpu(),
                'structural_kl': statistic.cpu(), 'choice': choice.cpu(),
            })
        compute_gae(records, len(instances), self.params.get('gamma', 1.0),
                    self.params.get('gae_lambda', 0.98))
        return records, env.makespan.mean().item()

    def update(self, records, batch_size):
        self.model.train()
        p = self.params
        minibatch = p.get('minibatch_size', 64)
        epochs = p.get('K_epochs', 3)
        eps = p.get('clip_range', 0.2)
        if min(minibatch, epochs) < 1 or not 0 < eps < 1:
            raise ValueError('Invalid PPO batch sizes, epoch count, or clip range.')

        by_instance = [[] for _ in range(batch_size)]
        for step, record in enumerate(records):
            for row, index in enumerate(record['batch'].tolist()):
                by_instance[index].append((step, row))
        if not all(by_instance):
            raise ValueError('Each training instance must contribute a complete rollout.')

        totals = dict(policy=0., value=0., exploration=0., loss=0.)
        count = 0

        for _ in range(epochs):
            permutation = torch.randperm(batch_size).tolist()
            refs = [ref for i in permutation for ref in by_instance[i]]

            # Every transition has the same weight in the accumulated gradient.
            refs.sort(key=lambda ref: records[ref[0]]['obs']['valid'].shape[1])
            chunks = [
                refs[i:i + minibatch]
                for i in range(0, len(refs), minibatch)
            ]

            self.optimizer.zero_grad(set_to_none=True)
            for chunk_index in torch.randperm(len(chunks)).tolist():
                chunk = chunks[chunk_index]
                obs, data = collate_records(records, chunk, self.device)
                dist, value = self.model(obs)

                ratio = (dist.log_prob(data['action']) - data['logprob']).exp()
                policy_loss = -torch.minimum(
                    ratio * data['advantage'],
                    ratio.clamp(1 - eps, 1 + eps) * data['advantage']
                ).mean()

                # Fixed SmoothL1 value loss (beta=1); no value clipping.
                value_loss = F.smooth_l1_loss(value, data['return'], beta=1.0)

                if self.aser is not None:
                    regularizer, statistic, _ = self.aser.penalty(
                        dist.probs, obs['eligible']
                    )
                else:
                    statistic = dist.entropy()
                    regularizer = -p.get('entropy_coeff', 0.01) * statistic.mean()

                loss = (
                    p.get('policy_coef', 1.0) * policy_loss
                    + p.get('vf_coef', 0.5) * value_loss
                    + regularizer
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite PPO loss.')

                (loss * (len(chunk) / len(refs))).backward()

                n = len(chunk)
                for key, val in (
                    ('policy', policy_loss),
                    ('value', value_loss),
                    ('exploration', statistic.mean()),
                    ('loss', loss)
                ):
                    totals[key] += float(val.detach()) * n
                count += n

            grad = nn.utils.clip_grad_norm_(
                self.model.parameters(), p.get('max_grad_norm', 1.0)
            )
            if not torch.isfinite(grad):
                raise FloatingPointError('Nonfinite PPO gradient.')
            self.optimizer.step()

        # ASER updates only its exploration coefficient, once per rollout.
        choice_count = sum(int(r['choice'].sum()) for r in records)
        rollout_kl = (
            sum(float(r['structural_kl'][r['choice']].sum()) for r in records)
            / choice_count
        ) if choice_count else 0.

        if self.aser is not None and choice_count:
            self.aser.update(rollout_kl)

        self.updates += 1
        metrics = {key: value / count for key, value in totals.items()}
        metrics['structural_kl'] = rollout_kl
        return metrics

    def train_batch(self, instances):
        records, score = self.collect(instances)
        metrics = self.update(records, len(instances))
        metrics['sample_makespan'] = score
        return metrics
