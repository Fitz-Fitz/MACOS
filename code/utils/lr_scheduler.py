import math


class IterationScheduler:
    """Per-iteration learning-rate schedule with an optional linear warm-up.

    kind='cosine':  warm-up, then cosine annealing from initial_lr to min_lr_ratio * initial_lr
    kind='poly':    warm-up, then polynomial decay to min_lr_ratio * initial_lr
    """

    def __init__(self, optimizer, initial_lr, max_iterations, warmup_iterations=0, kind='cosine', power=0.9, min_lr_ratio=0.01):
        if kind not in ('cosine', 'poly'):
            raise ValueError(f'unknown schedule {kind!r}')
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_iterations = max_iterations
        self.warmup_iterations = warmup_iterations
        self.kind = kind
        self.power = power
        self.min_lr_ratio = min_lr_ratio
        self.current_iteration = 0

    def lr_at(self, iteration):
        if iteration < self.warmup_iterations:
            return self.initial_lr * iteration / self.warmup_iterations
        progress = (iteration - self.warmup_iterations) / max(1, self.max_iterations - self.warmup_iterations)
        progress = min(progress, 1.0)
        floor = self.min_lr_ratio * self.initial_lr
        if self.kind == 'cosine':
            return floor + 0.5 * (self.initial_lr - floor) * (1 + math.cos(math.pi * progress))
        return floor + (self.initial_lr - floor) * (1 - progress) ** self.power

    def step(self):
        lr = self.lr_at(self.current_iteration)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.current_iteration += 1
        return lr

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
