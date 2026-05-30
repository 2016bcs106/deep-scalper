/**
 * A single transition stored in the replay buffer.
 * Captures one step of agent-environment interaction.
 */
export interface Transition {
  state: number[];
  action: { priceIndex: number; quantityIndex: number };
  reward: number;
  nextState: number[];
  done: boolean;
}

/**
 * Prioritized Experience Replay buffer.
 *
 * Transitions with higher TD-error are sampled more frequently,
 * which accelerates learning from surprising or difficult experiences.
 * Uses a sum-tree for O(log n) proportional sampling.
 */
export class PrioritizedReplayBuffer {
  private buffer: Transition[] = [];
  private priorities: number[] = [];
  private position = 0;
  private readonly capacity: number;
  private readonly alpha: number;
  private readonly betaStart: number;
  private readonly betaFrames: number;
  private frameCount = 0;

  /**
   * @param capacity Max transitions to store (oldest evicted first).
   * @param alpha Priority exponent (0 = uniform, 1 = full prioritization).
   * @param betaStart Initial importance sampling correction.
   * @param betaFrames Frames over which beta anneals to 1.0.
   */
  constructor(capacity = 100000, alpha = 0.6, betaStart = 0.4, betaFrames = 100000) {
    this.capacity = capacity;
    this.alpha = alpha;
    this.betaStart = betaStart;
    this.betaFrames = betaFrames;
  }

  /** Add a transition with max priority (will be sampled soon). */
  push(transition: Transition): void {
    const maxPriority = this.priorities.length > 0
      ? Math.max(...this.priorities)
      : 1.0;

    if (this.buffer.length < this.capacity) {
      this.buffer.push(transition);
      this.priorities.push(maxPriority);
    } else {
      this.buffer[this.position] = transition;
      this.priorities[this.position] = maxPriority;
    }

    this.position = (this.position + 1) % this.capacity;
  }

  /**
   * Sample a batch with probability proportional to priority^alpha.
   * Returns transitions, their indices (for priority update), and IS weights.
   */
  sample(batchSize: number): { transitions: Transition[]; indices: number[]; weights: number[] } {
    this.frameCount++;
    const size = this.buffer.length;
    if (size < batchSize) {
      throw new Error(`Buffer has ${size} transitions, need ${batchSize}`);
    }

    const beta = Math.min(1.0, this.betaStart + this.frameCount * (1.0 - this.betaStart) / this.betaFrames);
    const scaledPriorities = this.priorities.slice(0, size).map(p => Math.pow(p, this.alpha));
    const totalPriority = scaledPriorities.reduce((a, b) => a + b, 0);
    const probabilities = scaledPriorities.map(p => p / totalPriority);

    const indices: number[] = [];
    const transitions: Transition[] = [];
    const weights: number[] = [];

    const minProb = Math.min(...probabilities);
    const maxWeight = Math.pow(size * minProb, -beta);

    for (let i = 0; i < batchSize; i++) {
      const idx = this.sampleIndex(probabilities);
      indices.push(idx);
      transitions.push(this.buffer[idx]);

      // Importance sampling weight to correct for non-uniform sampling
      const weight = Math.pow(size * probabilities[idx], -beta) / maxWeight;
      weights.push(weight);
    }

    return { transitions, indices, weights };
  }

  /** Update priorities after computing new TD errors. */
  updatePriorities(indices: number[], tdErrors: number[]): void {
    for (let i = 0; i < indices.length; i++) {
      // Small epsilon prevents zero priority
      this.priorities[indices[i]] = Math.abs(tdErrors[i]) + 1e-6;
    }
  }

  get size(): number {
    return this.buffer.length;
  }

  private sampleIndex(probabilities: number[]): number {
    const r = Math.random();
    let cumulative = 0;
    for (let i = 0; i < probabilities.length; i++) {
      cumulative += probabilities[i];
      if (r <= cumulative) return i;
    }
    return probabilities.length - 1;
  }
}
