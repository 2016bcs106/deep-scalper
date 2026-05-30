import { describe, it, expect } from 'vitest';
import { PrioritizedReplayBuffer, Transition } from '../src/training/replay-buffer.js';

function makeTransition(reward: number): Transition {
  return {
    state: [1, 2, 3],
    action: { priceIndex: 0, quantityIndex: 2 },
    reward,
    nextState: [4, 5, 6],
    done: false,
  };
}

describe('PrioritizedReplayBuffer', () => {
  it('should store and retrieve transitions', () => {
    const buffer = new PrioritizedReplayBuffer(100);

    buffer.push(makeTransition(1.0));
    buffer.push(makeTransition(2.0));

    expect(buffer.size).toBe(2);
  });

  it('should evict oldest transitions when at capacity', () => {
    const buffer = new PrioritizedReplayBuffer(3);

    buffer.push(makeTransition(1));
    buffer.push(makeTransition(2));
    buffer.push(makeTransition(3));
    buffer.push(makeTransition(4)); // evicts transition with reward=1

    expect(buffer.size).toBe(3);

    const { transitions } = buffer.sample(3);
    const rewards = transitions.map(t => t.reward);
    expect(rewards).not.toContain(1); // oldest was evicted
  });

  it('should throw when sampling more than available', () => {
    const buffer = new PrioritizedReplayBuffer(100);
    buffer.push(makeTransition(1));

    expect(() => buffer.sample(5)).toThrow();
  });

  it('should return correct batch size on sample', () => {
    const buffer = new PrioritizedReplayBuffer(100);
    for (let i = 0; i < 20; i++) {
      buffer.push(makeTransition(i));
    }

    const { transitions, indices, weights } = buffer.sample(8);

    expect(transitions).toHaveLength(8);
    expect(indices).toHaveLength(8);
    expect(weights).toHaveLength(8);
  });

  it('should return importance sampling weights in [0, 1]', () => {
    const buffer = new PrioritizedReplayBuffer(100);
    for (let i = 0; i < 50; i++) {
      buffer.push(makeTransition(i));
    }

    const { weights } = buffer.sample(10);

    for (const w of weights) {
      expect(w).toBeGreaterThan(0);
      expect(w).toBeLessThanOrEqual(1.0 + 1e-6);
    }
  });

  it('should bias sampling toward higher priority transitions', () => {
    const buffer = new PrioritizedReplayBuffer(100, 1.0); // full prioritization
    for (let i = 0; i < 10; i++) {
      buffer.push(makeTransition(i));
    }

    // Set one transition to very high priority
    buffer.updatePriorities([5], [100.0]);

    // Sample many times and count how often index 5 appears
    let count = 0;
    for (let trial = 0; trial < 200; trial++) {
      const { indices } = buffer.sample(1);
      if (indices[0] === 5) count++;
    }

    // With full prioritization and priority=100 vs others ~1, should appear often
    expect(count).toBeGreaterThan(50);
  });

  it('should update priorities correctly', () => {
    const buffer = new PrioritizedReplayBuffer(100);
    for (let i = 0; i < 5; i++) {
      buffer.push(makeTransition(i));
    }

    // All start with same priority, update one to be very high
    buffer.updatePriorities([0, 1], [10.0, 0.001]);

    // Index 0 should now be sampled more than index 1
    let count0 = 0;
    let count1 = 0;
    for (let trial = 0; trial < 500; trial++) {
      const { indices } = buffer.sample(1);
      if (indices[0] === 0) count0++;
      if (indices[0] === 1) count1++;
    }

    expect(count0).toBeGreaterThan(count1);
  });
});
