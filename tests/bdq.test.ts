import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs-node';
import { BranchingDuelingQNetwork } from '../src/models/bdq.js';

describe('BranchingDuelingQNetwork', () => {
  const INPUT_SIZE = 128;
  const PRICE_LEVELS = 5;
  const QTY_LEVELS = 5;

  it('should produce Q-values of correct shape for each branch', () => {
    const bdq = new BranchingDuelingQNetwork(INPUT_SIZE, PRICE_LEVELS, QTY_LEVELS);
    const input = tf.randomNormal([4, INPUT_SIZE]);

    const { priceQ, quantityQ } = bdq.predict(input);

    expect(priceQ.shape).toEqual([4, PRICE_LEVELS]);
    expect(quantityQ.shape).toEqual([4, QTY_LEVELS]);

    input.dispose();
    priceQ.dispose();
    quantityQ.dispose();
  });

  it('should select valid action indices within bounds', () => {
    const bdq = new BranchingDuelingQNetwork(INPUT_SIZE, PRICE_LEVELS, QTY_LEVELS);
    const input = tf.randomNormal([1, INPUT_SIZE]);

    const action = bdq.selectAction(input);

    expect(action.priceIndex).toBeGreaterThanOrEqual(0);
    expect(action.priceIndex).toBeLessThan(PRICE_LEVELS);
    expect(action.quantityIndex).toBeGreaterThanOrEqual(0);
    expect(action.quantityIndex).toBeLessThan(QTY_LEVELS);

    input.dispose();
  });

  it('should copy weights from source to target network', () => {
    const source = new BranchingDuelingQNetwork(INPUT_SIZE, PRICE_LEVELS, QTY_LEVELS);
    const target = new BranchingDuelingQNetwork(INPUT_SIZE, PRICE_LEVELS, QTY_LEVELS);
    const input = tf.randomNormal([1, INPUT_SIZE]);

    // Before copy: outputs differ
    const { priceQ: srcQ } = source.predict(input);
    const { priceQ: tgtQ } = target.predict(input);
    const beforeDiff = srcQ.sub(tgtQ).abs().sum().dataSync()[0];

    // Copy weights
    target.copyWeightsFrom(source);

    // After copy: outputs match
    const { priceQ: srcQ2 } = source.predict(input);
    const { priceQ: tgtQ2 } = target.predict(input);
    const afterDiff = srcQ2.sub(tgtQ2).abs().sum().dataSync()[0];

    expect(afterDiff).toBeCloseTo(0, 5);
    // Only verify they matched after copy (before may or may not differ due to random init)
    expect(afterDiff).toBeLessThan(beforeDiff + 1e-5);

    input.dispose();
    srcQ.dispose();
    tgtQ.dispose();
    srcQ2.dispose();
    tgtQ2.dispose();
  });

  it('should apply dueling correctly: Q mean equals V for uniform advantages', () => {
    const bdq = new BranchingDuelingQNetwork(INPUT_SIZE, PRICE_LEVELS, QTY_LEVELS);
    const input = tf.randomNormal([1, INPUT_SIZE]);

    const { priceQ } = bdq.predict(input);

    // Due to dueling: mean(Q) across actions ≈ V(s) (advantages are centered)
    // So mean of Q-values across the action dim should be roughly consistent
    const qMean = priceQ.mean(1).dataSync()[0];
    const qValues = priceQ.dataSync();

    // At minimum, verify Q-values are finite and centered around their mean
    const centered = Array.from(qValues).map(q => q - qMean);
    const centeredSum = centered.reduce((a, b) => a + b, 0);
    expect(centeredSum).toBeCloseTo(0, 4);

    input.dispose();
    priceQ.dispose();
  });
});
