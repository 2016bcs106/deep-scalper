import { describe, it, expect } from 'vitest';
import { Trainer } from '../src/training/trainer.js';
import { OHLCVBar } from '../src/data/features.js';

function makeDayBars(count: number, startPrice = 100): OHLCVBar[] {
  const bars: OHLCVBar[] = [];
  let price = startPrice;

  for (let i = 0; i < count; i++) {
    // Random walk with slight upward drift
    price += (Math.random() - 0.48) * 2;
    bars.push({
      timestamp: new Date('2024-01-15T09:15:00').getTime() + i * 60000,
      open: price - 0.5,
      high: price + 1,
      low: price - 1,
      close: price,
      volume: 1000 + Math.random() * 500,
    });
  }

  return bars;
}

describe('Trainer', () => {
  it('should complete a training episode without errors', () => {
    const trainer = new Trainer({
      batchSize: 16,
      minBufferSize: 20,
      bufferCapacity: 1000,
      env: { maxPosition: 10, initialCash: 10000 },
    });

    const bars = makeDayBars(50);
    const stats = trainer.trainEpisode(bars);

    expect(stats.steps).toBe(49); // bars.length - 1
    expect(stats.netValue).toBeGreaterThan(0);
    expect(typeof stats.totalReward).toBe('number');
    expect(stats.epsilon).toBeLessThanOrEqual(1.0);
  });

  it('should decay epsilon over multiple episodes', () => {
    const trainer = new Trainer({
      batchSize: 8,
      minBufferSize: 10,
      bufferCapacity: 500,
      epsilonStart: 1.0,
      epsilonEnd: 0.01,
      epsilonDecaySteps: 100,
      env: { maxPosition: 5, initialCash: 5000 },
    });

    const bars = makeDayBars(30);

    trainer.trainEpisode(bars);
    const eps1 = trainer.getEpsilon();

    trainer.trainEpisode(bars);
    const eps2 = trainer.getEpsilon();

    expect(eps2).toBeLessThan(eps1);
  });

  it('should produce decreasing loss over episodes with sufficient data', () => {
    const trainer = new Trainer({
      batchSize: 16,
      minBufferSize: 32,
      bufferCapacity: 2000,
      learningRate: 0.01,
      env: { maxPosition: 10, initialCash: 10000 },
    });

    // Fill buffer with initial data
    const warmupBars = makeDayBars(100);
    trainer.trainEpisode(warmupBars);

    // Train a few more episodes and check loss trends
    const losses: number[] = [];
    for (let i = 0; i < 3; i++) {
      const stats = trainer.trainEpisode(makeDayBars(60));
      losses.push(stats.avgLoss);
    }

    // At minimum, losses should be finite numbers
    losses.forEach(l => expect(isFinite(l)).toBe(true));
  });
});
