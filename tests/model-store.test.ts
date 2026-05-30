import { describe, it, expect, afterEach } from 'vitest';
import { rmSync } from 'fs';
import * as tf from '@tensorflow/tfjs-node';
import { Trainer } from '../src/training/trainer.ts';
import { OHLCVBar } from '../src/data/features.ts';

const TEST_MODEL_DIR = 'models/TEST_SYMBOL';

function makeDayBars(count: number, startPrice = 100): OHLCVBar[] {
  const bars: OHLCVBar[] = [];
  let price = startPrice;

  for (let i = 0; i < count; i++) {
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

afterEach(() => {
  try { rmSync(TEST_MODEL_DIR, { recursive: true }); } catch {}
});

describe('Model save/load', () => {
  it('should save and load a model preserving predictions', async () => {
    const trainer = new Trainer({
      batchSize: 8,
      minBufferSize: 10,
      bufferCapacity: 500,
      env: { maxPosition: 5, initialCash: 5000 },
    });

    // Train a bit to get non-random weights
    trainer.trainEpisode(makeDayBars(30));

    // Save
    const dir = await trainer.saveModel('TEST_SYMBOL', 1, 1);
    expect(dir).toContain('TEST_SYMBOL');

    // Get prediction from original
    const testInput = tf.randomNormal([1, 19]);
    const originalAction = trainer.getQNetwork().selectAction(testInput);

    // Create a new trainer and load the saved model
    const trainer2 = new Trainer({
      batchSize: 8,
      minBufferSize: 10,
      bufferCapacity: 500,
      env: { maxPosition: 5, initialCash: 5000 },
    });

    const metadata = await trainer2.loadModel('TEST_SYMBOL');
    expect(metadata.symbol).toBe('TEST_SYMBOL');
    expect(metadata.epochs).toBe(1);

    // Loaded model should produce same predictions
    const loadedAction = trainer2.getQNetwork().selectAction(testInput);
    expect(loadedAction.priceIndex).toBe(originalAction.priceIndex);
    expect(loadedAction.quantityIndex).toBe(originalAction.quantityIndex);

    testInput.dispose();
  });
});
