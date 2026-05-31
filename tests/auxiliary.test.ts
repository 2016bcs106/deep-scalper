import { describe, it, expect } from 'vitest';
import tf from '../src/tf.ts';
import { VolatilityPredictor } from '../src/models/auxiliary.js';

describe('VolatilityPredictor', () => {
  it('should produce predictions of shape [batch, 1]', () => {
    const predictor = new VolatilityPredictor(128);
    const input = tf.randomNormal([4, 128]);

    const output = predictor.predict(input);

    expect(output.shape).toEqual([4, 1]);
    input.dispose();
    output.dispose();
  });

  it('should compute a scalar MSE loss', () => {
    const predictor = new VolatilityPredictor(64);
    const embedding = tf.randomNormal([8, 64]);
    const actual = tf.randomUniform([8, 1]);

    const loss = predictor.loss(embedding, actual);

    expect(loss.shape).toEqual([]);
    expect(loss.dataSync()[0]).toBeGreaterThan(0);

    embedding.dispose();
    actual.dispose();
    loss.dispose();
  });

  describe('computeVolatility', () => {
    it('should return 0 volatility for constant prices', () => {
      const prices = [100, 100, 100, 100, 100];
      const vols = VolatilityPredictor.computeVolatility(prices, 3);

      // All returns are 0, so variance = 0
      vols.forEach(v => expect(v).toBeCloseTo(0));
    });

    it('should return higher volatility for more erratic prices', () => {
      const steady = [100, 101, 102, 103, 104];
      const erratic = [100, 110, 90, 115, 85];

      const volSteady = VolatilityPredictor.computeVolatility(steady, 5);
      const volErratic = VolatilityPredictor.computeVolatility(erratic, 5);

      // Last value should show erratic has higher vol
      const lastSteady = volSteady[volSteady.length - 1];
      const lastErratic = volErratic[volErratic.length - 1];
      expect(lastErratic).toBeGreaterThan(lastSteady);
    });

    it('should return one value per price point', () => {
      const prices = [100, 101, 102, 103, 104, 105];
      const vols = VolatilityPredictor.computeVolatility(prices, 3);
      expect(vols).toHaveLength(6);
    });

    it('should handle single-bar edge case', () => {
      const vols = VolatilityPredictor.computeVolatility([100], 5);
      expect(vols).toEqual([0]);
    });

    it('should use window correctly', () => {
      // Window of 3: only looks at last 3 prices
      const prices = [100, 200, 100, 100, 100, 100];
      const vols = VolatilityPredictor.computeVolatility(prices, 3);

      // Last 3 prices [100,100,100] -> returns [0,0] -> vol=0
      expect(vols[5]).toBeCloseTo(0);
      // But earlier when window includes the spike, vol > 0
      expect(vols[2]).toBeGreaterThan(0);
    });
  });
});
