import { describe, it, expect, beforeEach } from 'vitest';
import { MetricsCalculator } from '../src/evaluation/metrics.js';

describe('MetricsCalculator', () => {
  let calc: MetricsCalculator;

  beforeEach(() => {
    calc = new MetricsCalculator();
  });

  describe('totalReturn', () => {
    it('should be positive when portfolio grows', () => {
      expect(calc.totalReturn([1.0, 1.05, 1.10])).toBeCloseTo(0.10);
    });

    it('should be negative when portfolio shrinks', () => {
      expect(calc.totalReturn([1.0, 0.95, 0.90])).toBeCloseTo(-0.10);
    });

    it('should be zero for flat portfolio', () => {
      expect(calc.totalReturn([1.0, 1.0, 1.0])).toBeCloseTo(0);
    });
  });

  describe('sharpeRatio', () => {
    it('should be positive for consistently positive returns', () => {
      const returns = [0.01, 0.02, 0.01, 0.03, 0.01];
      expect(calc.sharpeRatio(returns)).toBeGreaterThan(0);
    });

    it('should be zero for zero returns', () => {
      expect(calc.sharpeRatio([0, 0, 0])).toBe(0);
    });

    it('should be higher for less volatile positive returns', () => {
      const lessVol = [0.01, 0.015, 0.01, 0.012]; // low volatility
      const moreVol = [0.04, -0.02, 0.03, -0.01]; // higher volatility
      expect(calc.sharpeRatio(lessVol)).toBeGreaterThan(calc.sharpeRatio(moreVol));
    });
  });

  describe('maxDrawdown', () => {
    it('should be zero for monotonically increasing values', () => {
      expect(calc.maxDrawdown([1.0, 1.1, 1.2, 1.3])).toBe(0);
    });

    it('should capture the largest peak-to-trough decline', () => {
      // Peak at 1.2, trough at 0.9 -> drawdown = (1.2-0.9)/1.2 = 0.25
      const mdd = calc.maxDrawdown([1.0, 1.2, 1.0, 0.9, 1.1]);
      expect(mdd).toBeCloseTo(0.25);
    });

    it('should handle multiple drawdowns and pick the worst', () => {
      // First dip: 1.1->1.0 (9.09%), second dip: 1.3->1.0 (23.08%)
      const mdd = calc.maxDrawdown([1.0, 1.1, 1.0, 1.3, 1.0]);
      expect(mdd).toBeCloseTo((1.3 - 1.0) / 1.3);
    });
  });

  describe('sortinoRatio', () => {
    it('should be higher than sharpe when most volatility is upside', () => {
      // Big positive returns, small negatives
      const returns = [0.05, 0.04, -0.01, 0.06, 0.03];
      const sortino = calc.sortinoRatio(returns);
      const sharpe = calc.sharpeRatio(returns);
      expect(sortino).toBeGreaterThan(sharpe);
    });

    it('should return Infinity for all-positive returns', () => {
      const returns = [0.01, 0.02, 0.03];
      expect(calc.sortinoRatio(returns)).toBe(Infinity);
    });
  });

  describe('evaluate', () => {
    it('should return all metrics at once', () => {
      const netValues = [1.0, 1.02, 1.01, 1.05, 1.03, 1.08];
      const result = calc.evaluate(netValues);

      expect(result.totalReturn).toBeCloseTo(0.08);
      expect(result.sharpeRatio).toBeGreaterThan(0);
      expect(result.maxDrawdown).toBeGreaterThan(0);
      expect(result.calmarRatio).not.toBe(0);
      expect(result.sortinoRatio).toBeGreaterThan(0);
    });

    it('should handle edge case with fewer than 2 values', () => {
      const result = calc.evaluate([1.0]);
      expect(result.totalReturn).toBe(0);
      expect(result.sharpeRatio).toBe(0);
    });
  });
});
