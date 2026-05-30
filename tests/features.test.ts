import { describe, it, expect, beforeEach } from 'vitest';
import { FeatureCalculator, OHLCVBar } from '../src/data/features.js';

function makeBar(close: number, open = close, high = close, low = close, volume = 100): OHLCVBar {
  return { timestamp: Date.now(), open, high, low, close, volume };
}

describe('FeatureCalculator', () => {
  let calc: FeatureCalculator;

  beforeEach(() => {
    calc = new FeatureCalculator();
  });

  it('should throw on empty or out-of-bounds index', () => {
    expect(() => calc.computeAt(0)).toThrow(RangeError);

    calc.addBar(makeBar(100));
    expect(() => calc.computeAt(1)).toThrow(RangeError);
    expect(() => calc.computeAt(-1)).toThrow(RangeError);
  });

  it('should compute zopen, zhigh, zlow relative to close', () => {
    calc.addBar(makeBar(100, 98, 105, 95));

    const f = calc.computeAt(0);
    expect(f.zopen).toBeCloseTo(98 / 100 - 1);   // -0.02
    expect(f.zhigh).toBeCloseTo(105 / 100 - 1);  // +0.05
    expect(f.zlow).toBeCloseTo(95 / 100 - 1);    // -0.05
  });

  it('should compute zclose as return from previous bar', () => {
    calc.addBars([makeBar(100), makeBar(103)]);

    const f = calc.computeAt(1);
    expect(f.zclose).toBeCloseTo(103 / 100 - 1); // +0.03
  });

  it('should compute zclose as 0 for the first bar (no previous)', () => {
    calc.addBar(makeBar(100));

    const f = calc.computeAt(0);
    expect(f.zclose).toBeCloseTo(0);
  });

  it('should compute moving average indicators', () => {
    // 5 bars with closes: 100, 102, 104, 106, 108
    for (let i = 0; i < 5; i++) {
      calc.addBar(makeBar(100 + i * 2));
    }

    const f = calc.computeAt(4); // close = 108
    // MA(5) = (100+102+104+106+108)/5 = 104
    expect(f.zd_5).toBeCloseTo(104 / 108 - 1);
  });

  it('should handle partial window for moving averages gracefully', () => {
    // Only 3 bars, but zd_5 should use available data
    calc.addBars([makeBar(100), makeBar(102), makeBar(104)]);

    const f = calc.computeAt(2); // close = 104
    // Only 3 bars available, avg = (100+102+104)/3 = 102
    expect(f.zd_5).toBeCloseTo(102 / 104 - 1);
  });

  it('should return all 11 features from toArray', () => {
    calc.addBar(makeBar(100, 98, 105, 95));

    const f = calc.computeAt(0);
    const arr = FeatureCalculator.toArray(f);
    expect(arr).toHaveLength(11);
  });

  it('should compute all features for every bar', () => {
    calc.addBars([makeBar(100), makeBar(102), makeBar(104)]);

    const all = calc.computeAll();
    expect(all).toHaveLength(3);
  });

  it('should compute latest feature', () => {
    calc.addBars([makeBar(100), makeBar(105)]);

    const latest = calc.computeLatest();
    expect(latest.zclose).toBeCloseTo(105 / 100 - 1);
  });

  it('should reset state', () => {
    calc.addBar(makeBar(100));
    expect(calc.length).toBe(1);

    calc.reset();
    expect(calc.length).toBe(0);
  });
});
