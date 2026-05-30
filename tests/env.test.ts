import { describe, it, expect, beforeEach } from 'vitest';
import { TradingEnvironment } from '../src/env/trading-env.js';
import { OHLCVBar } from '../src/data/features.js';

function makeDayBars(prices: number[]): OHLCVBar[] {
  return prices.map((p, i) => ({
    timestamp: new Date('2024-01-15T09:15:00').getTime() + i * 60000,
    open: p,
    high: p + 1,
    low: p - 1,
    close: p,
    volume: 1000,
  }));
}

describe('TradingEnvironment', () => {
  let env: TradingEnvironment;

  beforeEach(() => {
    env = new TradingEnvironment({
      maxPosition: 10,
      feeRate: 0,
      priceLevels: 5,
      quantityLevels: 5,
      initialCash: 10000,
    });
  });

  it('should throw if reset with fewer than 2 bars', () => {
    expect(() => env.reset(makeDayBars([100]))).toThrow();
  });

  it('should return valid observation on reset', () => {
    const obs = env.reset(makeDayBars([100, 101, 102]));

    expect(obs.macro).toHaveLength(16); // 11 indicators + 5 OHLCV
    expect(obs.private).toHaveLength(3); // position, cash, time
    expect(obs.private[0]).toBe(0);      // no position
    expect(obs.private[1]).toBe(1);      // full cash
    expect(obs.private[2]).toBe(1);      // full time remaining
  });

  it('should accumulate reward when holding a position during price rise', () => {
    env.reset(makeDayBars([100, 101, 105, 110]));

    // Action: quantityIndex=4 (max buy) with 5 levels -> target position = +10
    const r1 = env.step({ priceIndex: 2, quantityIndex: 4 });
    // Bought at 100, price moved to 101, position=10 -> reward = (101-100)*10 = 10
    expect(r1.reward).toBe(10);
    expect(r1.info.position).toBe(10);

    // Hold: quantityIndex=4 again means target=+10, no change since already at 10
    const r2 = env.step({ priceIndex: 2, quantityIndex: 4 });
    // Price 101->105, still holding 10 -> reward = (105-101)*10 = 40
    expect(r2.reward).toBe(40);
  });

  it('should penalize reward when holding position during price drop', () => {
    env.reset(makeDayBars([100, 95, 90]));

    // Buy max
    const r1 = env.step({ priceIndex: 2, quantityIndex: 4 });
    // Price dropped 100->95, holding 10 -> reward = (95-100)*10 = -50
    expect(r1.reward).toBe(-50);
  });

  it('should give zero reward for no position', () => {
    env.reset(makeDayBars([100, 105, 110]));

    // Action: quantityIndex=2 (middle) -> target position = 0
    const r = env.step({ priceIndex: 2, quantityIndex: 2 });
    expect(r.reward).toBe(0);
    expect(r.info.position).toBe(0);
  });

  it('should end episode at last step and close positions', () => {
    env.reset(makeDayBars([100, 101, 102]));

    // Buy max
    env.step({ priceIndex: 2, quantityIndex: 4 });
    const r2 = env.step({ priceIndex: 2, quantityIndex: 4 });

    expect(r2.done).toBe(true);
    expect(r2.info.position).toBe(0); // force-closed
  });

  it('should throw when stepping after episode is done', () => {
    env.reset(makeDayBars([100, 101]));
    env.step({ priceIndex: 2, quantityIndex: 2 });

    expect(() => env.step({ priceIndex: 2, quantityIndex: 2 })).toThrow();
  });

  it('should maintain net value close to 1.0 with no trading', () => {
    env.reset(makeDayBars([100, 100, 100]));

    env.step({ priceIndex: 2, quantityIndex: 2 }); // hold
    const r = env.step({ priceIndex: 2, quantityIndex: 2 }); // hold

    expect(r.info.netValue).toBeCloseTo(1.0);
  });

  it('should deduct transaction fees when configured', () => {
    const envWithFees = new TradingEnvironment({
      maxPosition: 10,
      feeRate: 0.01, // 1% fee
      priceLevels: 5,
      quantityLevels: 5,
      initialCash: 10000,
    });

    envWithFees.reset(makeDayBars([100, 100, 100]));

    // Buy 10 shares at $100 with 1% fee = $100 cost
    envWithFees.step({ priceIndex: 2, quantityIndex: 4 });
    const r = envWithFees.step({ priceIndex: 2, quantityIndex: 4 });

    // Net value < 1 due to fees (buy fee + close fee)
    expect(r.info.netValue).toBeLessThan(1.0);
  });

  it('should correctly report time remaining in private state', () => {
    env.reset(makeDayBars([100, 101, 102, 103, 104])); // 4 steps

    const r1 = env.step({ priceIndex: 2, quantityIndex: 2 });
    // After 1 step: remaining = (4-1)/4 = 0.75
    expect(r1.observation.private[2]).toBeCloseTo(0.75);
  });
});
