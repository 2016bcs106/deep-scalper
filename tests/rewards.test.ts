import { describe, it, expect } from 'vitest';
import { RewardCalculator } from '../src/training/rewards.js';

describe('RewardCalculator', () => {
  describe('basicReward', () => {
    it('should return positive reward when price rises with long position', () => {
      const calc = new RewardCalculator({ feeRate: 0 });
      const reward = calc.basicReward(100, 103, 5, 0);
      expect(reward).toBe(15); // (103-100)*5
    });

    it('should return negative reward when price drops with long position', () => {
      const calc = new RewardCalculator({ feeRate: 0 });
      const reward = calc.basicReward(100, 97, 5, 0);
      expect(reward).toBe(-15); // (97-100)*5
    });

    it('should return positive reward when price drops with short position', () => {
      const calc = new RewardCalculator({ feeRate: 0 });
      const reward = calc.basicReward(100, 97, -5, 0);
      expect(reward).toBe(15); // (97-100)*(-5)
    });

    it('should deduct transaction fees', () => {
      const calc = new RewardCalculator({ feeRate: 0.01 });
      // Buy 10 shares at price 100, fee = 10*100*0.01 = 10
      const reward = calc.basicReward(100, 100, 0, 10);
      expect(reward).toBe(-10); // 0 pnl - 10 fee
    });

    it('should return zero reward with no position and no trade', () => {
      const calc = new RewardCalculator({ feeRate: 0 });
      expect(calc.basicReward(100, 110, 0, 0)).toBe(0);
    });
  });

  describe('hindsightBonus', () => {
    it('should reward holding long when future price is higher', () => {
      const calc = new RewardCalculator({ hindsightWeight: 0.1, hindsightHorizon: 3 });
      const prices = [100, 101, 102, 108, 110];

      // At t=0, position=1, look ahead 3 steps: price goes 100->108
      const bonus = calc.hindsightBonus(prices, 0, 1);
      expect(bonus).toBeCloseTo(0.1 * (108 - 100) * 1); // 0.8
    });

    it('should penalize holding long when future price is lower', () => {
      const calc = new RewardCalculator({ hindsightWeight: 0.1, hindsightHorizon: 3 });
      const prices = [100, 99, 98, 90, 88];

      const bonus = calc.hindsightBonus(prices, 0, 1);
      expect(bonus).toBeCloseTo(0.1 * (90 - 100) * 1); // -1.0
    });

    it('should penalize short position when future price rises', () => {
      const calc = new RewardCalculator({ hindsightWeight: 0.1, hindsightHorizon: 3 });
      const prices = [100, 102, 105, 110, 112];

      const bonus = calc.hindsightBonus(prices, 0, -2);
      expect(bonus).toBeCloseTo(0.1 * (110 - 100) * (-2)); // -2.0
    });

    it('should clamp horizon to end of available data', () => {
      const calc = new RewardCalculator({ hindsightWeight: 0.1, hindsightHorizon: 100 });
      const prices = [100, 105, 110]; // only 3 bars

      // Horizon 100 >> data length, should use last price (110)
      const bonus = calc.hindsightBonus(prices, 0, 1);
      expect(bonus).toBeCloseTo(0.1 * (110 - 100) * 1); // 1.0
    });
  });

  describe('enhancedReward', () => {
    it('should combine basic reward and hindsight bonus', () => {
      const calc = new RewardCalculator({ hindsightWeight: 0.1, hindsightHorizon: 3, feeRate: 0 });
      const prices = [100, 101, 102, 108];

      // t=0, position=2, no trade
      const reward = calc.enhancedReward(prices, 0, 2, 0);
      const expectedBasic = (101 - 100) * 2;           // 2
      const expectedBonus = 0.1 * (108 - 100) * 2;    // 1.6
      expect(reward).toBeCloseTo(expectedBasic + expectedBonus); // 3.6
    });
  });

  describe('computeEpisodeRewards', () => {
    it('should produce one reward per step (length - 1)', () => {
      const calc = new RewardCalculator({ hindsightWeight: 0, feeRate: 0 });
      const prices = [100, 101, 102, 103, 104];
      const positions = [1, 1, 1, 1, 1];
      const trades = [0, 0, 0, 0, 0];

      const rewards = calc.computeEpisodeRewards(prices, positions, trades);

      expect(rewards).toHaveLength(4);
      // Each step: (price_{t+1} - price_t) * 1 = 1
      rewards.forEach(r => expect(r).toBeCloseTo(1));
    });
  });
});
