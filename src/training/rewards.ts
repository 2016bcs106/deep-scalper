/**
 * Reward functions for DeepScalper training.
 *
 * Basic reward: change in P&L = (price_{t+1} - price_t) * position - fees
 * Hindsight bonus: looks H steps ahead to encourage long-term thinking.
 * The bonus is only applied during training, not evaluation.
 */

export interface RewardConfig {
  /** Weight of the hindsight bonus (lambda in the paper). */
  hindsightWeight: number;
  /** How many steps ahead to look (H in the paper). */
  hindsightHorizon: number;
  /** Transaction fee rate. */
  feeRate: number;
}

const DEFAULT_CONFIG: RewardConfig = {
  hindsightWeight: 0.1,
  hindsightHorizon: 180,
  feeRate: 0.0003,
};

/**
 * Computes enhanced rewards with hindsight bonus for training.
 *
 * reward_enhanced = reward_basic + lambda * (price_{t+H} - price_t) * position_t
 *
 * The hindsight bonus rewards holding through favorable trends and
 * penalizes premature exits before large moves.
 */
export class RewardCalculator {
  private config: RewardConfig;

  constructor(config: Partial<RewardConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /** Basic P&L reward: price change * position - transaction cost. */
  basicReward(priceBefore: number, priceAfter: number, position: number, tradeAmount: number): number {
    const pnl = (priceAfter - priceBefore) * position;
    const fee = Math.abs(tradeAmount) * priceBefore * this.config.feeRate;
    return pnl - fee;
  }

  /**
   * Enhanced reward with hindsight bonus (training only).
   * @param prices Array of close prices for the full day.
   * @param t Current time step index.
   * @param position Position held at time t.
   * @param tradeAmount Shares traded at time t (for fee calc).
   */
  enhancedReward(prices: number[], t: number, position: number, tradeAmount: number): number {
    if (t >= prices.length - 1) {
      return this.basicReward(prices[t], prices[t], position, tradeAmount);
    }

    const basic = this.basicReward(prices[t], prices[t + 1], position, tradeAmount);
    const bonus = this.hindsightBonus(prices, t, position);

    return basic + bonus;
  }

  /** Hindsight bonus: lambda * (price_{t+H} - price_t) * position. */
  hindsightBonus(prices: number[], t: number, position: number): number {
    const { hindsightWeight, hindsightHorizon } = this.config;

    // Clamp horizon to available data
    const futureIdx = Math.min(t + hindsightHorizon, prices.length - 1);
    const futurePrice = prices[futureIdx];
    const currentPrice = prices[t];

    return hindsightWeight * (futurePrice - currentPrice) * position;
  }

  /**
   * Compute all enhanced rewards for a full episode (batch mode).
   * Used after episode completes since hindsight requires future prices.
   */
  computeEpisodeRewards(
    prices: number[],
    positions: number[],
    tradeAmounts: number[],
  ): number[] {
    const rewards: number[] = [];

    for (let t = 0; t < prices.length - 1; t++) {
      rewards.push(this.enhancedReward(prices, t, positions[t], tradeAmounts[t]));
    }

    return rewards;
  }
}
