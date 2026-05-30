/**
 * Financial evaluation metrics from the DeepScalper paper (Section 5.2).
 * All metrics are computed from a sequence of net values over time.
 */

/** Complete evaluation results for a trading period. */
export interface EvaluationResult {
  totalReturn: number;
  sharpeRatio: number;
  calmarRatio: number;
  sortinoRatio: number;
  maxDrawdown: number;
}

/**
 * Computes risk-adjusted performance metrics for intraday trading.
 * Expects a time series of portfolio net values (normalized to 1.0 at start).
 */
export class MetricsCalculator {
  /** Compute all evaluation metrics from a net value series. */
  evaluate(netValues: number[]): EvaluationResult {
    if (netValues.length < 2) {
      return { totalReturn: 0, sharpeRatio: 0, calmarRatio: 0, sortinoRatio: 0, maxDrawdown: 0 };
    }

    const returns = this.computeReturns(netValues);
    const totalReturn = this.totalReturn(netValues);
    const sharpeRatio = this.sharpeRatio(returns);
    const maxDrawdown = this.maxDrawdown(netValues);
    const calmarRatio = this.calmarRatio(returns, maxDrawdown);
    const sortinoRatio = this.sortinoRatio(returns);

    return { totalReturn, sharpeRatio, calmarRatio, sortinoRatio, maxDrawdown };
  }

  /** TR: (final_value / initial_value) - 1. */
  totalReturn(netValues: number[]): number {
    return netValues[netValues.length - 1] / netValues[0] - 1;
  }

  /** SR: mean(returns) / std(returns). Higher = better risk-adjusted returns. */
  sharpeRatio(returns: number[]): number {
    const { mean, std } = this.meanAndStd(returns);
    return std === 0 ? 0 : mean / std;
  }

  /** CR: mean(returns) / max_drawdown. Reward vs. worst-case loss. */
  calmarRatio(returns: number[], maxDrawdown: number): number {
    if (maxDrawdown === 0) return 0;
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    return mean / maxDrawdown;
  }

  /** SoR: mean(returns) / downside_deviation. Only penalizes downside volatility. */
  sortinoRatio(returns: number[]): number {
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const downsideReturns = returns.filter(r => r < 0);

    if (downsideReturns.length === 0) return mean === 0 ? 0 : Infinity;

    const downsideVariance = downsideReturns.reduce((sum, r) => sum + r * r, 0) / downsideReturns.length;
    const downsideDev = Math.sqrt(downsideVariance);

    return downsideDev === 0 ? 0 : mean / downsideDev;
  }

  /** MDD: largest peak-to-trough decline in net value. */
  maxDrawdown(netValues: number[]): number {
    let peak = netValues[0];
    let maxDD = 0;

    for (const value of netValues) {
      if (value > peak) peak = value;
      const drawdown = (peak - value) / peak;
      if (drawdown > maxDD) maxDD = drawdown;
    }

    return maxDD;
  }

  /** Convert net values to step-wise returns. */
  computeReturns(netValues: number[]): number[] {
    const returns: number[] = [];
    for (let i = 1; i < netValues.length; i++) {
      returns.push(netValues[i] / netValues[i - 1] - 1);
    }
    return returns;
  }

  private meanAndStd(values: number[]): { mean: number; std: number } {
    if (values.length === 0) return { mean: 0, std: 0 };
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length;
    return { mean, std: Math.sqrt(variance) };
  }
}
