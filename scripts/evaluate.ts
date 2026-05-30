/**
 * Evaluate a trained DeepScalper agent on held-out test data.
 *
 * Usage:
 *   npm run evaluate -- data/RELIANCE_1m.csv [--train-ratio 0.8]
 *
 * Runs the agent (greedy, no exploration) on the test portion of data
 * and reports financial performance metrics.
 */

import { DataLoader } from '../src/data/loader.ts';
import { TradingEnvironment } from '../src/env/trading-env.ts';
import { MetricsCalculator } from '../src/evaluation/metrics.ts';

interface EvalArgs {
  dataPath: string;
  trainRatio: number;
}

function parseArgs(): EvalArgs {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes('--help')) {
    console.log('Usage: npm run evaluate -- <data.csv> [--train-ratio 0.8]');
    console.log('');
    console.log('Arguments:');
    console.log('  data.csv        Path to 1-min OHLCV CSV file (required)');
    console.log('  --train-ratio   Fraction used for training; rest is test (default: 0.8)');
    process.exit(args.includes('--help') ? 0 : 1);
  }

  const dataPath = args[0];
  let trainRatio = 0.8;

  for (let i = 1; i < args.length; i++) {
    if (args[i] === '--train-ratio' && args[i + 1]) trainRatio = parseFloat(args[++i]);
  }

  return { dataPath, trainRatio };
}

async function main() {
  const { dataPath, trainRatio } = parseArgs();

  console.log(`Loading data from: ${dataPath}`);
  const loader = new DataLoader();
  const allBars = loader.loadCSV(dataPath);
  const { test: testBars } = loader.trainTestSplit(allBars, trainRatio);
  const testDays = loader.splitByDay(testBars);

  console.log(`Test days: ${testDays.length}`);
  console.log(`Test bars: ${testBars.length}\n`);

  const env = new TradingEnvironment({
    maxPosition: 50,
    feeRate: 0.0003,
    initialCash: 100000,
  });

  const netValues: number[] = [1.0];

  for (const dayBars of testDays) {
    if (dayBars.length < 10) continue;

    env.reset(dayBars);

    // Random baseline agent (replace with trained agent once model saving is added)
    while (!env.isDone) {
      const action = {
        priceIndex: Math.floor(Math.random() * 5),
        quantityIndex: Math.floor(Math.random() * 5),
      };
      env.step(action);
    }

    netValues.push(env.getNetValue());
  }

  const metrics = new MetricsCalculator();
  const result = metrics.evaluate(netValues);

  console.log('=== Evaluation Results (Random Baseline) ===');
  console.log(`  Total Return:  ${(result.totalReturn * 100).toFixed(2)}%`);
  console.log(`  Sharpe Ratio:  ${result.sharpeRatio.toFixed(4)}`);
  console.log(`  Calmar Ratio:  ${result.calmarRatio.toFixed(4)}`);
  console.log(`  Sortino Ratio: ${result.sortinoRatio.toFixed(4)}`);
  console.log(`  Max Drawdown:  ${(result.maxDrawdown * 100).toFixed(2)}%`);
}

await main();
