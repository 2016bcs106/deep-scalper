/**
 * Evaluate a trained DeepScalper agent on held-out test data.
 *
 * Usage:
 *   npm run evaluate -- RELIANCE [--train-ratio 0.8]
 *
 * Loads model from models/<SYMBOL>/ and evaluates on the test portion
 * of data/<SYMBOL>_1m.csv.
 */

import tf from '../src/tf.ts';
import { DataLoader } from '../src/data/loader.ts';
import { TradingEnvironment } from '../src/env/trading-env.ts';
import { MetricsCalculator } from '../src/evaluation/metrics.ts';
import { Trainer } from '../src/training/trainer.ts';

interface EvalArgs {
  symbol: string;
  dataPath: string;
  trainRatio: number;
}

function parseArgs(): EvalArgs {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes('--help')) {
    console.log('Usage: npm run evaluate -- <SYMBOL> [--train-ratio 0.8]');
    console.log('');
    console.log('Arguments:');
    console.log('  SYMBOL          NSE symbol (required). Loads model and data for this symbol.');
    console.log('  --train-ratio   Fraction used for training; rest is test (default: 0.8)');
    process.exit(args.includes('--help') ? 0 : 1);
  }

  const symbol = args[0];
  let trainRatio = 0.8;

  for (let i = 1; i < args.length; i++) {
    if (args[i] === '--train-ratio' && args[i + 1]) trainRatio = parseFloat(args[++i]);
  }

  return { symbol, dataPath: `data/${symbol}_1m.csv`, trainRatio };
}

async function main() {
  const { symbol, dataPath, trainRatio } = parseArgs();

  console.log(`Evaluating DeepScalper for: ${symbol}`);
  console.log(`Loading data from: ${dataPath}`);

  const loader = new DataLoader();
  const allBars = loader.loadCSV(dataPath);
  const { test: testBars } = loader.trainTestSplit(allBars, trainRatio);
  const testDays = loader.splitByDay(testBars);

  console.log(`Test days: ${testDays.length}`);
  console.log(`Test bars: ${testBars.length}`);

  // Load trained model
  const trainer = new Trainer({
    env: { maxPosition: 500, feeRate: 0.0003, initialCash: 100000 },
  });

  console.log(`Loading model from: models/${symbol}/`);
  const metadata = await trainer.loadModel(symbol);
  console.log(`  Trained at: ${metadata.trainedAt}`);
  console.log(`  Epochs: ${metadata.epochs}, Episodes: ${metadata.episodes}\n`);

  const qNetwork = trainer.getQNetwork();
  const env = new TradingEnvironment({
    maxPosition: 500,
    feeRate: 0.0003,
    initialCash: 100000,
  });

  const netValues: number[] = [1.0];

  for (const dayBars of testDays) {
    if (dayBars.length < 10) continue;

    const obs = env.reset(dayBars);
    let currentObs = obs;

    while (!env.isDone) {
      const stateVec = [...currentObs.macro, ...currentObs.private];
      const stateTensor = tf.tensor2d([stateVec]);
      const action = qNetwork.selectAction(stateTensor);
      stateTensor.dispose();

      const result = env.step(action);
      currentObs = result.observation;
    }

    netValues.push(env.getNetValue());
  }

  const metrics = new MetricsCalculator();
  const result = metrics.evaluate(netValues);

  console.log(`=== Evaluation Results: ${symbol} ===`);
  console.log(`  Total Return:  ${(result.totalReturn * 100).toFixed(2)}%`);
  console.log(`  Sharpe Ratio:  ${result.sharpeRatio.toFixed(4)}`);
  console.log(`  Calmar Ratio:  ${result.calmarRatio.toFixed(4)}`);
  console.log(`  Sortino Ratio: ${result.sortinoRatio.toFixed(4)}`);
  console.log(`  Max Drawdown:  ${(result.maxDrawdown * 100).toFixed(2)}%`);
}

await main();
