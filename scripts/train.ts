/**
 * Train a DeepScalper agent on downloaded OHLCV data and save the model.
 *
 * Usage:
 *   npm run train -- RELIANCE [--epochs 5] [--train-ratio 0.8]
 *
 * Expects data at data/<SYMBOL>_1m.csv (output of download-data script).
 * Saves trained model to models/<SYMBOL>/.
 */

import tf from '../src/tf.ts';
import { DataLoader } from '../src/data/loader.ts';
import { Trainer, EpisodeStats } from '../src/training/trainer.ts';
import { TradingEnvironment } from '../src/env/trading-env.ts';
import { MetricsCalculator } from '../src/evaluation/metrics.ts';
import { EmailNotifier } from '../src/evaluation/notifier.ts';

interface TrainArgs {
  symbol: string;
  dataPath: string;
  epochs: number;
  trainRatio: number;
}

function parseArgs(): TrainArgs {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes('--help')) {
    console.log('Usage: npm run train -- <SYMBOL> [--epochs 5] [--train-ratio 0.8] [--quick]');
    console.log('');
    console.log('Arguments:');
    console.log('  SYMBOL          NSE symbol (required). Loads data/<SYMBOL>_1m.csv');
    console.log('  --epochs        Number of training passes over the data (default: 5)');
    console.log('  --train-ratio   Fraction of data used for training (default: 0.8)');
    console.log('  --quick         Use only 20 days for a fast test run (~30 seconds)');
    process.exit(args.includes('--help') ? 0 : 1);
  }

  const symbol = args[0];
  let epochs = 5;
  let trainRatio = 0.8;

  for (let i = 1; i < args.length; i++) {
    if (args[i] === '--epochs' && args[i + 1]) epochs = parseInt(args[++i]);
    if (args[i] === '--train-ratio' && args[i + 1]) trainRatio = parseFloat(args[++i]);
  }

  return { symbol, dataPath: `data/${symbol}_1m.csv`, epochs, trainRatio };
}

async function main() {
  const { symbol, dataPath, epochs, trainRatio } = parseArgs();

  console.log(`Training DeepScalper for: ${symbol}`);
  console.log(`Loading data from: ${dataPath}`);
  const loader = new DataLoader();
  const allBars = loader.loadCSV(dataPath);

  // 3-way split: train 60% / validation 20% / test 20% (prevents overfitting to test set)
  const { train: trainBars, val: valBars, test: testBars } = loader.trainValTestSplit(allBars, 0.6, 0.2);
  const tradingDays = loader.splitByDay(trainBars);
  const valDays = loader.splitByDay(valBars);

  console.log(`Total bars: ${allBars.length}`);
  console.log(`Train days: ${tradingDays.length} | Val days: ${valDays.length} | Test days: ${loader.splitByDay(testBars).length}`);
  console.log(`Epochs: ${epochs}\n`);

  // Use only first 20 days for quick test runs, full data otherwise
  const quickMode = process.argv.includes('--quick');
  const activeDays = quickMode ? tradingDays.slice(0, 20) : tradingDays;
  if (quickMode) console.log(`[Quick mode] Using only ${activeDays.length} days\n`);

  const totalStepsEstimate = activeDays.length * epochs * 375;
  const trainer = new Trainer({
    // Smaller batch + less buffer = lower memory (fits 1GB RAM)
    batchSize: 32,
    minBufferSize: 500,
    bufferCapacity: 50000,
    learningRate: 0.0005,
    // Update every 20 steps: balances learning vs memory/speed on micro
    trainEveryNSteps: quickMode ? 50 : 20,
    targetUpdateFreq: 2000,
    epsilonStart: 1.0,
    epsilonEnd: 0.02,
    epsilonDecaySteps: Math.floor(totalStepsEstimate * 0.6),
    gradientClipNorm: 1.0,
    auxiliaryWeight: 0.5,
    env: {
      // 100 shares max: keeps exposure at ~1.4L on RELIANCE (1.4x leverage)
      // Prevents catastrophic losses that go below 0 net value
      maxPosition: 100,
      feeRate: 0.0003,
      initialCash: 100000,
      priceLevels: 5,
      quantityLevels: 5,
    },
    reward: {
      hindsightWeight: 0.05,
      hindsightHorizon: 180,
      inactivityPenalty: 0.5,
      feeRate: 0.0003,
    },
  });

  let episode = 0;
  let lastEpochStats: EpisodeStats[] = [];

  for (let epoch = 0; epoch < epochs; epoch++) {
    console.log(`--- Epoch ${epoch + 1}/${epochs} ---`);
    const epochStats: EpisodeStats[] = [];
    lastEpochStats = epochStats;

    for (const dayBars of activeDays) {
      if (dayBars.length < 10) continue;

      const stats = trainer.trainEpisode(dayBars);
      stats.episode = ++episode;
      epochStats.push(stats);

      if (episode % 10 === 0) {
        const avgReward = epochStats.slice(-10).reduce((s, e) => s + e.totalReward, 0) / 10;
        const avgNetVal = epochStats.slice(-10).reduce((s, e) => s + e.netValue, 0) / 10;
        console.log(
          `  Episode ${episode} | ` +
          `Reward: ${avgReward.toFixed(2)} | ` +
          `Net Value: ${avgNetVal.toFixed(4)} | ` +
          `Epsilon: ${stats.epsilon.toFixed(3)} | ` +
          `Loss: ${stats.avgLoss.toFixed(4)}`
        );
      }
    }

    const epochAvgNV = epochStats.reduce((s, e) => s + e.netValue, 0) / epochStats.length;
    console.log(`  Epoch ${epoch + 1} train avg net value: ${epochAvgNV.toFixed(4)}`);

    // Validation: run greedy policy on validation days (no training)
    const valResults = valDays
      .filter(d => d.length >= 10)
      .slice(0, quickMode ? 5 : valDays.length)
      .map(d => trainer.validateEpisode(d));
    const valAvgNV = valResults.reduce((s, r) => s + r.netValue, 0) / valResults.length;
    const valTrades = valResults.reduce((s, r) => s + r.trades, 0);
    const gap = epochAvgNV - valAvgNV;

    console.log(`  Epoch ${epoch + 1} val avg net value:   ${valAvgNV.toFixed(4)} (gap: ${gap > 0 ? '+' : ''}${gap.toFixed(4)}) | val trades: ${valTrades}`);

    // Overfitting detection: train >> val means memorization
    if (gap > 0.05) {
      console.log(`  ⚠️  Overfitting detected (train-val gap > 5%). Stopping early.`);
      break;
    }
    console.log('');
  }

  // Save trained model
  const modelDir = await trainer.saveModel(symbol, epochs, episode);
  console.log(`Model saved to: ${modelDir}`);

  // Evaluate on held-out test data (never seen during training or validation)
  console.log('\nRunning evaluation on test data...');
  const testDays = loader.splitByDay(testBars);
  const env = new TradingEnvironment({ maxPosition: 100, feeRate: 0.0003, initialCash: 100000 });
  const qNet = trainer.getQNetwork();
  const netValues: number[] = [1.0];

  for (const dayBars of testDays) {
    if (dayBars.length < 10) continue;
    let obs = env.reset(dayBars);
    while (!env.isDone) {
      const state = tf.tensor2d([[...obs.macro, ...obs.private]]);
      const action = qNet.selectAction(state);
      state.dispose();
      obs = env.step(action).observation;
    }
    netValues.push(env.getNetValue());
  }

  const metrics = new MetricsCalculator();
  const evaluation = metrics.evaluate(netValues);

  console.log(`\n=== Evaluation Results: ${symbol} ===`);
  console.log(`  Total Return:  ${(evaluation.totalReturn * 100).toFixed(2)}%`);
  console.log(`  Sharpe Ratio:  ${evaluation.sharpeRatio.toFixed(4)}`);
  console.log(`  Calmar Ratio:  ${evaluation.calmarRatio.toFixed(4)}`);
  console.log(`  Sortino Ratio: ${evaluation.sortinoRatio.toFixed(4)}`);
  console.log(`  Max Drawdown:  ${(evaluation.maxDrawdown * 100).toFixed(2)}%`);

  // Send email notification
  try {
    const notifier = EmailNotifier.fromEnv();
    await notifier.sendTrainingReport({
      symbol,
      epochs,
      episodes: episode,
      trainingNetValue: lastEpochStats[lastEpochStats.length - 1]?.netValue ?? 0,
      evaluation,
    });
    console.log('\nEmail notification sent.');
  } catch (err: any) {
    console.log(`\nEmail notification skipped: ${err.message}`);
  }

  console.log('Training complete.');
}

await main();
