/**
 * Train a DeepScalper agent on downloaded OHLCV data and save the model.
 *
 * Usage:
 *   npm run train -- RELIANCE [--epochs 5] [--train-ratio 0.8]
 *
 * Expects data at data/<SYMBOL>_1m.csv (output of download-data script).
 * Saves trained model to models/<SYMBOL>/.
 */

import { DataLoader } from '../src/data/loader.ts';
import { Trainer, EpisodeStats } from '../src/training/trainer.ts';

interface TrainArgs {
  symbol: string;
  dataPath: string;
  epochs: number;
  trainRatio: number;
}

function parseArgs(): TrainArgs {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes('--help')) {
    console.log('Usage: npm run train -- <SYMBOL> [--epochs 5] [--train-ratio 0.8]');
    console.log('');
    console.log('Arguments:');
    console.log('  SYMBOL          NSE symbol (required). Loads data/<SYMBOL>_1m.csv');
    console.log('  --epochs        Number of training passes over the data (default: 5)');
    console.log('  --train-ratio   Fraction of data used for training (default: 0.8)');
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
  const { train: trainBars } = loader.trainTestSplit(allBars, trainRatio);
  const tradingDays = loader.splitByDay(trainBars);

  console.log(`Total bars: ${allBars.length}`);
  console.log(`Training days: ${tradingDays.length}`);
  console.log(`Epochs: ${epochs}\n`);

  const fast = tradingDays.length > 200;
  const trainer = new Trainer({
    batchSize: fast ? 32 : 64,
    minBufferSize: 200,
    bufferCapacity: 50000,
    learningRate: 0.001,
    trainEveryNSteps: fast ? 50 : 10,
    targetUpdateFreq: 500,
    epsilonStart: 1.0,
    epsilonEnd: 0.01,
    epsilonDecaySteps: tradingDays.length * epochs * 100,
    env: { maxPosition: 50, feeRate: 0.0003, initialCash: 100000 },
    reward: { hindsightWeight: 0.1, hindsightHorizon: 180, inactivityPenalty: 1.0 },
  });

  let episode = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    console.log(`--- Epoch ${epoch + 1}/${epochs} ---`);
    const epochStats: EpisodeStats[] = [];

    for (const dayBars of tradingDays) {
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
    console.log(`  Epoch ${epoch + 1} avg net value: ${epochAvgNV.toFixed(4)}\n`);
  }

  // Save trained model
  const modelDir = await trainer.saveModel(symbol, epochs, episode);
  console.log(`Model saved to: ${modelDir}`);
  console.log('Training complete.');
}

await main();
