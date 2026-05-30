/**
 * Train a DeepScalper agent on downloaded OHLCV data.
 *
 * Usage:
 *   npm run train -- data/RELIANCE_1m.csv [--epochs 5] [--train-ratio 0.8]
 */

import { DataLoader } from '../src/data/loader.ts';
import { Trainer, EpisodeStats } from '../src/training/trainer.ts';

interface TrainArgs {
  dataPath: string;
  epochs: number;
  trainRatio: number;
}

function parseArgs(): TrainArgs {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes('--help')) {
    console.log('Usage: npm run train -- <data.csv> [--epochs 5] [--train-ratio 0.8]');
    console.log('');
    console.log('Arguments:');
    console.log('  data.csv        Path to 1-min OHLCV CSV file (required)');
    console.log('  --epochs        Number of training passes over the data (default: 5)');
    console.log('  --train-ratio   Fraction of data used for training (default: 0.8)');
    process.exit(args.includes('--help') ? 0 : 1);
  }

  const dataPath = args[0];
  let epochs = 5;
  let trainRatio = 0.8;

  for (let i = 1; i < args.length; i++) {
    if (args[i] === '--epochs' && args[i + 1]) epochs = parseInt(args[++i]);
    if (args[i] === '--train-ratio' && args[i + 1]) trainRatio = parseFloat(args[++i]);
  }

  return { dataPath, epochs, trainRatio };
}

async function main() {
  const { dataPath, epochs, trainRatio } = parseArgs();

  console.log(`Loading data from: ${dataPath}`);
  const loader = new DataLoader();
  const allBars = loader.loadCSV(dataPath);
  const { train: trainBars } = loader.trainTestSplit(allBars, trainRatio);
  const tradingDays = loader.splitByDay(trainBars);

  console.log(`Total bars: ${allBars.length}`);
  console.log(`Training days: ${tradingDays.length}`);
  console.log(`Epochs: ${epochs}\n`);

  const trainer = new Trainer({
    batchSize: 64,
    minBufferSize: 500,
    bufferCapacity: 50000,
    learningRate: 0.001,
    epsilonStart: 1.0,
    epsilonEnd: 0.01,
    epsilonDecaySteps: tradingDays.length * epochs * 300,
    env: { maxPosition: 50, feeRate: 0.0003, initialCash: 100000 },
    reward: { hindsightWeight: 0.1, hindsightHorizon: 180 },
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

  console.log('Training complete.');
}

await main();
