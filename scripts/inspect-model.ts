import tf from '../src/tf.ts';
import { DataLoader } from '../src/data/loader.ts';
import { TradingEnvironment } from '../src/env/trading-env.ts';
import { Trainer } from '../src/training/trainer.ts';

const loader = new DataLoader();
const allBars = loader.loadCSV('data/RELIANCE_1m.csv');
const { test } = loader.trainTestSplit(allBars, 0.8);
const testDays = loader.splitByDay(test);

const trainer = new Trainer({ env: { maxPosition: 500, feeRate: 0.0003, initialCash: 100000 }});
await trainer.loadModel('RELIANCE');
const qNet = trainer.getQNetwork();
const env = new TradingEnvironment({ maxPosition: 500, feeRate: 0.0003, initialCash: 100000 });

console.log(`Evaluating on ${Math.min(20, testDays.length)} test days...\n`);

const netValues: number[] = [];
let totalTrades = 0;

for (let d = 0; d < Math.min(20, testDays.length); d++) {
  const dayBars = testDays[d];
  if (dayBars.length < 10) continue;

  let obs = env.reset(dayBars);
  let trades = 0;

  while (!env.isDone) {
    const stateVec = [...obs.macro, ...obs.private];
    const state = tf.tensor2d([stateVec]);
    const action = qNet.selectAction(state);
    state.dispose();

    if (action.quantityIndex !== 2) trades++;
    const result = env.step(action);
    obs = result.observation;
  }

  totalTrades += trades;
  netValues.push(env.getNetValue());
  console.log(`  Day ${String(d+1).padStart(2)} | Net Value: ${env.getNetValue().toFixed(4)} | Trades: ${trades}`);
}

console.log(`\n  Avg Net Value: ${(netValues.reduce((a,b)=>a+b)/netValues.length).toFixed(4)}`);
console.log(`  Total Trades: ${totalTrades}`);
console.log(`  Days with NV > 1: ${netValues.filter(v => v > 1).length}/${netValues.length}`);
