/**
 * Headless speed benchmark for FlappyEnv + SimulationRunner.
 *
 * Run with:  node src/examples/headless_speed_test.js
 *
 * Runs 200 episodes with a random-ish policy (no rendering) and
 * measures steps per second (SPS), average reward, and average score.
 */
import { FlappyEnv } from '../env/FlappyEnv.js';
import { SimulationRunner } from '../SimulationRunner.js';

const NUM_EPISODES = 200;
const SEED = 0;

// ── Random policy (flap ~30% of the time) ────────────────
let rngState = 12345;
function cheapRandom() {
    rngState = (rngState * 1664525 + 1013904223) & 0xffffffff;
    return (rngState >>> 0) / 4294967296;
}
const randomPolicy = () => (cheapRandom() < 0.3 ? 1 : 0);

// ── Run benchmark ────────────────────────────────────────
const env = new FlappyEnv();
const runner = new SimulationRunner(env, {
    actionProvider: randomPolicy,
});

console.log(`\n🚀 Headless speed benchmark`);
console.log(`   Episodes: ${NUM_EPISODES}`);
console.log(`   Policy:   random (30% flap)\n`);

const t0 = performance.now();

const results = runner.runSync({ episodes: NUM_EPISODES, seed: SEED });

const t1 = performance.now();
const elapsedMs = t1 - t0;
const totalSteps = results.reduce((s, r) => s + r.steps, 0);
const avgReward = results.reduce((s, r) => s + r.totalReward, 0) / results.length;
const avgScore = results.reduce((s, r) => s + r.score, 0) / results.length;
const avgSteps = totalSteps / results.length;
const sps = (totalSteps / elapsedMs) * 1000;

console.log(`── Results ─────────────────────────────────`);
console.log(`   Total steps:     ${totalSteps.toLocaleString()}`);
console.log(`   Time:            ${elapsedMs.toFixed(1)} ms`);
console.log(`   Steps/second:    ${sps.toFixed(0).toLocaleString()} SPS`);
console.log(`   Avg steps/ep:    ${avgSteps.toFixed(1)}`);
console.log(`   Avg reward/ep:   ${avgReward.toFixed(2)}`);
console.log(`   Avg score/ep:    ${avgScore.toFixed(2)}`);
console.log(`────────────────────────────────────────────\n`);

if (sps > 10000) {
    console.log('✅ High throughput confirmed (>10k SPS). Ready for RL training.');
} else {
    console.log('⚠️  SPS is below 10k. Performance might be an issue for training.');
}
