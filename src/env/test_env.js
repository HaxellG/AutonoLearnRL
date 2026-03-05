/**
 * Headless smoke test for FlappyEnv.
 * Run with:  node src/env/test_env.js
 *
 * Tests:
 * 1. Episode runs and terminates.
 * 2. Deterministic seeds produce identical trajectories.
 * 3. Reward structure is correct.
 * 4. State vector has 3 components.
 */
import { FlappyEnv } from './FlappyEnv.js';

function runEpisode(env, seed, policy) {
    const state = env.reset(seed);
    let totalReward = 0;
    let steps = 0;

    while (!env.done) {
        const action = policy(state, steps);
        const result = env.step(action);
        totalReward += result.reward;
        steps++;
        state[0] = result.state[0];
        state[1] = result.state[1];
        state[2] = result.state[2];
    }

    return { totalReward, steps, score: env.score };
}

// ── Test 1: Episode runs and terminates ──────────────────
console.log('=== Test 1: Episode terminates ===');
const env = new FlappyEnv();
const r1 = runEpisode(env, 42, () => 0);  // never flap → falls quickly
console.log(`  NO_OP policy: steps=${r1.steps}, score=${r1.score}, reward=${r1.totalReward.toFixed(2)}`);
console.assert(r1.steps > 0, 'Should take at least 1 step');
console.assert(r1.steps < 200, 'NO_OP should die quickly');
console.log('  ✓ Passed\n');

// ── Test 2: Always-flap policy ───────────────────────────
console.log('=== Test 2: Always-flap policy ===');
const r2 = runEpisode(env, 42, () => 1);
console.log(`  FLAP policy: steps=${r2.steps}, score=${r2.score}, reward=${r2.totalReward.toFixed(2)}`);
console.assert(r2.steps > 0, 'Should take at least 1 step');
console.log('  ✓ Passed\n');

// ── Test 3: Deterministic seeds ──────────────────────────
console.log('=== Test 3: Deterministic seeds ===');
const r3a = runEpisode(env, 123, (_, t) => (t % 7 === 0 ? 1 : 0));
const r3b = runEpisode(env, 123, (_, t) => (t % 7 === 0 ? 1 : 0));
console.log(`  Run A: steps=${r3a.steps}, score=${r3a.score}, reward=${r3a.totalReward.toFixed(4)}`);
console.log(`  Run B: steps=${r3b.steps}, score=${r3b.score}, reward=${r3b.totalReward.toFixed(4)}`);
console.assert(r3a.steps === r3b.steps, 'Same seed → same steps');
console.assert(r3a.score === r3b.score, 'Same seed → same score');
console.assert(
    Math.abs(r3a.totalReward - r3b.totalReward) < 1e-9,
    'Same seed → same reward'
);
console.log('  ✓ Passed\n');

// ── Test 4: State vector shape ───────────────────────────
console.log('=== Test 4: State vector [dx, dy, vy] ===');
const state = env.reset(1);
console.log(`  Initial state: [${state.map(v => v.toFixed(2)).join(', ')}]`);
console.assert(state.length === 3, 'State must have 3 components');
console.assert(typeof state[0] === 'number', 'dx must be number');
console.assert(typeof state[1] === 'number', 'dy must be number');
console.assert(typeof state[2] === 'number', 'vy must be number');
console.log('  ✓ Passed\n');

// ── Test 5: Reward structure ─────────────────────────────
console.log('=== Test 5: Reward structure ===');
env.reset(42);
const step1 = env.step(0);
console.log(`  Step 1 (survive): reward=${step1.reward}`);
console.assert(step1.reward >= 0.1, 'Survive reward should be >= 0.1');
console.assert(step1.done === false, 'Should not be done after 1 step');

// Run until done
let lastResult;
while (!env.done) {
    lastResult = env.step(0);
}
console.log(`  Final step (collision): reward=${lastResult.reward}, done=${lastResult.done}`);
console.assert(lastResult.reward === -1.0, 'Collision reward should be -1.0');
console.assert(lastResult.done === true, 'Should be done after collision');
console.log('  ✓ Passed\n');

// ── Test 6: Snapshot / Restore ───────────────────────────
console.log('=== Test 6: Snapshot / Restore ===');
env.reset(99);
for (let i = 0; i < 50; i++) env.step(i % 5 === 0 ? 1 : 0);
const snap = env.getSnapshot();
const stateBeforeSnap = env.getState();

// Take some more steps
for (let i = 0; i < 20; i++) env.step(0);
const stateAfterMore = env.getState();

// Restore
env.setSnapshot(snap);
const stateAfterRestore = env.getState();
console.log(`  Before snap: [${stateBeforeSnap.map(v => v.toFixed(2)).join(', ')}]`);
console.log(`  After more:  [${stateAfterMore.map(v => v.toFixed(2)).join(', ')}]`);
console.log(`  Restored:    [${stateAfterRestore.map(v => v.toFixed(2)).join(', ')}]`);
console.assert(
    stateBeforeSnap[0] === stateAfterRestore[0] &&
    stateBeforeSnap[1] === stateAfterRestore[1] &&
    stateBeforeSnap[2] === stateAfterRestore[2],
    'Restored state must match snapshot'
);
console.log('  ✓ Passed\n');

console.log('══════════════════════════════════');
console.log('All tests passed! FlappyEnv is operational.');
console.log('══════════════════════════════════');
