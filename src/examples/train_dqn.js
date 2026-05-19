/**
 * AutonoLearn RL — DQN Training Script
 *
 * Trains a Deep Q-Network agent on the Flappy Bird environment.
 * Requires: @tensorflow/tfjs-node (Node v20)
 * Results and model saved to: results/ and models_final/
 *
 * Usage:
 *   node src/examples/train_dqn.js
 */

import { CONFIG } from '../config.js';
import { FlappyEnv } from '../env/FlappyEnv.js';
import { SimulationRunner } from '../SimulationRunner.js';
import { DQNAgent } from '../agents/DQNAgent.js';
import fs from 'fs';
import path from 'path';

import * as tf from '@tensorflow/tfjs-node';
globalThis.tf = tf;

// ─── Logging (tee to file + terminal) ─────────────────────────
const difficulty = CONFIG.pipe.gap >= 150 ? 'easy' : (CONFIG.pipe.gap >= 120 ? 'medium' : 'hard');
const resultsDir = path.join(process.cwd(), 'results', difficulty);
fs.mkdirSync(resultsDir, { recursive: true });
const logStream = fs.createWriteStream(path.join(resultsDir, 'dqn_log.txt'), { flags: 'w' });
const _origWrite = process.stdout.write.bind(process.stdout);
process.stdout.write = (chunk, ...rest) => { logStream.write(chunk); return _origWrite(chunk, ...rest); };

// ─── Parameters ───────────────────────────────────────────────
const TOTAL_EPISODES      = 20000;
const LOG_INTERVAL        = 200;
const CHECKPOINT_INTERVAL = 2000;
const EVAL_EPISODES       = 100;
const EVAL_SEED           = 0;
const FINAL_EVAL_EPISODES = 200;

// ─── Utilities ────────────────────────────────────────────────

function computeStats(scores) {
    const n = scores.length;
    if (n === 0) return { mean: 0, max: 0, min: 0, stdDev: 0, median: 0 };
    const sorted = [...scores].sort((a, b) => a - b);
    const mean = scores.reduce((a, b) => a + b, 0) / n;
    const variance = scores.reduce((acc, v) => acc + (v - mean) ** 2, 0) / n;
    const median = n % 2 === 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2 : sorted[Math.floor(n / 2)];
    return {
        mean: Number(mean.toFixed(2)),
        max: Math.max(...scores),
        min: Math.min(...scores),
        stdDev: Number(Math.sqrt(variance).toFixed(2)),
        median
    };
}

function evaluateAgent(agent, env, episodes, startSeed) {
    const greedyPolicy = (state) => {
        const old = agent.epsilon;
        agent.epsilon = 0;
        const act = agent.act(state);
        agent.epsilon = old;
        return act;
    };
    const runner = new SimulationRunner(env, { actionProvider: greedyPolicy });
    const scores = [];
    for (let i = 0; i < episodes; i++) {
        const results = runner.runSync({ episodes: 1, seed: startSeed + i });
        scores.push(results[0].score);
    }
    return scores;
}

// ─── Main ─────────────────────────────────────────────────────

const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
console.log(`\n${'═'.repeat(60)}`);
console.log(`  AutonoLearn RL — DQN Training`);
console.log(`  Episodes: ${TOTAL_EPISODES} | Gap: ${CONFIG.pipe.gap}`);
console.log(`  Timestamp: ${timestamp}`);
console.log(`${'═'.repeat(60)}`);

const agent   = new DQNAgent();
agent.epsilonDecay = Math.pow(agent.epsilonMin / agent.epsilon, 1 / (TOTAL_EPISODES * 0.8));

const env     = new FlappyEnv();
const evalEnv = new FlappyEnv();

const learningCurve = [];
const checkpoints   = [];
let windowScores    = [];
let epCount         = 0;
let bestMean        = -Infinity;
let bestWeights     = null;

let lastState, lastAction;
const runner = new SimulationRunner(env, {
    actionProvider: (state) => {
        const action = agent.act(state);
        lastState = state;
        lastAction = action;
        return action;
    },
    onStep:       (result) => agent.learn(lastState, lastAction, result.reward, result.state, result.done),
    onEpisodeEnd: (summary) => {
        agent.endEpisode();
        epCount++;
        windowScores.push(summary.score);

        if (epCount % LOG_INTERVAL === 0) {
            const stats = computeStats(windowScores);
            learningCurve.push({ episode: epCount, avgScore: stats.mean, maxScore: stats.max, epsilon: Number(agent.epsilon.toFixed(4)) });
            process.stdout.write(`\r  Ep ${epCount}/${TOTAL_EPISODES} | Avg(${LOG_INTERVAL}): ${stats.mean} | Max: ${stats.max} | ε: ${agent.epsilon.toFixed(4)}`);
            windowScores = [];
        }

        if (epCount % CHECKPOINT_INTERVAL === 0) {
            const scores = evaluateAgent(agent, evalEnv, EVAL_EPISODES, EVAL_SEED);
            const stats  = computeStats(scores);
            checkpoints.push({ episode: epCount, ...stats });
            console.log(`\n  [Checkpoint @ ${epCount}] Eval(${EVAL_EPISODES} ep): Mean=${stats.mean}, Max=${stats.max}, StdDev=${stats.stdDev}`);
            if (stats.mean > bestMean) {
                bestMean = stats.mean;
                if (bestWeights) bestWeights.forEach(t => t.dispose());
                bestWeights = agent.mainNet.getWeights().map(t => t.clone());
                console.log(`  🌟 Mejor modelo guardado (Mean: ${stats.mean})`);
            }
        }
    }
});

const t0 = performance.now();
await runner.runHeadless({ episodes: TOTAL_EPISODES, yieldEvery: 32 });
const elapsed = ((performance.now() - t0) / 1000).toFixed(2);

if (bestWeights) {
    agent.mainNet.setWeights(bestWeights);
    bestWeights.forEach(t => t.dispose());
    agent._updateTargetNetwork();
    console.log(`\n  Restaurando mejor modelo (Mean: ${bestMean})`);
}
console.log(`\n  Entrenamiento completo en ${elapsed}s`);

// ─── Final greedy evaluation ──────────────────────────────────
console.log(`\n  Evaluación final (${FINAL_EVAL_EPISODES} episodios greedy)...`);
const finalScores = evaluateAgent(agent, evalEnv, FINAL_EVAL_EPISODES, EVAL_SEED);
const finalStats  = computeStats(finalScores);
const pad = (v, w = 6) => String(v).padStart(w);
console.log(`\n  🏆 DQN Final:`);
console.log(`  ┌──────────┬──────────┬──────────┬──────────┬──────────┐`);
console.log(`  │   Mean   │   Max    │   Min    │  StdDev  │  Median  │`);
console.log(`  ├──────────┼──────────┼──────────┼──────────┼──────────┤`);
console.log(`  │ ${pad(finalStats.mean, 8)} │ ${pad(finalStats.max, 8)} │ ${pad(finalStats.min, 8)} │ ${pad(finalStats.stdDev, 8)} │ ${pad(finalStats.median, 8)} │`);
console.log(`  └──────────┴──────────┴──────────┴──────────┴──────────┘`);

// ─── Save ─────────────────────────────────────────────────────
const modelsDir  = path.join(process.cwd(), 'models_final', difficulty);
const dqnDir     = path.join(modelsDir, 'dqn');
fs.mkdirSync(dqnDir,     { recursive: true });

await agent.save(tf.io.withSaveHandler(async (artifacts) => {
    const modelJSON = {
        modelTopology: artifacts.modelTopology,
        format: artifacts.format,
        generatedBy: artifacts.generatedBy,
        convertedBy: artifacts.convertedBy,
        weightsManifest: [{ paths: ['weights.bin'], weights: artifacts.weightSpecs }]
    };
    fs.writeFileSync(path.join(dqnDir, 'model.json'), JSON.stringify(modelJSON));
    if (artifacts.weightData) {
        fs.writeFileSync(path.join(dqnDir, 'weights.bin'), Buffer.from(artifacts.weightData));
    }
    return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
}));
console.log(`  💾 Modelo guardado en models_final/${difficulty}/dqn/`);

const resultData = {
    timestamp,
    agent: 'DQN',
    pipeGap: CONFIG.pipe.gap,
    trainingEpisodes: TOTAL_EPISODES,
    trainingTimeSec: Number(elapsed),
    hyperparameters: {
        gamma: 0.99,
        learningRate: agent.learningRate,
        epsilonStart: 1.0,
        epsilonEnd: agent.epsilonMin,
        epsilonDecay: agent.epsilonDecay,
        batchSize: agent.batchSize,
        replayBufferSize: 10000,
        targetUpdateFreq: agent.targetUpdateFreq,
        networkArchitecture: '3 → 24(ReLU) → 24(ReLU) → 2(Linear)'
    },
    learningCurve,
    checkpoints,
    finalEval: { ...finalStats, episodes: FINAL_EVAL_EPISODES, rawScores: finalScores }
};
fs.writeFileSync(path.join(resultsDir, 'dqn_results.json'), JSON.stringify(resultData, null, 2));
console.log(`  📊 Resultados guardados en results/dqn_results.json\n`);
