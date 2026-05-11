/**
 * AutonoLearn RL — Full Experiment Suite (v2: Q-Learning vs DQN vs Double DQN)
 * 
 * Experiment 1: Learning Curves
 * Experiment 2: Evaluation at Checkpoints (greedy, 100 identical-seed episodes)
 * Experiment 3: Final Head-to-Head (200 identical-seed episodes)
 * 
 * All results saved to /results/experiment_results.json
 */

import { CONFIG } from '../config.js';
import { FlappyEnv } from '../env/FlappyEnv.js';
import { SimulationRunner } from '../SimulationRunner.js';
import { QLearningAgent } from '../agents/QLearningAgent.js';
import { DQNAgent } from '../agents/DQNAgent.js';
import { DoubleDQNAgent } from '../agents/DoubleDQNAgent.js';
import fs from 'fs';
import path from 'path';

import * as tf from '@tensorflow/tfjs-node';
globalThis.tf = tf;

// ─── Experiment Parameters ───────────────────────────────────

const Q_TOTAL_EPISODES = 20000;
const DQN_TOTAL_EPISODES = 20000;
const DDQN_TOTAL_EPISODES = 30000;

const Q_LOG_INTERVAL = 200;
const DQN_LOG_INTERVAL = 200;
const DDQN_LOG_INTERVAL = 300;

const Q_CHECKPOINT_INTERVAL = 2000;
const DQN_CHECKPOINT_INTERVAL = 2000;
const DDQN_CHECKPOINT_INTERVAL = 3000;

const EVAL_EPISODES = 100;
const EVAL_SEED = 0;
const FINAL_EVAL_EPISODES = 200;

// ─── Utility Functions ────────────────────────────────────────

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
        if (typeof agent.bestAction === 'function') return agent.bestAction(state);
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

// ─── Trainer for Q-Learning (sync) ───────────────────────────

function trainQLearning(totalEpisodes, logInterval, checkpointInterval) {
    console.log(`\n  ── Q-Learning: ${totalEpisodes} episodes ──`);
    const agent = new QLearningAgent();
    agent.epsilonDecay = Math.pow(agent.epsilonEnd / agent.epsilon, 1 / (totalEpisodes * 0.8));
    const env = new FlappyEnv();
    const evalEnv = new FlappyEnv();

    const learningCurve = [];
    const checkpoints = [];
    let windowScores = [];
    let bestMean = -Infinity;
    let bestQTable = null;

    let lastState, lastAction;
    const runner = new SimulationRunner(env, {
        actionProvider: (state) => {
            const action = agent.act(state);
            lastState = state;
            lastAction = action;
            return action;
        },
        onStep: (result) => agent.learn(lastState, lastAction, result.reward, result.state, result.done),
        onEpisodeEnd: (summary) => {
            agent.endEpisode();
            windowScores.push(summary.score);
            const ep = agent.episodeCount;

            if (ep % logInterval === 0) {
                const stats = computeStats(windowScores);
                learningCurve.push({ episode: ep, avgScore: stats.mean, maxScore: stats.max, epsilon: Number(agent.epsilon.toFixed(4)), qTableSize: agent.statesVisited });
                process.stdout.write(`\r    Ep ${ep}/${totalEpisodes} | Avg(${logInterval}): ${stats.mean} | Max: ${stats.max} | ε: ${agent.epsilon.toFixed(4)} | Q-States: ${agent.statesVisited}`);
                windowScores = [];
            }

            if (ep % checkpointInterval === 0) {
                const scores = evaluateAgent(agent, evalEnv, EVAL_EPISODES, EVAL_SEED);
                const stats = computeStats(scores);
                checkpoints.push({ episode: ep, ...stats });
                console.log(`\n    [Checkpoint @ ${ep}] Eval(${EVAL_EPISODES} ep): Mean=${stats.mean}, Max=${stats.max}, StdDev=${stats.stdDev}`);
                
                if (stats.mean > bestMean) {
                    bestMean = stats.mean;
                    bestQTable = new Map();
                    for (const [k, v] of agent._qTable) bestQTable.set(k, [...v]);
                    console.log(`    🌟 Nuevo mejor modelo guardado (Mean: ${stats.mean})`);
                }
            }
        }
    });

    const t0 = performance.now();
    runner.runSync({ episodes: totalEpisodes });
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
    
    if (bestQTable) {
        agent._qTable = bestQTable;
        console.log(`\n  Restaurando mejor modelo (Mean: ${bestMean}) para evaluación final.`);
    }

    console.log(`\n  Training complete in ${elapsed}s\n`);

    return { agent, learningCurve, checkpoints, trainingTimeSec: Number(elapsed) };
}

// ─── Generic Trainer for DQN-family (async) ──────────────────

async function trainDQNFamily(agentName, agent, totalEpisodes, logInterval, checkpointInterval) {
    console.log(`\n  ── ${agentName}: ${totalEpisodes} episodes ──`);
    agent.epsilonDecay = Math.pow(agent.epsilonMin / agent.epsilon, 1 / (totalEpisodes * 0.8));
    const env = new FlappyEnv();
    const evalEnv = new FlappyEnv();

    const learningCurve = [];
    const checkpoints = [];
    let windowScores = [];
    let epCount = 0;
    let bestMean = -Infinity;
    let bestWeights = null;

    let lastState, lastAction;
    const runner = new SimulationRunner(env, {
        actionProvider: (state) => {
            const action = agent.act(state);
            lastState = state;
            lastAction = action;
            return action;
        },
        onStep: (result) => agent.learn(lastState, lastAction, result.reward, result.state, result.done),
        onEpisodeEnd: (summary) => {
            agent.endEpisode();
            epCount++;
            windowScores.push(summary.score);

            if (epCount % logInterval === 0) {
                const stats = computeStats(windowScores);
                learningCurve.push({ episode: epCount, avgScore: stats.mean, maxScore: stats.max, epsilon: Number(agent.epsilon.toFixed(4)) });
                process.stdout.write(`\r    Ep ${epCount}/${totalEpisodes} | Avg(${logInterval}): ${stats.mean} | Max: ${stats.max} | ε: ${agent.epsilon.toFixed(4)}`);
                windowScores = [];
            }

            if (epCount % checkpointInterval === 0) {
                const scores = evaluateAgent(agent, evalEnv, EVAL_EPISODES, EVAL_SEED);
                const stats = computeStats(scores);
                checkpoints.push({ episode: epCount, ...stats });
                console.log(`\n    [Checkpoint @ ${epCount}] Eval(${EVAL_EPISODES} ep): Mean=${stats.mean}, Max=${stats.max}, StdDev=${stats.stdDev}`);
                
                if (stats.mean > bestMean) {
                    bestMean = stats.mean;
                    if (bestWeights) bestWeights.forEach(t => t.dispose());
                    bestWeights = agent.mainNet.getWeights().map(t => t.clone());
                    console.log(`    🌟 Nuevo mejor modelo guardado (Mean: ${stats.mean})`);
                }
            }
        }
    });

    const t0 = performance.now();
    await runner.runHeadless({ episodes: totalEpisodes, yieldEvery: 32 });
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
    
    if (bestWeights) {
        agent.mainNet.setWeights(bestWeights);
        bestWeights.forEach(t => t.dispose());
        if (typeof agent._updateTargetNetwork === 'function') {
            agent._updateTargetNetwork();
        }
        console.log(`\n  Restaurando mejor modelo (Mean: ${bestMean}) para evaluación final.`);
    }

    console.log(`\n  Training complete in ${elapsed}s\n`);

    return { agent, learningCurve, checkpoints, trainingTimeSec: Number(elapsed) };
}

// ─── Save model weights ───────────────────────────────────────

function saveQAgent(agent, dir) {
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, 'qlearning.json'), JSON.stringify(agent.save(), null, 2));
}

async function saveNNAgent(agent, dir, subdir) {
    const modelDir = path.join(dir, subdir);
    fs.mkdirSync(modelDir, { recursive: true });
    await agent.save(tf.io.withSaveHandler(async (artifacts) => {
        const modelJSON = {
            modelTopology: artifacts.modelTopology,
            format: artifacts.format,
            generatedBy: artifacts.generatedBy,
            convertedBy: artifacts.convertedBy,
            weightsManifest: [{ paths: ['weights.bin'], weights: artifacts.weightSpecs }]
        };
        fs.writeFileSync(path.join(modelDir, 'model.json'), JSON.stringify(modelJSON));
        if (artifacts.weightData) {
            fs.writeFileSync(path.join(modelDir, 'weights.bin'), Buffer.from(artifacts.weightData));
        }
        return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
    }));
}

// ─── Main ─────────────────────────────────────────────────────

async function main() {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    console.log(`\n${'═'.repeat(60)}`);
    console.log(`  AutonoLearn RL — Full Experiment Suite v2`);
    console.log(`  Agents: Q-Learning, DQN, Double DQN`);
    console.log(`  Timestamp: ${timestamp}`);
    console.log(`  Q-Learning: ${Q_TOTAL_EPISODES} ep | DQN: ${DQN_TOTAL_EPISODES} ep | DDQN: ${DDQN_TOTAL_EPISODES} ep`);
    console.log(`  Eval episodes/checkpoint: ${EVAL_EPISODES} | Final H2H: ${FINAL_EVAL_EPISODES}`);
    console.log(`  Max steps/episode: ${CONFIG.env.maxStepsPerEpisode}`);
    console.log(`${'═'.repeat(60)}`);

    // Phase 1: Q-Learning
    console.log(`\n${'─'.repeat(60)}`);
    console.log(`  PHASE 1: Q-Learning Training`);
    console.log(`${'─'.repeat(60)}`);
    const qResult = trainQLearning(Q_TOTAL_EPISODES, Q_LOG_INTERVAL, Q_CHECKPOINT_INTERVAL);

    // Phase 2: DQN
    console.log(`\n${'─'.repeat(60)}`);
    console.log(`  PHASE 2: DQN Training`);
    console.log(`${'─'.repeat(60)}`);
    const dqnAgent = new DQNAgent();
    const dqnResult = await trainDQNFamily('DQN', dqnAgent, DQN_TOTAL_EPISODES, DQN_LOG_INTERVAL, DQN_CHECKPOINT_INTERVAL);

    // Phase 3: Double DQN
    console.log(`\n${'─'.repeat(60)}`);
    console.log(`  PHASE 3: Double DQN Training`);
    console.log(`${'─'.repeat(60)}`);
    const ddqnAgent = new DoubleDQNAgent();
    const ddqnResult = await trainDQNFamily('Double DQN', ddqnAgent, DDQN_TOTAL_EPISODES, DDQN_LOG_INTERVAL, DDQN_CHECKPOINT_INTERVAL);

    // Phase 4: Final Head-to-Head
    console.log(`\n${'─'.repeat(60)}`);
    console.log(`  PHASE 4: Final Head-to-Head (${FINAL_EVAL_EPISODES} episodes, same seeds)`);
    console.log(`${'─'.repeat(60)}`);

    const evalEnv = new FlappyEnv();
    const qScores = evaluateAgent(qResult.agent, evalEnv, FINAL_EVAL_EPISODES, EVAL_SEED);
    const dqnScores = evaluateAgent(dqnResult.agent, evalEnv, FINAL_EVAL_EPISODES, EVAL_SEED);
    const ddqnScores = evaluateAgent(ddqnResult.agent, evalEnv, FINAL_EVAL_EPISODES, EVAL_SEED);

    const qFinal = computeStats(qScores);
    const dqnFinal = computeStats(dqnScores);
    const ddqnFinal = computeStats(ddqnScores);

    const pad = (v, w = 6) => String(v).padStart(w);

    console.log(`\n  🏆 RESULTADOS FINALES:`);
    console.log(`  ┌────────────────┬────────┬────────┬────────┬────────┬────────┐`);
    console.log(`  │ Agente         │   Mean │    Max │    Min │ StdDev │ Median │`);
    console.log(`  ├────────────────┼────────┼────────┼────────┼────────┼────────┤`);
    console.log(`  │ Q-Learning     │ ${pad(qFinal.mean)} │ ${pad(qFinal.max)} │ ${pad(qFinal.min)} │ ${pad(qFinal.stdDev)} │ ${pad(qFinal.median)} │`);
    console.log(`  │ DQN            │ ${pad(dqnFinal.mean)} │ ${pad(dqnFinal.max)} │ ${pad(dqnFinal.min)} │ ${pad(dqnFinal.stdDev)} │ ${pad(dqnFinal.median)} │`);
    console.log(`  │ Double DQN     │ ${pad(ddqnFinal.mean)} │ ${pad(ddqnFinal.max)} │ ${pad(ddqnFinal.min)} │ ${pad(ddqnFinal.stdDev)} │ ${pad(ddqnFinal.median)} │`);
    console.log(`  └────────────────┴────────┴────────┴────────┴────────┴────────┘`);

    // Determine winner
    const allFinals = [
        { name: 'Q-Learning', mean: qFinal.mean },
        { name: 'DQN', mean: dqnFinal.mean },
        { name: 'Double DQN', mean: ddqnFinal.mean }
    ].sort((a, b) => b.mean - a.mean);

    console.log(`\n  🥇 ${allFinals[0].name} (${allFinals[0].mean})`);
    console.log(`  🥈 ${allFinals[1].name} (${allFinals[1].mean})`);
    console.log(`  🥉 ${allFinals[2].name} (${allFinals[2].mean})`);

    // Phase 5: Save
    console.log(`\n${'─'.repeat(60)}`);
    console.log(`  PHASE 5: Saving Results & Models`);
    console.log(`${'─'.repeat(60)}`);

    const resultsDir = path.join(process.cwd(), 'results');
    fs.mkdirSync(resultsDir, { recursive: true });

    const modelsDir = path.join(process.cwd(), 'models_final');
    saveQAgent(qResult.agent, modelsDir);
    await saveNNAgent(dqnResult.agent, modelsDir, 'dqn');
    await saveNNAgent(ddqnResult.agent, modelsDir, 'ddqn');
    console.log(`  💾 Models saved to /models_final/`);

    const experimentData = {
        metadata: {
            timestamp,
            environment: 'FlappyBird',
            maxStepsPerEpisode: CONFIG.env.maxStepsPerEpisode,
            evalSeed: EVAL_SEED,
            pipeGap: CONFIG.pipe.gap,
            rewards: CONFIG.rewards
        },
        qlearning: {
            trainingEpisodes: Q_TOTAL_EPISODES,
            trainingTimeSec: qResult.trainingTimeSec,
            hyperparameters: {
                alpha: CONFIG.rl.qlearning.alpha,
                gamma: CONFIG.rl.qlearning.gamma,
                epsilonStart: 1.0, epsilonEnd: 0.01,
                epsilonDecay: qResult.agent.epsilonDecay,
                bins: CONFIG.rl.qlearning.bins,
                qTableSize: qResult.agent.statesVisited
            },
            learningCurve: qResult.learningCurve,
            checkpoints: qResult.checkpoints,
            finalEval: { ...qFinal, episodes: FINAL_EVAL_EPISODES, rawScores: qScores }
        },
        dqn: {
            trainingEpisodes: DQN_TOTAL_EPISODES,
            trainingTimeSec: dqnResult.trainingTimeSec,
            hyperparameters: {
                gamma: 0.99, learningRate: 0.001,
                epsilonStart: 1.0, epsilonEnd: 0.01,
                epsilonDecay: dqnResult.agent.epsilonDecay,
                batchSize: 64, replayBufferSize: 10000,
                targetUpdateFreq: 5000,
                networkArchitecture: '3 → 24(ReLU) → 24(ReLU) → 2(Linear)',
                algorithm: 'Vanilla DQN: target = r + γ·max Q_target(s\',a\')'
            },
            learningCurve: dqnResult.learningCurve,
            checkpoints: dqnResult.checkpoints,
            finalEval: { ...dqnFinal, episodes: FINAL_EVAL_EPISODES, rawScores: dqnScores }
        },
        doubleDqn: {
            trainingEpisodes: DDQN_TOTAL_EPISODES,
            trainingTimeSec: ddqnResult.trainingTimeSec,
            hyperparameters: {
                gamma: 0.99, learningRate: 0.001,
                epsilonStart: 1.0, epsilonEnd: 0.01,
                epsilonDecay: ddqnResult.agent.epsilonDecay,
                batchSize: 64, replayBufferSize: 10000,
                targetUpdateFreq: 5000,
                networkArchitecture: '3 → 24(ReLU) → 24(ReLU) → 2(Linear)',
                algorithm: 'Double DQN: target = r + γ·Q_target(s\', argmax Q_main(s\',a\'))'
            },
            learningCurve: ddqnResult.learningCurve,
            checkpoints: ddqnResult.checkpoints,
            finalEval: { ...ddqnFinal, episodes: FINAL_EVAL_EPISODES, rawScores: ddqnScores }
        }
    };

    fs.writeFileSync(
        path.join(resultsDir, 'experiment_results_final.json'),
        JSON.stringify(experimentData, null, 2)
    );
    console.log(`  📊 Full results saved to /results/experiment_results_final.json`);

    console.log(`\n${'═'.repeat(60)}`);
    console.log(`  EXPERIMENT COMPLETE`);
    console.log(`${'═'.repeat(60)}\n`);
}

main().catch(console.error);
