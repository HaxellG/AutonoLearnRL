import { CONFIG } from '../config.js';
import { FlappyEnv } from '../env/FlappyEnv.js';
import { SimulationRunner } from '../SimulationRunner.js';
import { QLearningAgent } from '../agents/QLearningAgent.js';
import fs from 'fs';
import path from 'path';

const TOTAL_EPISODES = 20000;
const EVAL_EPISODES = 100;
const EVAL_SEED = 0;

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
    const greedyPolicy = (state) => agent.bestAction(state);
    const runner = new SimulationRunner(env, { actionProvider: greedyPolicy });
    const scores = [];
    for (let i = 0; i < episodes; i++) {
        const results = runner.runSync({ episodes: 1, seed: startSeed + i });
        scores.push(results[0].score);
    }
    return scores;
}

async function runTrainingStability() {
    console.log('--- Training Stability (10 runs) ---');
    const runs = [];
    for (let i = 0; i < 10; i++) {
        const agent = new QLearningAgent({ seed: 7777 + i * 13 });
        agent.epsilonDecay = Math.pow(0.01 / 1.0, 1 / (TOTAL_EPISODES * 0.8));
        const env = new FlappyEnv();
        const evalEnv = new FlappyEnv();
        
        let lastState, lastAction;
        let bestMean = -Infinity;
        let bestQTable = null;

        const runner = new SimulationRunner(env, {
            actionProvider: (state) => {
                const action = agent.act(state);
                lastState = state;
                lastAction = action;
                return action;
            },
            onStep: (result) => agent.learn(lastState, lastAction, result.reward, result.state, result.done),
            onEpisodeEnd: () => {
                agent.endEpisode();
                if (agent.episodeCount % 2000 === 0) {
                    const scores = evaluateAgent(agent, evalEnv, EVAL_EPISODES, EVAL_SEED);
                    const stats = computeStats(scores);
                    if (stats.mean > bestMean) {
                        bestMean = stats.mean;
                        bestQTable = new Map();
                        for (const [k, v] of agent._qTable) bestQTable.set(k, [...v]);
                    }
                }
            }
        });
        runner.runSync({ episodes: TOTAL_EPISODES });
        
        if (bestQTable) agent._qTable = bestQTable;
        
        const finalScores = evaluateAgent(agent, evalEnv, 200, 0);
        const finalStats = computeStats(finalScores);
        
        console.log(`Run ${String(i+1).padStart(2, ' ')}: Mean = ${finalStats.mean.toFixed(1).padStart(4, ' ')} | Max = ${String(finalStats.max).padStart(3, ' ')} | StdDev = ${finalStats.stdDev.toFixed(1).padStart(4, ' ')}`);
        runs.push(finalStats);
    }
    
    fs.mkdirSync('results/stability', { recursive: true });
    fs.writeFileSync('results/stability/training_runs_10.json', JSON.stringify(runs, null, 2));
    console.log('Saved to results/stability/training_runs_10.json\n');
}

async function main() {
    CONFIG.pipe.gap = 125;
    await runTrainingStability();
}
main();
