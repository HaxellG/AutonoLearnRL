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
    console.log('--- Training Stability (5 runs) ---');
    const runs = [];
    for (let i = 0; i < 5; i++) {
        const agent = new QLearningAgent({ seed: 7777 + i * 10 });
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
        
        // Restore best
        if (bestQTable) agent._qTable = bestQTable;
        
        // Final greedy eval (200 episodes)
        const finalScores = evaluateAgent(agent, evalEnv, 200, 0);
        const finalStats = computeStats(finalScores);
        
        console.log(`Run ${i+1}: Mean=${finalStats.mean}, Max=${finalStats.max}, StdDev=${finalStats.stdDev}`);
        runs.push(finalStats);
    }
    
    fs.mkdirSync('results/stability', { recursive: true });
    fs.writeFileSync('results/stability/training_runs.json', JSON.stringify(runs, null, 2));
    console.log('Saved to results/stability/training_runs.json\n');
}

async function runEvalStability() {
    console.log('--- Evaluation Stability (15 separate seeds) ---');
    const agent = new QLearningAgent();
    // Load the golden model
    const qData = JSON.parse(fs.readFileSync('models_final/medium/qlearning.json'));
    agent.load(qData);
    
    const env = new FlappyEnv();
    const scores = [];
    
    for (let i = 0; i < 15; i++) {
        // Evaluate on a specific seed (one run per seed to see variance)
        const scoreArr = evaluateAgent(agent, env, 1, 1000 + i);
        scores.push({ seed: 1000 + i, score: scoreArr[0] });
        console.log(`Eval Run ${i+1} (Seed ${1000 + i}): Score = ${scoreArr[0]}`);
    }
    
    const stats = computeStats(scores.map(s => s.score));
    console.log(`Eval Stats: Mean=${stats.mean}, StdDev=${stats.stdDev}`);
    
    fs.writeFileSync('results/stability/eval_runs.json', JSON.stringify({ runs: scores, stats }, null, 2));
    console.log('Saved to results/stability/eval_runs.json\n');
}

async function main() {
    // Force gap to 125 manually
    CONFIG.pipe.gap = 125;
    await runTrainingStability();
    await runEvalStability();
}
main();
