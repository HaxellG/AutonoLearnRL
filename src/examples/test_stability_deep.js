import { CONFIG } from '../config.js';
import { FlappyEnv } from '../env/FlappyEnv.js';
import { SimulationRunner } from '../SimulationRunner.js';
import { DQNAgent } from '../agents/DQNAgent.js';
import { DoubleDQNAgent } from '../agents/DoubleDQNAgent.js';
import fs from 'fs';
import path from 'path';
import * as tf from '@tensorflow/tfjs-node';

globalThis.tf = tf;

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
    const greedyPolicy = (state) => {
        const oldEps = agent.epsilon;
        agent.epsilon = 0;
        const act = agent.act(state);
        agent.epsilon = oldEps;
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

async function runTraining(AgentClass, name, runsCount) {
    console.log(`--- Training Stability ${name} (${runsCount} runs) ---`);
    const runs = [];
    for (let i = 0; i < runsCount; i++) {
        const agent = new AgentClass();
        agent.epsilonDecay = Math.pow(agent.epsilonMin / agent.epsilon, 1 / (TOTAL_EPISODES * 0.8));
        const env = new FlappyEnv();
        const evalEnv = new FlappyEnv();
        
        let lastState, lastAction;
        let bestMean = -Infinity;
        let bestWeights = null;
        let epCount = 0;

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
                epCount++;
                if (epCount % 2000 === 0) {
                    const scores = evaluateAgent(agent, evalEnv, EVAL_EPISODES, EVAL_SEED);
                    const stats = computeStats(scores);
                    if (stats.mean > bestMean) {
                        bestMean = stats.mean;
                        if (bestWeights) bestWeights.forEach(t => t.dispose());
                        bestWeights = agent.mainNet.getWeights().map(t => t.clone());
                    }
                }
            }
        });
        
        // Use yieldEvery to avoid blocking event loop completely
        await runner.runHeadless({ episodes: TOTAL_EPISODES, yieldEvery: 50 });
        
        if (bestWeights) {
            agent.mainNet.setWeights(bestWeights);
            bestWeights.forEach(t => t.dispose());
            agent._updateTargetNetwork();
        }
        
        const finalScores = evaluateAgent(agent, evalEnv, 200, 0);
        const finalStats = computeStats(finalScores);
        
        console.log(`Run ${i+1}: Mean=${finalStats.mean}, Max=${finalStats.max}, StdDev=${finalStats.stdDev}`);
        runs.push(finalStats);
    }
    return runs;
}

async function runEvalStability() {
    console.log('--- Evaluation Stability (15 separate seeds) ---');
    const env = new FlappyEnv();
    
    // DQN
    console.log('Evaluating DQN...');
    const dqnAgent = new DQNAgent();
    await dqnAgent.load('file://' + path.resolve('models_final/medium/dqn/model.json'));
    const dqnScores = [];
    for (let i = 0; i < 15; i++) {
        const scoreArr = evaluateAgent(dqnAgent, env, 1, 1000 + i);
        dqnScores.push({ seed: 1000 + i, score: scoreArr[0] });
        console.log(`DQN Eval Run ${i+1} (Seed ${1000 + i}): Score = ${scoreArr[0]}`);
    }
    const dqnStats = computeStats(dqnScores.map(s => s.score));
    console.log(`DQN Eval Stats: Mean=${dqnStats.mean}, StdDev=${dqnStats.stdDev}`);
    
    // DDQN
    console.log('Evaluating DDQN...');
    const ddqnAgent = new DoubleDQNAgent();
    await ddqnAgent.load('file://' + path.resolve('models_final/medium/ddqn/model.json'));
    const ddqnScores = [];
    for (let i = 0; i < 15; i++) {
        const scoreArr = evaluateAgent(ddqnAgent, env, 1, 1000 + i);
        ddqnScores.push({ seed: 1000 + i, score: scoreArr[0] });
        console.log(`DDQN Eval Run ${i+1} (Seed ${1000 + i}): Score = ${scoreArr[0]}`);
    }
    const ddqnStats = computeStats(ddqnScores.map(s => s.score));
    console.log(`DDQN Eval Stats: Mean=${ddqnStats.mean}, StdDev=${ddqnStats.stdDev}`);
    
    const results = {
        dqn: { runs: dqnScores, stats: dqnStats },
        ddqn: { runs: ddqnScores, stats: ddqnStats }
    };
    
    fs.mkdirSync('results/stability', { recursive: true });
    fs.writeFileSync('results/stability/eval_runs_deep.json', JSON.stringify(results, null, 2));
}

async function main() {
    CONFIG.pipe.gap = 125; // Normal difficulty
    
    // 1. Eval stability on golden models
    await runEvalStability();
    
    // 2. Train 5 DQNs and 5 DDQNs
    const dqnRuns = await runTraining(DQNAgent, 'DQN', 5);
    const ddqnRuns = await runTraining(DoubleDQNAgent, 'DDQN', 5);
    
    fs.writeFileSync('results/stability/training_runs_deep.json', JSON.stringify({
        dqn: dqnRuns,
        ddqn: ddqnRuns
    }, null, 2));
    
    console.log('All stability testing for deep networks completed!');
}

main().catch(err => console.error(err));
