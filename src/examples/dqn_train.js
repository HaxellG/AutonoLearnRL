import * as tf from '@tensorflow/tfjs';
import { FlappyEnv } from '../env/FlappyEnv.js';
import { DQNAgent } from '../agents/DQNAgent.js';
import { SimulationRunner } from '../SimulationRunner.js';
import fs from 'fs';

// 1. Hook TF into global so DQNAgent finds it
global.tf = tf;
globalThis.tf = tf;

const env = new FlappyEnv();
const agent = new DQNAgent();

// Adjust hyperparams for faster headless testing
agent.epsilonDecay = 0.999;
const episodes = 10000;

console.log(`\n🧠 DQN Training (Headless)`);
console.log(`   Episodes:   ${episodes}`);
console.log(`   α=${agent.learningRate}  γ=${agent.gamma}  ε=${agent.epsilon}→${agent.epsilonMin}`);
console.log(`   ε decay:    ×${agent.epsilonDecay}/episode\n`);

let lastState = null;
let lastAction = null;
let sumReward = 0;
let sumScore = 0;
let maxScore = 0;

let startTime = Date.now();
const printEvery = 100;

const runner = new SimulationRunner(env, {
    actionProvider: (state) => {
        const action = agent.act(state);
        lastState = state;
        lastAction = action;
        return action;
    },
    onStep: (result) => {
        agent.learn(lastState, lastAction, result.reward, result.state, result.done);
    },
    onEpisodeEnd: (summary) => {
        agent.endEpisode();

        sumReward += summary.totalReward;
        sumScore += summary.score;
        if (summary.score > maxScore) maxScore = summary.score;

        const ep = runner._episodeCount || 0; // We need to track it manually or use a loop

        // Actually in runHeadless, SimulationRunner loops for us and returns summaries.
        // We will just do a standard loop to have real-time printing.
    }
});

async function train() {
    let globalSteps = 0;

    for (let ep = 1; ep <= episodes; ep++) {
        let state = env.reset(ep); // Seed per episode for diversity
        let episodeReward = 0;
        let score = 0;

        while (!env.done) {
            const action = agent.act(state);
            const result = env.step(action);

            // Replay Buffer and Async trainOnBatch processing
            await agent.learn(state, action, result.reward, result.state, result.done);

            state = result.state;
            score = env.score;
            episodeReward += result.reward;
            globalSteps++;
        }

        agent.endEpisode();

        sumReward += episodeReward;
        sumScore += score;
        if (score > maxScore) maxScore = score;

        if (ep % printEvery === 0) {
            const avgR = (sumReward / printEvery).toFixed(1);
            const avgS = (sumScore / printEvery).toFixed(2);
            const epsDesc = agent.epsilon.toFixed(4);

            console.log(`  Ep ${ep.toString().padStart(6)} | ε=${epsDesc} | AvgR=${avgR.padStart(6)} | AvgS=${avgS.padStart(5)} | MaxS=${maxScore.toString().padStart(3)} | Buffer=${agent.memory.size}`);

            // Reset window stats
            sumReward = 0;
            sumScore = 0;
        }
    }

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
    console.log(`\n── Training Complete ────────────────────────`);
    console.log(`   Time:         ${elapsed}s`);
    console.log(`   Total steps:  ${globalSteps}`);
    console.log(`   Final ε:      ${agent.epsilon.toFixed(4)}`);

    // Save model
    // await agent.mainNet.save('file://./dqn_model');
}

train().catch(console.error);
