/**
 * Q-Learning Training Script
 *
 * Run with:  node src/examples/qlearn_train.js
 */
import { FlappyEnv } from '../env/FlappyEnv.js';
import { QLearningAgent } from '../agents/QLearningAgent.js';
import { QTrainer } from '../agents/QTrainer.js';
import { CONFIG } from '../config.js';
import { writeFileSync } from 'node:fs';

const cfg = CONFIG.rl.qlearning;
const EPISODES = cfg.training.episodes;
const LOG_EVERY = cfg.training.logEvery;
const MA_WINDOW = cfg.training.movingAvgWindow;

const env = new FlappyEnv();
const agent = new QLearningAgent({ seed: 42 });
const trainer = new QTrainer(agent, env);

console.log('\n🧠 Q-Learning Training');
console.log(`   Episodes:   ${EPISODES}`);
console.log(`   α=${agent.alpha}  γ=${agent.gamma}  ε=${agent.epsilon}→${agent.epsilonEnd}`);
console.log(`   ε decay:    ×${agent.epsilonDecay}/episode`);
console.log(`   States max: ${agent._discretizer.stateCount}`);
console.log('');

const t0 = performance.now();
const rewardHistory = [];
const scoreHistory = [];

trainer.train({
    episodes: EPISODES,
    onEpisode: (log) => {
        rewardHistory.push(log.totalReward);
        scoreHistory.push(log.score);

        if ((log.episode + 1) % LOG_EVERY === 0) {
            const start = Math.max(0, rewardHistory.length - MA_WINDOW);
            const rr = rewardHistory.slice(start);
            const ss = scoreHistory.slice(start);
            const avgR = rr.reduce((a, b) => a + b, 0) / rr.length;
            const avgS = ss.reduce((a, b) => a + b, 0) / ss.length;
            const maxS = Math.max(...ss);

            console.log(
                `  Ep ${String(log.episode + 1).padStart(6)} | ` +
                `ε=${log.epsilon.toFixed(4)} | ` +
                `AvgR=${avgR.toFixed(1).padStart(6)} | ` +
                `AvgS=${avgS.toFixed(2).padStart(6)} | ` +
                `MaxS=${String(maxS).padStart(3)} | ` +
                `States=${log.statesVisited}`
            );
        }
    },
});

const t1 = performance.now();
const totalSteps = rewardHistory.length > 0
    ? agent.totalSteps : 0;

console.log('');
console.log('── Training Complete ────────────────────────');
console.log(`   Time:         ${((t1 - t0) / 1000).toFixed(2)}s`);
console.log(`   Total steps:  ${totalSteps.toLocaleString()}`);
console.log(`   States seen:  ${agent.statesVisited}`);
console.log(`   Final ε:      ${agent.epsilon.toFixed(4)}`);

const last = rewardHistory.slice(-MA_WINDOW);
const lastS = scoreHistory.slice(-MA_WINDOW);
console.log('');
console.log(`── Final Performance (last ${MA_WINDOW} ep) ──────────`);
console.log(`   Avg Reward:   ${(last.reduce((a, b) => a + b, 0) / last.length).toFixed(2)}`);
console.log(`   Avg Score:    ${(lastS.reduce((a, b) => a + b, 0) / lastS.length).toFixed(2)}`);
console.log(`   Max Score:    ${Math.max(...lastS)}`);

const saveData = agent.save();
const savePath = 'qtable.json';
writeFileSync(savePath, JSON.stringify(saveData));
console.log(`\n💾 Q-table saved to ${savePath} (${agent.statesVisited} states)\n`);
