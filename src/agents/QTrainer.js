import { FlappyEnv } from '../env/FlappyEnv.js';
import { QLearningAgent } from './QLearningAgent.js';

/**
 * QTrainer — Headless training loop for Q-learning agent.
 */
export class QTrainer {
    /**
     * @param {QLearningAgent} agent
     * @param {FlappyEnv} env
     */
    constructor(agent, env) {
        this._agent = agent;
        this._env = env;
    }

    /**
     * Train the agent for N episodes.
     *
     * @param {object} opts
     * @param {number}  opts.episodes
     * @param {number}  [opts.seed] — fixed seed for all episodes (undefined = auto-seed)
     * @param {function} [opts.onEpisode] — callback(log) after each episode
     * @returns {object[]}
     */
    train({ episodes, seed, seedStart = 0, onEpisode }) {
        const logs = [];

        for (let ep = 0; ep < episodes; ep++) {
            // Use incrementing seeds for diverse episodes
            const epSeed = seed !== undefined ? seed : seedStart + ep;
            const log = this._runEpisode(epSeed);

            log.episode = ep;
            log.epsilon = this._agent.epsilon;
            log.statesVisited = this._agent.statesVisited;
            log.totalAgentSteps = this._agent.totalSteps;

            logs.push(log);
            this._agent.endEpisode();

            if (onEpisode) onEpisode(log);
        }

        return logs;
    }

    /**
     * Run a single episode.
     * @param {number} [seed]
     * @returns {{ totalReward: number, score: number, steps: number }}
     */
    _runEpisode(seed) {
        let state = this._env.reset(seed);
        let totalReward = 0;
        let steps = 0;

        while (!this._env.done) {
            const action = this._agent.act(state);
            const { state: nextState, reward, done } = this._env.step(action);

            this._agent.learn(state, action, reward, nextState, done);

            state = nextState;
            totalReward += reward;
            steps++;
        }

        return { totalReward, score: this._env.score, steps };
    }
}
