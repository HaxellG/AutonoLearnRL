import { CONFIG, Action } from '../config.js';
import { StateDiscretizer } from './StateDiscretizer.js';
import { SeededRandom } from '../utils/SeededRandom.js';

/**
 * Q-Learning Agent — Tabular RL with ε-greedy policy.
 *
 * Uses a Map-based Q-table keyed by discretized state "i|j|k".
 * Each entry stores [Q(NO_OP), Q(FLAP)].
 *
 * Epsilon decays multiplicatively per episode.
 */
export class QLearningAgent {
    constructor(options = {}) {
        const cfg = CONFIG.rl.qlearning;

        this.alpha = options.alpha ?? cfg.alpha;
        this.gamma = options.gamma ?? cfg.gamma;
        this.epsilon = options.epsilonStart ?? cfg.epsilonStart;
        this.epsilonEnd = options.epsilonEnd ?? cfg.epsilonEnd;
        this.epsilonDecay = options.epsilonDecay ?? cfg.epsilonDecay;
        this.qInit = options.qInit ?? cfg.qInit;

        this._discretizer = new StateDiscretizer(options.bins ?? cfg.bins);
        this._rng = new SeededRandom(options.seed ?? 7777);

        /** @type {Map<string, number[]>} */
        this._qTable = new Map();

        this.episodeCount = 0;
        this.totalSteps = 0;
    }

    /**
     * Select action using ε-greedy policy.
     * @param {number[]} state — [dx, dy, vy]
     * @returns {number}
     */
    act(state) {
        if (this._rng.next() < this.epsilon) {
            // 5% flap is roughly level flight, 50% flap shoots straight to ceiling
            return this._rng.next() < 0.05 ? Action.FLAP : Action.NO_OP;
        }
        const key = this._discretizer.discretize(state).key;
        const q = this._getQ(key);
        if (q[0] > q[1]) return Action.NO_OP;
        if (q[1] > q[0]) return Action.FLAP;
        return this._rng.next() < 0.5 ? Action.NO_OP : Action.FLAP;
    }

    /**
     * Greedy action (no exploration). For evaluation.
     * @param {number[]} state
     * @returns {number}
     */
    bestAction(state) {
        const key = this._discretizer.discretize(state).key;
        const q = this._getQ(key);
        if (q[0] > q[1]) return Action.NO_OP;
        if (q[1] > q[0]) return Action.FLAP;
        return Action.NO_OP;
    }

    /**
     * Q-learning TD update.
     */
    learn(state, action, reward, nextState, done) {
        const sKey = this._discretizer.discretize(state).key;
        const nsKey = this._discretizer.discretize(nextState).key;

        const q = this._getQ(sKey);
        const qNext = this._getQ(nsKey);

        const maxQNext = Math.max(qNext[0], qNext[1]);
        const target = reward + (done ? 0 : this.gamma * maxQNext);
        q[action] += this.alpha * (target - q[action]);

        this.totalSteps++;
    }

    /** Call at end of each episode to decay epsilon. */
    endEpisode() {
        this.episodeCount++;
        this.epsilon = Math.max(this.epsilonEnd, this.epsilon * this.epsilonDecay);
    }

    /** Serialize to JSON. */
    save() {
        const entries = [];
        for (const [key, qv] of this._qTable) {
            entries.push([key, qv[0], qv[1]]);
        }
        return {
            type: 'QLearningAgent',
            alpha: this.alpha,
            gamma: this.gamma,
            epsilon: this.epsilon,
            epsilonEnd: this.epsilonEnd,
            epsilonDecay: this.epsilonDecay,
            episodeCount: this.episodeCount,
            totalSteps: this.totalSteps,
            qTableSize: this._qTable.size,
            qTable: entries,
        };
    }

    /** Restore from JSON. */
    load(data) {
        this.alpha = data.alpha;
        this.gamma = data.gamma;
        this.epsilon = data.epsilon;
        this.epsilonEnd = data.epsilonEnd;
        this.epsilonDecay = data.epsilonDecay ?? 0.999;
        this.episodeCount = data.episodeCount;
        this.totalSteps = data.totalSteps ?? 0;
        this._qTable.clear();
        for (const e of data.qTable) {
            this._qTable.set(e[0], [e[1], e[2]]);
        }
    }

    /** @returns {number} */
    get statesVisited() {
        return this._qTable.size;
    }

    _getQ(key) {
        if (!this._qTable.has(key)) {
            this._qTable.set(key, [this.qInit, this.qInit]);
        }
        return this._qTable.get(key);
    }
}
