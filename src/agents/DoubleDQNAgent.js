import { CONFIG, Action } from '../config.js';
import { ReplayBuffer } from './ReplayBuffer.js';

/**
 * Double Deep Q-Network Agent
 * 
 * Addresses the overestimation bias of vanilla DQN (van Hasselt et al., 2016).
 * 
 * Key difference from DQN:
 *   - DQN:        target = r + γ · max_a' Q_target(s', a')
 *   - Double DQN: target = r + γ · Q_target(s', argmax_a' Q_main(s', a'))
 * 
 * The MAIN network selects the best action for the next state,
 * but the TARGET network evaluates its Q-value. This decoupling
 * prevents the same network from both proposing and judging,
 * reducing systematic overestimation of Q-values.
 */
export class DoubleDQNAgent {
    constructor() {
        this.tf = globalThis.tf || window?.tf;
        if (!this.tf) throw new Error("TensorFlow.js not loaded on global object.");

        // Hyperparameters (identical to DQN for fair comparison)
        this.gamma = 0.99;
        this.epsilon = 1.0;
        this.epsilonMin = 0.01;
        this.epsilonDecay = 0.999;

        this.learningRate = 0.001;
        this.batchSize = 64;
        this.targetUpdateFreq = 10000; // Increased to prevent rapid forgetting

        this.actionSpace = 2;
        this.stateSize = 3;

        // Tracking
        this.steps = 0;
        this.statesVisited = 0;

        // Modules
        this.memory = new ReplayBuffer(50000); // 5x larger to remember early game

        // Networks
        this.mainNet = this._buildModel();
        this.targetNet = this._buildModel();
        this._updateTargetNetwork();
    }

    _buildModel() {
        const model = this.tf.sequential();
        model.add(this.tf.layers.dense({ inputShape: [this.stateSize], units: 24, activation: 'relu' }));
        model.add(this.tf.layers.dense({ units: 24, activation: 'relu' }));
        model.add(this.tf.layers.dense({ units: this.actionSpace, activation: 'linear' }));

        model.compile({
            optimizer: this.tf.train.adam(this.learningRate),
            loss: 'meanSquaredError'
        });

        return model;
    }

    _updateTargetNetwork() {
        this.targetNet.setWeights(this.mainNet.getWeights());
    }

    /**
     * Normalizes the physical state into [-1, 1] ranges for the Neural Network.
     */
    _normalize(state) {
        const [dx, dy, vy] = state;
        const normDx = (dx / 288) * 2 - 1;
        const normDy = (dy / 400);
        const normVy = (vy / 15);
        return [normDx, normDy, normVy];
    }

    /**
     * Choose an action based on epsilon-greedy policy.
     */
    act(state) {
        if (Math.random() < this.epsilon) {
            return Math.random() < 0.05 ? Action.FLAP : Action.NO_OP;
        }

        return this.tf.tidy(() => {
            const normState = this._normalize(state);
            const stateTensor = this.tf.tensor2d([normState]);
            const qValues = this.mainNet.predict(stateTensor);
            return qValues.argMax(1).dataSync()[0];
        });
    }

    /**
     * Greedy action (no exploration). For evaluation.
     */
    bestAction(state) {
        return this.tf.tidy(() => {
            const normState = this._normalize(state);
            const stateTensor = this.tf.tensor2d([normState]);
            const qValues = this.mainNet.predict(stateTensor);
            return qValues.argMax(1).dataSync()[0];
        });
    }

    /**
     * Called every step to store the transition and occasionally train.
     */
    learn(state, action, reward, nextState, done) {
        this.memory.add(state, action, reward, nextState, done);

        if (this.memory.size >= this.batchSize) {
            this._replay();
        }

        this.steps++;
        if (this.steps % this.targetUpdateFreq === 0) {
            this._updateTargetNetwork();
        }
    }

    /**
     * DOUBLE DQN replay — the core algorithmic difference.
     * 
     * Instead of:  target = r + γ · max_a' Q_target(s', a')      [vanilla DQN]
     * We compute:  target = r + γ · Q_target(s', argmax_a' Q_main(s', a'))  [Double DQN]
     * 
     * Step 1: Main network selects the best action for each next state
     * Step 2: Target network evaluates Q-value of that selected action
     * This prevents the maximization bias where the same network both
     * proposes and evaluates actions.
     */
    async _replay() {
        if (this._isTraining) return;
        this._isTraining = true;

        const batch = this.memory.sample(this.batchSize);

        const states = batch.map(b => this._normalize(b.state));
        const actions = batch.map(b => b.action);
        const rewards = batch.map(b => b.reward);
        const nextStates = batch.map(b => this._normalize(b.nextState));
        const dones = batch.map(b => b.done);

        const { statesT, updatedQsT } = this.tf.tidy(() => {
            const sT = this.tf.tensor2d(states);
            const nsT = this.tf.tensor2d(nextStates);
            const rT = this.tf.tensor1d(rewards);
            const dT = this.tf.tensor1d(dones);

            // Current Q-values from main network
            const qs = this.mainNet.predict(sT).arraySync();

            // ═══════════ DOUBLE DQN CORE ═══════════
            // Step 1: MAIN network selects the best action for each next state
            const mainNextQs = this.mainNet.predict(nsT);
            const bestActions = mainNextQs.argMax(1).arraySync();

            // Step 2: TARGET network evaluates Q-value at the action selected by main
            const targetNextQs = this.targetNet.predict(nsT).arraySync();

            // Build target values using the decoupled selection/evaluation
            const targetValues = [];
            for (let i = 0; i < this.batchSize; i++) {
                if (dones[i]) {
                    targetValues.push(rewards[i]);
                } else {
                    // Q_target(s', argmax_a' Q_main(s', a'))
                    targetValues.push(rewards[i] + this.gamma * targetNextQs[i][bestActions[i]]);
                }
            }
            // ═══════════════════════════════════════

            // Update Q-values for the specific actions taken
            for (let i = 0; i < this.batchSize; i++) {
                qs[i][actions[i]] = targetValues[i];
            }

            return {
                statesT: this.tf.keep(sT),
                updatedQsT: this.tf.keep(this.tf.tensor2d(qs))
            };
        });

        await this.mainNet.trainOnBatch(statesT, updatedQsT);

        statesT.dispose();
        updatedQsT.dispose();

        this._isTraining = false;
    }

    endEpisode() {
        if (this.epsilon > this.epsilonMin) {
            this.epsilon *= this.epsilonDecay;
        }
    }

    async save(url) {
        await this.mainNet.save(url);
    }

    async load(url) {
        this.mainNet = await this.tf.loadLayersModel(url);
        this.mainNet.compile({
            optimizer: this.tf.train.adam(this.learningRate),
            loss: 'meanSquaredError'
        });
        this._updateTargetNetwork();
    }
}
