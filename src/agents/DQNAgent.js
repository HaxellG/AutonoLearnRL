import { CONFIG, Action } from '../config.js';
import { ReplayBuffer } from './ReplayBuffer.js';

/**
 * Deep Q-Network Agent
 * 
 * Approximates Q(s, a) using a Feed-Forward Neural Network via TensorFlow.js.
 * Utilizes a Replay Buffer and a Target Network for stability.
 */
export class DQNAgent {
    constructor() {
        // Ensure TF is available (Browser via CDN, Node via direct import to global)
        this.tf = globalThis.tf || window?.tf;
        if (!this.tf) throw new Error("TensorFlow.js not loaded on global object.");

        // Hyperparameters
        this.gamma = 0.99;       // Discount factor (look further ahead)
        this.epsilon = 1.0;      // Exploration rate
        this.epsilonMin = 0.01;
        this.epsilonDecay = 0.999;

        this.learningRate = 0.001;
        this.batchSize = 64;
        this.targetUpdateFreq = 10000; // Increased to prevent rapid forgetting

        this.actionSpace = 2; // [0, 1]
        this.stateSize = 3;   // [dx, dy, vy]

        // Tracking
        this.steps = 0;
        this.statesVisited = 0; // Keeping interface similar to QLearningAgent

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
     * This is CRITICAL. Without it, the network gradients explode and it learns nothing.
     */
    _normalize(state) {
        const [dx, dy, vy] = state;
        // dx ranges from roughly 0 to 288 (canvas width)
        const normDx = (dx / 288) * 2 - 1;
        // dy ranges from roughly -400 to 400
        const normDy = (dy / 400);
        // vy ranges from roughly -8 to +15
        const normVy = (vy / 15);
        return [normDx, normDy, normVy];
    }

    /**
     * Choose an action based on epsilon-greedy policy.
     * @param {Array<number>} state - Continuous state vector [dx, dy, vy]
     */
    act(state) {
        // Epsilon greedy exploration
        if (Math.random() < this.epsilon) {
            return Math.random() < 0.05 ? Action.FLAP : Action.NO_OP;
        }

        // Exploitation
        return this.tf.tidy(() => {
            const normState = this._normalize(state);
            const stateTensor = this.tf.tensor2d([normState]);
            const qValues = this.mainNet.predict(stateTensor);
            return qValues.argMax(1).dataSync()[0];
        });
    }

    /**
     * Greedy action (no exploration). For evaluation.
     * @param {Array<number>} state - Continuous state vector [dx, dy, vy]
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
        // 1. Store transition in replay buffer
        this.memory.add(state, action, reward, nextState, done);

        // 2. Train if we have enough samples
        if (this.memory.size >= this.batchSize) {
            this._replay();
        }

        // 3. Update target network periodically
        this.steps++;
        if (this.steps % this.targetUpdateFreq === 0) {
            this._updateTargetNetwork();
        }
    }

    /**
     * Train the neural network on a mini-batch from the ReplayBuffer.
     * Making this async so we can await trainOnBatch safely without blocking the UI thread completely,
     * while managing tensor memory manually instead of tf.tidy which doesn't support async.
     */
    async _replay() {
        if (this._isTraining) return;
        this._isTraining = true;

        const batch = this.memory.sample(this.batchSize);

        // Unzip the batch and normalize states!
        const states = batch.map(b => this._normalize(b.state));
        const actions = batch.map(b => b.action);
        const rewards = batch.map(b => b.reward);
        const nextStates = batch.map(b => this._normalize(b.nextState));
        const dones = batch.map(b => b.done);

        // We use tf.tidy for the synchronous part of the prediction
        const { statesT, updatedQsT } = this.tf.tidy(() => {
            const sT = this.tf.tensor2d(states);
            const nsT = this.tf.tensor2d(nextStates);
            const rT = this.tf.tensor1d(rewards);
            const dT = this.tf.tensor1d(dones);

            // Predict Q-values for current states from MAIN network
            const qs = this.mainNet.predict(sT).arraySync();

            // Predict Q-values for next states from TARGET network
            const nextQs = this.targetNet.predict(nsT);
            const maxNextQs = nextQs.max(1);

            // Compute target Q-values (Bellman equation)
            const targetQs = rT.add(
                maxNextQs.mul(this.gamma).mul(dT.logicalNot().cast('float32'))
            ).arraySync();

            // Update the Q-values for the specific actions taken
            for (let i = 0; i < this.batchSize; i++) {
                qs[i][actions[i]] = targetQs[i];
            }

            // Return the tensors we need for trainOnBatch (keep them so tidy doesn't destroy them)
            return {
                statesT: this.tf.keep(sT),
                updatedQsT: this.tf.keep(this.tf.tensor2d(qs))
            };
        });

        // Async training (won't kill the main thread entirely since it yields)
        await this.mainNet.trainOnBatch(statesT, updatedQsT);

        // Manual disposal of the retained tensors
        statesT.dispose();
        updatedQsT.dispose();

        this._isTraining = false;
    }

    /**
     * Called when the episode ends.
     */
    endEpisode() {
        if (this.epsilon > this.epsilonMin) {
            this.epsilon *= this.epsilonDecay;
        }
    }

    /**
     * Saves the model weights securely to a path/URL using TensorFlow backend
     * @param {string} url - Target destination (ex: 'file://./models/dqn' or 'localstorage://dqn')
     */
    async save(url) {
        await this.mainNet.save(url);
    }

    /**
     * Loads the model weights from a path/URL
     * @param {string} url - Location of the model.json (ex: 'http://localhost:5500/models/dqn/model.json' or 'file://...')
     */
    async load(url) {
        this.mainNet = await this.tf.loadLayersModel(url);
        
        // Re-compile since loading only gives architecture and weights
        this.mainNet.compile({
            optimizer: this.tf.train.adam(this.learningRate),
            loss: 'meanSquaredError'
        });

        // Sync target network
        this._updateTargetNetwork();
    }
}
