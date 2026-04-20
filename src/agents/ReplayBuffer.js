/**
 * Replay Buffer for Deep Q-Network (DQN).
 * Stores transitions (state, action, reward, nextState, done) and
 * allows sampling of random mini-batches for decorrelated training.
 */
export class ReplayBuffer {
    /**
     * @param {number} capacity - Maximum number of transitions to hold
     */
    constructor(capacity = 10000) {
        this.capacity = capacity;
        this.buffer = [];
        this.position = 0;
    }

    /**
     * Add a transition to the memory.
     */
    add(state, action, reward, nextState, done) {
        const transition = { state, action, reward, nextState, done };

        if (this.buffer.length < this.capacity) {
            this.buffer.push(transition);
        } else {
            this.buffer[this.position] = transition;
            this.position = (this.position + 1) % this.capacity;
        }
    }

    /**
     * Sample a random mini-batch of transitions.
     * @param {number} batchSize 
     * @returns {Array} Array of transitions
     */
    sample(batchSize) {
        const batch = [];
        for (let i = 0; i < batchSize; i++) {
            const index = Math.floor(Math.random() * this.buffer.length);
            batch.push(this.buffer[index]);
        }
        return batch;
    }

    /**
     * @returns {number} Current size of the buffer
     */
    get size() {
        return this.buffer.length;
    }
}
