/**
 * Mulberry32 — Seeded 32-bit PRNG.
 * Produces deterministic pseudo-random numbers given the same seed.
 * Used by FlappyEnv for reproducible pipe generation.
 */
export class SeededRandom {
    /**
     * @param {number} seed — unsigned 32-bit integer seed.
     */
    constructor(seed) {
        this._state = seed | 0;
    }

    /**
     * Returns the next pseudo-random float in [0, 1).
     * @returns {number}
     */
    next() {
        let t = (this._state += 0x6d2b79f5);
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }

    /**
     * Returns a random integer in [min, max] (inclusive).
     * @param {number} min
     * @param {number} max
     * @returns {number}
     */
    randInt(min, max) {
        return Math.floor(this.next() * (max - min + 1)) + min;
    }

    /**
     * Returns a random float in [min, max).
     * @param {number} min
     * @param {number} max
     * @returns {number}
     */
    randRange(min, max) {
        return this.next() * (max - min) + min;
    }
}

/**
 * Create a SeededRandom instance.
 * @param {number} seed
 * @returns {SeededRandom}
 */
export function createRNG(seed) {
    return new SeededRandom(seed);
}
