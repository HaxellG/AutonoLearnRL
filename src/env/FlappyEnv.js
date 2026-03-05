import { CONFIG, Action } from '../config.js';
import { SeededRandom } from '../utils/SeededRandom.js';

/**
 * FlappyEnv — Headless RL environment for Flappy Bird.
 *
 * Coordinate system:
 *   - Origin (0,0) is top-left of the canvas.
 *   - Y increases downward (gravity pulls bird toward higher Y).
 *   - vy > 0 means the bird is falling.
 *
 * Observation vector [dx, dy, vy]:
 *   dx : horizontal distance (px) from bird center to the left edge of the
 *        next pipe. Always ≥ 0 while the pipe is ahead.
 *   dy : gapCenterY − birdY.  Positive → bird is above the gap center.
 *   vy : bird's current vertical speed (px/frame). Positive = falling.
 *
 * Actions:  0 = NO_OP,  1 = FLAP.
 *
 * Rewards:
 *   +0.1  per step survived
 *   +1.0  when passing a pipe
 *   −1.0  on collision (episode terminates)
 */
export class FlappyEnv {
    constructor() {
        // ── Geometry constants (from measured sprites) ───────
        this._canvasW = CONFIG.canvas.width;
        this._canvasH = CONFIG.canvas.height;
        this._birdW = CONFIG.sprites.bird.width;
        this._birdH = CONFIG.sprites.bird.height;
        this._pipeW = CONFIG.sprites.pipe.width;
        this._pipeH = CONFIG.sprites.pipe.height;
        this._groundH = CONFIG.sprites.ground.height;
        this._groundY = this._canvasH - this._groundH;   // 302

        // ── Physics constants ───────────────────────────────
        this._gravity = CONFIG.bird.gravity;
        this._thrust = CONFIG.bird.thrust;
        this._dx = CONFIG.game.dx;
        this._pipeGap = CONFIG.pipe.gap;
        this._spawnInterval = CONFIG.pipe.spawnInterval;
        this._yOffsetBase = CONFIG.pipe.yOffsetBase;
        this._yMinF = CONFIG.pipe.yOffsetMinFactor;
        this._yMaxF = CONFIG.pipe.yOffsetMaxFactor;

        // ── Rewards ─────────────────────────────────────────
        this._rSurvive = CONFIG.rewards.survive;
        this._rPass = CONFIG.rewards.passPipe;
        this._rCollision = CONFIG.rewards.collision;
        this._maxSteps = CONFIG.env.maxStepsPerEpisode;

        // ── Collision helper (matches original Bird._checkCollision) ──
        //    r = birdH/4 + birdW/4
        this._birdR = this._birdH / 4 + this._birdW / 4;

        // ── Mutable state (set in reset) ────────────────────
        this._rng = null;
        this._deterministic = false;
        this.birdY = 0;
        this.birdVy = 0;
        this.pipes = [];
        this.score = 0;
        this.frames = 0;
        this.done = false;
        this._pipeMoved = true;   // flag matching original `pipe.moved`
    }

    // ══════════════════════════════════════════════════════
    // PUBLIC API
    // ══════════════════════════════════════════════════════

    /**
     * Reset the environment to a fresh episode.
     * @param {number} [seed] — if provided, pipe generation becomes deterministic.
     * @returns {number[]} initial observation [dx, dy, vy]
     */
    reset(seed) {
        if (seed !== undefined) {
            this._rng = new SeededRandom(seed);
            this._deterministic = true;
        } else {
            this._rng = null;
            this._deterministic = false;
        }

        this.birdY = CONFIG.bird.startY;
        this.birdVy = 0;
        this.pipes = [];
        this.score = 0;
        this.frames = 0;
        this.done = false;
        this._pipeMoved = true;

        return this.getState();
    }

    /**
     * Advance one step.
     * @param {number} action — 0 = NO_OP, 1 = FLAP
     * @returns {{ state: number[], reward: number, done: boolean, info: object }}
     */
    step(action) {
        if (this.done) {
            throw new Error('FlappyEnv: episode is done. Call reset() first.');
        }

        // ── 1. Apply action ─────────────────────────────────
        if (action === Action.FLAP && this.birdY > 0) {
            this.birdVy = -this._thrust;
        }

        // ── 2. Bird physics ─────────────────────────────────
        this.birdY += this.birdVy;
        this.birdVy += this._gravity;

        // ── 3. Pipe spawning & movement ─────────────────────
        if (this.frames % this._spawnInterval === 0) {
            this._spawnPipe();
        }

        for (let i = 0; i < this.pipes.length; i++) {
            this.pipes[i].x -= this._dx;
        }

        // Remove off-screen pipes
        if (this.pipes.length && this.pipes[0].x < -this._pipeW) {
            this.pipes.shift();
            this._pipeMoved = true;
        }

        // ── 4. Collision detection ──────────────────────────
        let reward = this._rSurvive;
        let collision = false;
        let passedPipe = false;

        const r = this._birdR;

        // Ground collision
        if (this.birdY + r >= this._groundY) {
            collision = true;
        }

        // Ceiling collision (bird goes above canvas)
        if (this.birdY - r <= 0) {
            // Clamp bird at ceiling — same as original (bird.flap checks y > 0)
            this.birdY = r;
            this.birdVy = 0;
        }

        // Pipe collision (matches original Bird._checkCollision exactly)
        if (!collision && this.pipes.length) {
            const p = this.pipes[0];
            const px = p.x;
            const py = p.y;
            const roof = py + this._pipeH;       // bottom edge of top pipe image
            const floor = roof + this._pipeGap;  // top edge of bottom pipe
            const w = this._pipeW;
            const birdX = CONFIG.bird.startX;     // bird X is fixed

            if (birdX + r >= px) {
                if (birdX + r < px + w) {
                    // Bird is horizontally within the pipe
                    if (this.birdY - r <= roof || this.birdY + r >= floor) {
                        collision = true;
                    }
                } else if (this._pipeMoved) {
                    // Bird has passed the pipe
                    this.score++;
                    this._pipeMoved = false;
                    passedPipe = true;
                    reward += this._rPass;
                }
            }
        }

        // ── 5. Termination ──────────────────────────────────
        if (collision) {
            reward = this._rCollision;
            this.done = true;
        }

        this.frames++;

        if (this.frames >= this._maxSteps) {
            this.done = true;
        }

        // ── 6. Build result ─────────────────────────────────
        const state = this.getState();
        const info = {
            score: this.score,
            timestep: this.frames,
            passedPipe,
            collision,
            birdY: this.birdY,
            birdVy: this.birdVy,
            nextPipeX: this.pipes.length ? this.pipes[0].x : null,
            gapCenterY: this.pipes.length
                ? this.pipes[0].y + this._pipeH + this._pipeGap / 2
                : null,
        };

        return { state, reward, done: this.done, info };
    }

    /**
     * Get the current observation vector.
     * @returns {number[]} [dx, dy, vy]
     */
    getState() {
        if (!this.pipes.length) {
            // No pipe visible yet → dx is distance to right edge, dy = 0
            return [this._canvasW - CONFIG.bird.startX, 0, this.birdVy];
        }

        const p = this.pipes[0];
        const birdX = CONFIG.bird.startX;
        const dx = p.x - birdX;
        const gapCenterY = p.y + this._pipeH + this._pipeGap / 2;
        const dy = gapCenterY - this.birdY;

        return [dx, dy, this.birdVy];
    }

    /**
     * Serialize the full internal state for debug / snapshot.
     * @returns {object}
     */
    getSnapshot() {
        return {
            birdY: this.birdY,
            birdVy: this.birdVy,
            pipes: this.pipes.map((p) => ({ ...p })),
            score: this.score,
            frames: this.frames,
            done: this.done,
            pipeMoved: this._pipeMoved,
        };
    }

    /**
     * Restore from a snapshot.
     * @param {object} snap
     */
    setSnapshot(snap) {
        this.birdY = snap.birdY;
        this.birdVy = snap.birdVy;
        this.pipes = snap.pipes.map((p) => ({ ...p }));
        this.score = snap.score;
        this.frames = snap.frames;
        this.done = snap.done;
        this._pipeMoved = snap.pipeMoved;
    }

    // ══════════════════════════════════════════════════════
    // PRIVATE
    // ══════════════════════════════════════════════════════

    _spawnPipe() {
        const rand = this._deterministic ? this._rng.next() : Math.random();
        const factor = Math.min(rand + this._yMinF, this._yMaxF);
        this.pipes.push({
            x: this._canvasW,
            y: this._yOffsetBase * factor,
        });
    }
}
