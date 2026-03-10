import { CONFIG, GameState, Action } from './config.js';
import { FlappyEnv } from './env/FlappyEnv.js';
import { Bird } from './env/Bird.js';
import { Pipe } from './env/Pipe.js';
import { Ground } from './env/Ground.js';
import { Background } from './env/Background.js';
import { UI } from './ui/UI.js';
import { createSFX } from './ui/SFX.js';

const RAD = Math.PI / 180;

/**
 * Main Game controller (human-play mode).
 *
 * Uses FlappyEnv internally for physics during play state.
 * Renders using the existing sprite-based classes.
 * Human input is translated to env actions.
 * Supports speed multiplier (stepsPerFrame) for fast-forward.
 */
export class Game {
    /**
     * @param {HTMLCanvasElement} canvas
     */
    constructor(canvas) {
        this._canvas = canvas;
        this._ctx = canvas.getContext('2d');
        canvas.tabIndex = 1;

        // ── State ──────────────────────────────────────────
        this._state = GameState.getReady;
        this._frames = 0;

        // ── Speed control ─────────────────────────────────
        this._stepsPerFrame = 1;

        // ── Pending action for env ────────────────────────
        this._pendingAction = Action.NO_OP;

        // Getter passed to sub-systems so they can read state.
        this._getState = () => this._state;

        // ── SFX ────────────────────────────────────────────
        this._sfx = createSFX();

        // ── RL Environment (headless physics) ──────────────
        this._env = new FlappyEnv();

        // ── Visual-only objects (rendering) ────────────────
        this._ground = new Ground(this._ctx, this._canvas, this._getState);
        this._bg = new Background(this._ctx, this._canvas);
        this._pipe = new Pipe(this._ctx, this._canvas, this._getState);
        this._ui = new UI(this._ctx, this._canvas, this._getState);

        this._bird = new Bird(
            this._ctx,
            this._ground,
            this._pipe,
            this._sfx,
            this._getState,
            () => { /* score handled by env now */ }
        );

        // ── Input ──────────────────────────────────────────
        this._bindInput();

        // ── rAF handle ─────────────────────────────────────
        this._rafId = null;
        this._running = false;
    }

    // ────────────────────────────────────────────────────
    // Public API
    // ────────────────────────────────────────────────────

    /** Start the game loop (requestAnimationFrame). */
    start() {
        this._running = true;
        this._lastTime = performance.now();
        this._accumulator = 0;
        this._scheduleFrame();
    }

    /** Stop the game loop. */
    stop() {
        this._running = false;
        if (this._rafId !== null) {
            cancelAnimationFrame(this._rafId);
            this._rafId = null;
        }
    }

    /**
     * Set the simulation speed multiplier.
     * At speed N, each animation frame executes N env steps.
     * @param {number} multiplier — 1, 4, 16, 64, etc.
     */
    setSpeed(multiplier) {
        this._stepsPerFrame = Math.max(1, multiplier | 0);
    }

    /** @returns {number} current speed multiplier */
    getSpeed() {
        return this._stepsPerFrame;
    }

    /** 
     * Force-render an external environment state (used by Agent runner).
     * @param {import('./env/FlappyEnv.js').FlappyEnv} env 
     */
    renderEnv(env) {
        // Hijack internal env state and force sync & draw
        this._env = env;
        this._state = GameState.play;  // Skip "Get Ready" screening
        this._ui.score.curr = env.score; // Keep UI score synced
        this._syncFromEnv();
        this._draw();
    }

    // ────────────────────────────────────────────────────
    // Private — Loop
    // ────────────────────────────────────────────────────

    _scheduleFrame() {
        if (!this._running) return;
        this._rafId = requestAnimationFrame((now) => this._loop(now));
    }

    _loop(now) {
        if (!this._running) return;

        // Fixed timestep accumulator — run physics at CONFIG.game.frameInterval rate
        const dt = now - this._lastTime;
        this._lastTime = now;
        this._accumulator += dt;

        const interval = CONFIG.game.frameInterval;

        while (this._accumulator >= interval) {
            this._accumulator -= interval;
            this._update();
            this._frames++;
        }

        this._draw();
        this._scheduleFrame();
    }

    _update() {
        const st = this._state;

        if (st === GameState.play) {
            // Execute stepsPerFrame env steps per game tick
            for (let i = 0; i < this._stepsPerFrame; i++) {
                if (this._state !== GameState.play) break;

                const { done, info } = this._env.step(this._pendingAction);
                this._pendingAction = Action.NO_OP;   // consume on first step

                // ── SFX triggers ──────────────────────────────
                if (info.passedPipe) this._sfx.score.play();
                if (info.collision) this._sfx.hit.play();

                // ── Score ─────────────────────────────────────
                this._ui.score.curr = this._env.score;

                // ── Game over transition ──────────────────────
                if (done && info.collision) {
                    this._state = GameState.gameOver;
                }
            }

            // Sync visuals from final env state
            this._syncFromEnv();

            // Bird animation frame
            this._bird.frame += this._frames % CONFIG.bird.animRatePlay === 0 ? 1 : 0;
            this._bird.frame = this._bird.frame % this._bird.animations.length;

        } else if (st === GameState.getReady) {
            // Idle bobbing animation (visual only)
            this._bird.rotation = 0;
            this._bird.y += this._frames % CONFIG.bird.animRateIdle === 0
                ? Math.sin(this._frames * RAD) : 0;
            this._bird.frame += this._frames % CONFIG.bird.animRateIdle === 0 ? 1 : 0;
            this._bird.frame = this._bird.frame % this._bird.animations.length;

        } else if (st === GameState.gameOver) {
            // Bird death fall animation (visual only)
            const r = CONFIG.sprites.bird.width / 2;
            this._bird.frame = 1;
            if (this._bird.y + r < this._ground.y) {
                this._bird.y += this._bird.speed;
                this._updateBirdRotation();
                this._bird.speed += CONFIG.bird.gravity * 2;
            } else {
                this._bird.speed = 0;
                this._bird.y = this._ground.y - r;
                this._bird.rotation = 90;
                if (!this._sfx.played) {
                    this._sfx.die.play();
                    this._sfx.played = true;
                }
            }
        }

        // Ground scrolling (only during play)
        this._ground.update();
        // UI tap animation (only during non-play)
        this._ui.update(this._frames);
    }

    /** Sync visual render objects from FlappyEnv's internal state. */
    _syncFromEnv() {
        this._bird.y = this._env.birdY;
        this._bird.speed = this._env.birdVy;
        this._updateBirdRotation();
        this._pipe.pipes = this._env.pipes;
        this._pipe.moved = this._env._pipeMoved;
    }

    /** Calculate bird visual rotation from speed (same formula as original). */
    _updateBirdRotation() {
        const speed = this._bird.speed;
        const thrust = CONFIG.bird.thrust;
        if (speed <= 0) {
            this._bird.rotation = Math.max(-25, (-25 * speed) / (-1 * thrust));
        } else {
            this._bird.rotation = Math.min(90, (90 * speed) / (thrust * 2));
        }
    }

    _draw() {
        this._ctx.fillStyle = '#30c0df';
        this._ctx.fillRect(0, 0, this._canvas.width, this._canvas.height);
        this._bg.draw();
        this._pipe.draw();
        this._bird.draw();
        this._ground.draw();
        this._ui.draw();
    }

    /** Handle the primary action (start / flap / restart). */
    _handleAction() {
        switch (this._state) {
            case GameState.getReady:
                this._state = GameState.play;
                this._env.reset();
                this._sfx.start.play();
                break;
            case GameState.play:
                this._pendingAction = Action.FLAP;
                this._sfx.flap.play();
                break;
            case GameState.gameOver:
                this._resetVisuals();
                break;
        }
    }

    /** Reset visual objects for a new round. */
    _resetVisuals() {
        this._state = GameState.getReady;
        this._bird.reset();
        this._pipe.reset();
        this._ui.resetScore();
        this._sfx.played = false;
        this._pendingAction = Action.NO_OP;
    }

    _bindInput() {
        this._canvas.addEventListener('click', () => this._handleAction());

        this._canvas.onkeydown = (e) => {
            if (e.keyCode === 32 || e.keyCode === 87 || e.keyCode === 38) {
                // Space / W / Arrow Up
                this._handleAction();
            }
        };
    }
}
