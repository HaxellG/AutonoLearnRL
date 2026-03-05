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

        // ── Timer handle ───────────────────────────────────
        this._intervalId = null;
    }

    // ────────────────────────────────────────────────────
    // Public API
    // ────────────────────────────────────────────────────

    /** Start the game loop. */
    start() {
        this._intervalId = setInterval(() => this._loop(), CONFIG.game.frameInterval);
    }

    /** Stop the game loop. */
    stop() {
        if (this._intervalId !== null) {
            clearInterval(this._intervalId);
            this._intervalId = null;
        }
    }

    // ────────────────────────────────────────────────────
    // Private
    // ────────────────────────────────────────────────────

    _loop() {
        this._update();
        this._draw();
        this._frames++;
    }

    _update() {
        const st = this._state;

        if (st === GameState.play) {
            // ── Physics via FlappyEnv ──────────────────────
            const { reward, done, info } = this._env.step(this._pendingAction);
            this._pendingAction = Action.NO_OP;     // consume action

            // ── Sync visual objects from env state ─────────
            this._syncFromEnv();

            // ── SFX triggers ──────────────────────────────
            if (info.passedPipe) {
                this._sfx.score.play();
            }
            if (info.collision) {
                this._sfx.hit.play();
            }

            // ── Score ──────────────────────────────────────
            this._ui.score.curr = this._env.score;

            // ── Game over transition ──────────────────────
            if (done && info.collision) {
                this._state = GameState.gameOver;
            }

            // ── Bird animation frame ──────────────────────
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
        // Bird position & speed
        this._bird.y = this._env.birdY;
        this._bird.speed = this._env.birdVy;
        this._updateBirdRotation();

        // Pipes — copy positions from env to visual Pipe manager
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
                this._env.reset();                // start a new episode
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
