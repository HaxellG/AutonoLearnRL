import { CONFIG, GameState } from './config.js';
import { Bird } from './env/Bird.js';
import { Pipe } from './env/Pipe.js';
import { Ground } from './env/Ground.js';
import { Background } from './env/Background.js';
import { UI } from './ui/UI.js';
import { createSFX } from './ui/SFX.js';

/**
 * Main Game controller.
 * Orchestrates the game loop, state transitions, input, and all sub-systems.
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

        // Getter passed to sub-systems so they can read (but not write) state.
        this._getState = () => this._state;

        // ── SFX ────────────────────────────────────────────
        this._sfx = createSFX();

        // ── Game objects ───────────────────────────────────
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
            () => {
                this._ui.score.curr++;
            }
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
        const gameOver = this._bird.update(this._frames);
        if (gameOver) {
            this._state = GameState.gameOver;
        }
        this._ground.update();
        this._pipe.update(this._frames);
        this._ui.update(this._frames);
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
                this._sfx.start.play();
                break;
            case GameState.play:
                this._bird.flap();
                break;
            case GameState.gameOver:
                this._reset();
                break;
        }
    }

    /** Reset all game objects for a new round. */
    _reset() {
        this._state = GameState.getReady;
        this._bird.reset();
        this._pipe.reset();
        this._ui.resetScore();
        this._sfx.played = false;
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
