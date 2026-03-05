import { CONFIG, GameState } from '../config.js';
import { SeededRandom } from '../utils/SeededRandom.js';

export class Pipe {
    /**
     * @param {CanvasRenderingContext2D} ctx
     * @param {HTMLCanvasElement} canvas
     * @param {function} getState — returns current GameState
     */
    constructor(ctx, canvas, getState) {
        this._ctx = ctx;
        this._canvas = canvas;
        this._getState = getState;

        this.topSprite = new Image();
        this.topSprite.src = CONFIG.assets.images.topPipe;

        this.botSprite = new Image();
        this.botSprite.src = CONFIG.assets.images.botPipe;

        this.gap = CONFIG.pipe.gap;
        this.moved = true;
        this.pipes = [];

        // Seeded PRNG for visual-mode pipe generation (no Math.random)
        this._rng = new SeededRandom(Date.now() ^ 0);
    }

    /** Reset pipes for a new episode. */
    reset() {
        this.pipes = [];
        this.moved = true;
    }

    draw() {
        for (let i = 0; i < this.pipes.length; i++) {
            const p = this.pipes[i];
            this._ctx.drawImage(this.topSprite, p.x, p.y);
            this._ctx.drawImage(
                this.botSprite,
                p.x,
                p.y + parseFloat(this.topSprite.height) + this.gap
            );
        }
    }

    update(frames) {
        if (this._getState() !== GameState.play) return;

        if (frames % CONFIG.pipe.spawnInterval === 0) {
            this.pipes.push({
                x: parseFloat(this._canvas.width),
                y: CONFIG.pipe.yOffsetBase *
                    Math.min(
                        this._rng.next() + CONFIG.pipe.yOffsetMinFactor,
                        CONFIG.pipe.yOffsetMaxFactor
                    ),
            });
        }

        const dx = CONFIG.game.dx;
        this.pipes.forEach((pipe) => {
            pipe.x -= dx;
        });

        if (this.pipes.length && this.pipes[0].x < -this.topSprite.width) {
            this.pipes.shift();
            this.moved = true;
        }
    }
}
