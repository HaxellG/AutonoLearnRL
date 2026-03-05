import { CONFIG, GameState } from '../config.js';

export class Ground {
    /**
     * @param {CanvasRenderingContext2D} ctx
     * @param {HTMLCanvasElement} canvas
     * @param {function} getState — returns current GameState
     */
    constructor(ctx, canvas, getState) {
        this._ctx = ctx;
        this._canvas = canvas;
        this._getState = getState;

        this.sprite = new Image();
        this.sprite.src = CONFIG.assets.images.ground;

        this.x = 0;
        this.y = 0;
    }

    draw() {
        this.y = parseFloat(this._canvas.height - this.sprite.height);
        this._ctx.drawImage(this.sprite, this.x, this.y);
    }

    update() {
        if (this._getState() !== GameState.play) return;
        this.x -= CONFIG.game.dx;
        this.x = this.x % (this.sprite.width / 2);
    }
}
