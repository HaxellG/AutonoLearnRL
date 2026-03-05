import { CONFIG } from '../config.js';

export class Background {
    /**
     * @param {CanvasRenderingContext2D} ctx
     * @param {HTMLCanvasElement} canvas
     */
    constructor(ctx, canvas) {
        this._ctx = ctx;
        this._canvas = canvas;

        this.sprite = new Image();
        this.sprite.src = CONFIG.assets.images.bg;

        this.x = 0;
        this.y = 0;
    }

    draw() {
        const y = parseFloat(this._canvas.height - this.sprite.height);
        this._ctx.drawImage(this.sprite, this.x, y);
    }
}
