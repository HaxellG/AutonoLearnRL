import { CONFIG, GameState } from '../config.js';

export class UI {
    /**
     * @param {CanvasRenderingContext2D} ctx
     * @param {HTMLCanvasElement} canvas
     * @param {function} getState — returns current GameState
     */
    constructor(ctx, canvas, getState) {
        this._ctx = ctx;
        this._canvas = canvas;
        this._getState = getState;

        this.getReadySprite = new Image();
        this.getReadySprite.src = CONFIG.assets.images.getReady;

        this.gameOverSprite = new Image();
        this.gameOverSprite.src = CONFIG.assets.images.gameOver;

        this.tap = CONFIG.assets.images.tap.map((src) => {
            const img = new Image();
            img.src = src;
            return { sprite: img };
        });

        this.score = { curr: 0, best: 0 };
        this.frame = 0;
    }

    /** Reset score for a new episode. */
    resetScore() {
        this.score.curr = 0;
    }

    draw() {
        const st = this._getState();
        let x, y, tx, ty;

        switch (st) {
            case GameState.getReady:
                y = parseFloat(this._canvas.height - this.getReadySprite.height) / 2;
                x = parseFloat(this._canvas.width - this.getReadySprite.width) / 2;
                tx = parseFloat(this._canvas.width - this.tap[0].sprite.width) / 2;
                ty = y + this.getReadySprite.height - this.tap[0].sprite.height;
                this._ctx.drawImage(this.getReadySprite, x, y);
                this._ctx.drawImage(this.tap[this.frame].sprite, tx, ty);
                break;

            case GameState.gameOver:
                y = parseFloat(this._canvas.height - this.gameOverSprite.height) / 2;
                x = parseFloat(this._canvas.width - this.gameOverSprite.width) / 2;
                tx = parseFloat(this._canvas.width - this.tap[0].sprite.width) / 2;
                ty = y + this.gameOverSprite.height - this.tap[0].sprite.height;
                this._ctx.drawImage(this.gameOverSprite, x, y);
                this._ctx.drawImage(this.tap[this.frame].sprite, tx, ty);
                break;
        }

        this._drawScore();
    }

    _drawScore() {
        const st = this._getState();
        this._ctx.fillStyle = '#FFFFFF';
        this._ctx.strokeStyle = '#000000';

        switch (st) {
            case GameState.play:
                this._ctx.lineWidth = '2';
                this._ctx.font = '35px Squada One';
                this._ctx.fillText(this.score.curr, this._canvas.width / 2 - 5, 50);
                this._ctx.strokeText(this.score.curr, this._canvas.width / 2 - 5, 50);
                break;

            case GameState.gameOver:
                this._ctx.lineWidth = '2';
                this._ctx.font = '40px Squada One';
                const sc = `SCORE :     ${this.score.curr}`;
                try {
                    this.score.best = Math.max(
                        this.score.curr,
                        localStorage.getItem('best')
                    );
                    localStorage.setItem('best', this.score.best);
                    const bs = `BEST  :     ${this.score.best}`;
                    this._ctx.fillText(sc, this._canvas.width / 2 - 80, this._canvas.height / 2 + 0);
                    this._ctx.strokeText(sc, this._canvas.width / 2 - 80, this._canvas.height / 2 + 0);
                    this._ctx.fillText(bs, this._canvas.width / 2 - 80, this._canvas.height / 2 + 30);
                    this._ctx.strokeText(bs, this._canvas.width / 2 - 80, this._canvas.height / 2 + 30);
                } catch (e) {
                    this._ctx.fillText(sc, this._canvas.width / 2 - 85, this._canvas.height / 2 + 15);
                    this._ctx.strokeText(sc, this._canvas.width / 2 - 85, this._canvas.height / 2 + 15);
                }
                break;
        }
    }

    update(frames) {
        if (this._getState() === GameState.play) return;
        this.frame += frames % 10 === 0 ? 1 : 0;
        this.frame = this.frame % this.tap.length;
    }
}
