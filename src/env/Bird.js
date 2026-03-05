import { CONFIG, GameState } from '../config.js';

const RAD = Math.PI / 180;

export class Bird {
    /**
     * @param {CanvasRenderingContext2D} ctx
     * @param {object} ground  — Ground instance (used for floor collision)
     * @param {object} pipeManager — Pipe instance (used for pipe collision)
     * @param {object} sfx — SFX object
     * @param {function} getState — returns current GameState
     * @param {function} onScore — callback when bird passes a pipe
     */
    constructor(ctx, ground, pipeManager, sfx, getState, onScore) {
        this._ctx = ctx;
        this._ground = ground;
        this._pipeManager = pipeManager;
        this._sfx = sfx;
        this._getState = getState;
        this._onScore = onScore;

        this.animations = CONFIG.assets.images.bird.map((src) => {
            const img = new Image();
            img.src = src;
            return { sprite: img };
        });

        this.x = CONFIG.bird.startX;
        this.y = CONFIG.bird.startY;
        this.speed = 0;
        this.gravity = CONFIG.bird.gravity;
        this.thrust = CONFIG.bird.thrust;
        this.rotation = 0;
        this.frame = 0;
    }

    /** Reset bird to initial position (for episode restart). */
    reset() {
        this.speed = 0;
        this.y = CONFIG.bird.startY;
        this.rotation = 0;
        this.frame = 0;
    }

    draw() {
        const h = this.animations[this.frame].sprite.height;
        const w = this.animations[this.frame].sprite.width;
        this._ctx.save();
        this._ctx.translate(this.x, this.y);
        this._ctx.rotate(this.rotation * RAD);
        this._ctx.drawImage(this.animations[this.frame].sprite, -w / 2, -h / 2);
        this._ctx.restore();
    }

    update(frames) {
        const r = parseFloat(this.animations[0].sprite.width) / 2;
        const st = this._getState();

        switch (st) {
            case GameState.getReady:
                this.rotation = 0;
                this.y += frames % CONFIG.bird.animRateIdle === 0 ? Math.sin(frames * RAD) : 0;
                this.frame += frames % CONFIG.bird.animRateIdle === 0 ? 1 : 0;
                break;

            case GameState.play:
                this.frame += frames % CONFIG.bird.animRatePlay === 0 ? 1 : 0;
                this.y += this.speed;
                this._setRotation();
                this.speed += this.gravity;

                if (this.y + r >= this._ground.y || this._checkCollision()) {
                    return true; // signal game over
                }
                break;

            case GameState.gameOver:
                this.frame = 1;
                if (this.y + r < this._ground.y) {
                    this.y += this.speed;
                    this._setRotation();
                    this.speed += this.gravity * 2;
                } else {
                    this.speed = 0;
                    this.y = this._ground.y - r;
                    this.rotation = 90;
                    if (!this._sfx.played) {
                        this._sfx.die.play();
                        this._sfx.played = true;
                    }
                }
                break;
        }

        this.frame = this.frame % this.animations.length;
        return false;
    }

    flap() {
        if (this.y > 0) {
            this._sfx.flap.play();
            this.speed = -this.thrust;
        }
    }

    _setRotation() {
        if (this.speed <= 0) {
            this.rotation = Math.max(-25, (-25 * this.speed) / (-1 * this.thrust));
        } else if (this.speed > 0) {
            this.rotation = Math.min(90, (90 * this.speed) / (this.thrust * 2));
        }
    }

    _checkCollision() {
        const pipes = this._pipeManager.pipes;
        if (!pipes.length) return false;

        const birdSprite = this.animations[0].sprite;
        const x = pipes[0].x;
        const y = pipes[0].y;
        const r = birdSprite.height / 4 + birdSprite.width / 4;
        const roof = y + parseFloat(this._pipeManager.topSprite.height);
        const floor = roof + this._pipeManager.gap;
        const w = parseFloat(this._pipeManager.topSprite.width);

        if (this.x + r >= x) {
            if (this.x + r < x + w) {
                if (this.y - r <= roof || this.y + r >= floor) {
                    this._sfx.hit.play();
                    return true;
                }
            } else if (this._pipeManager.moved) {
                this._onScore();
                this._sfx.score.play();
                this._pipeManager.moved = false;
            }
        }
        return false;
    }
}
