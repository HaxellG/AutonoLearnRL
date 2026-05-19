import { CONFIG, GameState, Action } from './config.js';
import { FlappyEnv } from './env/FlappyEnv.js';
import { QLearningAgent } from './agents/QLearningAgent.js';
import { DQNAgent } from './agents/DQNAgent.js';
import { DoubleDQNAgent } from './agents/DoubleDQNAgent.js';
import { Background } from './env/Background.js';
import { Ground } from './env/Ground.js';
import { createSFX } from './ui/SFX.js';
import { UI } from './ui/UI.js';

const RAD = Math.PI / 180;

/**
 * BattleMode — Runs Q-Learning, DQN, and DDQN agents simultaneously
 * on the same pipe layout. All three birds are rendered in a single
 * canvas with colored tint overlays and name labels. When all agents
 * die, a ranking overlay is shown.
 */
export class BattleMode {
    constructor(canvas, mode = 'battle') {
        this._canvas = canvas;
        this._ctx = canvas.getContext('2d');
        this._mode = mode; // 'battle' or 'vs'
        this._canvas.tabIndex = 1; // ensure it can receive key events

        this._running = false;
        this._rafId = null;
        this._frames = 0;
        this._state = GameState.getReady;

        // ── Agent definitions (color, name, env, agent, state) ──
        if (this._mode === 'vs') {
            this._agents = [
                { name: 'P1 (YOU)', color: '#f1c40f', env: null, agent: null, isHuman: true, pendingAction: Action.NO_OP, alive: true, score: 0, deathOrder: 0 },
                { name: 'DDQN',     color: '#e74c3c', env: null, agent: null, alive: true, score: 0, deathOrder: 0 },
            ];
            this._sfx = createSFX();
            this._bindInput();
        } else {
            this._agents = [
                { name: 'Q-LEARN', color: '#73bf2e', env: null, agent: null, alive: true, score: 0, deathOrder: 0 },
                { name: 'DQN',     color: '#3498db', env: null, agent: null, alive: true, score: 0, deathOrder: 0 },
                { name: 'DDQN',    color: '#e74c3c', env: null, agent: null, alive: true, score: 0, deathOrder: 0 },
            ];
        }
        this._deathCounter = 0;
        this._battleDone = false;

        // ── Visual resources ──
        this._birdSprites = CONFIG.assets.images.bird.map(src => {
            const img = new Image();
            img.src = src;
            return img;
        });

        this._pipeTopImg = new Image();
        this._pipeTopImg.src = CONFIG.assets.images.topPipe;
        this._pipeBotImg = new Image();
        this._pipeBotImg.src = CONFIG.assets.images.botPipe;

        // Background & Ground (shared rendering objects)
        const getState = () => this._state;
        this._bg = new Background(this._ctx, this._canvas);
        this._ground = new Ground(this._ctx, this._canvas, getState);
        this._ui = new UI(this._ctx, this._canvas, getState);

        // Offscreen canvas for bird tinting
        this._offCanvas = document.createElement('canvas');
        this._offCanvas.width = 64;
        this._offCanvas.height = 64;
        this._offCtx = this._offCanvas.getContext('2d');

        // Results overlay callback
        this._onBattleEnd = null;
    }

    /**
     * Load all three pretrained models and start the battle.
     */
    async start() {
        this._battleDone = false;
        this._deathCounter = 0;
        this._frames = 0;

        // Determine difficulty folder (matches training script logic)
        const difficulty = CONFIG.pipe.gap >= 120 ? 'medium' : 'hard';
        const basePath = `./models_final/${difficulty}`;

        // Use the fixed seed or random based on config
        const EVAL_SEED = CONFIG.game.seedMode === 'random' ? Math.floor(Math.random() * 10000) : 0;

        // Status element
        const statusEl = document.getElementById('battle-status');
        if (statusEl) statusEl.textContent = 'Loading models...';

        try {
            if (this._mode === 'battle') {
                // ── Load Q-Learning (best model) ──
                const qAgent = new QLearningAgent();
                const qReq = await fetch(`${basePath}/qlearning.json`);
                if (!qReq.ok) throw new Error(`Q-Learning model not found in ${basePath}/`);
                const qData = await qReq.json();
                qAgent.load(qData);
                qAgent.epsilon = 0; // greedy
                this._agents[0].agent = qAgent;

                // ── Load DQN (best model) ──
                const dqnAgent = new DQNAgent();
                await dqnAgent.load(`${basePath}/dqn/model.json`);
                dqnAgent.epsilon = 0;
                this._agents[1].agent = dqnAgent;

                // ── Load DDQN (best model) ──
                const ddqnAgent = new DoubleDQNAgent();
                await ddqnAgent.load(`${basePath}/ddqn/model.json`);
                ddqnAgent.epsilon = 0;
                this._agents[2].agent = ddqnAgent;
            } else if (this._mode === 'vs') {
                // ── Load DDQN (best model) for AI ──
                const ddqnAgent = new DoubleDQNAgent();
                await ddqnAgent.load(`${basePath}/ddqn/model.json`);
                ddqnAgent.epsilon = 0;
                this._agents[1].agent = ddqnAgent; // index 1 is DDQN, index 0 is Human
            }

        } catch (err) {
            console.error('Battle: Failed to load models:', err);
            if (statusEl) statusEl.textContent = `Error: ${err.message}`;
            return;
        }

        if (statusEl) statusEl.textContent = '';

        // ── Create environments with SAME seed (EVAL_SEED = 0) ──
        for (const a of this._agents) {
            a.env = new FlappyEnv();
            a.env.reset(EVAL_SEED);
            a.alive = true;
            a.score = 0;
            a.deathOrder = 0;
        }

        this._running = true;
        this._state = GameState.getReady;
        this._bindInput(); // ALWAYS bind input for battle mode so we can click to start
        this._scheduleFrame();
    }

    stop() {
        this._running = false;
        if (this._rafId !== null) {
            cancelAnimationFrame(this._rafId);
            this._rafId = null;
        }
        if (this._mode === 'vs') {
            this._canvas.removeEventListener('click', this._clickHandler);
            this._canvas.onkeydown = null;
        } else {
            this._canvas.removeEventListener('click', this._clickHandler);
            this._canvas.onkeydown = null;
        }
    }

    _bindInput() {
        if (!this._sfx) this._sfx = createSFX();

        this._clickHandler = () => this._handleAction();
        this._canvas.addEventListener('click', this._clickHandler);

        this._canvas.onkeydown = (e) => {
            if (e.keyCode === 32 || e.keyCode === 87 || e.keyCode === 38) {
                // Space / W / Arrow Up
                this._handleAction();
            }
        };
    }

    _handleAction() {
        if (!this._running) return;

        if (this._state === GameState.getReady) {
            this._state = GameState.play;
            this._sfx.start.play();
            return;
        }

        if (this._mode === 'vs') {
            const humanAgent = this._agents.find(a => a.isHuman);
            if (humanAgent && humanAgent.alive) {
                humanAgent.pendingAction = Action.FLAP;
                this._sfx.flap.play();
            }
        }
    }

    /** Set callback for when battle finishes. */
    onBattleEnd(fn) {
        this._onBattleEnd = fn;
    }

    // ──────────────────────────────────────────────────────
    // Loop
    // ──────────────────────────────────────────────────────

    _scheduleFrame() {
        if (!this._running) return;
        this._rafId = requestAnimationFrame(() => this._tick());
    }

    _tick() {
        if (!this._running) return;

        this._frames++;

        if (this._state === GameState.getReady) {
            // Just update UI and Ground
            this._ground.update();
            this._ui.update(this._frames);
            this._draw();
            this._scheduleFrame();
            return;
        }

        // Step each alive agent
        for (const a of this._agents) {
            if (!a.alive) continue;

            const state = a.env.getState();
            let action;
            if (a.isHuman) {
                action = a.pendingAction;
                a.pendingAction = Action.NO_OP;
            } else if (a.agent.bestAction) {
                action = a.agent.bestAction(state);
            } else {
                action = a.agent.act(state);
            }

            const prevScore = a.score;
            const result = a.env.step(action);
            a.score = a.env.score;

            if (a.isHuman && a.score > prevScore) {
                this._sfx.score.play();
            }

            if (result.done) {
                a.alive = false;
                this._deathCounter++;
                a.deathOrder = this._deathCounter;
                if (a.isHuman) {
                    this._sfx.hit.play();
                    this._sfx.die.play();
                }
            }
        }

        // Render
        this._draw();

        // Check if all dead
        if (this._agents.every(a => !a.alive)) {
            this._battleDone = true;
            this._running = false;
            this._showResults();
            return;
        }

        this._scheduleFrame();
    }

    // ──────────────────────────────────────────────────────
    // Rendering
    // ──────────────────────────────────────────────────────

    _draw() {
        const ctx = this._ctx;
        const W = this._canvas.width;
        const H = this._canvas.height;

        // Clear
        ctx.fillStyle = '#79c3f4';
        ctx.fillRect(0, 0, W, H);

        // Background
        this._bg.draw();

        // Pipes — use the first alive agent's pipes, or any agent's pipes
        const refAgent = this._agents.find(a => a.alive) || this._agents[0];
        const pipes = refAgent.env.pipes;
        const pipeGap = CONFIG.pipe.gap;
        const pipeH = CONFIG.sprites.pipe.height;

        for (const p of pipes) {
            // Top pipe
            ctx.drawImage(this._pipeTopImg, p.x, p.y);
            // Bottom pipe
            ctx.drawImage(this._pipeBotImg, p.x, p.y + pipeH + pipeGap);
        }

        // Draw each alive bird
        for (let i = 0; i < this._agents.length; i++) {
            const a = this._agents[i];
            if (a.alive) this._drawBird(a, i);
        }

        // Ground
        this._ground.update();
        this._ground.draw();

        // HUD: Score labels for each agent
        this._drawHUD();

        // Get Ready screen if applicable
        if (this._state === GameState.getReady) {
            this._ui.draw();
        }
    }

    _drawBird(agentData, index) {
        const ctx = this._ctx;
        const env = agentData.env;

        // Bird animation frame
        const frameIdx = agentData.alive
            ? Math.floor(this._frames / 5) % this._birdSprites.length
            : 1;
        const sprite = this._birdSprites[frameIdx];

        const birdY = env.birdY;
        const birdX = CONFIG.bird.startX;
        const birdVy = env.birdVy;

        // Calculate rotation
        let rotation;
        const thrust = CONFIG.bird.thrust;
        if (birdVy <= 0) {
            rotation = Math.max(-25, (-25 * birdVy) / (-1 * thrust));
        } else {
            rotation = Math.min(90, (90 * birdVy) / (thrust * 2));
        }

        if (!agentData.alive) rotation = 90;

        const w = sprite.width;
        const h = sprite.height;

        let birdYToDraw = birdY;
        if (this._state === GameState.getReady) {
            // Hover animation exactly like Game.js -> Bird.js
            birdYToDraw += this._frames % CONFIG.bird.animRateIdle === 0 ? Math.sin(this._frames * RAD) * 5 : 0;
            // The multiplication by 5 exaggerates the sine wave just a bit so we can see it clearly, or we can just use the base env
        }

        if (agentData.isHuman || this._state === GameState.getReady) {
            // Draw original sprite for human player or during get ready
            ctx.save();
            ctx.translate(birdX, birdYToDraw);
            ctx.rotate(rotation * RAD);
            ctx.drawImage(sprite, -w / 2, -h / 2);
            ctx.restore();
        } else {
            // Tint the bird on the offscreen canvas for AI
            const oc = this._offCtx;
            this._offCanvas.width = w;
            this._offCanvas.height = h;
            oc.clearRect(0, 0, w, h);
            oc.drawImage(sprite, 0, 0);
            oc.globalCompositeOperation = 'source-atop';
            oc.fillStyle = agentData.color;
            oc.globalAlpha = agentData.alive ? 0.5 : 0.7;
            oc.fillRect(0, 0, w, h);
            oc.globalAlpha = 1.0;
            oc.globalCompositeOperation = 'source-over';

            // Draw the tinted bird on the main canvas
            ctx.save();
            ctx.translate(birdX, birdYToDraw);
            ctx.rotate(rotation * RAD);
            ctx.drawImage(this._offCanvas, -w / 2, -h / 2);
            ctx.restore();
        }

        // Name label above bird (only if alive)
        if (agentData.alive) {
            ctx.save();
            ctx.font = '7px "Press Start 2P"';
            ctx.fillStyle = agentData.color;
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 2;
            ctx.textAlign = 'center';
            const labelY = birdY - 18;
            ctx.strokeText(agentData.name, birdX, labelY);
            ctx.fillText(agentData.name, birdX, labelY);
            ctx.restore();
        }
    }

    _drawHUD() {
        const ctx = this._ctx;
        const W = this._canvas.width;

        // Draw scores at top of screen
        ctx.save();
        ctx.font = '7px "Press Start 2P"';
        ctx.textAlign = 'left';

        let y = 16;
        for (const a of this._agents) {
            const status = a.alive ? `${a.score}` : `☠ ${a.score}`;
            ctx.fillStyle = '#000';
            ctx.fillText(`${a.name}: ${status}`, 9, y + 1);
            ctx.fillStyle = a.alive ? a.color : '#666';
            ctx.fillText(`${a.name}: ${status}`, 8, y);
            y += 14;
        }

        ctx.restore();
    }

    // ──────────────────────────────────────────────────────
    // Results
    // ──────────────────────────────────────────────────────

    _showResults() {
        // Sort: last to die = winner (lower deathOrder = died earlier = worse)
        const ranked = [...this._agents].sort((a, b) => b.deathOrder - a.deathOrder);

        // If tied in death order, sort by score
        ranked.sort((a, b) => {
            if (a.score !== b.score) return b.score - a.score;
            return b.deathOrder - a.deathOrder;
        });

        if (this._onBattleEnd) {
            this._onBattleEnd(ranked.map((a, i) => ({
                rank: i + 1,
                name: a.name,
                score: a.score,
                color: a.color,
            })));
        }
    }
}
