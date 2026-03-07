import { Action } from './config.js';

/**
 * SimulationRunner — Controls FlappyEnv execution in visual or headless mode.
 *
 * Visual mode:  requestAnimationFrame loop, stepsPerFrame × env.step() + render.
 * Headless mode: async batched loop, yields periodically to avoid blocking UI.
 *
 * Usage:
 *   const runner = new SimulationRunner(env, {
 *     actionProvider: (state) => agent.act(state),
 *     renderer:       (env)   => drawToCanvas(env),
 *     onStep:         (result) => { ... },
 *     onEpisodeEnd:   (summary) => { ... },
 *   });
 *   runner.setStepsPerFrame(16);
 *   runner.startVisual(seed);
 *   // or
 *   await runner.runHeadless({ episodes: 200, seed: 42 });
 */
export class SimulationRunner {
    /**
     * @param {import('./env/FlappyEnv.js').FlappyEnv} env
     * @param {object} options
     * @param {function} [options.actionProvider] — (state) => action (0 or 1)
     * @param {function} [options.renderer]       — (env) => void, draws env state
     * @param {function} [options.onStep]         — (result) => void
     * @param {function} [options.onEpisodeEnd]   — (summary) => void
     * @param {function} [options.onReset]        — (seed) => void
     */
    constructor(env, options = {}) {
        this._env = env;
        this._actionProvider = options.actionProvider || (() => Action.NO_OP);
        this._renderer = options.renderer || null;
        this._onStep = options.onStep || null;
        this._onEpisodeEnd = options.onEpisodeEnd || null;
        this._onReset = options.onReset || null;

        this._stepsPerFrame = 1;
        this._renderEnabled = true;
        this._running = false;
        this._paused = false;
        this._rafId = null;

        // Episode tracking
        this._state = null;
        this._episodeReward = 0;
        this._episodeSteps = 0;
        this._seed = undefined;         // fixed seed (reuse across episodes)
        this._autoResetOnDone = true;   // auto-start new episode when done
    }

    // ══════════════════════════════════════════════════════
    // Configuration
    // ══════════════════════════════════════════════════════

    /** Set how many env steps to execute per animation frame. */
    setStepsPerFrame(n) {
        this._stepsPerFrame = Math.max(1, n | 0);
    }

    /** Enable or disable rendering in visual mode. */
    setRenderEnabled(enabled) {
        this._renderEnabled = !!enabled;
    }

    /** Replace the action provider at runtime (e.g., swap agent). */
    setActionProvider(fn) {
        this._actionProvider = fn;
    }

    /** Replace the renderer at runtime. */
    setRenderer(fn) {
        this._renderer = fn;
    }

    /** Whether to auto-reset the env when an episode ends. */
    setAutoReset(enabled) {
        this._autoResetOnDone = !!enabled;
    }

    // ══════════════════════════════════════════════════════
    // Visual mode (browser, requestAnimationFrame)
    // ══════════════════════════════════════════════════════

    /**
     * Start the visual loop.
     * @param {number} [seed] — env seed (reused on auto-reset if provided).
     */
    startVisual(seed) {
        this._seed = seed;
        this._resetEpisode(seed);
        this._running = true;
        this._paused = false;
        this._scheduleFrame();
    }

    /** Pause the visual loop (can resume). */
    pause() {
        this._paused = true;
    }

    /** Resume from pause. */
    resume() {
        if (!this._running) return;
        this._paused = false;
        this._scheduleFrame();
    }

    /** Stop the visual loop entirely. */
    stop() {
        this._running = false;
        this._paused = false;
        if (this._rafId !== null) {
            cancelAnimationFrame(this._rafId);
            this._rafId = null;
        }
    }

    /** @returns {boolean} */
    get running() { return this._running; }

    /** @returns {boolean} */
    get paused() { return this._paused; }

    // ══════════════════════════════════════════════════════
    // Headless mode (Node + browser, async batched)
    // ══════════════════════════════════════════════════════

    /**
     * Run episodes headlessly (no rendering).
     * Yields periodically to avoid blocking the event loop.
     *
     * @param {object} opts
     * @param {number} opts.episodes    — number of episodes to run
     * @param {number} [opts.seed]      — env seed (undefined = auto-seed each episode)
     * @param {number} [opts.yieldEvery=2000] — yield to event loop every N steps
     * @returns {Promise<object[]>} — array of episode summaries
     */
    async runHeadless({ episodes, seed, yieldEvery = 2000 }) {
        const results = [];
        let globalSteps = 0;

        for (let ep = 0; ep < episodes; ep++) {
            this._resetEpisode(seed);

            while (!this._env.done) {
                const action = this._actionProvider(this._state);
                const result = this._env.step(action);
                this._state = result.state;
                this._episodeReward += result.reward;
                this._episodeSteps++;
                globalSteps++;

                this._onStep?.(result);

                // Yield to event loop periodically
                if (globalSteps % yieldEvery === 0) {
                    await new Promise((r) => setTimeout(r, 0));
                }
            }

            const summary = this._buildSummary();
            this._onEpisodeEnd?.(summary);
            results.push(summary);
        }

        return results;
    }

    /**
     * Run episodes headlessly and synchronously (maximum speed, Node only).
     * Does NOT yield — will block the event loop.
     *
     * @param {object} opts
     * @param {number} opts.episodes
     * @param {number} [opts.seed]
     * @returns {object[]} — array of episode summaries
     */
    runSync({ episodes, seed }) {
        const results = [];

        for (let ep = 0; ep < episodes; ep++) {
            this._resetEpisode(seed);

            while (!this._env.done) {
                const action = this._actionProvider(this._state);
                const result = this._env.step(action);
                this._state = result.state;
                this._episodeReward += result.reward;
                this._episodeSteps++;
            }

            const summary = this._buildSummary();
            this._onEpisodeEnd?.(summary);
            results.push(summary);
        }

        return results;
    }

    // ══════════════════════════════════════════════════════
    // Private
    // ══════════════════════════════════════════════════════

    _scheduleFrame() {
        if (!this._running || this._paused) return;
        this._rafId = requestAnimationFrame(() => this._visualTick());
    }

    _visualTick() {
        if (!this._running || this._paused) return;

        // Execute stepsPerFrame env steps
        for (let i = 0; i < this._stepsPerFrame; i++) {
            if (this._env.done) {
                const summary = this._buildSummary();
                this._onEpisodeEnd?.(summary);

                if (this._autoResetOnDone) {
                    this._resetEpisode(this._seed);
                } else {
                    this.stop();
                    return;
                }
            }

            const action = this._actionProvider(this._state);
            const result = this._env.step(action);
            this._state = result.state;
            this._episodeReward += result.reward;
            this._episodeSteps++;

            this._onStep?.(result);
        }

        // Render (optional)
        if (this._renderEnabled && this._renderer) {
            this._renderer(this._env);
        }

        this._scheduleFrame();
    }

    _resetEpisode(seed) {
        this._state = this._env.reset(seed);
        this._episodeReward = 0;
        this._episodeSteps = 0;
        this._onReset?.(this._env.seed);
    }

    _buildSummary() {
        return {
            score: this._env.score,
            steps: this._episodeSteps,
            totalReward: this._episodeReward,
            seed: this._env.seed,
        };
    }
}
