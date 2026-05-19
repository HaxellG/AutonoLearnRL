/**
 * AutonoLearn RL — Centralized Configuration
 * Modify these values to tune physics, difficulty, or rendering
 * without touching game logic.
 */

export const CONFIG = {
    // ── Canvas ──────────────────────────────────────────────
    canvas: {
        width: 276,
        height: 414,
    },

    // ── Bird physics ────────────────────────────────────────
    bird: {
        startX: 50,
        startY: 100,
        gravity: 0.125,
        thrust: 3.6,
        // Animation frame cycle rate (frames modulo)
        animRatePlay: 5,
        animRateIdle: 10,
    },

    // ── Pipes ───────────────────────────────────────────────
    pipe: {
        gap: 125,               // 125 is Normal, 100 is Hard
        spawnInterval: 100,     // spawn a new pipe every N frames
        // y offset range: y = -210 * clamp(random()+1, 1, 1.8)
        yOffsetBase: -210,
        yOffsetMinFactor: 1.0,
        yOffsetMaxFactor: 1.8,
    },

    // ── Game ────────────────────────────────────────────────
    game: {
        dx: 2,                  // horizontal scroll speed (pixels/frame)
        frameInterval: 20,      // ms between frames (~50 FPS)
        seedMode: 'fixed',      // 'fixed' (always 0) or 'random'
    },

    // ── Sprite dimensions (measured from PNGs) ──────────────
    sprites: {
        bird: { width: 34, height: 26 },
        pipe: { width: 52, height: 400 },
        ground: { width: 552, height: 112 },
        bg: { width: 276, height: 228 },
    },

    // ── RL Rewards ──────────────────────────────────────────
    rewards: {
        survive: 0.1,       // +0.1 per step alive
        passPipe: 1.0,      // +1.0 when passing a pipe
        collision: -1.0,    // -1.0 on crash (episode ends)
        proximityScale: 0.5, // reward shaping: bonus for being near gap center
    },

    // ── Environment limits ──────────────────────────────────
    env: {
        maxStepsPerEpisode: 300000,  // Increased from 3000 to allow scores > 28
    },

    // ── Q-Learning RL ─────────────────────────────────────
    rl: {
        qlearning: {
            alpha: 0.1,              // learning rate
            gamma: 0.95,            // discount factor
            epsilonStart: 1.0,      // initial exploration rate
            epsilonEnd: 0.01,       // minimum exploration rate
            epsilonDecay: 0.9999,   // Slower decay for more episodes
            qInit: 0,               // initial Q-value

            bins: {
                dx: { min: -50, max: 300, count: 15 },
                dy: { min: -200, max: 200, count: 20 },
                vy: { min: -8, max: 8, count: 15 },
            },

            training: {
                episodes: 50000,
                logEvery: 2500,
                movingAvgWindow: 500,
            },
        },
    },

    // ── Asset paths ─────────────────────────────────────────
    assets: {
        images: {
            bg: 'img/BG.png',
            ground: 'img/ground.png',
            topPipe: 'img/toppipe.png',
            botPipe: 'img/botpipe.png',
            getReady: 'img/getready.png',
            gameOver: 'img/go.png',
            tap: ['img/tap/t0.png', 'img/tap/t1.png'],
            bird: ['img/bird/b0.png', 'img/bird/b1.png', 'img/bird/b2.png', 'img/bird/b0.png'],
        },
        sfx: {
            start: 'sfx/start.wav',
            flap: 'sfx/flap.wav',
            score: 'sfx/score.wav',
            hit: 'sfx/hit.wav',
            die: 'sfx/die.wav',
        },
    },
};

// ── Game states (enum-like) ───────────────────────────────
export const GameState = {
    getReady: 0,
    play: 1,
    gameOver: 2,
};

// ── RL Actions ────────────────────────────────────────────
export const Action = {
    NO_OP: 0,
    FLAP: 1,
};
