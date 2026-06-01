import { Game } from './Game.js';
import { BattleMode } from './BattleMode.js';
import { SimulationRunner } from './SimulationRunner.js';
import { FlappyEnv } from './env/FlappyEnv.js';
import { QLearningAgent } from './agents/QLearningAgent.js';
import { DQNAgent } from './agents/DQNAgent.js';
import { DoubleDQNAgent } from './agents/DoubleDQNAgent.js';
import { Telemetry } from './utils/Telemetry.js';
import { Dashboard } from './ui/Dashboard.js';
import { Leaderboard } from './ui/Leaderboard.js';
import { Evaluator } from './utils/Evaluator.js';
import { CONFIG } from './config.js';

// ── UI Elements ──────────────────────────────────────────
const canvas = document.getElementById('canvas');
const modeBtns = document.querySelectorAll('#mode-selector [data-mode]');
const diffBtns = document.querySelectorAll('#diff-selector-panel [data-diff]');
const seedBtns = document.querySelectorAll('#diff-selector-panel [data-seed]');
const speedBtns = document.querySelectorAll('.speed-btn');
const speedControls = document.getElementById('speed-controls');
const leaderboardBtn = document.getElementById('leaderboard-btn');

// ── Global State ─────────────────────────────────────────
let humanGame = null;
let aiRunner = null;
let battleMode = null;
let env = null;
let agent = null;
let telemetry = null;
let dashboard = null;

// Initialize Human Mode by default
startHumanMode();

// ── Leaderboard ──────────────────────────────────────────
const leaderboard = new Leaderboard();
leaderboardBtn.addEventListener('click', () => leaderboard.open());

// ── Mode Switcher ────────────────────────────────────────
modeBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
        if (btn.disabled) return;
        const mode = btn.dataset.mode;

        // Update active button highlight
        modeBtns.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');

        stopCurrentModes();
        if (mode === 'human') {
            startHumanMode();
        } else if (mode === 'battle' || mode === 'vs') {
            startBattleMode(mode);
        } else {
            startAIMode(mode);
        }
    });
});

// ── Difficulty Switcher ──────────────────────────────────
diffBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
        const diff = btn.dataset.diff;
        
        diffBtns.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');

        if (diff === 'normal') {
            CONFIG.pipe.gap = 125;
        } else if (diff === 'hard') {
            CONFIG.pipe.gap = 100;
        }

        // Restart current mode to apply new difficulty
        const activeModeBtn = document.querySelector('#mode-selector [data-mode].active');
        if (activeModeBtn) {
            activeModeBtn.click();
        }
    });
});

// ── Seed Mode Switcher ───────────────────────────────────
seedBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
        const seedMode = btn.dataset.seed;
        
        seedBtns.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');

        CONFIG.game.seedMode = seedMode;

        // Restart current mode to apply new seed mode
        const activeModeBtn = document.querySelector('#mode-selector [data-mode].active');
        if (activeModeBtn) {
            activeModeBtn.click();
        }
    });
});

// ── Speed Controls ───────────────────────────────────────
speedBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
        if (btn.disabled) return;
        const speed = parseInt(btn.dataset.speed, 10);

        if (humanGame) humanGame.setSpeed(speed);
        if (aiRunner) aiRunner.setStepsPerFrame(speed);

        speedBtns.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
    });
});

// ── Logic ───────────────────────────────────────────────

function stopCurrentModes() {
    if (humanGame) {
        humanGame.stop();
        humanGame = null;
    }
    if (aiRunner) {
        aiRunner.stop();
        aiRunner = null;
    }
    if (dashboard) {
        dashboard.hide();
    }
    if (battleMode) {
        battleMode.stop();
        battleMode = null;
    }
    document.getElementById('ai-indicators').style.display = 'none';
    document.getElementById('benchmark-dashboard').classList.add('hidden');
    document.getElementById('canvas-frame').style.display = 'inline-block';
    document.getElementById('battle-status').textContent = '';
}

function startHumanMode() {
    if (speedControls) speedControls.style.display = 'none';
    humanGame = new Game(canvas);

    // Force 1x speed for manual mode
    speedBtns.forEach((b) => b.classList.remove('active'));
    const speed1Btn = document.querySelector('.speed-btn[data-speed="1"]');
    if (speed1Btn) speed1Btn.classList.add('active');
    
    humanGame.setSpeed(1);

    humanGame.start();
}

function startAIMode(agentType) {
    if (speedControls) speedControls.style.display = 'flex';
    
    // Disable 5000x for DQN/DDQN (performance limitation)
    const speed5000Btn = document.querySelector('.speed-btn[data-speed="5000"]');
    if (speed5000Btn) {
        if (agentType === 'dqn' || agentType === 'ddqn') {
            speed5000Btn.disabled = true;
            speed5000Btn.style.opacity = '0.4';
            speed5000Btn.style.cursor = 'not-allowed';
            if (speed5000Btn.classList.contains('active')) {
                speed5000Btn.classList.remove('active');
                const speed100Btn = document.querySelector('.speed-btn[data-speed="100"]');
                if (speed100Btn) speed100Btn.classList.add('active');
            }
        } else {
            speed5000Btn.disabled = false;
            speed5000Btn.style.opacity = '';
            speed5000Btn.style.cursor = '';
        }
    }

    // 1. Initialize RL components
    env = new FlappyEnv();

    let maxEpisodes = 20000;

    if (agentType === 'dqn') {
        agent = new DQNAgent();
        agent.epsilonDecay = Math.pow(agent.epsilonMin / agent.epsilon, 1 / (maxEpisodes * 0.8));
    } else if (agentType === 'ddqn') {
        maxEpisodes = 30000;
        agent = new DoubleDQNAgent();
        agent.epsilonDecay = Math.pow(agent.epsilonMin / agent.epsilon, 1 / (maxEpisodes * 0.8));
    } else {
        agent = new QLearningAgent();
        agent.epsilonDecay = Math.pow(agent.epsilonEnd / agent.epsilon, 1 / (maxEpisodes * 0.8));
    }

    // Smooth moving average: Q-Learning is noisy (100 window), neural nets are smoother (50 window)
    telemetry = new Telemetry(agentType === 'qlearning' ? 100 : 50);
    telemetry.currentEpsilon = agent.epsilon;
    dashboard = new Dashboard(telemetry);
    dashboard.show();
    dashboard.update();
    document.getElementById('ai-indicators').style.display = 'flex';

    // 2. We use Game as a pure renderer for the AI
    // Assign to humanGame so stopCurrentModes() will stop its internal loop if mode switches
    humanGame = new Game(canvas);
    const renderer = humanGame;

    // 3. Runner orchestrates training + rendering
    let lastState = null;
    let lastAction = null;

    aiRunner = new SimulationRunner(env, {
        actionProvider: (state) => {
            const action = agent.act(state);
            lastState = state;
            lastAction = action;
            return action;
        },
        renderer: (e) => {
            renderer.renderEnv(e);
            
            const activeSpeedBtn = document.querySelector('.speed-btn.active');
            const activeSpeed = activeSpeedBtn ? parseInt(activeSpeedBtn.dataset.speed, 10) : 1;
            const indicatorsPanel = document.getElementById('ai-indicators');
            
            if (activeSpeed !== 1) {
                indicatorsPanel.style.opacity = '0.4';
                document.getElementById('ind-jump').classList.remove('active');
                document.getElementById('ind-nojump').classList.remove('active');
                return;
            } else {
                indicatorsPanel.style.opacity = '1';
            }

            if (lastAction === 1) {
                document.getElementById('ind-jump').classList.add('active');
                document.getElementById('ind-nojump').classList.remove('active');
                
                clearTimeout(window.jumpTimeout);
                window.jumpTimeoutIsActive = true;
                window.jumpTimeout = setTimeout(() => {
                    window.jumpTimeoutIsActive = false;
                    document.getElementById('ind-jump').classList.remove('active');
                    document.getElementById('ind-nojump').classList.add('active');
                }, 500);
            } else if (!window.jumpTimeoutIsActive) {
                document.getElementById('ind-jump').classList.remove('active');
                document.getElementById('ind-nojump').classList.add('active');
            }
        },
        onStep: (result) => {
            // TD Update during play!
            agent.learn(lastState, lastAction, result.reward, result.state, result.done);
        },
        onEpisodeEnd: (summary) => {
            agent.endEpisode();
            // Record metrics
            telemetry.addEpisode(summary.totalReward, summary.score, agent.epsilon, agent.statesVisited || "N/A");
            // Update UI
            dashboard.update();

            // Stop training at maxEpisodes
            if (telemetry.episodes >= maxEpisodes) {
                aiRunner.stop();
                console.log(`Training complete (${maxEpisodes} episodes reached).`);
            }
        }
    });

    // Maintain speed selection
    const activeSpeed = document.querySelector('.speed-btn.active');
    if (activeSpeed) aiRunner.setStepsPerFrame(parseInt(activeSpeed.dataset.speed, 10));

    // Wait for user to click to start, using Game's native Get Ready state
    renderer.start();

    const checkStart = () => {
        // If mode switched while waiting, abort
        if (humanGame !== renderer) return;

        if (renderer._getState() === 1) { // GameState.play
            // User clicked! Stop the manual loop and start the AI loop
            renderer.stop();
            
            let seedCount = 0;
            aiRunner._onReset = () => {
                if (CONFIG.game.seedMode === 'random') {
                    env.seed = Math.floor(Math.random() * 10000);
                } else {
                    env.seed = 0; // Fixed EVAL_SEED
                }
            };
            const initialSeed = CONFIG.game.seedMode === 'random' ? Math.floor(Math.random() * 10000) : 0;
            aiRunner.startVisual(initialSeed);
        } else {
            requestAnimationFrame(checkStart);
        }
    };
    checkStart();
}

async function startBattleMode(mode) {
    if (speedControls) speedControls.style.display = 'none';

    battleMode = new BattleMode(canvas, mode);
    battleMode.onBattleEnd((results) => {
        showBattleResults(results, mode);
    });
    await battleMode.start();
}

function showBattleResults(results, mode) {
    const overlay = document.getElementById('battle-results-overlay');
    const podium = document.getElementById('battle-podium');
    const medals = ['🥇', '🥈', '🥉'];

    podium.innerHTML = '';
    results.forEach((r, i) => {
        const row = document.createElement('div');
        row.className = 'podium-row' + (i === 0 ? ' gold' : '');
        row.innerHTML = `
            <span class="podium-rank">${medals[i] || (i+1)}</span>
            <span class="podium-name" style="color:${r.color}">${r.name}</span>
            <span class="podium-score">SCORE: ${r.score}</span>
        `;
        podium.appendChild(row);
    });

    overlay.classList.add('open');

    // Close button
    document.getElementById('battle-results-close').onclick = () => {
        overlay.classList.remove('open');
    };

    // Rematch button
    document.getElementById('battle-replay-btn').onclick = () => {
        overlay.classList.remove('open');
        // Restart battle
        const battleBtn = document.querySelector(`#mode-selector [data-mode="${mode}"]`);
        if (battleBtn) battleBtn.click();
    };
}

