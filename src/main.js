/**
 * AutonoLearn RL — Entry Point
 *
 * Bootstraps the game and wires the speed control UI.
 */
import { Game } from './Game.js';

// ── Game ──────────────────────────────────────────────
const canvas = document.getElementById('canvas');
const game = new Game(canvas);
game.start();

// ── Speed controls ───────────────────────────────────
const speedBtns = document.querySelectorAll('.speed-btn');

speedBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
        const speed = parseInt(btn.dataset.speed, 10);
        game.setSpeed(speed);

        // Update active class
        speedBtns.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
    });
});
