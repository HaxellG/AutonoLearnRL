/**
 * AutonoLearn RL — Entry Point
 *
 * This module bootstraps the game. It is loaded as an ES module from index.html.
 */
import { Game } from './Game.js';

const canvas = document.getElementById('canvas');
const game = new Game(canvas);
game.start();
