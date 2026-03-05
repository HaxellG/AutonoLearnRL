import { CONFIG } from '../config.js';

/**
 * Sound effects module.
 * Exports a single SFX object matching the original structure.
 */
export function createSFX() {
    const sfx = {
        start: new Audio(),
        flap: new Audio(),
        score: new Audio(),
        hit: new Audio(),
        die: new Audio(),
        played: false,
    };

    sfx.start.src = CONFIG.assets.sfx.start;
    sfx.flap.src = CONFIG.assets.sfx.flap;
    sfx.score.src = CONFIG.assets.sfx.score;
    sfx.hit.src = CONFIG.assets.sfx.hit;
    sfx.die.src = CONFIG.assets.sfx.die;

    return sfx;
}
