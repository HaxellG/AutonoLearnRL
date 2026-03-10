import { CONFIG } from '../config.js';

export class StateDiscretizer {
  constructor(cfgBins = null) {
    // Custom non-uniform bins optimized for Flappy Bird
    // These give high resolution where it matters (near the pipe gap)
    this.dxBins = [-40, -20, 0, 20, 40, 60, 80, 100, 140, 200];
    this.dyBins = [-80, -40, -20, -10, 0, 10, 20, 40, 80];
    this.vyBins = [-4, -2, -0.5, 0.5, 2, 4];
  }

  discretize(state) {
    const [dx, dy, vy] = state;

    const iDx = this._getBinIndex(dx, this.dxBins);
    const iDy = this._getBinIndex(dy, this.dyBins);
    const iVy = this._getBinIndex(vy, this.vyBins);

    return {
      key: `${iDx}|${iDy}|${iVy}`,
      indices: [iDx, iDy, iVy],
    };
  }

  get stateCount() {
    return (this.dxBins.length + 1) * (this.dyBins.length + 1) * (this.vyBins.length + 1);
  }

  _getBinIndex(val, bins) {
    for (let i = 0; i < bins.length; i++) {
      if (val < bins[i]) return i;
    }
    return bins.length;
  }
}
