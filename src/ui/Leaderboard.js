// src/ui/Leaderboard.js
// Loads results JSON from /results/ and renders a ranked leaderboard table

export class Leaderboard {
    constructor() {
        this.overlay = document.getElementById('leaderboard-overlay');
        this.closeBtn = document.getElementById('leaderboard-close');
        this.tbody = document.getElementById('leaderboard-tbody');
        this.difficultyBtns = document.querySelectorAll('.lb-diff-btn');
        this.currentDifficulty = 'medium';
        this.cache = {};     // cache fetched results per difficulty
        this.isOpen = false;

        this._bindEvents();
    }

    _bindEvents() {
        // Close button
        this.closeBtn.addEventListener('click', () => this.close());

        // Click outside panel to close
        this.overlay.addEventListener('click', (e) => {
            if (e.target === this.overlay) this.close();
        });

        // Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen) this.close();
        });

        // Difficulty tabs
        this.difficultyBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                this.difficultyBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.currentDifficulty = btn.dataset.diff;
                this._loadAndRender();
            });
        });
    }

    async open() {
        this.isOpen = true;
        this.overlay.classList.add('visible');
        // small delay for transition
        requestAnimationFrame(() => {
            this.overlay.querySelector('.leaderboard-panel').classList.add('visible');
        });
        await this._loadAndRender();
    }

    close() {
        this.isOpen = false;
        this.overlay.querySelector('.leaderboard-panel').classList.remove('visible');
        setTimeout(() => {
            this.overlay.classList.remove('visible');
        }, 350); // matches CSS transition duration
    }

    async _loadAndRender() {
        const diff = this.currentDifficulty;

        if (!this.cache[diff]) {
            this.tbody.innerHTML = `
                <tr>
                    <td colspan="7" class="lb-loading">LOADING...</td>
                </tr>`;

            const agents = ['qlearning', 'dqn', 'ddqn'];
            const results = [];

            for (const agent of agents) {
                try {
                    const res = await fetch(`./results/${diff}/${agent}_results.json`);
                    if (res.ok) {
                        const data = await res.json();
                        results.push(data);
                    }
                } catch (e) {
                    console.warn(`Could not load ${agent} results for ${diff}:`, e);
                }
            }

            this.cache[diff] = results;
        }

        this._render(this.cache[diff]);
    }

    _render(results) {
        if (!results || results.length === 0) {
            this.tbody.innerHTML = `
                <tr>
                    <td colspan="7" class="lb-loading">NO DATA AVAILABLE</td>
                </tr>`;
            return;
        }

        // Sort by finalEval.mean descending (best first)
        const sorted = [...results].sort((a, b) => b.finalEval.mean - a.finalEval.mean);

        const medals = ['🥇', '🥈', '🥉'];
        const rankClasses = ['lb-rank-gold', 'lb-rank-silver', 'lb-rank-bronze'];

        this.tbody.innerHTML = sorted.map((r, i) => {
            const fe = r.finalEval;
            const hp = r.hyperparameters;
            const medal = medals[i] || `${i + 1}`;
            const rankClass = rankClasses[i] || '';
            const agentShort = r.agent;

            // Format training time
            const timeStr = this._formatTime(r.trainingTimeSec);

            return `
                <tr class="lb-row ${rankClass}">
                    <td class="lb-cell lb-cell--rank">${medal}</td>
                    <td class="lb-cell lb-cell--agent">${agentShort}</td>
                    <td class="lb-cell lb-cell--highlight">${fe.mean.toFixed(2)}</td>
                    <td class="lb-cell">${fe.median}</td>
                    <td class="lb-cell">${fe.max}</td>
                    <td class="lb-cell">${fe.min}</td>
                    <td class="lb-cell">${fe.stdDev.toFixed(2)}</td>
                </tr>`;
        }).join('');
    }

    _formatTime(sec) {
        if (sec < 60) return `${sec.toFixed(1)}s`;
        const mins = Math.floor(sec / 60);
        const secs = Math.floor(sec % 60);
        if (mins < 60) return `${mins}m ${secs}s`;
        const hrs = Math.floor(mins / 60);
        const remainMins = mins % 60;
        return `${hrs}h ${remainMins}m`;
    }
}
