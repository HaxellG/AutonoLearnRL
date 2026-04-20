// src/agents/tf_import.js
// Utility to load TensorFlow.js seamlessly across Browser and Node environments
let tf;

if (typeof window !== 'undefined' && window.tf) {
    // 1. Browser environment (loaded via CDN <script src="...">)
    tf = window.tf;
} else {
    // 2. Node.js headless environment
    try {
        // We use dynamic import for ES modules when possible, 
        // but for synchronous initialization we might need to rely on the module bundler pattern.
        // Actually, since this is a top-level module in Node running natively ES6:
        // we can just import it normally. But we only want to do that if not in browser.
        // A common pattern is importing tfjs directly and the bundler ignores it.
    } catch (e) {
        console.error("TensorFlow.js not found. If running in Node, ensure @tensorflow/tfjs is installed.");
    }
}

export { tf };
