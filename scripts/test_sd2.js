// Test SD2 model
const { DQuilane } = require('/home/z/my-project/download/llm/web/app.js');
const fs = require('fs');

const modelData = JSON.parse(fs.readFileSync('/home/z/my-project/download/llm/web/dquilane_model.json', 'utf8'));
const model = new DQuilane(modelData);

console.log(`Model loaded. Vocab: ${model.config.vocab_size}, Params: ${model.w.tok_emb_weight.length / 4 * model.config.d_model}`);
console.log();

const prompts = [
    'Halo',
    'Satu tambah satu?',
    'Dua dikali dua?',
    'Lima dikali lima?',
    'Sepuluh dibagi dua?',
    'Ibu kota Indonesia?',
    'Apa ibu kota Jepang?',
    'Apa warna langit?',
    'Berapa hari dalam seminggu?',
    'Berapa menit dalam satu jam?',
    'Berapa benua di dunia?',
    'Apa mata uang Indonesia?',
    'Pola satu dua tiga empat. Berikutnya?',
    'Translate halo to English',
    'Apa itu air?',
    'Indonesia',
    'English is',
    '日本語',
];

(async () => {
    console.log('=== SD2 TEST ===');
    for (const p of prompts) {
        const t0 = Date.now();
        const out = await model.generate(p, 40, 0.5, 10, 0.9, 1.2);
        const elapsed = (Date.now() - t0) / 1000;
        console.log(`[${p}] (${elapsed.toFixed(2)}s)`);
        console.log(`  -> ${out.substring(0, 120)}`);
    }
})();
