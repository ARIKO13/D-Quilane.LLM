// Test SD2 + EOS
const { DQuilane } = require('/home/z/my-project/download/llm/web/app.js');
const fs = require('fs');

const modelData = JSON.parse(fs.readFileSync('/home/z/my-project/download/llm/web/dquilane_model.json', 'utf8'));
const model = new DQuilane(modelData);
const EOS_ID = modelData.config.eos_id || modelData.config.vocab_size - 1;
console.log(`Model loaded. Vocab: ${model.config.vocab_size}, EOS_ID: ${EOS_ID}`);

const prompts = [
    'Halo',
    'Siapa presiden pertama Indonesia?',
    'Siapa presiden Indonesia?',
    'Satu tambah satu?',
    'Dua dikali dua?',
    'Lima dikali lima?',
    'Ibu kota Indonesia?',
    'Apa ibu kota Jepang?',
    'Apa warna langit?',
    'Berapa hari dalam seminggu?',
    'Apa mata uang Indonesia?',
    'Pola satu dua tiga empat. Berikutnya?',
    'Translate halo to English',
    'Terima kasih',
    'Indonesia',
];

(async () => {
    console.log('\n=== SD2 + EOS TEST ===');
    for (const p of prompts) {
        const t0 = Date.now();
        // Test with stop on EOS_ID
        const out = await model.generate(p, 100, 0.5, 10, 0.9, 1.2, undefined, EOS_ID);
        const elapsed = (Date.now() - t0) / 1000;
        console.log(`[${p}] (${elapsed.toFixed(2)}s)`);
        console.log(`  -> ${out}`);
    }
})();
