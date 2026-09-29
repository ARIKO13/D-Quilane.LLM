// Test v0.10 with all topics
const { DQuilane } = require('/home/z/my-project/download/llm/web/app.js');
const fs = require('fs');
const modelData = JSON.parse(fs.readFileSync('/home/z/my-project/download/llm/web/dquilane_model.json', 'utf8'));
const model = new DQuilane(modelData);
const EOS = modelData.config.eos_id;
console.log(`Vocab: ${model.config.vocab_size}, d_model: ${model.config.d_model}, n_layers: ${model.config.n_layers}`);

const prompts = [
    // Emotions (NEW!)
    'Apa itu senang?',
    'Apa itu takut?',
    'Apa itu lapar?',
    // Body parts (NEW!)
    'Apa fungsi mata?',
    'Berapa jari di tangan?',
    // Colors
    'Apa warna langit?',
    'Apa warna rumput?',
    // Animals + sounds (NEW!)
    'Apa suara kucing?',
    'Apa suara sapi?',
    'Apa suara anjing?',
    // Family (NEW!)
    'Siapa kakek?',
    'Siapa paman?',
    // Plants (NEW!)
    'Apa fungsi akar?',
    'Apa yang dibutuhkan tanaman?',
    // Tech (NEW!)
    'Apa itu komputer?',
    'Apa itu YouTube?',
    'Apa itu Minecraft?',
    // Maps (NEW!)
    'Matahari terbit di?',
    'Apa itu kompas?',
    // Daily routines (NEW!)
    'Apa itu mandi?',
    'Apa itu PR?',
    // Reading (NEW!)
    'Sebutkan huruf A sampai Z',
    // Standard
    'Halo',
    'Satu tambah satu?',
    'Siapa presiden pertama Indonesia?',
    'tokyo itu apa',
    'ibu kota jepang apa',
    'Apa warna darah?',
];

(async () => {
    console.log('\n=== v0.10 TEST ===');
    for (const p of prompts) {
        const t0 = Date.now();
        const out = await model.generate(p, 100, 0.4, 10, 0.9, 1.2, undefined, EOS);
        const elapsed = (Date.now() - t0) / 1000;
        console.log(`[${p}] (${elapsed.toFixed(2)}s)`);
        console.log(`  -> ${out.substring(0, 110)}`);
    }
})();
