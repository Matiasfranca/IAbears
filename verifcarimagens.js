const fs = require("fs");
const path = require("path")
const sizeOf = require('image-size');

for (let index = 1; index <= 100; index++) {
    if (fs.existsSync(__dirname + `/ursos_pardos/up${index}.png`)) {
        let image = fs.readFileSync(__dirname + `/ursos_pardos/up${index}.png`)
        try {
            const dimensions = sizeOf(image)
        } catch (error) {
            console.log(error);
            fs.unlinkSync(__dirname + `/ursos_pardos/up${index}.png`, (err) => {
                if (err) {
                  console.error('Erro ao apagar o arquivo:', err);
                } else {
                  console.log('Arquivo apagado com sucesso.');
                }
              })
        }
    }
}


const pasta = __dirname + `/ursos_pardos/`
// Lê os arquivos na pasta
fs.readdir(__dirname + `/ursos_pardos/`, (err, arquivos) => {
  if (err) {
    console.error('Erro ao ler a pasta:', err);
    return;
  }
  let i = 1
  // Loop através dos arquivos
  for (const arquivo of arquivos) {
    const caminhoAtual = path.join(pasta, arquivo);

    // Renomeia o arquivo
    fs.rename(caminhoAtual, path.join(pasta, `up${i}.png`), (err) => {
      if (err) {
        console.error(`Erro ao renomear ${arquivo}:`, err);
      } else {
        console.log(`${arquivo} renomeado para up${i}.png`);
      }
    });
    i++;
  }
});

