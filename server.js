const express = require("express");
const cors = require('cors');
const PORT = 5000
const app = express();

const tf = require('@tensorflow/tfjs');
require("@tensorflow/tfjs-backend-webgl")
const sharp = require('sharp');
const fs = require('fs');
const path = require('path');
const up = { path: __dirname + "/ursos_pardos", labels: 1 }
const ub = { path: __dirname + "/ursos_brancos", labels: 0 }

main([up, ub]).then(()=>console.log("pronto"))

const validation = 8

const inputShape = [224, 224, 4]
const numClasses = 2

const tensorTrain = { x: [], y: [] }
const tensorValidation = { x: [], y: [] }

app.use(cors()); // habilitando o cors da aplicação para todas as rotas

app.get("/train", async (req, res) => {
    console.log("alguem chamou");
    tensorTrain.x.dat = Array.from(tensorTrain.x.dataSync())
    // tensorTrain.y.dat = Array.from(tensorTrain.x.dataSync())
    // tensorValidation.x.dat = Array.from(tensorValidation.x.dataSync())
    // tensorValidation.y.dat = Array.from(tensorValidation.x.dataSync())
    const payload = {
        tensorTrainx: tensorTrain.x.dat,
        shape: tensorTrain.x.shape,
        dtype: tensorTrain.x.dtype
    }

    res.json(payload)
});

app.get("/trainy", async (req, res) => {
    // tensorTrain.x.dat = Array.from(tensorTrain.x.dataSync())
    tensorTrain.y.dat = Array.from(tensorTrain.y.dataSync())
    // tensorValidation.x.dat = Array.from(tensorValidation.x.dataSync())
    // tensorValidation.y.dat = Array.from(tensorValidation.x.dataSync())
    const payload = {
        tensorTrainy: tensorTrain.y.dat,
        shape: tensorTrain.y.shape,
        dtype: tensorTrain.y.dtype
    }

    res.json(payload)
});

app.get("/validation", async (req, res) => {
    // tensorTrain.x.dat = Array.from(tensorTrain.x.dataSync())
    // tensorTrain.y.dat = Array.from(tensorTrain.y.dataSync())
    tensorValidation.x.dat = Array.from(tensorValidation.x.dataSync())
    // tensorValidation.y.dat = Array.from(tensorValidation.x.dataSync())
    const payload = {
        tensorValidationx: tensorValidation.x.dat,
        shape: tensorValidation.x.shape,
        dtype: tensorValidation.x.dtype
    }

    res.json(payload)
});

app.get("/validationy", async (req, res) => {
    // tensorTrain.x.dat = Array.from(tensorTrain.x.dataSync())
    // tensorTrain.y.dat = Array.from(tensorTrain.y.dataSync())
    // tensorValidation.x.dat = Array.from(tensorValidation.x.dataSync())
    tensorValidation.y.dat = Array.from(tensorValidation.y.dataSync())
    const payload = {
        tensorValidationy: tensorValidation.y.dat,
        shape: tensorValidation.y.shape,
        dtype: tensorValidation.y.dtype
    }

    res.json(payload)
});

app.listen(PORT, ()=>console.log("Server running " + PORT))



// main([up, ub]).then(async () => {
//     // Modelo que espera tensores 4D
//     // tensorTrain.y.print()
//     // tensorTrain.x.print()
//     // const model = createModel4D(inputShape, numClasses);


//     // // Configurando callbacks para mostrar a precisão e a perda durante o treinamento

//     // // Treinando o modelo
//     // const numEpochs = 3;
//     // await model.fit(tensorTrain.x, tensorTrain.y, {
//     //     batchSize: 32,
//     //     epochs: numEpochs,
//     //     // validationData: [tensorValidation.x, tensorValidation.y],
//     //     shuffle: true,
//     //     callbacks: {
//     //         onEpochEnd: async (epoch, logs) => {
//     //             console.log(`Epoch ${epoch + 1}, Loss: ${logs.loss}, Accuracy: ${logs.acc}`);
//     //         }
//     //     }
//     // }).then(() => {
//     //     console.log('Treinamento concluído.');
//     // });;
// })

async function main(pasta) {
    const arquivosCompletos = []
    const arquivosTeste = []

    // URSOS PARDOS
    fs.readdir(pasta[0].path, (err, arquivos) => {
        if (err) {
            console.error('Erro ao ler a pasta:', err);
            return;
        }

        for (let index = 0; index < arquivos.length; index++) {
            if (index <= 7)
                arquivosTeste.push({ path: path.join(pasta[0].path, arquivos[index]), labels: pasta[0].labels })
            else
                arquivosCompletos.push({ path: path.join(pasta[0].path, arquivos[index]), labels: pasta[0].labels })
        }

        // Filtrar apenas os arquivos, não incluindo subpastas
        // const arquivosCompletos = arquivos
        //     .filter(arquivo => fs.statSync(path.join(pasta.path, arquivo)).isFile())
        //     .map(arquivo => { return { path: path.join(pasta.path, arquivo), labels: pasta.labels } });

        // processImages(arquivosCompletos).then(tensor => tensor);
    });

    //URSOS BRANCOS
    fs.readdir(pasta[1].path, (err, arquivos) => {
        if (err) {
            console.error('Erro ao ler a pasta:', err);
            return;
        }

        for (let index = 0; index < arquivos.length; index++) {
            if (index <= 7)
                arquivosTeste.push({ path: path.join(pasta[1].path, arquivos[index]), labels: pasta[1].labels })
            else
                arquivosCompletos.push({ path: path.join(pasta[1].path, arquivos[index]), labels: pasta[1].labels })
        }

        // Filtrar apenas os arquivos, não incluindo subpastas
        // const arquivosCompletos = arquivos
        //     .filter(arquivo => fs.statSync(path.join(pasta.path, arquivo)).isFile())
        //     .map(arquivo => { return { path: path.join(pasta.path, arquivo), labels: pasta.labels } });

        // processImages(arquivosCompletos).then(tensor => tensor);
    });

    return new Promise((resolve, reject) => {
        setTimeout(async () => {
            await processImages(arquivosCompletos).then(tensor => { tensorTrain.x = tensor.tensorImage; tensorTrain.y = tensor.tensorLabels })
            await processImages(arquivosTeste).then(tensor => { tensorValidation.x = tensor.tensorImage; tensorValidation.y = tensor.tensorLabels })

            resolve()
        }, 1000)
    })
}

// Função para processar várias imagens
async function processImages(arquivosCompletos) {
    const tensor = { tensorImage: [], tensorLabels: [] }
    tensor.tensorImage = await Promise.all(arquivosCompletos.map((arquivo) => processSingleImage(arquivo.path)));
    tensor.tensorLabels = await Promise.all(arquivosCompletos.map((arquivo) => processSingleLabel(arquivo.labels)));
    tensor.tensorImage = tf.concat(tensor.tensorImage); // Concatena os tensores ao longo do eixo do lote
    tensor.tensorLabels = tf.concat(tensor.tensorLabels); // Concatena os tensores ao longo do eixo do lote
    tensor.tensorLabels = tensor.tensorLabels.reshape([tensor.tensorLabels.size,1])
    // console.log(tensor.tensorImage);

    // tensor.tensorLabels = tf.concat(tensor.tensorLabels); // Concatena os tensores ao longo do eixo do lote
    // tensor.labels = tf.fill([tensor.tensorImage.shape[0]], arquivosCompletos[0].labels)
    return tensor
}

// Função para processar uma única imagem
async function processSingleImage(path) {
    const size = 64
    let channels = 3
    // Redimensionar a imagem para 224x224 com sharp
    const resizedImage = await sharp(path)
        .resize(size, size)
        .raw()
        .toBuffer({ resolveWithObject: true });

    const { data, info } = resizedImage;

    if (info.channels !== 4)
        channels = 3
    else
        channels = 4

    // Converter os dados brutos para um array de floats normalizados
    const normalizedImage = channels === 3
        ? Array.from(data).map(v => v / 255).concat(Array.from({ length: size * size }, () => 1)) // Adiciona um canal adicional com todos os valores 1
        : Array.from(data).map(v => v / 255);

    // Criar um tensor 4D [1, 224, 224, 3] para uma única imagem
    const tensorImage = tf.tensor4d(normalizedImage, [1, size, size, 4]);

    return tensorImage
}

async function processSingleLabel(labels) {
    const tensorLabel = tf.tensor1d([labels], "int32")
    // console.log(tensorLabel);
    return tensorLabel
}


// // Fazendo previsões com o modelo nas imagens processadas
// const predictions = model.predict(processedImages);

// // Exibindo as previsões
// predictions.print();

// Função para criar um modelo que espera tensores 4D
function createModel4D(inputShape, numClasses) {
    const model = tf.sequential();

    // // Adicionando uma camada de convolução 2D
    // model.add(tf.layers.conv2d({
    //     inputShape,
    //     filters: 32,
    //     kernelSize: 3,
    //     activation: 'relu',
    //     // kernelInitializer: "heNormal"
    // }));

    // model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

    // // Adicionando outra camada de convolução 2D
    // model.add(tf.layers.conv2d({
    //     filters: 64,
    //     kernelSize: 3,
    //     activation: 'relu',
    //     // kernelInitializer: "heNormal"
    // }));

    // model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));


    // Adicione camadas ao seu modelo aqui...
    model.add(tf.layers.flatten({ inputShape }));
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({
        optimizer: "adam",
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    // model.summary();

    return model;
}




// // Função para processar várias imagens
// async function processImages(arquivosCompletos) {
//     const tensor = { tensorImage: [], tensorLabels: [] }
//     await Promise.all(arquivosCompletos.map(async function (arquivo) {
//         tensor.tensorImage.push(await processSingleImage(arquivo.path))
//         tensor.tensorLabels.push(await processSingleLabel(arquivo.labels))

//         return
//     }));
//     // const tensorlabels = await Promise.all(arquivosCompletos.map(processSingleLabel));
//     tensor.tensorImage = tf.concat(tensor.tensorImage); // Concatena os tensores ao longo do eixo do lote
//     // tensor.tensorLabels = tf.concat(tensor.tensorLabels); // Concatena os tensores ao longo do eixo do lote

//     tensor.labels = tf.fill([tensor.tensorImage.shape[0]], arquivosCompletos[0].labels)
//     // console.log(tensor.labels);
//     return tensor
// }