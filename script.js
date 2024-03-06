const inputShape = [64, 64, 4]

document.addEventListener('DOMContentLoaded', async () => {
    // Inicialize o TensorFlow.js
    await tf.ready().then(() => {
        // Seu código TensorFlow.js aqui
    });
    document.getElementById("test").addEventListener("click", test)
    // run()
});

async function test() {
    let resultado = null
    let tensorx = null
    let tensory = null
    const model = await tf.loadLayersModel('http://localhost:3000/model/teste.json');
    await fetch('http://localhost:3000/test').then(json => json.json().then(data => {
        tensorx = tf.tensor(data[0].image, data[0].shape, data[0].dtype);
        tensory = tf.tensor(data[1].label, data[1].shape, data[1].dtype);

        resultado = model.predict(tensorx);

        // const corretos = tensory.reduce((acumulador, valor, índice) => {
        //     return valor === resultado[índice] ? acumulador + 1 : acumulador;
        // }, 0);
    
        // const precisao = corretos / tensory.length;
    
        // console.log('Precisão:', precisao);
    }))

    console.log(tensory);
    console.log(resultado);

    // const inputElement = document.getElementById('inputImage');
    // const imagemInput = document.getElementById('imagemInput');
    // const resultadoElement = document.getElementById('resultado');

    // const arquivo = inputElement.files[0];
    // if (arquivo) {
    //     const imagemTensor = await carregarImagem(arquivo);
    //     imagemInput.src = URL.createObjectURL(arquivo);

    //     // Substitua 'modelo/model.json' pelo caminho do seu modelo
    //     const resultado = model.predict(imagemTensor);

    //     // Exibir resultados ou fazer o que for necessário com 'resultado'
    //     resultadoElement.innerHTML = JSON.stringify(resultado.dataSync(), null, 2);
    // } else {
    //     alert('Selecione uma imagem antes de processar.');
    // }
    console.log(resultado.dataSync()+"\n"+tensory.dataSync());
}

async function carregarImagem(arquivo) {
    return new Promise((resolve, reject) => {
        const leitor = new FileReader();
        leitor.onload = (evento) => {
            const imagem = new Image();
            imagem.onload = () => {
                const tensor = tf.browser.fromPixels(imagem).toFloat().div(tf.scalar(255));
                console.log(tensor);
                resolve(tensor);
            };
            imagem.src = evento.target.result;
        };
        leitor.readAsDataURL(arquivo);
    });
}


async function run() {
    tf.disposeVariables()
    const data = await getdData();
    // console.log(data);
    const tensorx = tf.tensor(data.x.tensorTrainx, data.x.shape, data.x.dtype);
    const tensory = tf.tensor(data.y.tensorTrainy, data.y.shape, data.y.dtype);
    const validationx = tf.tensor(data.z.tensorValidationx, data.z.shape, data.z.dtype);
    const validationy = tf.tensor(data.k.tensorValidationy, data.k.shape, data.k.dtype);
    // data.tensorTrain.x = tf.tensor4d(data.tensorTrain.x.size , [data.tensorTrain.x.shape[0], data.tensorTrain.x.shape[1], data.tensorTrain.x.shape[2], data.tensorTrain.x.shape[3]]);
    // data.tensorTrain.x = tf.tensor(await data.tensorTrain.x.data(), data.tensorTrain.x.shape, data.tensorTrain.x.dtype);
    console.log("treinar");

    const model = createModel4D(inputShape);
    tfvis.show.modelSummary({ name: 'Model Architecture', tab: 'Model' }, model);

    await train(model, tensorx, tensory, validationx, validationy);

    // await showAccuracy(model, data);
    // await showConfusion(model, data);
}

async function getdData() {
    let dad = { x: null, y: null, z: null, k: null }
    // if(localStorage.getItem("data")){
    //     dad = JSON.parse(localStorage.getItem("data"))
    //     return dad
    // }
    await fetch("http://localhost:5000/train").then(x => x.json().then(data => dad.x = data))
    await fetch("http://localhost:5000/trainy").then(x => x.json().then(data => dad.y = data))
    await fetch("http://localhost:5000/validation").then(x => x.json().then(data => dad.z = data))
    await fetch("http://localhost:5000/validationy").then(x => x.json().then(data => dad.k = data))
    // localStorage.setItem("data", JSON.stringify(dad))
    console.log("deu certo");
    return dad
}

async function train(model, tensorx, tensory, validationx, validationy) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
        name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const BATCH_SIZE = 32;

    return model.fit(tensorx, tensory, {
        batchSize: BATCH_SIZE,
        validationData: [validationx, validationy],
        epochs: 100,
        shuffle: true,
        callbacks: fitCallbacks
        //     onEpochEnd: async (epoch, logs) => {
        //         console.log(`Epoch ${epoch + 1}, Loss: ${logs.loss}, Accuracy: ${logs.acc}`);
        //     }
        // }
    }).then(async () => {
        console.log('Treinamento concluído.');
        await model.save('downloads://teste');
    });;
}

// // Treinando o modelo
// const numEpochs = 3;
// await model.fit(tensorTrain.x, tensorTrain.y, {
//     batchSize: 32,
//     epochs: numEpochs,
//     // validationData: [tensorValidation.x, tensorValidation.y],
//     shuffle: true,
// callbacks: {
//     onEpochEnd: async (epoch, logs) => {
//         console.log(`Epoch ${epoch + 1}, Loss: ${logs.loss}, Accuracy: ${logs.acc}`);
//     }
// }
// }).then(() => {
//     console.log('Treinamento concluído.');
// });;


function createModel4D(inputShape, numClasses) {
    const model = tf.sequential();

    // Adicionando uma camada de convolução 2D
    model.add(tf.layers.conv2d({
        inputShape,
        filters: 32,
        kernelSize: 3,
        activation: 'relu',
        kernelInitializer: "heNormal"
    }));

    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

    // Adicionando outra camada de convolução 2D
    model.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        activation: 'relu',
    }));

    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));


    // Adicione camadas ao seu modelo aqui...
    model.add(tf.layers.flatten());
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

function getModel(inputShape) {
    const model = tf.sequential();

    // In the first layer of our convolutional neural network we have
    // to specify the input shape. Then we specify some parameters for
    // the convolution operation that takes place in this layer.
    // model.add(tf.layers.conv2d({
    //     inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    //     kernelSize: 5,
    //     filters: 8,
    //     strides: 1,
    //     activation: 'relu',
    //     kernelInitializer: 'varianceScaling'
    // }));

    // // The MaxPooling layer acts as a sort of downsampling using max values
    // // in a region instead of averaging.
    // model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    // // Repeat another conv2d + maxPooling stack.
    // // Note that we have more filters in the convolution.
    // model.add(tf.layers.conv2d({
    //     kernelSize: 5,
    //     filters: 16,
    //     strides: 1,
    //     activation: 'relu',
    //     kernelInitializer: 'varianceScaling'
    // }));
    // model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    // model.add(tf.layers.dense({
    //     units: NUM_OUTPUT_CLASSES,
    //     kernelInitializer: 'varianceScaling',
    //     activation: 'softmax'
    // }));

    model.add(tf.layers.flatten([inputShape]));
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));


    // Choose an optimizer, loss function and accuracy metric,
    // then compile and return the model
    const optimizer = tf.train.adam();

    model.compile({
        optimizer: optimizer,
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });


    // model.compile({
    //     optimizer: optimizer,
    //     loss: 'categoricalCrossentropy',
    //     metrics: ['accuracy'],
    // });

    return model;
}


// function doPrediction(model, data, testDataSize = 500) {
//     const IMAGE_WIDTH = 28;
//     const IMAGE_HEIGHT = 28;
//     const testData = data.nextTestBatch(testDataSize);
//     const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
//     const labels = testData.labels.argMax(-1);
//     const preds = model.predict(testxs).argMax(-1);

//     testxs.dispose();
//     return [preds, labels];
// }

// async function showAccuracy(model, data) {
//     const [preds, labels] = doPrediction(model, data);
//     const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
//     const container = { name: 'Accuracy', tab: 'Evaluation' };
//     tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

//     labels.dispose();
// }

// async function showConfusion(model, data) {
//     const [preds, labels] = doPrediction(model, data);
//     const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
//     const container = { name: 'Confusion Matrix', tab: 'Evaluation' };
//     tfvis.render.confusionMatrix(container, { values: confusionMatrix, tickLabels: classNames });

//     labels.dispose();
// }