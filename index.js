const IA = "AIzaSyC6xpV6X3I3e51maGfdaOZXbQvP_EQBONg"
const SEARCH = "348dfd8a0715645ee"
const fs = require("fs")
const path = require("path")

let image

async function testse(start) {
    let x = await fetch(`https://www.googleapis.com/customsearch/v1?key=${IA}&cx=${SEARCH}&q=Ursus arctos&num=10&searchType=image${start !== 1 ? `&start=${start}` : ""}`).then(
        res => {
            res.json().then(json => baixar(json, start))
        }
    ).catch(err => console.log(err))
}


function baixar(json, name) {
    for (let index = 0; index < json.items.length; index++) {
        let x = json.items[index].link;
        console.log(x);
        fetch(x).then(res => res.blob().then(blob => blob.arrayBuffer().then(blobBuffer => {
            image = new Buffer.from(blobBuffer, "binary")
            fs.writeFileSync(path.join(__dirname, "ursos_pardos", `up${name + index}.png`), image)
        }).catch(err => console.log(err))).catch(err => console.log(err))).catch(err => console.log(err))
    }
    if (name + 10 < 101)
    testse(name + 10)

}

testse(1)