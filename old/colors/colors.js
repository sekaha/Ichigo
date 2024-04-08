let b_polynomials = {};

const satBoost = document.getElementById("saturation-boost");
const paletteType = document.getElementById("palette-type");
const colorInputOne = document.getElementById("color1");
const colorInputTwo = document.getElementById("color2");
const colorInputThree = document.getElementById("color3");
const blindnessMode = document.getElementById("color-blindness");
const paletteCount = document.getElementById("palette-count-input");


function run() {
    gen();

    [paletteType, colorInputOne, colorInputTwo, colorInputThree, paletteCount, blindnessMode].forEach(
        (input) => input.addEventListener("input", () => {
            gen();
        })
    );
}

function gen() {
    const generatorFunctions = {
        "sequential_direct": generateSequential,
        "sequential_hue": generateSequentialHue,
        "bezier_direct": generateBezier,
        "bezier_hue": generateBezierHue,
        "diverging": generateDiverging,
        "spectrum": generateEquidistance,
        "triad": generateEquidistance,
    };

    col1 = blackFix(colorInputOne.value);
    col2 = blackFix(colorInputTwo.value);
    col3 = blackFix(colorInputThree.value);

    const args = [
        [[col1, col2], Math.max(2, Math.min(99, paletteCount.value))],
        [col1, col2, Math.max(2, Math.min(99, paletteCount.value))],
        [[col1, col3, col2], Math.max(2, Math.min(99, paletteCount.value))],
        [[col1, col3, col2], Math.max(2, Math.min(99, paletteCount.value))],
        [col1, col2, Math.max(2, Math.min(99, paletteCount.value))],
        [col1, Math.max(2, Math.min(99, paletteCount.value))],
        [col1, Math.max(2, Math.min(99, paletteCount.value))],
    ];

    createPanels(generatorFunctions[paletteType.value](...args[paletteType.selectedIndex]));
}

function darkToLight(cols, paletteCount) {
    let colors = generateSequential(cols, paletteCount);
    let new_colors = [];

    for (let i = 0; i < colors.length; i++) {
        let [l, c, h] = chroma(colors[i]).oklch();
        new_c = (1 - Math.abs(2 * (i / (colors.length) - 0.5))) * c// c * 
        new_colors.push(chroma.oklch(0.125 + (i / (colors.length)) * 0.75, new_c, h))
        console.log(new_c)
    }

    return new_colors
}

// blue shifted ought to be darker shifted and yellow shifted ligher shifted
function generateEquidistance(col, splits, blindnessMode, degrees = 360) {
    let colors = Array.from({ length: splits }, (_, i) => {
        return chroma(col).set('oklch.h', '+' + (i * (degrees / splits)).toString());
    });

    color_strings = colors.map(s => "\'" + s.hex().replace(/^#/, "#") + "\'")
    console.log("[" + color_strings.join(", ") + "]");

    return colors
}

function generateSequential(cols, paletteCount) {
    console.log(cols);
    const colors = chroma.scale(cols).mode("oklab").colors(paletteCount);

    color_strings = colors.map(s => "\'" + s.replace(/^#/, "#") + "\'")
    console.log("[" + color_strings.join(", ") + "]");

    return colors
}

function generateSequentialHue(col1, col2, paletteCount) {
    const satBoost = document.getElementById("saturation-boost");
    [l1, c1, h1] = chroma(col1).oklch();
    [l2, c2, h2] = chroma(col2).oklch();

    if (h1 < h2) {
        h1 += 360;
    }

    h_diff = (h2 - h1);
    l_diff = (l2 - l1);
    c_diff = (c2 - c1);
    colors = [];

    for (let i = 0; i < paletteCount; i++) {
        colors.push(chroma.oklch(l1 + i * (l_diff / paletteCount),
            (c1 + i * (c_diff / paletteCount)) * (satBoost.value / 100),
            (h1 + i * (h_diff / paletteCount)) % 360));
    }

    color_strings = colors.map(s => "\'" + s.hex().replace(/^#/, "#") + "\'")
    console.log("[" + color_strings.join(", ") + "]");

    return colors
}

function generateDiverging(col1, col2, paletteCount) {
    col1 = chroma(col1).oklch()
    col2 = chroma(col2).oklch()
    let [l1, c1, h1] = col1;
    let [l2, c2, h2] = col2;

    let brightness = (l1 + l2) / 2;
    let sat = (c1 + c2) / 2
    let bright_diff = chroma("white").oklch()[0] - brightness;
    let sat_diff = chroma("white").oklch()[1] - sat;
    // let new_col1 = chroma.oklch(brightness, sat, h1);
    // let new_col2 = chroma.oklch(brightness, sat, h2);

    let new_col1_dark = chroma.oklch(brightness - bright_diff, sat - sat_diff, h1);
    let new_col2_dark = chroma.oklch(brightness - bright_diff, sat - sat_diff, h2);
    let new_col1_light = chroma.oklch(brightness + bright_diff, sat + sat_diff, h1);
    let new_col2_light = chroma.oklch(brightness + bright_diff, sat + sat_diff, h2);

    const colors = chroma.scale([new_col1_dark, new_col1_light, new_col2_dark]).mode("oklch").colors(paletteCount);

    color_strings = colors.map(s => "\'" + chroma(s).hex().replace(/^#/, "#") + "\'")
    console.log("[" + color_strings.join(", ") + "]");

    return colors;
}

function choose(n, k) {
    let res = 1;

    for (let i = 1; i < k + 1; i++) {
        res *= n - i + 1;
        res = Math.floor(res / i);
    }

    return res;
}

function getBernsteinPolynomial(n) {
    if (!(n in b_polynomials)) {
        let deg = n - 1;

        b_polynomials[n] = Array.from({ length: n }, (_, i) => (
            (j) => (t) => {
                return choose(deg, j) * ((1 - t) ** (deg - j)) * (t ** j);
            })(i)
        );
    }

    return b_polynomials[n];
}

function generateBezier(cols, paletteCount) {
    cols = ["#fcfdbf", "#fc8961", "#51127c", "#060518"]
    cols = ["#ff0000", "#00ff00", "#0000ff"]
    let color_vectors = cols.map(c => chroma(c).oklab());

    let new_colors = [];
    let polynomials = getBernsteinPolynomial(color_vectors.length);

    for (let i = 0; i < paletteCount; i++) {
        let t = i / paletteCount;
        let new_col = [0, 0, 0];

        for (let j = 0; j < color_vectors.length; j++) {
            new_col = new_col.map((val, k) => val + color_vectors[j][k] * polynomials[j](t));
        }

        new_colors.push(chroma.oklab(new_col));
    }

    color_strings = new_colors.map(s => "\'" + s.hex().replace(/^#/, "#") + "\'")
    console.log("[" + color_strings.join(", ") + "]");

    return new_colors;
}

function generateBezierHue(cols, paletteCount) {
    const satBoost = document.getElementById("saturation-boost");
    // cols = ["#fef7a6", cols[0], "#190423"]

    //cols = ["#060518", "#b73779", "#fcfdbf"]
    //cols = ["#fcfdbf", "#b73779", "#060518"]

    let color_vectors = cols.map(c => chroma(c).oklch());

    for (let i = 0; i < color_vectors.length - 1; i++) {
        if (color_vectors[i + 1][2] > color_vectors[i][2]) {
            color_vectors[i][2] += 360;
        }
    }

    // color_vectors = [
    //     [0, color_vectors[0][1], color_vectors[0][2]],
    //     color_vectors[0],
    //     color_vectors[1],
    //     [1, color_vectors[1][1], color_vectors[1][2]]
    // ] // Math.min(color_vectors[1][0] * 4, 1)


    let new_colors = [];
    let polynomials = getBernsteinPolynomial(color_vectors.length);

    for (let i = 0; i < paletteCount; i++) {
        let t = i / paletteCount;
        let new_color = [0, 0, 0];

        for (let j = 0; j < color_vectors.length; j++) {
            new_color = [new_color[0] + color_vectors[j][0] * polynomials[j](t),
            new_color[1] + color_vectors[j][1] * polynomials[j](t),
            new_color[2] + color_vectors[j][2] * polynomials[j](t)
            ];
        }

        new_color[1] = new_color[1] * (satBoost.value / 100);

        new_colors.push(chroma.oklch(new_color));
    }

    color_strings = new_colors.map(s => "\'" + s.hex().replace(/^#/, "#") + "\'")
    console.log("[" + color_strings.join(", ") + "]");

    return new_colors;
}


function createPanels(colors) {
    const palettes = document.getElementById("palette");
    palettes.innerHTML = "";

    colors.forEach((color) => {
        const panel = document.createElement("div");
        const colorDisplay = document.createElement("span");
        colorDisplay.classList.add("color-display");

        const contrastColor = chroma.contrast(color, chroma(color).darken(3));
        const textColor = contrastColor > 2 ? chroma(color).darken(3) : chroma(color).luminance(3);

        panel.style.setProperty("--text-color", textColor);
        colorDisplay.appendChild(document.createTextNode(color));
        panel.appendChild(colorDisplay);

        panel.classList.add("panel");
        panel.style.setProperty("--palette-color", colorBlindSim(color, blindnessMode.value));
        palettes.appendChild(panel);
    });
}

// Gradients just don't work well with pure black
function blackFix(col) {
    if (col == "#000000") {
        return "#010101";
    } else {
        return col;
    }
}

window.addEventListener("load", run);
